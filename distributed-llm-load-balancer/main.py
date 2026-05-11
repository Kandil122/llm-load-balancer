# main.py
import asyncio
import argparse
import csv
import os
import time
from datetime import datetime
from common.config import config
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer
from lb.least_connections import LeastConnectionsBalancer
from lb.load_aware import LoadAwareBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test
from metrics.collector import MetricsCollector
from metrics.dashboard import print_summary, build_dashboard, console
from rag.indexer import get_collection


STRATEGIES = {
    "round_robin": RoundRobinBalancer,
    "least_connections": LeastConnectionsBalancer,
    "load_aware": LoadAwareBalancer,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed LLM Load Balancer")
    parser.add_argument("--strategy", choices=STRATEGIES.keys(),
                        default=config.lb_strategy)
    parser.add_argument("--users", type=int, default=config.num_users)
    parser.add_argument("--workers", type=int, default=config.num_workers)
    parser.add_argument("--distributed", action="store_true",
                        help="Run against Docker containers")
    parser.add_argument("--no-fault", action="store_true",
                        help="Disable fault simulation")
    parser.add_argument("--save-results", action="store_true",
                        help="Save metrics to CSV")
    parser.add_argument("--save-output", action="store_true",
                        help="Save LLM text responses with timing to file")
    return parser.parse_args()


def format_time(ts):
    return datetime.fromtimestamp(ts).strftime('%H:%M:%S.%f')[:-3]


async def run_distributed_load_test(collector, num_users, strategy, workers_count, no_fault):
    """Sends requests to the Master Docker container with live monitoring and real-time events."""
    import aiohttp
    from client.load_generator import SAMPLE_QUERIES
    from master.http_scheduler import HttpScheduler
    from rich.live import Live
    
    print(f"🚀 [Distributed] Attempting to connect to Master at http://localhost:8000")
    
    events = ["System starting. Connecting to cluster..."]
    last_worker_states = {i: True for i in range(workers_count)} 
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        use_hybrid = False
        try:
            async with session.get("http://localhost:8000/workers/status", timeout=2) as resp:
                if resp.status == 200:
                    events.append("Master container connected.")
                else:
                    raise Exception("Master returned error status")
        except Exception as e:
            events.append("Master container unreachable. Switching to HYBRID MODE.")
            use_hybrid = True

        responses_data = []

        if use_hybrid:
            worker_urls = [f"http://localhost:{8001 + i}" for i in range(workers_count)]
            scheduler = HttpScheduler(urls=worker_urls)
            scheduler._session = session 
            heartbeat_task = asyncio.create_task(scheduler.heartbeat_loop())

            async def get_hybrid_status():
                status = []
                for i, url in enumerate(scheduler.worker_urls):
                    try:
                        async with session.get(f"{url}/metrics", timeout=0.5) as r:
                            data = await r.json()
                            alive = data.get("is_alive", False)
                            if i in last_worker_states and last_worker_states[i] != alive:
                                events.append(f"Worker {i} {'RECOVERED' if alive else 'FAILED'}!")
                            last_worker_states[i] = alive
                            status.append(data)
                    except:
                        if last_worker_states.get(i, False):
                            events.append(f"Worker {i} is UNREACHABLE!")
                        last_worker_states[i] = False
                        status.append({"worker_id": i, "url": url, "is_alive": False, "error": "unreachable"})
                return status

            collector.start_timer()
            with Live(build_dashboard([], mode="HYBRID MASTER", current_counts=collector.worker_counts, events=events), 
                      console=console, refresh_per_second=2) as live:
                
                async def update_live():
                    while not collector._end_time:
                        status_data = await get_hybrid_status()
                        live.update(build_dashboard([], status_data=status_data, mode="HYBRID MASTER", 
                                                   current_counts=collector.worker_counts, events=events))
                        await asyncio.sleep(0.5)

                update_task = asyncio.create_task(update_live())
                sem = asyncio.Semaphore(workers_count * 2) 

                async def wrap_task(i):
                    async with sem:
                        query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
                        sent_at = time.time()
                        res = await scheduler.handle(i, query)
                        recv_at = time.time()
                        
                        if isinstance(res, dict) and res.get("success"):
                            collector.record_success(res.get("worker_id", 0), res.get("latency", 0))
                            responses_data.append({
                                "id": i, "query": query, "response": res.get("result", ""), 
                                "worker": res.get("worker_id"), "sent_at": sent_at, "recv_at": recv_at, "lat": recv_at - sent_at
                            })
                        else:
                            collector.record_failure(-1)
                            responses_data.append({
                                "id": i, "query": query, "response": "FAILED", 
                                "worker": -1, "sent_at": sent_at, "recv_at": recv_at, "lat": recv_at - sent_at
                            })
                        collector.record_request_done()
                        return res

                tasks = [wrap_task(i) for i in range(num_users)]
                await asyncio.gather(*tasks, return_exceptions=True)
                collector.stop_timer()
                await update_task
                
            for i, is_alive in scheduler._alive.items():
                if not is_alive:
                    collector.mark_worker_dead(i)
            heartbeat_task.cancel()
        else:
            # Normal Distributed Mode
            async def get_remote_status():
                try:
                    async with session.get("http://localhost:8000/workers/status", timeout=1) as r:
                        data = await r.json()
                        for i, d in enumerate(data):
                            w_id = d.get("worker_id", i)
                            alive = d.get("is_alive", False) if "error" not in d else False
                            if w_id in last_worker_states and last_worker_states[w_id] != alive:
                                events.append(f"Worker {w_id} {'RECOVERED' if alive else 'FAILED'}!")
                            last_worker_states[w_id] = alive
                        return data
                except: return []

            collector.start_timer()
            with Live(build_dashboard([], mode="CONTAINER MASTER", current_counts=collector.worker_counts, events=events), 
                      console=console, refresh_per_second=2) as live:
                
                async def update_live():
                    while not collector._end_time:
                        data = await get_remote_status()
                        if data: 
                            live.update(build_dashboard([], status_data=data, mode="CONTAINER MASTER", 
                                                       current_counts=collector.worker_counts, events=events))
                        await asyncio.sleep(0.5)

                update_task = asyncio.create_task(update_live())
                sem = asyncio.Semaphore(workers_count * 2)

                async def wrap_request(i):
                    async with sem:
                        url = "http://localhost:8000/request"
                        query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
                        sent_at = time.time()
                        try:
                            async with session.post(url, json={"id": i, "query": query}) as resp:
                                recv_at = time.time()
                                if resp.status == 200:
                                    data = await resp.json()
                                    collector.record_success(data.get("worker_id", 0), data.get("latency", 0))
                                    responses_data.append({
                                        "id": i, "query": query, "response": data.get("result", ""), 
                                        "worker": data.get("worker_id"), "sent_at": sent_at, "recv_at": recv_at, "lat": recv_at - sent_at
                                    })
                                    collector.record_request_done()
                                    return data
                        except: pass
                        recv_at = time.time()
                        collector.record_failure(-1)
                        responses_data.append({
                            "id": i, "query": query, "response": "FAILED", 
                            "worker": -1, "sent_at": sent_at, "recv_at": recv_at, "lat": recv_at - sent_at
                        })
                        collector.record_request_done()
                        return None

                tasks = [wrap_request(i) for i in range(num_users)]
                await asyncio.gather(*tasks, return_exceptions=True)
                collector.stop_timer()
                await update_task
                
    collector.stop_timer()
    return responses_data


async def main():
    args = parse_args()

    print("=" * 60)
    print("  DISTRIBUTED LLM LOAD BALANCER")
    print(f"  Mode     : {'DISTRIBUTED (Docker)' if args.distributed else 'SIMULATION (Local)'}")
    print(f"  Strategy : {args.strategy}")
    print(f"  Workers  : {args.workers}")
    print(f"  Users    : {args.users}")
    print(f"  Model    : {config.ollama_model}")
    print("=" * 60)

    # Step 1: Initialize Metrics
    collector = MetricsCollector(num_workers=args.workers)
    workers = []
    final_outputs = []

    if args.distributed:
        # Distributed Mode with Failover
        final_outputs = await run_distributed_load_test(collector, args.users, args.strategy, args.workers, args.no_fault)
    else:
        # Simulation Mode: Local Workers
        from metrics.dashboard import build_dashboard, console
        from rich.live import Live
        
        print("\n📚 Initializing RAG knowledge base...")
        get_collection()
        
        config.worker_failure_simulation = not args.no_fault
        workers = [GPUWorker(i) for i in range(args.workers)]
        lb = STRATEGIES[args.strategy]()
        scheduler = Scheduler(lb, workers, collector)
        
        await scheduler.start()
        
        print(f"🚀 Starting local simulation with {args.users} users...")
        
        events = ["Simulation initialized."]
        last_worker_states = {w.id: True for w in workers}

        with Live(build_dashboard(workers, mode="SIMULATION", current_counts=collector.worker_counts, events=events), 
                  console=console, refresh_per_second=2) as live:
            
            async def update_live():
                while not collector._end_time:
                    for w in workers:
                        if last_worker_states[w.id] != w.status.is_alive:
                            label = "RECOVERED" if w.status.is_alive else "FAILED"
                            events.append(f"Worker {w.id} {label}!")
                            last_worker_states[w.id] = w.status.is_alive
                    
                    live.update(build_dashboard(workers, mode="SIMULATION", current_counts=collector.worker_counts, events=events))
                    await asyncio.sleep(0.5)

            update_task = asyncio.create_task(update_live())
            sem = asyncio.Semaphore(args.workers * 2)
            
            async def wrap_sim_task(i):
                async with sem:
                    from client.load_generator import SAMPLE_QUERIES, Request
                    query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
                    request = Request(id=i, query=query)
                    sent_at = time.time()
                    resp = await scheduler.handle_request(request)
                    recv_at = time.time()
                    if resp.success:
                        final_outputs.append({
                            "id": i, "query": query, "response": resp.result, "worker": resp.worker_id,
                            "sent_at": sent_at, "recv_at": recv_at, "lat": recv_at - sent_at
                        })
                    else:
                        final_outputs.append({
                            "id": i, "query": query, "response": "FAILED", "worker": -1,
                            "sent_at": sent_at, "recv_at": recv_at, "lat": recv_at - sent_at
                        })
                    collector.record_request_done()
                    return resp

            tasks = [wrap_sim_task(i) for i in range(args.users)]
            collector.start_timer()
            await asyncio.gather(*tasks, return_exceptions=True)
            collector.stop_timer()
            await update_task

        await scheduler.stop()
        
    # Step 2: Print summary (for both modes)
    print_summary(workers, collector, args.strategy)

    # Step 3: Save results (Metrics)
    if args.save_results:
        os.makedirs("results", exist_ok=True)
        filename = f"results/{'dist' if args.distributed else 'sim'}_{args.strategy}_{args.users}users_metrics.csv"
        summary = collector.get_summary()
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in summary.items():
                writer.writerow([k, v])
        print(f"\n💾 Metrics saved to {filename}")

    # Step 4: Save outputs (LLM Responses + Tracing)
    if args.save_output:
        os.makedirs("results", exist_ok=True)
        filename = f"results/{'dist' if args.distributed else 'sim'}_{args.strategy}_{args.users}users_outputs.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["request_id", "query", "worker_id", "sent_at", "received_at", "total_latency", "llm_response"])
            for out in sorted(final_outputs, key=lambda x: x['id']):
                writer.writerow([
                    out['id'], 
                    out['query'], 
                    out['worker'], 
                    format_time(out['sent_at']), 
                    format_time(out['recv_at']), 
                    f"{out['lat']:.3f}s",
                    out['response']
                ])
        print(f"💾 Detailed LLM Tracing saved to {filename}")


if __name__ == "__main__":
    asyncio.run(main())
