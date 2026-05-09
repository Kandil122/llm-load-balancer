# main.py
import asyncio
import argparse
import csv
import os
from common.config import config
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer
from lb.least_connections import LeastConnectionsBalancer
from lb.load_aware import LoadAwareBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test
from metrics.collector import MetricsCollector
from metrics.dashboard import print_summary
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
                        help="Save results to CSV")
    return parser.parse_args()


async def run_distributed_load_test(collector, num_users, strategy, workers_count, no_fault):
    """Sends requests to the Master Docker container with live monitoring and clean session management."""
    import aiohttp
    from client.load_generator import SAMPLE_QUERIES
    from master.http_scheduler import HttpScheduler
    from metrics.dashboard import build_table, console
    from rich.live import Live
    
    print(f"🚀 [Distributed] Attempting to connect to Master at http://localhost:8000")
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        use_hybrid = False
        try:
            async with session.get("http://localhost:8000/workers/status", timeout=2) as resp:
                if resp.status != 200:
                    raise Exception("Master container returned unhealthy status")
        except Exception as e:
            print(f"\n⚠️  [FAILOVER] Master container unreachable or unhealthy: {e}")
            print(f"🔄 Switching to HYBRID MODE (Local Master + Docker Workers)...")
            use_hybrid = True

        if use_hybrid:
            worker_urls = [f"http://localhost:{8001 + i}" for i in range(workers_count)]
            print(f"📡 Local Master connecting to workers directly...")
            
            scheduler = HttpScheduler(urls=worker_urls)
            scheduler._session = session 
            heartbeat_task = asyncio.create_task(scheduler.heartbeat_loop())

            async def get_hybrid_status():
                status = []
                for i, url in enumerate(scheduler.worker_urls):
                    try:
                        async with session.get(f"{url}/metrics", timeout=0.5) as r:
                            status.append(await r.json())
                    except:
                        status.append({"worker_id": i, "url": url, "is_alive": False})
                return status

            collector.start_timer()
            with Live(build_table([], status_data=[], mode="HYBRID MASTER", current_counts=collector.worker_counts), console=console, refresh_per_second=2) as live:
                async def update_live():
                    while not collector._end_time:
                        live.update(build_table([], status_data=await get_hybrid_status(), mode="HYBRID MASTER", current_counts=collector.worker_counts))
                        await asyncio.sleep(0.5)

                update_task = asyncio.create_task(update_live())
                
                sem = asyncio.Semaphore(workers_count * 2) 

                async def wrap_task(i):
                    async with sem:
                        query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
                        res = await scheduler.handle(i, query)
                        if isinstance(res, dict) and res.get("success"):
                            collector.record_success(res.get("worker_id", 0), res.get("latency", 0))
                        else:
                            collector.record_failure(-1)
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
                        return await r.json()
                except: return []

            collector.start_timer()
            with Live(build_table([], status_data=[], mode="CONTAINER MASTER", current_counts=collector.worker_counts), console=console, refresh_per_second=2) as live:
                async def update_live():
                    while not collector._end_time:
                        data = await get_remote_status()
                        if data: live.update(build_table([], status_data=data, mode="CONTAINER MASTER", current_counts=collector.worker_counts))
                        await asyncio.sleep(0.5)

                update_task = asyncio.create_task(update_live())
                sem = asyncio.Semaphore(workers_count * 2)

                async def wrap_request(i):
                    async with sem:
                        url = "http://localhost:8000/request"
                        query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
                        try:
                            async with session.post(url, json={"id": i, "query": query}) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    collector.record_success(data.get("worker_id", 0), data.get("latency", 0))
                                    collector.record_request_done()
                                    return data
                        except: pass
                        collector.record_failure(-1)
                        collector.record_request_done()
                        return None

                tasks = [wrap_request(i) for i in range(num_users)]
                await asyncio.gather(*tasks, return_exceptions=True)
                collector.stop_timer()
                await update_task
                
    collector.stop_timer()


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

    if args.distributed:
        # Distributed Mode with Failover
        await run_distributed_load_test(collector, args.users, args.strategy, args.workers, args.no_fault)
    else:
        # Simulation Mode: Local Workers
        print("\n📚 Initializing RAG knowledge base...")
        get_collection()
        
        config.worker_failure_simulation = not args.no_fault
        workers = [GPUWorker(i) for i in range(args.workers)]
        lb = STRATEGIES[args.strategy]()
        scheduler = Scheduler(lb, workers, collector)
        
        await scheduler.start()
        await run_load_test(scheduler, collector, num_users=args.users)
        await scheduler.stop()
        
    # Step 2: Print summary (for both modes)
    print_summary(workers, collector, args.strategy)

    # Step 3: Save results
    if args.save_results:
        os.makedirs("results", exist_ok=True)
        filename = f"results/{'dist' if args.distributed else 'sim'}_{args.strategy}_{args.users}users.csv"
        summary = collector.get_summary()
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in summary.items():
                writer.writerow([k, v])
        print(f"\n💾 Results saved to {filename}")


if __name__ == "__main__":
    asyncio.run(main())
