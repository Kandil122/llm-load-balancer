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
    parser.add_argument("--no-fault", action="store_true",
                        help="Disable fault simulation")
    parser.add_argument("--save-results", action="store_true",
                        help="Save results to CSV")
    return parser.parse_args()


async def main():
    args = parse_args()

    print("=" * 60)
    print("  DISTRIBUTED LLM LOAD BALANCER")
    print(f"  Strategy : {args.strategy}")
    print(f"  Workers  : {args.workers}")
    print(f"  Users    : {args.users}")
    print(f"  Model    : {config.ollama_model}")
    print("=" * 60)

    # Override config with CLI args
    config.worker_failure_simulation = not args.no_fault

    # Step 1: Initialize RAG
    print("\n📚 Initializing RAG knowledge base...")
    get_collection()

    # Step 2: Create workers
    workers = [GPUWorker(i) for i in range(args.workers)]

    # Step 3: Create load balancer
    lb = STRATEGIES[args.strategy]()

    # Step 4: Create metrics collector
    collector = MetricsCollector(num_workers=args.workers)

    # Step 5: Create scheduler
    scheduler = Scheduler(lb, workers, collector)
    await scheduler.start()

    # Step 6: Run load test
    responses = await run_load_test(scheduler, collector, num_users=args.users)

    # Step 7: Stop scheduler
    await scheduler.stop()

    # Step 8: Print summary
    print_summary(workers, collector, args.strategy)

    # Step 9: Save results
    if args.save_results:
        os.makedirs("results", exist_ok=True)
        filename = f"results/{args.strategy}_{args.users}users_{args.workers}workers.csv"
        summary = collector.get_summary()
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in summary.items():
                writer.writerow([k, v])
        print(f"\n💾 Results saved to {filename}")


if __name__ == "__main__":
    asyncio.run(main())
