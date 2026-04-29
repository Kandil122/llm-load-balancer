# client/load_generator.py
import asyncio
import time
from typing import List
from common.models import Request
from master.scheduler import Scheduler
from metrics.collector import MetricsCollector

SAMPLE_QUERIES = [
    "What is load balancing and why is it important?",
    "How does round robin scheduling work?",
    "Explain fault tolerance in distributed systems.",
    "What is the difference between latency and throughput?",
    "How do GPU clusters handle parallel inference?",
    "What is RAG and how does it improve LLM responses?",
    "How does the least connections algorithm work?",
    "What happens when a node fails in a distributed system?",
    "Explain the role of a master node in a cluster.",
    "What is ChromaDB used for?",
]


async def simulate_user(user_id: int, scheduler: Scheduler, collector: MetricsCollector):
    """A single simulated user sending one request."""
    query = SAMPLE_QUERIES[user_id % len(SAMPLE_QUERIES)]
    request = Request(id=user_id, query=query)

    response = await scheduler.handle_request(request)
    collector.record_request_done()

    return response


async def run_load_test(scheduler: Scheduler, collector: MetricsCollector,
                        num_users: int = 100) -> List:
    """
    Simulate num_users concurrent users using asyncio.gather.
    Returns list of all responses.
    """
    print(f"\n🚀 Starting load test with {num_users} concurrent users...")
    collector.start_timer()

    tasks = [
        simulate_user(i, scheduler, collector)
        for i in range(num_users)
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    collector.stop_timer()

    return responses
