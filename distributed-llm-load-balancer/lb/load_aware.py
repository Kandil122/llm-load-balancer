# lb/load_aware.py
from typing import List, Optional
from lb.load_balancer import LoadBalancer
from workers.gpu_worker import GPUWorker


class LoadAwareBalancer(LoadBalancer):
    """
    Picks the worker with the lowest load score.
    Score = active_connections * avg_latency (lower is better).
    """

    async def get_next_worker(self, workers: List[GPUWorker]) -> Optional[GPUWorker]:
        alive = self.alive_workers(workers)
        if not alive:
            return None
        return min(alive, key=lambda w: w.status.load_score)
