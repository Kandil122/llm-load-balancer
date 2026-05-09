# lb/least_connections.py
from typing import List, Optional
from lb.load_balancer import LoadBalancer
from workers.gpu_worker import GPUWorker


class LeastConnectionsBalancer(LoadBalancer):
    """Picks the worker with the fewest active connections."""

    async def get_next_worker(self, workers: List[GPUWorker]) -> Optional[GPUWorker]:
        alive = self.alive_workers(workers)
        if not alive:
            return None
        return min(alive, key=lambda w: w.status.active_connections)
