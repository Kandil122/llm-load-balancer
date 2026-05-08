# lb/round_robin.py
import asyncio
from typing import List, Optional
from lb.load_balancer import LoadBalancer
from workers.gpu_worker import GPUWorker


class RoundRobinBalancer(LoadBalancer):
    """Cycles through workers in order, skipping dead ones."""

    def __init__(self):
        self._index = 0
        self._lock = asyncio.Lock()

    async def get_next_worker(self, workers: List[GPUWorker]) -> Optional[GPUWorker]:
        alive = self.alive_workers(workers)
        if not alive:
            return None
        async with self._lock:
            worker = alive[self._index % len(alive)]
            self._index = (self._index + 1) % len(alive)
        return worker
