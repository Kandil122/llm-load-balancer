# lb/load_balancer.py
from abc import ABC, abstractmethod
from typing import List, Optional
from workers.gpu_worker import GPUWorker


class LoadBalancer(ABC):
    """Abstract base class for all load balancing strategies."""

    @abstractmethod
    async def get_next_worker(self, workers: List[GPUWorker]) -> Optional[GPUWorker]:
        """Select the next worker to handle a request."""
        pass

    def alive_workers(self, workers: List[GPUWorker]) -> List[GPUWorker]:
        """Filter and return only alive workers."""
        return [w for w in workers if w.status.is_alive]
