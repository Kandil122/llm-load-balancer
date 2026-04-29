# common/models.py
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class Request:
    id: int
    query: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Response:
    id: int
    result: str
    latency: float
    worker_id: int
    success: bool = True
    error: Optional[str] = None


@dataclass
class WorkerStatus:
    id: int
    is_alive: bool = True
    active_connections: int = 0
    avg_latency: float = 0.0
    total_processed: int = 0
    total_failed: int = 0
    latency_sum: float = 0.0

    def update_latency(self, latency: float):
        self.total_processed += 1
        self.latency_sum += latency
        self.avg_latency = self.latency_sum / self.total_processed

    @property
    def load_score(self) -> float:
        """Used by load-aware strategy: lower is better."""
        return self.active_connections * max(self.avg_latency, 0.1)
