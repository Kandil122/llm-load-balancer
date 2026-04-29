# metrics/collector.py
import time
import threading
from typing import Dict, List


class MetricsCollector:
    def __init__(self, num_workers: int):
        self._lock = threading.Lock()
        self.num_workers = num_workers

        # Counters
        self.total_requests = 0
        self.completed = 0
        self.failed = 0
        self.latencies: List[float] = []

        # Per-worker
        self.worker_counts: Dict[int, int] = {i: 0 for i in range(num_workers)}
        self.worker_failures: Dict[int, int] = {i: 0 for i in range(num_workers)}
        self.dead_workers = set()

        # Timing
        self._start_time = None
        self._end_time = None

    def start_timer(self):
        self._start_time = time.time()

    def stop_timer(self):
        self._end_time = time.time()

    def record_success(self, worker_id: int, latency: float):
        with self._lock:
            self.completed += 1
            self.latencies.append(latency)
            if worker_id in self.worker_counts:
                self.worker_counts[worker_id] += 1

    def record_failure(self, worker_id: int):
        with self._lock:
            self.failed += 1
            if worker_id in self.worker_failures:
                self.worker_failures[worker_id] += 1

    def record_request_done(self):
        with self._lock:
            self.total_requests += 1

    def mark_worker_dead(self, worker_id: int):
        with self._lock:
            self.dead_workers.add(worker_id)

    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def throughput(self) -> float:
        if self._start_time is None or self._end_time is None:
            return 0.0
        elapsed = self._end_time - self._start_time
        return self.completed / elapsed if elapsed > 0 else 0.0

    def get_summary(self) -> dict:
        return {
            "total": self.total_requests,
            "completed": self.completed,
            "failed": self.failed,
            "avg_latency": round(self.avg_latency, 3),
            "p95_latency": round(self.p95_latency, 3),
            "throughput": round(self.throughput, 2),
            "worker_counts": self.worker_counts,
            "dead_workers": list(self.dead_workers),
        }
