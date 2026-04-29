# tests/test_fault_tolerance.py
import asyncio
import pytest
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer
from metrics.collector import MetricsCollector
from master.scheduler import Scheduler


@pytest.mark.asyncio
async def test_dead_worker_is_skipped():
    workers = [GPUWorker(i) for i in range(3)]
    workers[1].simulate_failure()

    lb = RoundRobinBalancer()
    alive = lb.alive_workers(workers)

    assert len(alive) == 2
    assert all(w.id != 1 for w in alive)


@pytest.mark.asyncio
async def test_worker_revive():
    worker = GPUWorker(0)
    worker.simulate_failure()
    assert not worker.status.is_alive
    worker.revive()
    assert worker.status.is_alive


def test_metrics_collector_records_correctly():
    collector = MetricsCollector(num_workers=4)
    collector.record_success(0, 1.2)
    collector.record_success(1, 0.8)
    collector.record_failure(2)

    assert collector.completed == 2
    assert collector.failed == 1
    assert abs(collector.avg_latency - 1.0) < 0.01
