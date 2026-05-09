# tests/test_load_balancer.py
import asyncio
import pytest
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer
from lb.least_connections import LeastConnectionsBalancer
from lb.load_aware import LoadAwareBalancer


def make_workers(n=4):
    return [GPUWorker(i) for i in range(n)]


@pytest.mark.asyncio
async def test_round_robin_cycles():
    workers = make_workers(3)
    lb = RoundRobinBalancer()
    selected = [await lb.get_next_worker(workers) for _ in range(6)]
    ids = [w.id for w in selected]
    assert ids == [0, 1, 2, 0, 1, 2]


@pytest.mark.asyncio
async def test_round_robin_skips_dead():
    workers = make_workers(3)
    workers[1].status.is_alive = False
    lb = RoundRobinBalancer()
    for _ in range(10):
        w = await lb.get_next_worker(workers)
        assert w.id != 1


@pytest.mark.asyncio
async def test_least_connections_picks_lowest():
    workers = make_workers(3)
    workers[0].status.active_connections = 5
    workers[1].status.active_connections = 2
    workers[2].status.active_connections = 8
    lb = LeastConnectionsBalancer()
    w = await lb.get_next_worker(workers)
    assert w.id == 1


@pytest.mark.asyncio
async def test_load_aware_picks_lowest_score():
    workers = make_workers(3)
    workers[0].status.active_connections = 3
    workers[0].status.avg_latency = 2.0   # score = 6.0
    workers[1].status.active_connections = 1
    workers[1].status.avg_latency = 1.0   # score = 1.0  ← lowest
    workers[2].status.active_connections = 4
    workers[2].status.avg_latency = 1.5   # score = 6.0
    lb = LoadAwareBalancer()
    w = await lb.get_next_worker(workers)
    assert w.id == 1


@pytest.mark.asyncio
async def test_all_dead_returns_none():
    workers = make_workers(3)
    for w in workers:
        w.status.is_alive = False
    lb = RoundRobinBalancer()
    result = await lb.get_next_worker(workers)
    assert result is None
