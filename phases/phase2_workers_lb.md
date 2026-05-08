# Phase 2 — Workers + Load Balancer
> `workers/gpu_worker.py` · `lb/load_balancer.py` · `lb/round_robin.py` · `lb/least_connections.py` · `lb/load_aware.py`

---

## What You're Building

The worker nodes that process requests, and the three load balancing
strategies that decide which worker handles each request.
This is the core distributed systems logic of the project.

---

## Architecture of This Phase

```
Incoming Request
      │
      ▼
┌─────────────────────────────────────┐
│         Load Balancer               │
│                                     │
│  filters: [w for w if w.is_alive]  │
│                                     │
│  Strategy 1: Round Robin            │
│  Strategy 2: Least Connections      │
│  Strategy 3: Load Aware             │
└──────────┬──────────────────────────┘
           │ selects one worker
           ▼
    ┌─────────────┐
    │  GPUWorker  │
    │  .process() │  ← increments active_connections
    │             │  ← calls RAG + LLM (stubbed for now)
    │             │  ← decrements active_connections
    └─────────────┘
```

---

## The Strategy Pattern

All three load balancers inherit from the same abstract base class.
This means you can swap strategies with one line change — this is what
makes your architecture clean and worth explaining in the report.

```python
# Abstract base — one method to implement
class LoadBalancer(ABC):
    async def get_next_worker(self, workers) -> GPUWorker: ...

# Three concrete implementations
class RoundRobinBalancer(LoadBalancer): ...
class LeastConnectionsBalancer(LoadBalancer): ...
class LoadAwareBalancer(LoadBalancer): ...
```

---

## ✅ Phase 2 Tests

### Test 1 — Workers instantiate correctly

```bash
python3 -c "
from workers.gpu_worker import GPUWorker

workers = [GPUWorker(i) for i in range(4)]

for w in workers:
    print(f'Worker {w.id}: alive={w.status.is_alive}, '
          f'connections={w.status.active_connections}, '
          f'processed={w.status.total_processed}')

print()
print('✅ All workers created successfully')
"
```

Expected output:
```
Worker 0: alive=True, connections=0, processed=0
Worker 1: alive=True, connections=0, processed=0
Worker 2: alive=True, connections=0, processed=0
Worker 3: alive=True, connections=0, processed=0
✅ All workers created successfully
```

---

### Test 2 — Round Robin distributes evenly

```bash
python3 -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer

async def test():
    workers = [GPUWorker(i) for i in range(4)]
    lb = RoundRobinBalancer()

    selected = []
    for _ in range(8):
        w = await lb.get_next_worker(workers)
        selected.append(w.id)

    print(f'Round Robin sequence (8 requests, 4 workers):')
    print(f'  {selected}')
    assert selected == [0, 1, 2, 3, 0, 1, 2, 3], 'Round robin not cycling correctly!'
    print()
    print('✅ Round Robin cycles correctly')

asyncio.run(test())
"
```

Expected output:
```
Round Robin sequence (8 requests, 4 workers):
  [0, 1, 2, 3, 0, 1, 2, 3]
✅ Round Robin cycles correctly
```

---

### Test 3 — Round Robin skips dead workers

```bash
python3 -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer

async def test():
    workers = [GPUWorker(i) for i in range(4)]
    workers[1].simulate_failure()  # kill worker 1
    workers[3].simulate_failure()  # kill worker 3

    lb = RoundRobinBalancer()
    selected = []
    for _ in range(6):
        w = await lb.get_next_worker(workers)
        selected.append(w.id)

    print(f'Workers 1 and 3 are dead.')
    print(f'Selected workers: {selected}')
    assert 1 not in selected, 'Dead worker 1 was selected!'
    assert 3 not in selected, 'Dead worker 3 was selected!'
    print()
    print('✅ Dead workers are correctly skipped')

asyncio.run(test())
"
```

---

### Test 4 — Least Connections picks lowest

```bash
python3 -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.least_connections import LeastConnectionsBalancer

async def test():
    workers = [GPUWorker(i) for i in range(4)]

    # Simulate different loads
    workers[0].status.active_connections = 8
    workers[1].status.active_connections = 2   # ← should be picked
    workers[2].status.active_connections = 5
    workers[3].status.active_connections = 6

    lb = LeastConnectionsBalancer()
    selected = await lb.get_next_worker(workers)

    print(f'Active connections: {[w.status.active_connections for w in workers]}')
    print(f'Selected worker: {selected.id} (expected: 1)')
    assert selected.id == 1
    print()
    print('✅ Least Connections picks worker with fewest active connections')

asyncio.run(test())
"
```

---

### Test 5 — Load Aware picks by score

```bash
python3 -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.load_aware import LoadAwareBalancer

async def test():
    workers = [GPUWorker(i) for i in range(3)]

    # Worker 0: 3 connections × 2.0s latency = score 6.0
    workers[0].status.active_connections = 3
    workers[0].status.avg_latency = 2.0

    # Worker 1: 1 connection × 1.0s latency = score 1.0  ← best
    workers[1].status.active_connections = 1
    workers[1].status.avg_latency = 1.0

    # Worker 2: 4 connections × 1.5s latency = score 6.0
    workers[2].status.active_connections = 4
    workers[2].status.avg_latency = 1.5

    lb = LoadAwareBalancer()
    selected = await lb.get_next_worker(workers)

    scores = [w.status.load_score for w in workers]
    print(f'Load scores: {[round(s, 2) for s in scores]}')
    print(f'Selected worker: {selected.id} (expected: 1 with score {scores[1]:.1f})')
    assert selected.id == 1
    print()
    print('✅ Load Aware picks worker with lowest load score')

asyncio.run(test())
"
```

---

### Test 6 — All workers dead returns None

```bash
python3 -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer

async def test():
    workers = [GPUWorker(i) for i in range(3)]
    for w in workers:
        w.simulate_failure()

    lb = RoundRobinBalancer()
    result = await lb.get_next_worker(workers)

    print(f'All workers dead. Result: {result}')
    assert result is None
    print()
    print('✅ Returns None when all workers are dead (no crash)')

asyncio.run(test())
"
```

---

### Test 7 — Run pytest for load balancer

```bash
python3 -m pytest tests/test_load_balancer.py -v```

Expected output:
```
PASSED tests/test_load_balancer.py::test_round_robin_cycles
PASSED tests/test_load_balancer.py::test_round_robin_skips_dead
PASSED tests/test_load_balancer.py::test_least_connections_picks_lowest
PASSED tests/test_load_balancer.py::test_load_aware_picks_lowest_score
PASSED tests/test_load_balancer.py::test_all_dead_returns_none

5 passed in Xs
```

---

### Test 8 — Compare strategies side by side

```bash
python3 -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer
from lb.least_connections import LeastConnectionsBalancer
from lb.load_aware import LoadAwareBalancer

async def test():
    # Give workers different loads
    def make_loaded_workers():
        workers = [GPUWorker(i) for i in range(4)]
        workers[0].status.active_connections = 1
        workers[0].status.avg_latency = 0.5
        workers[1].status.active_connections = 8
        workers[1].status.avg_latency = 3.0
        workers[2].status.active_connections = 3
        workers[2].status.avg_latency = 1.0
        workers[3].status.active_connections = 2
        workers[3].status.avg_latency = 0.8
        return workers

    strategies = {
        'Round Robin      ': RoundRobinBalancer(),
        'Least Connections': LeastConnectionsBalancer(),
        'Load Aware       ': LoadAwareBalancer(),
    }

    print('Strategy comparison (same worker loads):')
    print(f'  Workers active_conn : [1, 8, 3, 2]')
    print(f'  Workers avg_latency : [0.5, 3.0, 1.0, 0.8]')
    print()

    for name, lb in strategies.items():
        workers = make_loaded_workers()
        selected = await lb.get_next_worker(workers)
        print(f'  {name} → Worker {selected.id}')

asyncio.run(test())
"
```

Expected output (your results may vary on round robin):
```
Strategy comparison (same worker loads):
  Workers active_conn : [1, 8, 3, 2]
  Workers avg_latency : [0.5, 3.0, 1.0, 0.8]

  Round Robin       → Worker 0
  Least Connections → Worker 0
  Load Aware        → Worker 0
```

---

## ✅ Phase 2 Complete Checklist

- [ ] Workers instantiate with correct default status values
- [ ] Round Robin cycles through all workers in order
- [ ] Round Robin correctly skips dead workers
- [ ] Least Connections always picks the worker with fewest connections
- [ ] Load Aware picks the worker with lowest `connections × latency` score
- [ ] All strategies return `None` (not crash) when all workers are dead
- [ ] `simulate_failure()` sets `is_alive = False`
- [ ] `revive()` sets `is_alive = True`
- [ ] All 5 pytest tests pass

---

## Common Issues

**`ImportError: cannot import name GPUWorker`**
Make sure all `__init__.py` files exist:
```bash
ls common/__init__.py lb/__init__.py workers/__init__.py
```

**`asyncio.run() error`**
Make sure you are calling with `asyncio.run()` wrapping an `async def` function.

**Round Robin not cycling correctly**
The `asyncio.Lock()` in `RoundRobinBalancer.__init__()` must be created there,
not at class level, because locks belong to the running event loop.

---

➡️ **Next: Phase 3 — RAG Pipeline**
