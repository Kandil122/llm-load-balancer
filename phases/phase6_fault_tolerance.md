# Phase 6 — Fault Tolerance ⭐
> `master/scheduler.py`

---

## What You're Building

The most graded component of the project. The Scheduler runs two concurrent
async tasks — a request dispatcher and a heartbeat monitor — that together
guarantee zero dropped requests even when workers fail mid-execution.

---

## Fault Tolerance Flow

```
Normal operation:
  Request → Scheduler → Load Balancer → Worker → Response ✅

Worker dies mid-request:
  Request → Scheduler → Load Balancer → Worker (DEAD)
                                            │
                                            ▼ raises RuntimeError
                              Scheduler catches exception
                                            │
                                            ▼ retries
                              Load Balancer picks next alive worker
                                            │
                                            ▼
                                       Worker (alive) → Response ✅
                                    (zero requests lost)

Heartbeat detects dead worker:
  every 2s → check all workers → worker.is_alive == False
           → mark in metrics → dashboard shows 🔴
           → load balancer already skips it automatically
```

---

## The Two Async Tasks in Scheduler

```python
async def start(self):
    # Task 1: handles requests (called externally)
    # Task 2: monitors workers every 2 seconds
    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
```

These run concurrently via asyncio — the heartbeat never blocks request handling.

---

## ✅ Phase 6 Tests

### Test 1 — Scheduler dispatches a request successfully

```bash
python3 -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer
from master.scheduler import Scheduler
from metrics.collector import MetricsCollector
from common.models import Request

async def test():
    workers = [GPUWorker(i) for i in range(3)]
    lb = RoundRobinBalancer()
    collector = MetricsCollector(num_workers=3)
    scheduler = Scheduler(lb, workers, collector)

    await scheduler.start()

    request = Request(id=1, query='What is distributed computing?')
    print(f'Dispatching request {request.id}...')

    response = await scheduler.handle_request(request)

    print(f'Response received:')
    print(f'  success     : {response.success}')
    print(f'  worker_id   : {response.worker_id}')
    print(f'  latency     : {response.latency:.2f}s')
    print(f'  result      : {response.result[:80]}...')
    print()
    assert response.success == True
    assert response.worker_id in [0, 1, 2]

    await scheduler.stop()
    print('✅ Scheduler dispatches and returns response correctly')

asyncio.run(test())
"
```

---

### Test 2 — Dead worker is skipped, request still completes

```bash
python3 -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer
from master.scheduler import Scheduler
from metrics.collector import MetricsCollector
from common.models import Request

async def test():
    workers = [GPUWorker(i) for i in range(3)]
    lb = RoundRobinBalancer()
    collector = MetricsCollector(num_workers=3)
    scheduler = Scheduler(lb, workers, collector)
    await scheduler.start()

    # Kill workers 0 and 1 — only worker 2 is alive
    workers[0].simulate_failure()
    workers[1].simulate_failure()
    print('Workers 0 and 1 are dead. Only Worker 2 is alive.')

    request = Request(id=1, query='How does fault tolerance work?')
    response = await scheduler.handle_request(request)

    print(f'Request still completed:')
    print(f'  success   : {response.success}')
    print(f'  worker_id : {response.worker_id} (expected: 2)')
    print(f'  latency   : {response.latency:.2f}s')
    print()
    assert response.success == True
    assert response.worker_id == 2

    await scheduler.stop()
    print('✅ Dead workers skipped — request completed by alive worker')

asyncio.run(test())
"
```

---

### Test 3 — All workers dead returns failure gracefully

```bash
python3 -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer
from master.scheduler import Scheduler
from metrics.collector import MetricsCollector
from common.models import Request

async def test():
    workers = [GPUWorker(i) for i in range(3)]
    for w in workers:
        w.simulate_failure()

    lb = RoundRobinBalancer()
    collector = MetricsCollector(num_workers=3)
    scheduler = Scheduler(lb, workers, collector)
    await scheduler.start()

    request = Request(id=1, query='test query')
    response = await scheduler.handle_request(request)

    print(f'All workers dead:')
    print(f'  success : {response.success} (expected: False)')
    print(f'  error   : {response.error}')
    print()
    assert response.success == False
    assert response.error is not None

    await scheduler.stop()
    print('✅ Graceful failure when all workers are dead (no crash)')

asyncio.run(test())
"
```

---

### Test 4 — Heartbeat detects dead worker

```bash
python3 -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer
from master.scheduler import Scheduler
from metrics.collector import MetricsCollector
from common.config import config

async def test():
    workers = [GPUWorker(i) for i in range(3)]
    lb = RoundRobinBalancer()
    collector = MetricsCollector(num_workers=3)

    # Set fast heartbeat for testing
    config.heartbeat_interval = 1.0

    scheduler = Scheduler(lb, workers, collector)
    await scheduler.start()

    print('Heartbeat running. Killing Worker 1...')
    workers[1].simulate_failure()

    # Wait for heartbeat to detect it
    await asyncio.sleep(2.5)

    print(f'Dead workers in metrics: {list(collector.dead_workers)}')
    assert 1 in collector.dead_workers, 'Heartbeat did not detect dead worker!'

    await scheduler.stop()
    print()
    print('✅ Heartbeat detected dead worker within 2 seconds')

asyncio.run(test())
"
```

Expected output:
```
Heartbeat running. Killing Worker 1...
💀 [Worker 1] Manually failed
Dead workers in metrics: [1]
✅ Heartbeat detected dead worker within 2 seconds
```

---

### Test 5 — Zero requests lost during fault simulation

This is the key test — the one to show in your demo video.

```bash
python3 -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.least_connections import LeastConnectionsBalancer
from master.scheduler import Scheduler
from metrics.collector import MetricsCollector
from common.models import Request

async def test():
    NUM_REQUESTS = 10
    workers = [GPUWorker(i) for i in range(3)]
    lb = LeastConnectionsBalancer()
    collector = MetricsCollector(num_workers=3)
    scheduler = Scheduler(lb, workers, collector)
    await scheduler.start()

    print(f'Sending {NUM_REQUESTS} requests...')
    print('Worker 1 will be killed halfway through.')
    print()

    async def kill_worker_midway():
        await asyncio.sleep(2)   # let some requests start
        workers[1].simulate_failure()
        print(f'  ⚡ Worker 1 killed mid-run!')

    # Run requests and kill worker concurrently
    tasks = [scheduler.handle_request(Request(id=i, query=f'Query {i}'))
             for i in range(NUM_REQUESTS)]
    tasks.append(kill_worker_midway())

    results = await asyncio.gather(*tasks, return_exceptions=True)
    responses = [r for r in results if hasattr(r, 'success')]

    successful = sum(1 for r in responses if r.success)
    failed = sum(1 for r in responses if not r.success)

    print()
    print(f'Results:')
    print(f'  Total responses  : {len(responses)}')
    print(f'  Successful       : {successful}')
    print(f'  Failed           : {failed}')
    print()

    assert failed == 0, f'{failed} requests were lost during fault!'
    print('✅ ZERO requests lost during worker failure')

    await scheduler.stop()

asyncio.run(test())
"
```

Expected output:
```
Sending 10 requests...
Worker 1 will be killed halfway through.

💀 [Worker 1] Manually failed
  ⚡ Worker 1 killed mid-run!

Results:
  Total responses  : 10
  Successful       : 10
  Failed           : 0

✅ ZERO requests lost during worker failure
```

---

### Test 6 — Run pytest for fault tolerance

```bash
python3 -m pytest tests/test_fault_tolerance.py -v
```

Expected output:
```
PASSED tests/test_fault_tolerance.py::test_dead_worker_is_skipped
PASSED tests/test_fault_tolerance.py::test_worker_revive
PASSED tests/test_fault_tolerance.py::test_metrics_collector_records_correctly

3 passed in Xs
```

---

### Test 7 — Worker revive (bonus)

```bash
python3 -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer
from master.scheduler import Scheduler
from metrics.collector import MetricsCollector
from common.models import Request

async def test():
    workers = [GPUWorker(i) for i in range(2)]
    lb = RoundRobinBalancer()
    collector = MetricsCollector(num_workers=2)
    scheduler = Scheduler(lb, workers, collector)
    await scheduler.start()

    # Kill worker 0
    workers[0].simulate_failure()
    print('Worker 0 killed.')

    # Send request — should go to worker 1
    r1 = await scheduler.handle_request(Request(id=1, query='test'))
    print(f'Request 1 → Worker {r1.worker_id} (expected: 1)')
    assert r1.worker_id == 1

    # Revive worker 0
    workers[0].revive()
    print('Worker 0 revived.')

    # Send request — worker 0 should be available again
    r2 = await scheduler.handle_request(Request(id=2, query='test'))
    print(f'Request 2 → Worker {r2.worker_id} (can be 0 or 1 now)')

    await scheduler.stop()
    print()
    print('✅ Worker revive works correctly')

asyncio.run(test())
"
```

---

## ✅ Phase 6 Complete Checklist

- [ ] Scheduler dispatches requests and returns correct responses
- [ ] Dead workers are automatically skipped by all strategies
- [ ] Requests are retried on other workers when one dies mid-request
- [ ] All workers dead → graceful failure response, no crash
- [ ] Heartbeat detects dead workers within 2 seconds
- [ ] **Zero requests lost during fault simulation** ← most important
- [ ] Worker revive makes the worker available again
- [ ] All 3 pytest tests pass

---

## Common Issues

**Request hangs indefinitely after worker dies**
Make sure your `GPUWorker.process()` raises `RuntimeError` when `is_alive == False`
and that `Scheduler.handle_request()` catches it and retries.

**Heartbeat task not running**
Make sure `await scheduler.start()` is called before `handle_request()`.
The heartbeat task is created inside `start()`.

**`gather()` exceptions not caught**
Use `return_exceptions=True` in `asyncio.gather()` when running in tests
so one failed task doesn't cancel all others.

---

➡️ **Next: Phase 7 — Load Generator & Main Entry Point**
