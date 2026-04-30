# Phase 7 — Load Generator & Main Entry Point
> `client/load_generator.py` · `main.py`

---

## What You're Building

The client simulator that fires concurrent users using `asyncio.gather()`,
and the main entry point that wires everything together with CLI arguments
for switching strategies, user counts, and worker counts.

---

## Why asyncio.gather() and Not Threads

```python
# ❌ Threading approach (wrong for this project)
threads = [Thread(target=simulate_user, args=(i,)) for i in range(1000)]
# Problem: Python GIL limits true parallelism
# Problem: 1000 threads = massive memory overhead
# Problem: thread synchronization complexity

# ✅ asyncio.gather() (correct approach)
tasks = [simulate_user(i, scheduler, collector) for i in range(1000)]
responses = await asyncio.gather(*tasks)
# 1000 coroutines share one thread — lightweight
# Perfect for I/O-bound work like HTTP calls to Ollama
# No GIL issues for async I/O
```

This distinction is worth a paragraph in your report.

---

## ✅ Phase 7 Tests

### Test 1 — Load generator with 5 users (smoke test)

```bash
python3 -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer
from master.scheduler import Scheduler
from metrics.collector import MetricsCollector
from client.load_generator import run_load_test
from common.config import config

# Disable fault simulation for clean smoke test
config.worker_failure_simulation = False

async def test():
    NUM_USERS = 5
    workers = [GPUWorker(i) for i in range(2)]
    lb = RoundRobinBalancer()
    collector = MetricsCollector(num_workers=2)
    scheduler = Scheduler(lb, workers, collector)
    await scheduler.start()

    responses = await run_load_test(scheduler, collector, num_users=NUM_USERS)

    print(f'Total responses      : {len(responses)}')
    successful = sum(1 for r in responses if hasattr(r, 'success') and r.success)
    print(f'Successful responses : {successful}')

    for r in responses:
        if hasattr(r, 'worker_id'):
            print(f'  Request {r.id:3d} → Worker {r.worker_id} | {r.latency:.2f}s')

    await scheduler.stop()
    print()
    print('✅ Load generator works with 5 users')

asyncio.run(test())
"
```

---

### Test 2 — Verify requests distributed across workers

```bash
python3 -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer
from master.scheduler import Scheduler
from metrics.collector import MetricsCollector
from client.load_generator import run_load_test
from common.config import config

config.worker_failure_simulation = False

async def test():
    NUM_USERS = 8
    NUM_WORKERS = 4
    workers = [GPUWorker(i) for i in range(NUM_WORKERS)]
    lb = RoundRobinBalancer()
    collector = MetricsCollector(num_workers=NUM_WORKERS)
    scheduler = Scheduler(lb, workers, collector)
    await scheduler.start()

    responses = await run_load_test(scheduler, collector, num_users=NUM_USERS)

    # Count per worker
    from collections import Counter
    worker_counts = Counter(r.worker_id for r in responses if hasattr(r, 'worker_id'))
    print(f'Requests per worker (Round Robin, {NUM_USERS} users, {NUM_WORKERS} workers):')
    for wid, count in sorted(worker_counts.items()):
        bar = '█' * count
        print(f'  Worker {wid}: {bar} ({count})')

    # All workers should have received ~equal load
    counts = list(worker_counts.values())
    spread = max(counts) - min(counts)
    print(f'Load spread (max - min): {spread} (should be ≤ 1 for round robin)')
    assert spread <= 1

    await scheduler.stop()
    print()
    print('✅ Load distributed evenly across workers')

asyncio.run(test())
"
```

---

### Test 3 — main.py CLI with round_robin

```bash
# Start with small user count first (real LLM calls are slow)
python3 main.py --strategy round_robin --users 5 --workers 3 --no-fault
```

Expected output:
```
============================================================
  DISTRIBUTED LLM LOAD BALANCER
  Strategy : round_robin
  Workers  : 3
  Users    : 5
  Model    : llama3.2:1b
============================================================

📚 Initializing RAG knowledge base...
✅ ChromaDB already has 30 documents

🚀 Starting load test with 5 concurrent users...

[Scheduler] Dispatching requests...
...

══════════════════ FINAL RESULTS ══════════════════
Strategy          round_robin
Total Requests    5
Completed         5
Failed            0
Avg Latency       3.4s
P95 Latency       4.1s
Throughput        0.89 req/s
Dead Workers      []
...
```

---

### Test 4 — Compare all three strategies

```bash
# Run each strategy with 5 users and compare results
for strategy in round_robin least_connections load_aware; do
    echo "==============================="
    echo "Strategy: $strategy"
    echo "==============================="
    python3 main.py --strategy $strategy --users 5 --workers 3 --no-fault
    echo ""
    sleep 2
done
```

Record the avg latency and throughput for each strategy — this is your report table data.

---

### Test 5 — Fault simulation via main.py

```bash
# Run WITH fault simulation enabled (worker 1 dies after 50 requests)
# Use enough users that worker 1 actually hits the limit
python3 main.py --strategy round_robin --users 10 --workers 3
```

Watch for:
```
💀 [Worker 1] SIMULATED FAILURE after 50 requests
⚠️  [Scheduler] Worker 1 failed on request X, retrying (attempt 1)...
```

And in the final summary:
```
Dead Workers      [1]
Failed            0        ← zero dropped despite worker death
```

---

### Test 6 — Save results to CSV

```bash
python3 main.py --strategy round_robin --users 5 --workers 4 \
    --no-fault --save-results

ls results/
cat results/round_robin_5users_4workers.csv
```

Expected output:
```
metric,value
total,5
completed,5
failed,0
avg_latency,3.42
p95_latency,4.1
throughput,0.88
worker_counts,"{'0': 2, '1': 1, '2': 1, '3': 1}"
dead_workers,[]
```

---

### Test 7 — Help text works

```bash
python3 main.py --help
```

Expected output:
```
usage: main.py [-h] [--strategy {round_robin,least_connections,load_aware}]
               [--users USERS] [--workers WORKERS] [--no-fault] [--save-results]

Distributed LLM Load Balancer

options:
  --strategy    Load balancing strategy (default: round_robin)
  --users       Number of concurrent users (default: 20)
  --workers     Number of GPU workers (default: 4)
  --no-fault    Disable fault simulation
  --save-results Save results to CSV in results/ folder
```

---

## Ramp-Up Load Test (for Report Data)

Run this script to collect all your report table numbers.
**Warning: with real LLM calls this will take time. Start small.**

```bash
#!/bin/bash
# save as run_benchmarks.sh and run: bash run_benchmarks.sh

WORKERS=4

for STRATEGY in round_robin least_connections load_aware; do
    for USERS in 5 10 20; do
        echo "Running: strategy=$STRATEGY users=$USERS workers=$WORKERS"
        python3 main.py \
            --strategy $STRATEGY \
            --users $USERS \
            --workers $WORKERS \
            --no-fault \
            --save-results
        sleep 3
    done
done

echo ""
echo "All benchmarks complete. Results in results/"
ls results/
```

---

## ✅ Phase 7 Complete Checklist

- [ ] Load generator fires N coroutines with `asyncio.gather()`
- [ ] All users receive a response (none silently dropped)
- [ ] Round Robin distributes ≤ 1 request difference across workers
- [ ] `main.py --help` shows all arguments
- [ ] `main.py --strategy round_robin --users 5 --workers 3 --no-fault` runs cleanly
- [ ] All three strategies run without errors
- [ ] Fault simulation triggers correctly (worker 1 dies, zero failures)
- [ ] `--save-results` creates CSV in `results/` folder
- [ ] Results are different between strategies (latency varies)

---

## Common Issues

**Very slow with many users**
With real LLM calls, 20 users × ~4s per request = only 5 req/s.
For load testing beyond 20 users, consider adding a stub mode that
skips Ollama and just sleeps for a fixed time:

```python
# In llm/inference.py — add a stub mode
import os
if os.getenv("LLM_STUB") == "1":
    await asyncio.sleep(0.5)  # simulate inference
    return f"Stub response for: {query}"
```

Then test:
```bash
LLM_STUB=1 python3 main.py --strategy round_robin --users 100 --workers 4
```

**`asyncio.gather` returns exceptions as values**
The load generator uses `return_exceptions=True` — exceptions appear
as items in the responses list. Filter them with:
```python
real_responses = [r for r in responses if not isinstance(r, Exception)]
```

---

➡️ **Next: Phase 8 — Testing & Final Validation**
