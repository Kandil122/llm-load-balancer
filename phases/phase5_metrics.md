# Phase 5 — Metrics & Dashboard
> `metrics/collector.py` · `metrics/dashboard.py`

---

## What You're Building

A thread-safe metrics collector that aggregates all system statistics,
and a live Rich terminal dashboard that displays them in real time
alongside real hardware utilization from `nvidia-smi` and `psutil`.

---

## What the Dashboard Looks Like

```
╭──────────────── DISTRIBUTED LLM SYSTEM ─────────────────╮
│  Strategy: round_robin   Workers: 4   Users: 100         │
├──────────┬────────────┬─────────────┬──────────┬─────────┤
│ Worker   │ Status     │ Active Conn │ Processed│Avg Lat  │
├──────────┼────────────┼─────────────┼──────────┼─────────┤
│ Worker-0 │ 🟢 alive   │ 3           │ 28       │ 3.21s   │
│ Worker-1 │ 🔴 dead    │ 0           │ 12       │ 3.45s   │
│ Worker-2 │ 🟢 alive   │ 2           │ 31       │ 3.18s   │
│ Worker-3 │ 🟢 alive   │ 4           │ 29       │ 3.27s   │
╰──────────┴────────────┴─────────────┴──────────┴─────────╯
GPU: 87%  VRAM: 1823MB/4096MB  Temp: 74°C
CPU: 43%  RAM: 6.2GB / 16GB
Completed: 87  Failed: 0  Avg: 3.27s  P95: 4.1s  RPS: 2.3
```

---

## ✅ Phase 5 Tests

### Test 1 — MetricsCollector basic operations

```bash
python3 -c "
from metrics.collector import MetricsCollector

collector = MetricsCollector(num_workers=4)

# Simulate some results
collector.start_timer()
collector.record_success(0, 1.2)
collector.record_success(0, 0.9)
collector.record_success(1, 2.1)
collector.record_success(2, 1.5)
collector.record_failure(3)
collector.record_failure(3)
collector.record_request_done()
collector.mark_worker_dead(3)
import time; time.sleep(0.1)
collector.stop_timer()

print(f'Completed    : {collector.completed}')
print(f'Failed       : {collector.failed}')
print(f'Avg latency  : {collector.avg_latency:.3f}s')
print(f'P95 latency  : {collector.p95_latency:.3f}s')
print(f'Throughput   : {collector.throughput:.2f} req/s')
print(f'Dead workers : {list(collector.dead_workers)}')
print()
print('Full summary:')
import json
print(json.dumps(collector.get_summary(), indent=2))
print()
print('✅ MetricsCollector works correctly')
"
```

Expected output:
```
Completed    : 4
Failed       : 2
Avg latency  : 1.425s
P95 latency  : 2.100s
Throughput   : ~40.0 req/s
Dead workers : [3]

Full summary:
{
  "total": 1,
  "completed": 4,
  "failed": 2,
  ...
}
✅ MetricsCollector works correctly
```

---

### Test 2 — Thread-safety under concurrent writes

```bash
python3 -c "
import asyncio
import threading
from metrics.collector import MetricsCollector

collector = MetricsCollector(num_workers=4)

# Simulate 100 threads writing simultaneously
def write_metrics(worker_id, n):
    for _ in range(n):
        collector.record_success(worker_id % 4, 1.0)

threads = [threading.Thread(target=write_metrics, args=(i, 25)) for i in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f'Total recorded: {collector.completed} (expected: 100)')
assert collector.completed == 100, f'Race condition! Got {collector.completed}'
print()
print('✅ Thread-safe: no race conditions under 100 concurrent writes')
"
```

---

### Test 3 — GPU stats function

```bash
python3 -c "
from metrics.dashboard import get_gpu_stats
import psutil

gpu_info = get_gpu_stats()
print(f'GPU stats: {gpu_info}')

cpu = psutil.cpu_percent(interval=1)
ram = psutil.virtual_memory()
print(f'CPU usage: {cpu}%')
print(f'RAM: {ram.used // (1024**3)}GB used / {ram.total // (1024**3)}GB total')
print()
print('✅ Hardware stats readable')
"
```

Expected output:
```
GPU stats: GPU: 0%  VRAM: 5MB/4096MB  Temp: 44°C
CPU usage: 12.3%
RAM: 8GB used / 16GB total
✅ Hardware stats readable
```

---

### Test 4 — Dashboard table renders correctly

```bash
python3 -c "
from workers.gpu_worker import GPUWorker
from metrics.collector import MetricsCollector
from metrics.dashboard import build_table, print_summary
from rich.console import Console

# Setup
workers = [GPUWorker(i) for i in range(4)]
collector = MetricsCollector(num_workers=4)

# Simulate some data
workers[0].status.total_processed = 28
workers[0].status.avg_latency = 3.21
workers[0].status.active_connections = 3

workers[1].status.total_processed = 12
workers[1].status.avg_latency = 3.45
workers[1].simulate_failure()

workers[2].status.total_processed = 31
workers[2].status.avg_latency = 3.18
workers[2].status.active_connections = 2

workers[3].status.total_processed = 29
workers[3].status.avg_latency = 3.27
workers[3].status.active_connections = 4

collector.record_success(0, 3.21)
collector.record_success(2, 3.18)
collector.record_failure(1)

# Render
console = Console()
table = build_table(workers, collector, 'round_robin')
console.print(table)
print()
print('✅ Dashboard table renders correctly')
"
```

---

### Test 5 — Full summary output

```bash
python3 -c "
from workers.gpu_worker import GPUWorker
from metrics.collector import MetricsCollector
from metrics.dashboard import print_summary
import time

workers = [GPUWorker(i) for i in range(4)]
collector = MetricsCollector(num_workers=4)

# Simulate completed run
collector.start_timer()
for i in range(20):
    collector.record_success(i % 4, 2.5 + (i * 0.1))
    collector.record_request_done()
collector.record_failure(1)
collector.mark_worker_dead(1)
time.sleep(0.1)
collector.stop_timer()

workers[1].simulate_failure()
workers[1].status.total_processed = 5

print_summary(workers, collector, 'round_robin')
"
```

---

### Test 6 — P95 latency calculation is correct

```bash
python3 -c "
from metrics.collector import MetricsCollector

collector = MetricsCollector(num_workers=2)

# Add 100 latency values: 0.1, 0.2, ..., 10.0
for i in range(100):
    collector.record_success(i % 2, (i + 1) * 0.1)

p95 = collector.p95_latency
avg = collector.avg_latency

print(f'100 requests, latencies from 0.1s to 10.0s')
print(f'Avg latency : {avg:.2f}s (expected ~5.05s)')
print(f'P95 latency : {p95:.2f}s (expected ~9.50s)')

assert 5.0 <= avg <= 5.1, f'Wrong avg: {avg}'
assert 9.4 <= p95 <= 9.6, f'Wrong P95: {p95}'
print()
print('✅ P95 latency calculated correctly')
"
```

---

## ✅ Phase 5 Complete Checklist

- [ ] `MetricsCollector` tracks completed, failed, latencies correctly
- [ ] Thread-safe: 100 concurrent writes produce exact count
- [ ] `avg_latency` and `p95_latency` calculated correctly
- [ ] `throughput` (req/s) calculated from start/stop timer
- [ ] `get_gpu_stats()` reads from `nvidia-smi` successfully
- [ ] `psutil` reads CPU% and RAM correctly
- [ ] Dashboard table renders with correct worker status colors
- [ ] Dead workers show 🔴, alive show 🟢
- [ ] `print_summary()` shows complete final report

---

## Common Issues

**`nvidia-smi not found` in GPU stats**
The `get_gpu_stats()` function silently falls back to `"GPU: N/A"` if
`nvidia-smi` fails. Since you have the GTX 960M this should work.
If not, check:
```bash
which nvidia-smi
nvidia-smi
```

**Race condition in thread-safety test**
If Test 2 gives a number other than 100, there's a missing `threading.Lock()`
in `metrics/collector.py`. Check the `_lock` is used in every write method.

**Rich rendering issues in terminal**
```bash
# If colors look wrong, force rich to use basic mode:
TERM=xterm-256color python3 -c "..."
```

---

➡️ **Next: Phase 6 — Fault Tolerance**
