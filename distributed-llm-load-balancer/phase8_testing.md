# Phase 8 — Testing & Final Validation
> Full system test · pytest · Demo prep · Report data

---

## What You're Doing

Running the complete test suite, validating the full system end to end,
collecting all numbers for your report, and preparing for the demo video.

---

## ✅ Phase 8 Tests

### Test 1 — Run the complete pytest suite

```bash
uv run pytest tests/ -v --tb=short
```

Expected output:
```
tests/test_load_balancer.py::test_round_robin_cycles              PASSED
tests/test_load_balancer.py::test_round_robin_skips_dead          PASSED
tests/test_load_balancer.py::test_least_connections_picks_lowest  PASSED
tests/test_load_balancer.py::test_load_aware_picks_lowest_score   PASSED
tests/test_load_balancer.py::test_all_dead_returns_none           PASSED
tests/test_fault_tolerance.py::test_dead_worker_is_skipped        PASSED
tests/test_fault_tolerance.py::test_worker_revive                 PASSED
tests/test_fault_tolerance.py::test_metrics_collector_records     PASSED
tests/test_rag.py::test_collection_initializes                    PASSED
tests/test_rag.py::test_retriever_returns_context                 PASSED
tests/test_rag.py::test_retriever_relevance                       PASSED

11 passed in Xs
```

**All 11 must pass before proceeding.**

---

### Test 2 — Full system smoke test (5 users, no fault)

```bash
uv run python main.py --strategy round_robin --users 5 --workers 4 --no-fault
```

Verify:
- All 5 requests complete successfully
- Each response contains real text from gemma3:270m
- Latency numbers are shown
- No errors or tracebacks

---

### Test 3 — Full fault tolerance demo run

This is the exact sequence to record for your demo video:

```bash
# Terminal 1: Watch GPU live
watch -n 1 nvidia-smi

# Terminal 2: Run with fault simulation
uv run python main.py --strategy least_connections --users 10 --workers 4
```

Verify in the output:
```
💀 [Worker 1] SIMULATED FAILURE after 50 requests
⚠️  [Scheduler] Worker 1 failed on request X, retrying...

FINAL RESULTS
─────────────────────────────
Completed    : 10
Failed       : 0           ← this must be 0
Dead Workers : [1]
```

---

### Test 4 — All three strategies produce different results

```bash
# Run each and note avg latency + throughput
uv run python main.py --strategy round_robin       --users 8 --workers 4 --no-fault --save-results
uv run python main.py --strategy least_connections --users 8 --workers 4 --no-fault --save-results
uv run python main.py --strategy load_aware        --users 8 --workers 4 --no-fault --save-results

# View saved results
for f in results/*.csv; do echo "=== $f ==="; cat $f; echo; done
```

---

### Test 5 — Stub mode load test (for large user counts)

For your report you need numbers at 100+ users. Use stub mode so
you're testing the distributed system logic, not waiting for real LLM calls:

```bash
# Add to .env temporarily
echo "LLM_STUB=1" >> .env

# Run large load tests
LLM_STUB=1 uv run python main.py --strategy round_robin       --users 100 --workers 4 --no-fault --save-results
LLM_STUB=1 uv run python main.py --strategy least_connections --users 100 --workers 4 --no-fault --save-results
LLM_STUB=1 uv run python main.py --strategy load_aware        --users 100 --workers 4 --no-fault --save-results

LLM_STUB=1 uv run python main.py --strategy round_robin       --users 500 --workers 4 --no-fault --save-results
LLM_STUB=1 uv run python main.py --strategy least_connections --users 500 --workers 4 --no-fault --save-results
LLM_STUB=1 uv run python main.py --strategy load_aware        --users 500 --workers 4 --no-fault --save-results

# Remove stub mode after
sed -i '/LLM_STUB/d' .env
```

> **Note for report:** Be transparent that large-scale load tests use a
> simulated inference stub to isolate and measure the distributed system
> layer independently from LLM inference time. This is standard practice
> in distributed systems research.

---

### Test 6 — Verify GPU utilization logging

```bash
# Log GPU stats during a real LLM run
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu \
           --format=csv --loop=2 > results/gpu_log.csv &
GPU_PID=$!

uv run python main.py --strategy round_robin --users 5 --workers 4 --no-fault

kill $GPU_PID
echo ""
echo "GPU log during inference:"
cat results/gpu_log.csv
```

Record the peak GPU utilization from this log for your report.

---

### Test 7 — Repository check

```bash
# Make sure .env is not committed
git status

# Initialize git if not done yet
git init
git add .
git status

# Verify .env is in .gitignore (should not appear in git add)
grep ".env" .gitignore
```

---

## 📊 Report Data Collection

Fill this table by running Test 4 and Test 5 results:

```
┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Strategy            │  5 users │ 10 users │100 users*│500 users*│
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Round Robin         │          │          │          │          │
│   avg latency       │  ____s   │  ____s   │  ____s   │  ____s   │
│   throughput        │ __req/s  │ __req/s  │ __req/s  │ __req/s  │
│   p95 latency       │  ____s   │  ____s   │  ____s   │  ____s   │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Least Connections   │          │          │          │          │
│   avg latency       │  ____s   │  ____s   │  ____s   │  ____s   │
│   throughput        │ __req/s  │ __req/s  │ __req/s  │ __req/s  │
│   p95 latency       │  ____s   │  ____s   │  ____s   │  ____s   │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Load Aware          │          │          │          │          │
│   avg latency       │  ____s   │  ____s   │  ____s   │  ____s   │
│   throughput        │ __req/s  │ __req/s  │ __req/s  │ __req/s  │
│   p95 latency       │  ____s   │  ____s   │  ____s   │  ____s   │
└─────────────────────┴──────────┴──────────┴──────────┴──────────┘
* stub mode (LLM_STUB=1)
```

Also record:
```
Hardware:
  GPU              : NVIDIA GeForce GTX 960M
  VRAM             : 4096 MB
  Peak GPU util    : ____%
  Peak VRAM usage  : ____MB
  Peak GPU temp    : ____°C
  CPU during test  : ____%

Fault Tolerance:
  Workers killed   : 1 (Worker 1)
  Requests lost    : 0
  Recovery time    : < 2s (heartbeat interval)
```

---

## 🎬 Demo Video Script

Record in this order — total should be 5–8 minutes:

**Scene 1 — System startup (30s)**
```bash
# Show Ollama running
ollama list
ollama ps   # shows gemma3:270m loaded

# Show nvidia-smi idle
nvidia-smi
```

**Scene 2 — RAG pipeline demo (45s)**
```bash
uv run python -c "
import asyncio
from rag.retriever import retrieve_context
from llm.inference import run_llm

async def demo():
    query = 'How does load balancing improve system performance?'
    print(f'User query: {query}')
    ctx = await retrieve_context(query)
    print(f'RAG retrieved: {ctx[:200]}...')
    result = await run_llm(query, ctx)
    print(f'LLM answer: {result}')

asyncio.run(demo())
"
```

**Scene 3 — Load test with live dashboard (2min)**
```bash
# Split screen: nvidia-smi on left, this on right
watch -n 1 nvidia-smi &
uv run python main.py --strategy round_robin --users 8 --workers 4 --no-fault
```

**Scene 4 — Strategy comparison (1min)**
```bash
uv run python main.py --strategy round_robin       --users 5 --workers 4 --no-fault
uv run python main.py --strategy least_connections --users 5 --workers 4 --no-fault
uv run python main.py --strategy load_aware        --users 5 --workers 4 --no-fault
# Point out latency differences between strategies
```

**Scene 5 — Fault tolerance live kill (2min)**
```bash
uv run python main.py --strategy least_connections --users 10 --workers 4
# Wait for 💀 Worker 1 SIMULATED FAILURE message
# Show zero failed requests in summary
```

**Scene 6 — Final summary (30s)**
```bash
# Show results CSV files
ls results/
cat results/round_robin_8users_4workers.csv
```

---

## ✅ Final Complete Checklist

### Code
- [ ] All 11 pytest tests pass
- [ ] `main.py --help` works
- [ ] All 3 strategies selectable via CLI
- [ ] Fault simulation works (worker dies, zero failures)
- [ ] Results save to CSV
- [ ] `.env` not committed to git

### Performance
- [ ] Real LLM responses are coherent and relevant
- [ ] GPU utilization visible during inference
- [ ] Latency numbers recorded for all strategies
- [ ] Stub mode tested for 100+ users

### Report Must Include
- [ ] System architecture diagram
- [ ] Explanation of 3 load balancing strategies with comparison
- [ ] RAG pipeline explanation with ChromaDB diagram
- [ ] Fault tolerance flow diagram
- [ ] Performance table (latency, throughput, P95 per strategy)
- [ ] GPU utilization numbers from real tests
- [ ] Discussion of asyncio vs threading design choice
- [ ] Limitations section
- [ ] References (include the GitHub repos from research)

### Demo Video
- [ ] Shows Ollama + gemma3:270m running locally
- [ ] Shows real GPU utilization in nvidia-smi
- [ ] Shows all 3 load balancing strategies
- [ ] Shows fault tolerance (worker dies, zero dropped requests)
- [ ] Shows real AI responses end to end
- [ ] Under 10 minutes total

---

## Common Issues at This Stage

**Some pytest tests fail after code changes**
Run with `-s` to see print output:
```bash
uv run pytest tests/ -v -s
```

**GPU utilization stays at 0% during inference**
Ollama may have fallen back to CPU. Fix:
```bash
# Check Ollama GPU detection
OLLAMA_DEBUG=1 ollama run gemma3:270m "hi" 2>&1 | grep -i gpu
```

**Demo video: terminal font too small**
```bash
# Increase terminal font to 16pt minimum before recording
# Use a dark theme (dark background, light text) for better visibility
```

---

## 🎓 You're Done!

```
✅ Phase 1 — Foundation (models + config)
✅ Phase 2 — Workers + Load Balancer (3 strategies)
✅ Phase 3 — RAG Pipeline (ChromaDB + embeddings)
✅ Phase 4 — LLM Integration (gemma3:270m via Ollama)
✅ Phase 5 — Metrics & Dashboard (rich live UI)
✅ Phase 6 — Fault Tolerance (heartbeat + retry)
✅ Phase 7 — Load Generator (asyncio.gather)
✅ Phase 8 — Testing & Validation (11 tests pass)
```

Good luck with the demo! 🚀
