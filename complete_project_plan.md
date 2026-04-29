# Distributed LLM Load Balancer — Complete Project Plan
### CSE354: Distributed Computing — 2nd Semester 2025/2026
### Ain Shams University · Faculty of Engineering

---

> **Project Title:** Efficient Load Balancing and GPU Cluster Task Distribution for Handling 1000+ Concurrent LLM Requests
> **Model:** llama3.2:1b via Ollama (local, no API key)
> **Hardware:** NVIDIA GeForce GTX 960M 4GB · i7-4720HQ · 16GB RAM · Linux

---

## Table of Contents

1. [Tech Stack](#tech-stack)
2. [Hardware Setup](#hardware-setup)
3. [Project Structure](#project-structure)
4. [System Architecture](#system-architecture)
5. [Request Lifecycle](#request-lifecycle)
6. [Setup Steps](#setup-steps)
7. [Phase 1 — Foundation](#phase-1--foundation)
8. [Phase 2 — Workers + Load Balancer](#phase-2--workers--load-balancer)
9. [Phase 3 — RAG Pipeline](#phase-3--rag-pipeline)
10. [Phase 4 — LLM Integration](#phase-4--llm-integration)
11. [Phase 5 — Metrics & Dashboard](#phase-5--metrics--dashboard)
12. [Phase 6 — Fault Tolerance](#phase-6--fault-tolerance)
13. [Phase 7 — Load Generator & Main](#phase-7--load-generator--main)
14. [Phase 8 — Testing & Final Validation](#phase-8--testing--final-validation)
15. [Report Data Table](#report-data-table)
16. [Demo Video Checklist](#demo-video-checklist)
17. [Key Points for Report](#key-points-for-report)
18. [GitHub Repository](#github-repository)

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Package manager | **uv** | Fastest, modern, handles venvs automatically |
| Local LLM Runner | **Ollama** | Runs LLMs locally, REST API at `localhost:11434` |
| LLM Model | **llama3.2:1b** | Fits in 4GB VRAM, fast enough for demo |
| GPU Monitoring | **gputil + psutil** | Real hardware utilization metrics |
| RAG Vector DB | **ChromaDB** | Lightweight, runs locally, no server needed |
| Embeddings | **sentence-transformers** | Free, local, `all-MiniLM-L6-v2` model |
| Concurrency | **asyncio** | Correct for I/O-bound async work |
| HTTP Client | **aiohttp** | Async HTTP calls to Ollama REST API |
| Metrics UI | **rich** | Live terminal dashboard |
| Config | **pydantic-settings** | Type-safe env/config management |
| Testing | **pytest + pytest-asyncio** | Async-friendly testing |

---

## Hardware Setup

### Primary Machine (Lenovo Y50-70) — Run the system here
```
CPU    : Intel i7-4720HQ @ 2.60GHz
RAM    : 16GB
GPU    : NVIDIA GeForce GTX 960M — 4GB VRAM ✅
CUDA   : 13.0 ✅ (Ollama uses GPU automatically)
Driver : 580.126.09
OS     : Linux (Ubuntu)
```

### Verify GPU before starting
```bash
nvidia-smi
# Should show: GTX 960M | 4096MiB | CUDA 13.0
```

### Why this hardware is sufficient
- `llama3.2:1b` needs ~1.3GB VRAM — fits in 4GB with 2.7GB to spare
- CUDA 13.0 means Ollama detects GPU automatically, no configuration needed
- 16GB RAM handles ChromaDB + sentence-transformers + Python runtime comfortably
- Real GPU utilization (60–90% during inference) is visible in `nvidia-smi`

---

## Project Structure

```
distributed-llm-load-balancer/
│
├── .env                          # API keys, config (never commit)
├── .env.example                  # Safe to commit
├── pyproject.toml                # uv project file
├── uv.lock                       # Commit this
├── README.md
├── main.py                       # Entry point + CLI
│
├── common/
│   ├── __init__.py
│   ├── models.py                 # Request, Response, WorkerStatus dataclasses
│   └── config.py                 # pydantic-settings, loads .env
│
├── lb/
│   ├── __init__.py
│   ├── load_balancer.py          # Abstract base class (Strategy Pattern)
│   ├── round_robin.py            # Strategy 1
│   ├── least_connections.py      # Strategy 2
│   └── load_aware.py             # Strategy 3
│
├── master/
│   ├── __init__.py
│   └── scheduler.py              # Dispatch + heartbeat + fault recovery
│
├── workers/
│   ├── __init__.py
│   └── gpu_worker.py             # Worker node + failure simulation
│
├── rag/
│   ├── __init__.py
│   ├── indexer.py                # Populates ChromaDB on startup
│   └── retriever.py              # Async vector search → context string
│
├── llm/
│   ├── __init__.py
│   └── inference.py              # Async POST to Ollama REST API
│
├── client/
│   ├── __init__.py
│   └── load_generator.py         # asyncio.gather() concurrent user simulator
│
├── metrics/
│   ├── __init__.py
│   ├── collector.py              # Thread-safe latency, throughput, errors
│   └── dashboard.py              # Rich live terminal UI + real HW stats
│
├── results/                      # CSV output from benchmarks
│
└── tests/
    ├── test_load_balancer.py     # 5 tests
    ├── test_fault_tolerance.py   # 3 tests
    └── test_rag.py               # 3 tests
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│         1000 async coroutines via asyncio.gather()              │
│         Each sends one Request(id, query, timestamp)            │
└─────────────────────────┬───────────────────────────────────────┘
                          │ 1000 concurrent Request objects
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MASTER SCHEDULER                            │
│   • Receives all requests                                       │
│   • Runs heartbeat loop every 2s (detects dead workers)         │
│   • Catches worker failures and requeues requests               │
│   • Guarantees zero dropped requests                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LOAD BALANCER                              │
│   Filters: [w for w in workers if w.is_alive]                   │
│                                                                 │
│   Strategy 1 — Round Robin:                                     │
│     Cycles workers in order, even distribution                  │
│                                                                 │
│   Strategy 2 — Least Connections:                               │
│     Picks worker with fewest active_connections                 │
│                                                                 │
│   Strategy 3 — Load Aware:                                      │
│     Score = active_connections × avg_latency (lower = better)  │
└──────────┬──────────────┬──────────────┬────────────────────────┘
           │              │              │
           ▼              ▼              ▼
      ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
      │Worker 0 │   │Worker 1 │   │Worker 2 │   │Worker 3 │
      │ alive ✅│   │ DEAD ❌ │   │ alive ✅│   │ alive ✅│
      └────┬────┘   └─────────┘   └────┬────┘   └────┬────┘
           │         ↑ heartbeat        │              │
           │         ↑ detects,         │              │
           │         ↑ reroutes         │              │
           └─────────────────┬──────────┘──────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RAG MODULE                                │
│   1. Embed query → 384-dim vector (sentence-transformers)       │
│   2. ChromaDB cosine similarity search → top-3 documents        │
│   3. Return context string                                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │ query + context
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                LOCAL LLM — Ollama + llama3.2:1b                 │
│   • Endpoint: http://localhost:11434/api/generate               │
│   • Model loaded into GTX 960M VRAM (4GB)                       │
│   • Real GPU inference — 2–8s per response                      │
│   • Returns generated text                                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   METRICS COLLECTOR                             │
│   Records: latency, worker_id, success/failure                  │
│   Reads:   CPU% via psutil, GPU% via nvidia-smi subprocess      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              RICH LIVE DASHBOARD (terminal)                     │
│   Worker table · System resources · Throughput · P95 latency    │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
              Response returned to client
          Response(id, result, latency, worker_id)
```

---

## Request Lifecycle

Every single request follows these exact steps:

**Step 1 — Client spawns coroutines**
`asyncio.gather()` fires all N requests simultaneously. Each coroutine
creates `Request(id, query, timestamp)` and calls `scheduler.handle_request()`.

**Step 2 — Scheduler receives the request**
Wraps the dispatch in try/except. If the selected worker fails mid-execution,
the exception is caught here and the request is retried on another worker.

**Step 3 — Load balancer picks a worker**
Filters `[w for w in workers if w.is_alive]` first. Then applies
the active strategy's selection logic. Returns a `GPUWorker` reference.

**Step 4 — Worker increments active_connections**
`self.status.active_connections += 1` before processing. This counter
is read by Least Connections and Load Aware strategies in real time.

**Step 5 — RAG retrieval**
`retrieve_context(query)` runs in a thread pool (via `run_in_executor`)
to avoid blocking the async event loop. Queries ChromaDB for top-3
semantically similar documents. Returns a context string.

**Step 6 — Prompt construction**
```
You are a helpful assistant.
Use the following context to answer concisely.

Context:
- {document 1}
- {document 2}
- {document 3}

Question: {user query}

Answer in 2-3 sentences:
```

**Step 7 — Ollama generates the response**
Async HTTP POST to `localhost:11434/api/generate`. llama3.2:1b runs
on the GTX 960M GPU. Real inference: 2–8 seconds. GPU-Util: 60–90%.

**Step 8 — Worker decrements active_connections**
`finally` block: `self.active_connections -= 1`, `latency_sum += elapsed`,
`avg_latency` updated. Fault simulation checked after N requests.

**Step 9 — Metrics recorded**
`collector.record_success(worker_id, latency)` — thread-safe via `threading.Lock()`.

**Step 10 — Response returned**
`Response(id, result, latency, worker_id, success=True)` travels back to
the original coroutine in the client layer.

---

## Setup Steps

### Step 1 — Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### Step 2 — Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 3 — Pull llama3.2:1b
```bash
ollama pull llama3.2:1b
# Downloads ~1.3GB — one time only
```

### Step 4 — Start Ollama server
```bash
# Keep this terminal open for the entire project
ollama serve
# Runs at http://localhost:11434
```

### Step 5 — Run the setup script
```bash
bash setup_project.sh
cd distributed-llm-load-balancer
```

### Step 6 — Install dependencies
```bash
uv sync
```

### Step 7 — Verify .env
```bash
cat .env
# Should contain:
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.2:1b
# NUM_WORKERS=4
# NUM_USERS=20
# LB_STRATEGY=round_robin
# WORKER_FAILURE_SIMULATION=true
# FAILURE_AFTER_N_REQUESTS=50
# HEARTBEAT_INTERVAL=2.0
```

### Step 8 — Verify GPU
```bash
nvidia-smi
# Should show: GTX 960M | 4096MiB | CUDA 13.0
```

---

## Phase 1 — Foundation

**Files:** `common/models.py`, `common/config.py`

### What it contains

**`models.py`** — three dataclasses used everywhere:
```python
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

    def update_latency(self, latency: float): ...

    @property
    def load_score(self) -> float:
        return self.active_connections * max(self.avg_latency, 0.1)
```

**`config.py`** — loads `.env` with type validation:
```python
class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    num_workers: int = 4
    num_users: int = 20
    lb_strategy: str = "round_robin"
    worker_failure_simulation: bool = True
    failure_after_n_requests: int = 50
    heartbeat_interval: float = 2.0
```

### Tests
```bash
# Test 1 — Models import
uv run python -c "
from common.models import Request, Response, WorkerStatus
req = Request(id=1, query='test')
ws = WorkerStatus(id=0)
ws.update_latency(1.2)
print(f'avg_latency: {ws.avg_latency}')
print('✅ Models OK')
"

# Test 2 — Config loads
uv run python -c "
from common.config import config
print(f'Model: {config.ollama_model}')
assert config.ollama_model == 'llama3.2:1b'
print('✅ Config OK')
"

# Test 3 — Ollama reachable
uv run python -c "
import asyncio, aiohttp
async def check():
    async with aiohttp.ClientSession() as s:
        async with s.get('http://localhost:11434/api/tags') as r:
            data = await r.json()
            models = [m['name'] for m in data.get('models', [])]
            print(f'Models available: {models}')
            print('✅ Ollama reachable')
asyncio.run(check())
"
```

### ✅ Phase 1 Checklist
- [ ] `uv sync` completes with no errors
- [ ] Models import and work correctly
- [ ] Config loads `llama3.2:1b` from `.env`
- [ ] Ollama is running and reachable at `localhost:11434`

---

## Phase 2 — Workers + Load Balancer

**Files:** `workers/gpu_worker.py`, `lb/load_balancer.py`,
`lb/round_robin.py`, `lb/least_connections.py`, `lb/load_aware.py`

### Strategy Pattern Design

One abstract base class, three concrete implementations:

```python
class LoadBalancer(ABC):
    @abstractmethod
    async def get_next_worker(self, workers) -> Optional[GPUWorker]: ...

    def alive_workers(self, workers):
        return [w for w in workers if w.status.is_alive]
```

### The Three Strategies

| Strategy | Algorithm | Best When |
|---|---|---|
| Round Robin | `index = (index + 1) % len(alive)` | Uniform request sizes |
| Least Connections | `min(alive, key=lambda w: w.status.active_connections)` | Variable duration |
| Load Aware | `min(alive, key=lambda w: w.status.load_score)` | Mixed workloads |

All three call `alive_workers()` first — dead workers are never selected.

### Worker Design
```python
class GPUWorker:
    async def process(self, request) -> Response:
        if not self.status.is_alive:
            raise RuntimeError(f"Worker {self.id} is dead")
        self.status.active_connections += 1
        try:
            context = await retrieve_context(request.query)
            result = await run_llm(request.query, context)
            ...
        finally:
            self.status.active_connections -= 1
```

### Tests
```bash
# Round robin cycles
uv run python -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer

async def test():
    workers = [GPUWorker(i) for i in range(4)]
    lb = RoundRobinBalancer()
    ids = [(await lb.get_next_worker(workers)).id for _ in range(8)]
    assert ids == [0,1,2,3,0,1,2,3]
    print(f'Sequence: {ids}')
    print('✅ Round Robin OK')

asyncio.run(test())
"

# Least connections
uv run python -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.least_connections import LeastConnectionsBalancer

async def test():
    workers = [GPUWorker(i) for i in range(4)]
    workers[0].status.active_connections = 8
    workers[1].status.active_connections = 2
    workers[2].status.active_connections = 5
    workers[3].status.active_connections = 6
    lb = LeastConnectionsBalancer()
    w = await lb.get_next_worker(workers)
    assert w.id == 1
    print(f'Selected: Worker {w.id} (expected: 1)')
    print('✅ Least Connections OK')

asyncio.run(test())
"

# Dead workers skipped
uv run python -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer

async def test():
    workers = [GPUWorker(i) for i in range(4)]
    workers[1].simulate_failure()
    workers[3].simulate_failure()
    lb = RoundRobinBalancer()
    for _ in range(10):
        w = await lb.get_next_worker(workers)
        assert w.id not in [1, 3]
    print('✅ Dead workers skipped OK')

asyncio.run(test())
"

# Run pytest
uv run pytest tests/test_load_balancer.py -v
```

### ✅ Phase 2 Checklist
- [ ] Round Robin cycles all workers in order
- [ ] Least Connections picks worker with fewest connections
- [ ] Load Aware picks worker with lowest `connections × latency` score
- [ ] All strategies skip dead workers automatically
- [ ] Returns `None` (no crash) when all workers are dead
- [ ] All 5 pytest tests pass

---

## Phase 3 — RAG Pipeline

**Files:** `rag/indexer.py`, `rag/retriever.py`

### How RAG Works

```
Startup (once):
  30 knowledge base documents
       ↓ sentence-transformers all-MiniLM-L6-v2
  384-dimensional embeddings
       ↓
  ChromaDB (stored on disk at ./chroma_db)

Per request:
  query string
       ↓ same embedding model
  384-dim query vector
       ↓ cosine similarity search
  top-3 matching documents
       ↓
  context string → LLM prompt
```

### Knowledge Base Topics
The 30 documents cover: load balancing, fault tolerance, distributed systems,
GPU computing, LLM inference, RAG, ChromaDB, asyncio, heartbeat mechanisms,
latency/throughput metrics, CUDA, master/worker architecture, and more.

### Key Implementation Detail
ChromaDB queries are synchronous but run in a thread pool via
`asyncio.run_in_executor()` to avoid blocking the async event loop:
```python
async def retrieve_context(query: str) -> str:
    loop = asyncio.get_event_loop()
    context = await loop.run_in_executor(None, _query_chromadb)
    return context
```

### Tests
```bash
# Index knowledge base
uv run python -c "
from rag.indexer import get_collection
collection, model = get_collection()
assert collection.count() == 30
print(f'Documents indexed: {collection.count()}')
print('✅ RAG Indexer OK')
"

# Retrieve relevant context
uv run python -c "
import asyncio
from rag.retriever import retrieve_context

async def test():
    ctx = await retrieve_context('What is load balancing?')
    print(f'Context: {ctx[:200]}')
    assert 'load' in ctx.lower() or 'balanc' in ctx.lower()
    print('✅ RAG Retriever OK')

asyncio.run(test())
"

# Run pytest
uv run pytest tests/test_rag.py -v
```

### ✅ Phase 3 Checklist
- [ ] `all-MiniLM-L6-v2` model downloaded (~90MB, one time)
- [ ] ChromaDB created at `./chroma_db/`
- [ ] 30 documents indexed on first run
- [ ] Second run does NOT re-index (idempotent)
- [ ] Retrieved context is topically relevant to query
- [ ] Concurrent retrievals run without blocking
- [ ] All 3 pytest tests pass

---

## Phase 4 — LLM Integration

**Files:** `llm/inference.py`

### Ollama REST API

```python
async def run_llm(query: str, context: str) -> str:
    prompt = f"""You are a helpful assistant.
Use the following context to answer concisely.

Context:
{context}

Question: {query}

Answer in 2-3 sentences:"""

    payload = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 150,    # limit tokens for speed
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            data = await resp.json()
            return data["response"].strip()
```

### GPU Monitoring During Inference

Watch GPU in a second terminal while running tests:
```bash
watch -n 1 nvidia-smi
# Expect GPU-Util: 60-90% during inference
# Expect VRAM: ~1800MB / 4096MB used
```

### Tests
```bash
# Single LLM call
uv run python -c "
import asyncio, time
from llm.inference import run_llm

async def test():
    start = time.time()
    result = await run_llm('What is load balancing?',
                           'Load balancing distributes requests across servers.')
    elapsed = time.time() - start
    print(f'Response ({elapsed:.2f}s): {result[:100]}')
    assert len(result) > 10
    print('✅ LLM Integration OK')

asyncio.run(test())
"

# Full RAG → LLM pipeline
uv run python -c "
import asyncio, time
from rag.retriever import retrieve_context
from llm.inference import run_llm

async def test():
    query = 'How does fault tolerance work in distributed systems?'
    ctx = await retrieve_context(query)
    start = time.time()
    result = await run_llm(query, ctx)
    elapsed = time.time() - start
    print(f'Query  : {query}')
    print(f'Context: {ctx[:100]}...')
    print(f'Answer ({elapsed:.2f}s): {result}')
    print('✅ Full RAG → LLM pipeline OK')

asyncio.run(test())
"
```

### Numbers to Record for Report
```
Single request latency    : ____s
Average latency (3 calls) : ____s
Peak GPU utilization      : ____%
VRAM usage during infer   : ____MB / 4096MB
GPU temperature           : ____°C
```

### ✅ Phase 4 Checklist
- [ ] Single LLM call returns coherent text response
- [ ] Response is relevant to the query
- [ ] Latency measured (expect 2–8s on GTX 960M)
- [ ] `nvidia-smi` shows GPU utilization during inference
- [ ] Worker's `process()` correctly calls RAG then LLM
- [ ] No timeout errors (120s timeout is sufficient)

---

## Phase 5 — Metrics & Dashboard

**Files:** `metrics/collector.py`, `metrics/dashboard.py`

### Dashboard Preview
```
╭──────────────── DISTRIBUTED LLM SYSTEM ────────────────╮
│  Strategy: round_robin   Workers: 4   Users: 100        │
├──────────┬────────────┬─────────────┬──────────┬────────┤
│ Worker   │ Status     │ Active Conn │ Processed│Avg Lat │
├──────────┼────────────┼─────────────┼──────────┼────────┤
│ Worker-0 │ 🟢 alive   │ 3           │ 28       │ 3.21s  │
│ Worker-1 │ 🔴 dead    │ 0           │ 12       │ 3.45s  │
│ Worker-2 │ 🟢 alive   │ 2           │ 31       │ 3.18s  │
│ Worker-3 │ 🟢 alive   │ 4           │ 29       │ 3.27s  │
╰──────────┴────────────┴─────────────┴──────────┴────────╯
GPU: 87%  VRAM: 1823MB/4096MB  Temp: 74°C
CPU: 43%  RAM: 6.2GB / 16GB
Completed: 87  Failed: 0  Avg: 3.27s  P95: 4.1s  RPS: 2.3
```

### MetricsCollector Design

Thread-safe using `threading.Lock()` for all write operations:
```python
class MetricsCollector:
    def record_success(self, worker_id, latency):
        with self._lock:
            self.completed += 1
            self.latencies.append(latency)
            self.worker_counts[worker_id] += 1

    @property
    def p95_latency(self) -> float:
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[idx]

    @property
    def throughput(self) -> float:
        elapsed = self._end_time - self._start_time
        return self.completed / elapsed
```

### Tests
```bash
# Basic collector operations
uv run python -c "
from metrics.collector import MetricsCollector
import time

c = MetricsCollector(num_workers=4)
c.start_timer()
c.record_success(0, 1.2)
c.record_success(1, 0.8)
c.record_failure(2)
c.mark_worker_dead(2)
time.sleep(0.1)
c.stop_timer()

print(f'Completed : {c.completed}')
print(f'Failed    : {c.failed}')
print(f'Avg lat   : {c.avg_latency:.2f}s')
print(f'P95 lat   : {c.p95_latency:.2f}s')
print(f'Dead      : {list(c.dead_workers)}')
print('✅ MetricsCollector OK')
"

# GPU stats
uv run python -c "
from metrics.dashboard import get_gpu_stats
print(get_gpu_stats())
print('✅ GPU stats OK')
"
```

### ✅ Phase 5 Checklist
- [ ] `MetricsCollector` tracks all counters correctly
- [ ] Thread-safe: 100 concurrent writes produce exact count
- [ ] `avg_latency` and `p95_latency` computed correctly
- [ ] `throughput` (req/s) calculated from start/stop timer
- [ ] `nvidia-smi` subprocess returns GPU stats
- [ ] Dashboard table renders with 🟢 / 🔴 status indicators

---

## Phase 6 — Fault Tolerance

**Files:** `master/scheduler.py`

> ⭐ This is the most graded component of the project.

### Two Concurrent Async Tasks

```python
async def start(self):
    # Task 2 runs permanently in background
    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

async def _heartbeat_loop(self):
    while True:
        await asyncio.sleep(config.heartbeat_interval)  # every 2s
        for worker in self.workers:
            if not worker.status.is_alive:
                self.collector.mark_worker_dead(worker.id)
```

### Fault Tolerance Flow

```
Normal:
  Request → Scheduler → LB → Worker → Response ✅

Worker dies:
  Request → Scheduler → LB → Worker (DEAD) → RuntimeError
                                  ↓
                     Scheduler catches exception
                                  ↓
                     Retries with next alive worker
                                  ↓
                          Response ✅ (zero lost)

Heartbeat detects:
  Every 2s → check is_alive → False detected
           → mark in metrics → dashboard shows 🔴
           → LB already skips it automatically
```

### Retry Logic

```python
async def handle_request(self, request: Request) -> Response:
    max_retries = len(self.workers)
    for attempt in range(max_retries):
        worker = await self.lb.get_next_worker(self.workers)
        if worker is None:
            return Response(..., success=False, error="No alive workers")
        try:
            response = await worker.process(request)
            self.collector.record_success(worker.id, response.latency)
            return response
        except RuntimeError:
            # Worker died — retry with another
            self.collector.record_failure(worker.id)
            continue
    return Response(..., success=False, error="All retries exhausted")
```

### Tests
```bash
# Zero requests lost during fault
uv run python -c "
import asyncio
from workers.gpu_worker import GPUWorker
from lb.least_connections import LeastConnectionsBalancer
from master.scheduler import Scheduler
from metrics.collector import MetricsCollector
from common.models import Request

async def test():
    workers = [GPUWorker(i) for i in range(3)]
    scheduler = Scheduler(LeastConnectionsBalancer(), workers,
                          MetricsCollector(num_workers=3))
    await scheduler.start()

    async def kill_midway():
        await asyncio.sleep(2)
        workers[1].simulate_failure()

    tasks = [scheduler.handle_request(Request(id=i, query=f'Query {i}'))
             for i in range(8)]
    tasks.append(kill_midway())

    results = await asyncio.gather(*tasks, return_exceptions=True)
    responses = [r for r in results if hasattr(r, 'success')]
    failed = sum(1 for r in responses if not r.success)

    assert failed == 0, f'{failed} requests lost!'
    print(f'All {len(responses)} requests completed. Failed: {failed}')
    print('✅ Zero requests lost during fault simulation')
    await scheduler.stop()

asyncio.run(test())
"

# Run pytest
uv run pytest tests/test_fault_tolerance.py -v
```

### ✅ Phase 6 Checklist
- [ ] Scheduler dispatches requests correctly
- [ ] Dead workers are automatically skipped
- [ ] Requests retry on next alive worker after failure
- [ ] All workers dead → graceful `Response(success=False)`, no crash
- [ ] Heartbeat detects dead workers within 2 seconds
- [ ] **Zero requests lost during fault simulation** ← critical
- [ ] All 3 pytest tests pass

---

## Phase 7 — Load Generator & Main

**Files:** `client/load_generator.py`, `main.py`

### Why asyncio.gather() Not Threads

```python
# ❌ Threads — wrong for this project
threads = [Thread(target=simulate_user, args=(i,)) for i in range(1000)]
# Problem: GIL limits CPU parallelism, 1000 threads = high memory overhead

# ✅ asyncio.gather() — correct
tasks = [simulate_user(i, scheduler, collector) for i in range(1000)]
responses = await asyncio.gather(*tasks)
# 1000 coroutines, one thread, lightweight, no GIL issues for I/O-bound work
```

This distinction is worth explaining in your report.

### CLI Interface

```bash
# Basic usage
uv run python main.py --strategy round_robin --users 10 --workers 4

# No fault simulation
uv run python main.py --strategy least_connections --users 20 --workers 4 --no-fault

# Save results to CSV
uv run python main.py --strategy load_aware --users 5 --workers 4 --save-results

# Stub mode for large load tests (skips real LLM)
LLM_STUB=1 uv run python main.py --strategy round_robin --users 500 --workers 4
```

### Tests
```bash
# Smoke test — 5 users
uv run python main.py --strategy round_robin --users 5 --workers 3 --no-fault

# Fault simulation test
uv run python main.py --strategy least_connections --users 10 --workers 4

# All three strategies
for s in round_robin least_connections load_aware; do
    uv run python main.py --strategy $s --users 5 --workers 4 --no-fault --save-results
done
```

### Benchmark Script (for report data)
```bash
# Save as run_benchmarks.sh
for STRATEGY in round_robin least_connections load_aware; do
    for USERS in 5 10 20; do
        uv run python main.py --strategy $STRATEGY \
            --users $USERS --workers 4 --no-fault --save-results
        sleep 2
    done
done

# Large scale with stub
for STRATEGY in round_robin least_connections load_aware; do
    for USERS in 100 500 1000; do
        LLM_STUB=1 uv run python main.py --strategy $STRATEGY \
            --users $USERS --workers 4 --no-fault --save-results
    done
done
```

### ✅ Phase 7 Checklist
- [ ] 5-user smoke test completes with all successful responses
- [ ] Round Robin distributes ≤ 1 request difference across workers
- [ ] All 3 strategies selectable via `--strategy` CLI arg
- [ ] `--no-fault` disables fault simulation
- [ ] `--save-results` writes CSV to `results/` folder
- [ ] Fault simulation triggers correctly with default settings
- [ ] Stub mode (`LLM_STUB=1`) runs fast for 100+ users

---

## Phase 8 — Testing & Final Validation

### Run Complete Test Suite
```bash
uv run pytest tests/ -v --tb=short
```

**All 11 tests must pass:**
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

11 passed
```

### Full System End-to-End Test
```bash
# Real LLM, fault simulation, 10 users
uv run python main.py --strategy least_connections --users 10 --workers 4

# Verify output:
# ✅ All 10 requests completed
# ✅ Worker 1 died mid-run
# ✅ Failed: 0
# ✅ Dead Workers: [1]
```

### GPU Utilization Logging
```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu \
           --format=csv --loop=2 > results/gpu_log.csv &
GPU_PID=$!

uv run python main.py --strategy round_robin --users 5 --workers 4 --no-fault

kill $GPU_PID
cat results/gpu_log.csv
```

---

## Report Data Table

Fill this during Phase 8 by running benchmarks:

| Strategy | 5 users | 10 users | 100 users* | 500 users* |
|---|---|---|---|---|
| Round Robin — avg latency | ____s | ____s | ____s | ____s |
| Round Robin — throughput | __r/s | __r/s | __r/s | __r/s |
| Round Robin — P95 | ____s | ____s | ____s | ____s |
| Least Connections — avg | ____s | ____s | ____s | ____s |
| Least Connections — r/s | __r/s | __r/s | __r/s | __r/s |
| Least Connections — P95 | ____s | ____s | ____s | ____s |
| Load Aware — avg | ____s | ____s | ____s | ____s |
| Load Aware — r/s | __r/s | __r/s | __r/s | __r/s |
| Load Aware — P95 | ____s | ____s | ____s | ____s |

> \* stub mode (`LLM_STUB=1`) — measures distributed system layer only

**Hardware metrics during real inference:**
```
GPU utilization   : ____%
VRAM usage        : ____MB / 4096MB
GPU temperature   : ____°C
CPU utilization   : ____%
RAM usage         : ____GB / 16GB
```

**Fault tolerance results:**
```
Workers killed    : 1 (Worker 1, after 50 requests)
Requests lost     : 0
Detection time    : < 2s (heartbeat interval)
```

---

## Demo Video Checklist

Record in this order (target: 6–8 minutes):

- [ ] **Scene 1 (30s)** — Show `ollama list` and `nvidia-smi` idle
- [ ] **Scene 2 (45s)** — Single RAG → LLM call with real response printed
- [ ] **Scene 3 (2min)** — Live load test, split screen with `nvidia-smi` showing GPU spike
- [ ] **Scene 4 (1min)** — Run same workload with all 3 strategies, compare latency
- [ ] **Scene 5 (2min)** — Fault tolerance live: worker dies, system recovers, `Failed: 0`
- [ ] **Scene 6 (30s)** — Show `results/` CSV files with all benchmark data

---

## Key Points for Report

### Design Decisions to Explain

**asyncio over threading**
Python threads are limited by the GIL for CPU-bound work. For I/O-bound
tasks like HTTP calls to Ollama and vector DB queries, asyncio coroutines
are the correct choice — they're lightweight, composable, and avoid GIL issues.

**Strategy Pattern for load balancing**
One abstract base class with a `get_next_worker()` method. Three concrete
implementations. Adding a 4th strategy requires zero changes to existing code
(open/closed principle). Clean, extensible, and testable in isolation.

**ChromaDB local persistence**
No external server dependency. The vector database lives at `./chroma_db/`
and persists between runs. Fully reproducible demo environment — anyone who
clones the repo gets the same knowledge base after running `get_collection()`.

**Pydantic-settings for config**
Type-safe configuration that validates on startup. If `OLLAMA_MODEL` is
missing from `.env`, the process crashes immediately with a clear error
rather than failing silently 3 layers deep.

**run_in_executor for ChromaDB**
ChromaDB's Python client is synchronous. Wrapping it in
`asyncio.run_in_executor()` runs it in a thread pool without blocking
the main event loop — other requests continue processing concurrently.

**thread_lock in MetricsCollector**
1000 coroutines write to the collector concurrently. `threading.Lock()`
guarantees atomic writes and prevents race conditions in the counters.

### Limitations to Include

- Single GPU means workers share hardware — true parallelism is limited by
  the GPU being saturated. In production each worker would have a dedicated GPU.
- llama3.2:1b is a small model — response quality is limited. A production
  system would use 7B+ parameter models on multi-GPU nodes.
- Fault tolerance simulates failures via a software flag, not actual process
  crashes or network partitions.
- The 1000-user load test with real LLM calls is replaced by stub mode for
  performance reasons — a production system would use a dedicated inference
  server like vLLM with continuous batching.

### References to Cite
- `thushan/olla` — production LLM load balancer with circuit breakers
- `K2/olol` — distributed Ollama inference with gRPC
- `open-webui/loadbalancer.py` — clean Python implementation of LB strategies
- ChromaDB documentation — vector database design
- Ollama documentation — local LLM serving
- Python asyncio documentation — concurrent I/O design

---

## GitHub Repository

**Repository name:**
```
distributed-llm-load-balancer
```

**Description:**
```
Distributed system for handling 1000+ concurrent LLM requests using llama3.2:1b
via Ollama, with async load balancing (Round Robin, Least Connections, Load-Aware),
RAG pipeline via ChromaDB, and fault-tolerant GPU worker simulation.
Built with Python asyncio & uv. CSE354 — Ain Shams University.
```

**Topics to add:**
```
distributed-systems  load-balancing  llm  ollama  rag  chromadb
asyncio  python  fault-tolerance  gpu
```

---

## Quick Reference — Most Used Commands

```bash
# Start Ollama (keep open in background terminal)
ollama serve

# Watch GPU
watch -n 1 nvidia-smi

# Run tests
uv run pytest tests/ -v

# Run system (real LLM)
uv run python main.py --strategy round_robin --users 10 --workers 4

# Run system (stub mode, fast)
LLM_STUB=1 uv run python main.py --strategy round_robin --users 1000 --workers 4

# Run with fault simulation
uv run python main.py --strategy least_connections --users 20 --workers 4

# Save benchmark results
uv run python main.py --strategy load_aware --users 10 --workers 4 --save-results

# View results
ls results/
cat results/*.csv
```

---

*Built for CSE354 Distributed Computing — Ain Shams University — 2nd Semester 2025/2026*
