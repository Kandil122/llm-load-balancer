# CSE354 Distributed Computing Project — Build Plan
### Efficient Load Balancing and GPU Cluster Task Distribution for Handling 1000+ Concurrent LLM Requests
> Ain Shams University | Faculty of Engineering | 2nd Semester 2025/2026

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| Package manager | **uv** | Fastest, modern, handles venvs too |
| LLM | **Google Gemini 2.0 Flash** | Free via AI Studio, fast, 1M tokens/day |
| Gemini SDK | **google-genai** | New unified SDK, clean async support |
| RAG Vector DB | **ChromaDB** | Lightweight, runs locally, no server needed |
| Embeddings | **sentence-transformers** | Free, local, good quality |
| Concurrency | **asyncio** | Far better than threads for 1000 concurrent requests |
| Metrics | **rich** | Beautiful live terminal dashboard |
| Config | **pydantic-settings** | Clean env/config management |
| Testing | **pytest + pytest-asyncio** | Async-friendly testing |

---

## 📁 Project Structure

```
distributed-llm/
│
├── .env                          # API keys, config (never commit)
├── .env.example                  # committed to git (no secrets)
├── pyproject.toml                # uv project file
├── uv.lock                       # committed to git
├── README.md
├── main.py                       # entry point
│
├── common/
│   ├── __init__.py
│   ├── models.py                 # Request, Response, WorkerStatus dataclasses
│   └── config.py                 # pydantic-settings config
│
├── lb/
│   ├── __init__.py
│   ├── load_balancer.py          # abstract base class (strategy pattern)
│   ├── round_robin.py            # strategy 1
│   ├── least_connections.py      # strategy 2
│   └── load_aware.py             # strategy 3
│
├── master/
│   ├── __init__.py
│   └── scheduler.py              # orchestrates lb + heartbeat + fault detection
│
├── workers/
│   ├── __init__.py
│   └── gpu_worker.py             # worker node + failure simulation
│
├── rag/
│   ├── __init__.py
│   ├── indexer.py                # populates ChromaDB on startup
│   └── retriever.py              # async ChromaDB query logic
│
├── llm/
│   ├── __init__.py
│   └── inference.py              # Gemini 2.0 Flash via google-genai SDK
│
├── client/
│   ├── __init__.py
│   └── load_generator.py         # async 1000-user simulator
│
├── metrics/
│   ├── __init__.py
│   ├── collector.py              # aggregates latency, throughput, errors
│   └── dashboard.py              # rich live terminal display
│
└── tests/
    ├── test_load_balancer.py
    ├── test_fault_tolerance.py
    └── test_rag.py
```

---

## ⚙️ Setup Steps

### Step 1 — Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### Step 2 — Get your Gemini API Key
1. Go to **aistudio.google.com**
2. Click **"Get API Key"** → **"Create API key"**
3. Copy it — no credit card needed

### Step 3 — Initialize the project
```bash
uv init distributed-llm
cd distributed-llm
```

### Step 4 — Add all dependencies
```bash
uv add google-genai chromadb sentence-transformers \
       pydantic-settings rich \
       pytest pytest-asyncio python-dotenv
```

### Step 5 — Create your .env
```bash
# .env
GEMINI_API_KEY=AIza...
NUM_WORKERS=4
NUM_USERS=1000
LB_STRATEGY=round_robin          # round_robin | least_connections | load_aware
WORKER_FAILURE_SIMULATION=true
FAILURE_AFTER_N_REQUESTS=50      # kill a worker after 50 requests
RATE_LIMIT_RPM=14                # stay just under Gemini's 15 RPM free limit
```

### Step 6 — Run
```bash
uv run python main.py

# or with CLI args:
uv run python main.py --strategy least_connections --users 500 --workers 6
```

---

## 🏗️ Build Order (Phase by Phase)

---

### Phase 1 — Foundation
**Files:** `common/models.py`, `common/config.py`

`models.py` defines the core dataclasses used everywhere:
```
Request        → id, query, timestamp
Response       → id, result, latency, worker_id
WorkerStatus   → id, is_alive, active_connections, avg_latency, total_processed
```

`config.py` loads `.env` via pydantic-settings — type-safe, validates on startup,
crashes early if `GEMINI_API_KEY` is missing rather than failing silently later.

**✅ Goal:** import config and models with no errors.

---

### Phase 2 — Workers + Load Balancer
**Files:** `workers/gpu_worker.py`, `lb/load_balancer.py`, `lb/round_robin.py`, `lb/least_connections.py`, `lb/load_aware.py`

**Worker** gets:
- `async def process(request)` — main handler
- `is_alive` flag — toggled by fault simulation
- `active_connections` counter — used by least connections strategy
- `avg_latency` tracker — used by load aware strategy
- `simulate_failure()` — sets `is_alive = False` after N requests

**Load Balancer** uses the **Strategy Pattern** — one abstract base class with a single
`get_next_worker(workers)` method, three concrete implementations:

| Strategy | Logic | Best When |
|---|---|---|
| Round Robin | cycles workers in order | uniform request sizes |
| Least Connections | picks lowest `active_connections` | variable request duration |
| Load Aware | scores by `active_connections × avg_latency` | mixed workloads |

All three filter out dead workers with `[w for w in workers if w.is_alive]` before selecting.

**✅ Goal:** 10-request test shows round robin distributing evenly across workers in terminal.

---

### Phase 3 — RAG Pipeline
**Files:** `rag/indexer.py`, `rag/retriever.py`

**`indexer.py`** runs once on startup:
- Loads a small knowledge base (30–50 documents — tech facts, science, anything consistent)
- Embeds them using `sentence-transformers` model `all-MiniLM-L6-v2` (fast, free, local)
- Stores embeddings in a local ChromaDB collection on disk

**`retriever.py`**:
- `async def retrieve_context(query)`
- Embeds the query, searches ChromaDB for top-3 most similar documents
- Returns them concatenated as a context string passed to the LLM

**✅ Goal:** query the retriever in isolation, get back relevant context for a test question.

---

### Phase 4 — Gemini LLM Integration
**Files:** `llm/inference.py`

```python
# llm/inference.py
from google import genai

client = genai.Client(api_key=config.GEMINI_API_KEY)

async def run_llm(query: str, context: str) -> str:
    prompt = f"""You are a helpful assistant.
Use the following context to answer the question concisely.

Context: {context}

Question: {query}

Answer:"""

    response = await client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text
```

**Rate limiter** — wraps `run_llm` with an `asyncio.Semaphore(14)` to stay under
the 15 RPM free tier limit. Without this, 1000 concurrent users will instantly
trigger 429 errors.

**Model:** `gemini-2.0-flash` — fastest on the free tier, low latency, ideal for demo.

**✅ Goal:** one real end-to-end call: query → RAG context → Gemini response printed in terminal.

---

### Phase 5 — Metrics & Dashboard
**Files:** `metrics/collector.py`, `metrics/dashboard.py`

**`collector.py`** — async-safe counters using `asyncio.Lock()`:
- `record_success(worker_id, latency)`
- `record_failure(worker_id)`
- `get_summary()` → returns dict with all stats

Tracks:
```
total_requests       completed            failed
avg_latency          p95_latency          requests_per_second
per_worker_counts    failed_workers       active_connections
```

**`dashboard.py`** — uses `rich.Live` + `rich.Table` for a live terminal UI
that refreshes every second showing all metrics while the simulation runs.

**✅ Goal:** run 100 users and watch the dashboard update live in the terminal.

---

### Phase 6 — Fault Tolerance ⭐
**Files:** `master/scheduler.py`

This is the most graded section. The scheduler runs two concurrent async tasks:

**Task 1 — Request Handler:** dispatches incoming requests through the load balancer to workers

**Task 2 — Heartbeat Loop:** every 2 seconds checks all workers, marks dead ones, logs the event

**Fault flow:**
```
Worker dies (is_alive = False)
    → Heartbeat detects it within 2 seconds
    → Marks worker as failed in metrics
    → Load balancer automatically skips it (filters is_alive)
    → In-flight request on dead worker raises exception
    → Scheduler catches exception, requeues the request
    → Request reassigned to next live worker
    → Zero requests lost
```

**✅ Goal:** start 1000-user simulation → worker 2 dies at request 50 → dashboard
shows failure event → all 1000 requests complete successfully.

---

### Phase 7 — Client Load Generator
**Files:** `client/load_generator.py`, `main.py`

Uses `asyncio.gather()` — not threads. This is architecturally correct for
I/O-bound async work and worth explaining in the report.

**Ramp-up mode** for report data:
```
Run at 100 users  → record metrics → save to results/100.csv
Run at 250 users  → record metrics → save to results/250.csv
Run at 500 users  → record metrics → save to results/500.csv
Run at 1000 users → record metrics → save to results/1000.csv
```

**`main.py` CLI:**
```bash
uv run python main.py --strategy round_robin --users 1000 --workers 4
uv run python main.py --strategy least_connections --users 1000 --workers 4
uv run python main.py --strategy load_aware --users 1000 --workers 4
```

---

### Phase 8 — Testing
**Files:** `tests/`

```bash
uv run pytest tests/ -v
```

| Test File | What It Tests |
|---|---|
| `test_load_balancer.py` | Round robin cycles correctly, least connections picks lowest, load aware scores correctly, dead workers skipped |
| `test_fault_tolerance.py` | Worker dies → detected within 2s → requests rerouted → zero lost |
| `test_rag.py` | Index docs → query retriever → relevant context returned |

---

## 📊 Report Data Table

Run all three strategies at all four load levels and fill this in:

| Strategy | 100 users | 250 users | 500 users | 1000 users |
|---|---|---|---|---|
| Round Robin | latency / throughput | ... | ... | ... |
| Least Connections | ... | ... | ... | ... |
| Load Aware | ... | ... | ... | ... |

> **Target:** requests lost during fault simulation = **0**

---

## 🎬 Demo Video Checklist

- [ ] Show live dashboard with 1000 users running
- [ ] Switch strategies via CLI arg between runs
- [ ] **Kill a worker live** — show dashboard react, show zero dropped requests
- [ ] Show one real Gemini response end-to-end (RAG context → LLM answer printed)
- [ ] Show final metrics summary table

---

## ⚡ Key Points to Highlight in Report

- **asyncio over threading** — threads are limited by Python's GIL for CPU-bound work;
  async is correct for I/O-bound tasks like API calls
- **Strategy Pattern for load balancing** — open/closed principle, trivial to add a 4th strategy
- **Rate limiter with Semaphore** — necessary design decision when integrating a free-tier
  API into a high-concurrency system
- **ChromaDB local** — no external server dependency, fully reproducible demo environment
- **Pydantic-settings for config** — type-safe, fails fast on missing keys rather than
  cryptic runtime errors

---

*Ready to start? Say **"start Phase 1"** to begin writing the code.*
