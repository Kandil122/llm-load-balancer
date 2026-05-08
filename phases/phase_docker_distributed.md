# Phase 9 — Docker Containerization & True Distributed Workers

**CSE354 Distributed Computing — Ain Shams University**
**Upgrade:** Single-process simulation → Real Docker microservices

---

## Overview

In the original system, all workers are Python objects living in the same process and sharing the same memory. "Worker failure" is just a flag (`is_alive = False`) — not a real crash. This phase transforms each worker into an independent Docker container running a FastAPI HTTP service. The master scheduler communicates with workers over a real network. Failures are real process deaths. Resource utilization is measured per container.

**What changes:**

| Aspect | Before | After |
|---|---|---|
| Workers | In-memory Python objects | Docker containers (FastAPI) |
| Communication | Direct method calls | HTTP REST over Docker network |
| Failure simulation | `is_alive = False` flag | `docker stop worker-1` |
| Resource metrics | `psutil` for whole process | Per-container via `docker stats` + `/metrics` endpoint |
| Isolation | None (shared memory) | Full process + network isolation |
| Scaling | Change `num_workers` int | Add a service block in `docker-compose.yml` |

---

## Updated Project Structure

```
distributed-llm-load-balancer/
│
├── docker/
│   ├── Dockerfile.worker          ← NEW
│   └── Dockerfile.master          ← NEW
│
├── docker-compose.yml             ← NEW
│
├── worker/
│   ├── __init__.py
│   └── app.py                     ← NEW: FastAPI worker microservice
│
├── master/
│   ├── __init__.py
│   ├── app.py                     ← NEW: FastAPI master service
│   └── http_scheduler.py          ← REPLACES: scheduler.py
│
├── client/
│   └── load_generator.py          ← UPDATED: sends HTTP to master:8000
│
├── common/                        ← unchanged
├── lb/                            ← unchanged
├── rag/                           ← unchanged
├── llm/                           ← unchanged
├── metrics/
│   ├── collector.py               ← UPDATED: polls worker /metrics endpoints
│   └── dashboard.py               ← UPDATED: shows per-container stats
│
└── main.py                        ← UPDATED: starts system via docker-compose
```

---

## System Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │            docker-compose network            │
                        │                                              │
  curl / load_gen ────► │  ┌─────────────────────────────────────┐    │
  POST /request         │  │     master container  :8000          │    │
                        │  │  FastAPI · HttpScheduler · LB        │    │
                        │  └───────┬───────┬───────┬─────────┬───┘    │
                        │          │ HTTP  │       │         │         │
                        │          ▼       ▼       ▼         ▼         │
                        │  ┌──────────┐ ┌──────────┐ ┌──────────┐     │
                        │  │worker-0  │ │worker-1  │ │worker-2  │ ... │
                        │  │  :8001   │ │  :8002   │ │  :8003   │     │
                        │  │/process  │ │/process  │ │/process  │     │
                        │  │/health   │ │/health   │ │/health   │     │
                        │  │/metrics  │ │/metrics  │ │/metrics  │     │
                        │  └────┬─────┘ └────┬─────┘ └────┬─────┘     │
                        │       │             │             │           │
                        │       └─────────────┼─────────────┘           │
                        │                     ▼                         │
                        │         ┌───────────────────────┐             │
                        │         │  chroma-data volume   │             │
                        │         │  (shared read/write)  │             │
                        │         └───────────────────────┘             │
                        │                     │                         │
                        └─────────────────────┼─────────────────────────┘
                                              ▼
                                  Ollama  (host :11434)
                              GTX 960M — shared by all workers
```

> **Note on GPU sharing:** Since only one GTX 960M is available, all worker containers call the same Ollama instance on the host via `host.docker.internal:11434`. Workers are isolated in CPU, RAM, and network — GPU access is shared through the HTTP interface. In a real multi-GPU cluster, each worker would have its own GPU node.

---

## Files to Create

### `worker/app.py`

Each worker is a standalone FastAPI application. It exposes four endpoints and tracks its own `WorkerStatus` in memory.

```python
# worker/app.py
import os, time
import psutil
from fastapi import FastAPI
from pydantic import BaseModel
from rag.retriever import retrieve_context
from llm.inference import run_llm
from common.models import WorkerStatus

app = FastAPI()
WORKER_ID = int(os.environ.get("WORKER_ID", 0))
status = WorkerStatus(id=WORKER_ID)


class ProcessRequest(BaseModel):
    id: int
    query: str


@app.on_event("startup")
async def startup():
    # Pre-load ChromaDB index on startup so first request is not slow
    from rag.indexer import get_collection
    get_collection()
    print(f"[Worker {WORKER_ID}] Ready on port {8001 + WORKER_ID}")


@app.post("/process")
async def process(req: ProcessRequest):
    if not status.is_alive:
        return {"success": False, "error": "Worker is dead", "worker_id": WORKER_ID}

    start = time.time()
    status.active_connections += 1
    try:
        context = await retrieve_context(req.query)
        result = await run_llm(req.query, context)
        latency = time.time() - start
        status.update_latency(latency)
        return {
            "id": req.id,
            "result": result,
            "latency": round(latency, 3),
            "worker_id": WORKER_ID,
            "success": True,
        }
    except Exception as e:
        status.total_failed += 1
        return {
            "success": False,
            "error": str(e),
            "worker_id": WORKER_ID,
            "latency": round(time.time() - start, 3),
        }
    finally:
        status.active_connections = max(0, status.active_connections - 1)


@app.get("/health")
async def health():
    return {"alive": status.is_alive, "worker_id": WORKER_ID}


@app.get("/metrics")
async def metrics():
    proc = psutil.Process()
    return {
        "worker_id": WORKER_ID,
        "is_alive": status.is_alive,
        "active_connections": status.active_connections,
        "total_processed": status.total_processed,
        "total_failed": status.total_failed,
        "avg_latency": round(status.avg_latency, 3),
        "cpu_percent": round(proc.cpu_percent(interval=0.1), 1),
        "memory_mb": round(proc.memory_info().rss / 1e6, 1),
        "memory_percent": round(proc.memory_percent(), 2),
    }


@app.post("/kill")
async def kill():
    """Simulate failure — sets is_alive=False without stopping the container."""
    status.is_alive = False
    return {"killed": WORKER_ID}


@app.post("/revive")
async def revive():
    status.is_alive = True
    return {"revived": WORKER_ID}
```

---

### `master/app.py`

The master exposes a `/request` endpoint and a `/workers/status` endpoint that aggregates live metrics from all worker containers.

```python
# master/app.py
import asyncio
import aiohttp
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from master.http_scheduler import HttpScheduler
from common.config import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = HttpScheduler.instance()
    asyncio.create_task(scheduler.heartbeat_loop())
    yield


app = FastAPI(lifespan=lifespan)

WORKER_URLS = [
    f"http://worker-{i}:{8001 + i}"
    for i in range(config.num_workers)
]


class UserRequest(BaseModel):
    id: int
    query: str


@app.post("/request")
async def handle_request(req: UserRequest):
    return await HttpScheduler.instance().handle(req.id, req.query)


@app.get("/workers/status")
async def workers_status():
    """Poll all worker /metrics endpoints — real per-container utilization."""
    results = []
    async with aiohttp.ClientSession() as session:
        for url in WORKER_URLS:
            try:
                async with session.get(
                    f"{url}/metrics",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as r:
                    results.append(await r.json())
            except Exception:
                results.append({"error": "unreachable", "url": url})
    return results


@app.get("/health")
async def health():
    return {"status": "ok", "workers": config.num_workers}
```

---

### `master/http_scheduler.py`

Replaces `master/scheduler.py`. All worker calls are now HTTP requests. Dead workers are detected by network timeouts and HTTP errors — not software flags.

```python
# master/http_scheduler.py
import asyncio
import aiohttp
from common.config import config

WORKER_URLS = [
    f"http://worker-{i}:{8001 + i}"
    for i in range(config.num_workers)
]


class HttpScheduler:
    _instance = None

    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._index = 0
        self._lock = asyncio.Lock()
        self._alive = {i: True for i in range(len(WORKER_URLS))}

    async def handle(self, request_id: int, query: str) -> dict:
        max_retries = len(WORKER_URLS)
        for attempt in range(max_retries):
            worker_idx = await self._next_alive()
            if worker_idx is None:
                return {"success": False, "error": "No alive workers"}
            url = WORKER_URLS[worker_idx]
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{url}/process",
                        json={"id": request_id, "query": query},
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("success"):
                                return data
                        # Non-200 = worker is unhealthy
                        self._alive[worker_idx] = False
            except Exception:
                # Network error = container is truly down
                self._alive[worker_idx] = False
                print(f"[Scheduler] worker-{worker_idx} unreachable, retrying...")
        return {"success": False, "error": "All retry attempts exhausted"}

    async def _next_alive(self):
        """Round-robin over alive workers (swap in other strategies here)."""
        async with self._lock:
            alive = [i for i, ok in self._alive.items() if ok]
            if not alive:
                return None
            chosen = alive[self._index % len(alive)]
            self._index = (self._index + 1) % len(alive)
            return chosen

    async def heartbeat_loop(self):
        """Poll /health on every worker every 2 seconds.
        Revives workers that come back online (e.g. after docker start)."""
        while True:
            await asyncio.sleep(config.heartbeat_interval)
            async with aiohttp.ClientSession() as session:
                for i, url in enumerate(WORKER_URLS):
                    try:
                        async with session.get(
                            f"{url}/health",
                            timeout=aiohttp.ClientTimeout(total=1)
                        ) as r:
                            data = await r.json()
                            self._alive[i] = data.get("alive", False)
                    except Exception:
                        self._alive[i] = False
```

---

### `docker/Dockerfile.worker`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] aiohttp \
    chromadb sentence-transformers \
    pydantic-settings psutil

COPY . .

# WORKER_ID injected by docker-compose per service
ENV WORKER_ID=0
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

# Port is computed: 8001 + WORKER_ID
CMD ["sh", "-c", "uvicorn worker.app:app --host 0.0.0.0 --port $((8001 + WORKER_ID))"]
```

---

### `docker/Dockerfile.master`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] aiohttp \
    pydantic-settings psutil rich

COPY . .

ENV NUM_WORKERS=4
ENV LB_STRATEGY=round_robin

EXPOSE 8000
CMD ["uvicorn", "master.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### `docker-compose.yml`

```yaml
version: "3.9"

networks:
  llm-net:
    driver: bridge

volumes:
  chroma-data:   # all workers share the same ChromaDB index

services:

  master:
    build:
      context: .
      dockerfile: docker/Dockerfile.master
    ports:
      - "8000:8000"
    networks: [llm-net]
    environment:
      - NUM_WORKERS=4
      - LB_STRATEGY=round_robin
      - HEARTBEAT_INTERVAL=2.0
    depends_on:
      - worker-0
      - worker-1
      - worker-2
      - worker-3
    restart: on-failure

  worker-0:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    environment:
      - WORKER_ID=0
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    ports: ["8001:8001"]
    networks: [llm-net]
    volumes:
      - chroma-data:/app/chroma_db
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 2G
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: on-failure

  worker-1:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    environment:
      - WORKER_ID=1
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    ports: ["8002:8002"]
    networks: [llm-net]
    volumes:
      - chroma-data:/app/chroma_db
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 2G
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: on-failure

  worker-2:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    environment:
      - WORKER_ID=2
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    ports: ["8003:8003"]
    networks: [llm-net]
    volumes:
      - chroma-data:/app/chroma_db
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 2G
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: on-failure

  worker-3:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    environment:
      - WORKER_ID=3
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    ports: ["8004:8004"]
    networks: [llm-net]
    volumes:
      - chroma-data:/app/chroma_db
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 2G
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: on-failure
```

---

## Setup Steps

### Step 1 — Install Docker
```bash
# Ubuntu
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker compose version
```

### Step 2 — Make sure Ollama is running on the host
```bash
# Ollama must be on host (not inside Docker) so GPU is accessible
ollama serve &
ollama pull gemma3:270m

# Verify it's reachable from the host
curl http://localhost:11434/api/tags
```

### Step 3 — Build all containers
```bash
cd distributed-llm-load-balancer
docker compose build
```

### Step 4 — Start the system
```bash
docker compose up -d

# Watch logs from all containers
docker compose logs -f
```

### Step 5 — Verify all workers are healthy
```bash
curl http://localhost:8001/health   # worker-0
curl http://localhost:8002/health   # worker-1
curl http://localhost:8003/health   # worker-2
curl http://localhost:8004/health   # worker-3

# All should return: {"alive": true, "worker_id": N}
```

### Step 6 — Verify master can see all workers
```bash
curl http://localhost:8000/workers/status | python3 -m json.tool
```

Expected output:
```json
[
  {"worker_id": 0, "is_alive": true, "cpu_percent": 0.3, "memory_mb": 312.4, ...},
  {"worker_id": 1, "is_alive": true, "cpu_percent": 0.1, "memory_mb": 308.1, ...},
  {"worker_id": 2, "is_alive": true, "cpu_percent": 0.2, "memory_mb": 310.5, ...},
  {"worker_id": 3, "is_alive": true, "cpu_percent": 0.1, "memory_mb": 305.8, ...}
]
```

---

## Running Tests

### Single request
```bash
curl -X POST http://localhost:8000/request \
  -H "Content-Type: application/json" \
  -d '{"id": 1, "query": "What is load balancing?"}'
```

### Load test — 20 concurrent requests
```bash
python3 - <<'EOF'
import asyncio, aiohttp, time

async def send(session, i):
    async with session.post("http://localhost:8000/request",
                            json={"id": i, "query": "Explain fault tolerance"}) as r:
        return await r.json()

async def main():
    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [send(session, i) for i in range(20)]
        results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    ok = sum(1 for r in results if r.get("success"))
    print(f"Completed: {ok}/20 in {elapsed:.2f}s")
    for r in results:
        print(f"  Request {r.get('id')} → Worker {r.get('worker_id')} "
              f"({r.get('latency')}s)")

asyncio.run(main())
EOF
```

### Fault tolerance test — kill worker-1 mid-run
```bash
# Terminal 1: start a load test
python3 your_load_test.py &

# Terminal 2: kill worker-1 (real container death)
docker stop worker-1

# Observe master logs — it detects the failure and reroutes
docker compose logs master | grep "unreachable"

# Revive it
docker start worker-1
```

### Per-container utilization — live view
```bash
docker stats worker-0 worker-1 worker-2 worker-3
```

Example output during active inference:
```
NAME       CPU %     MEM USAGE / LIMIT   NET I/O        BLOCK I/O
worker-0   87.3%     1.21GiB / 2GiB      412MB / 38MB   0B / 0B
worker-1   0.1%      312MiB / 2GiB       1.2MB / 800kB  0B / 0B
worker-2   92.1%     1.19GiB / 2GiB      395MB / 36MB   0B / 0B
worker-3   4.2%      318MiB / 2GiB       8.4MB / 2.1MB  0B / 0B
```

This is the real distributed utilization data — each row is a separate container.

### Log container stats to CSV (for report)
```bash
docker stats --format \
  "{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.NetIO}}" \
  --no-stream >> results/container_stats.csv
```

---

## Metrics to Record for Report

Run with real LLM (`gemma3:270m`) for 5–10 users, then record from `docker stats` and `/workers/status`:

| Worker | CPU % (peak) | CPU % (idle) | RAM used | Requests handled | Avg latency |
|--------|-------------|-------------|----------|-----------------|-------------|
| worker-0 | ____% | ____% | ____MB | ____ | ____s |
| worker-1 | ____% | ____% | ____MB | ____ | ____s |
| worker-2 | ____% | ____% | ____MB | ____ | ____s |
| worker-3 | ____% | ____% | ____MB | ____ | ____s |

**Fault tolerance results:**
```
Container killed  : worker-1 (docker stop)
Requests lost     : 0
Detection time    : < 2s (heartbeat interval)
Requests rerouted : automatic (round-robin skips dead worker)
Revival test      : docker start worker-1 → back in rotation within 2s
```

---

## Phase 9 Checklist

- [ ] `docker compose build` completes without errors
- [ ] All 4 worker containers start and pass `/health` check
- [ ] Master `/workers/status` shows all 4 workers alive with real metrics
- [ ] Single request via `POST /request` returns a real LLM response
- [ ] Load test routes requests across multiple workers (check `worker_id` in responses)
- [ ] `docker stop worker-1` → master detects failure within 2s → routes to others
- [ ] Zero requests lost during fault simulation
- [ ] `docker start worker-1` → worker rejoins rotation automatically
- [ ] `docker stats` shows per-container CPU and RAM during inference
- [ ] Container stats logged to `results/container_stats.csv`
- [ ] All 3 load balancing strategies still work (change `LB_STRATEGY` env var)

---

## Key Points for Report

**Why Docker containers make this truly distributed:**
In the original system, all workers share one Python interpreter, one memory heap, and one event loop. There is no isolation — a crash in one worker crashes everything. In the Docker version, each worker is a separate Linux process with its own network namespace, memory limit, and filesystem. A container crash (`docker stop`, OOM kill, or process segfault) is invisible to other containers. The master detects it only through a failed HTTP call, exactly as a real distributed system would.

**Why `host.docker.internal` for Ollama:**
Docker containers cannot access `localhost` of the host machine. `host.docker.internal` is a special DNS name Docker resolves to the host's IP from inside the container. This allows all worker containers to reach the Ollama instance running on the host GPU, without running Ollama inside Docker (which would lose GPU access).

**Why ChromaDB uses a shared volume:**
All workers need to query the same knowledge base. Rather than each worker indexing its own copy (slow startup, inconsistent), a named Docker volume (`chroma-data`) is mounted at `/app/chroma_db` in every worker container. The first worker to start indexes the documents; subsequent workers find the database already populated.

**Limitation — single GPU:**
All worker containers share one GTX 960M through the Ollama HTTP interface. True multi-GPU distribution would require separate Ollama instances on separate machines, each with a dedicated GPU. Within this constraint, CPU, RAM, and network are still fully isolated per container, which is the relevant demonstration for the load balancing and fault tolerance portions of the project.

---

*Phase 9 — Distributed Workers via Docker · CSE354 · Ain Shams University · 2025/2026*
