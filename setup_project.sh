#!/bin/bash

# ============================================================
#  distributed-llm-load-balancer — Project Setup Script
#  Model: llama3.2:1b via Ollama
#  CSE354 Distributed Computing — Ain Shams University
# ============================================================

set -e

PROJECT_NAME="distributed-llm-load-balancer"

echo "========================================"
echo "  Setting up $PROJECT_NAME"
echo "========================================"

# ── 1. Create root folder ───────────────────────────────────
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# ── 2. Create all directories ───────────────────────────────
mkdir -p common lb master workers rag llm client metrics tests results

echo "✅ Directories created"

# ── 3. Create all __init__.py files ─────────────────────────
for pkg in common lb master workers rag llm client metrics tests; do
  touch "$pkg/__init__.py"
done

echo "✅ __init__.py files created"

# ── 4. Create all source files ──────────────────────────────

# ── common/models.py ────────────────────────────────────────
cat > common/models.py << 'EOF'
# common/models.py
from dataclasses import dataclass, field
from typing import Optional
import time


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

    def update_latency(self, latency: float):
        self.total_processed += 1
        self.latency_sum += latency
        self.avg_latency = self.latency_sum / self.total_processed

    @property
    def load_score(self) -> float:
        """Used by load-aware strategy: lower is better."""
        return self.active_connections * max(self.avg_latency, 0.1)
EOF

# ── common/config.py ────────────────────────────────────────
cat > common/config.py << 'EOF'
# common/config.py
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:1b")

    # System
    num_workers: int = Field(default=4)
    num_users: int = Field(default=100)
    lb_strategy: str = Field(default="round_robin")

    # Fault tolerance
    worker_failure_simulation: bool = Field(default=True)
    failure_after_n_requests: int = Field(default=50)
    heartbeat_interval: float = Field(default=2.0)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global singleton
config = Settings()
EOF

# ── lb/load_balancer.py ─────────────────────────────────────
cat > lb/load_balancer.py << 'EOF'
# lb/load_balancer.py
from abc import ABC, abstractmethod
from typing import List, Optional
from workers.gpu_worker import GPUWorker


class LoadBalancer(ABC):
    """Abstract base class for all load balancing strategies."""

    @abstractmethod
    async def get_next_worker(self, workers: List[GPUWorker]) -> Optional[GPUWorker]:
        """Select the next worker to handle a request."""
        pass

    def alive_workers(self, workers: List[GPUWorker]) -> List[GPUWorker]:
        """Filter and return only alive workers."""
        return [w for w in workers if w.status.is_alive]
EOF

# ── lb/round_robin.py ───────────────────────────────────────
cat > lb/round_robin.py << 'EOF'
# lb/round_robin.py
import asyncio
from typing import List, Optional
from lb.load_balancer import LoadBalancer
from workers.gpu_worker import GPUWorker


class RoundRobinBalancer(LoadBalancer):
    """Cycles through workers in order, skipping dead ones."""

    def __init__(self):
        self._index = 0
        self._lock = asyncio.Lock()

    async def get_next_worker(self, workers: List[GPUWorker]) -> Optional[GPUWorker]:
        alive = self.alive_workers(workers)
        if not alive:
            return None
        async with self._lock:
            worker = alive[self._index % len(alive)]
            self._index = (self._index + 1) % len(alive)
        return worker
EOF

# ── lb/least_connections.py ─────────────────────────────────
cat > lb/least_connections.py << 'EOF'
# lb/least_connections.py
from typing import List, Optional
from lb.load_balancer import LoadBalancer
from workers.gpu_worker import GPUWorker


class LeastConnectionsBalancer(LoadBalancer):
    """Picks the worker with the fewest active connections."""

    async def get_next_worker(self, workers: List[GPUWorker]) -> Optional[GPUWorker]:
        alive = self.alive_workers(workers)
        if not alive:
            return None
        return min(alive, key=lambda w: w.status.active_connections)
EOF

# ── lb/load_aware.py ────────────────────────────────────────
cat > lb/load_aware.py << 'EOF'
# lb/load_aware.py
from typing import List, Optional
from lb.load_balancer import LoadBalancer
from workers.gpu_worker import GPUWorker


class LoadAwareBalancer(LoadBalancer):
    """
    Picks the worker with the lowest load score.
    Score = active_connections * avg_latency (lower is better).
    """

    async def get_next_worker(self, workers: List[GPUWorker]) -> Optional[GPUWorker]:
        alive = self.alive_workers(workers)
        if not alive:
            return None
        return min(alive, key=lambda w: w.status.load_score)
EOF

# ── workers/gpu_worker.py ───────────────────────────────────
cat > workers/gpu_worker.py << 'EOF'
# workers/gpu_worker.py
import asyncio
import time
from common.models import Request, Response, WorkerStatus
from common.config import config


class GPUWorker:
    def __init__(self, worker_id: int):
        self.id = worker_id
        self.status = WorkerStatus(id=worker_id)
        self._lock = asyncio.Lock()

    async def process(self, request: Request) -> Response:
        """Process a request through RAG → LLM pipeline."""
        if not self.status.is_alive:
            raise RuntimeError(f"Worker {self.id} is dead")

        start = time.time()
        self.status.active_connections += 1

        try:
            # Import here to avoid circular imports
            from rag.retriever import retrieve_context
            from llm.inference import run_llm

            # Step 1: RAG retrieval
            context = await retrieve_context(request.query)

            # Step 2: LLM inference
            result = await run_llm(request.query, context)

            latency = time.time() - start
            self.status.update_latency(latency)

            # Simulate failure after N requests
            if (config.worker_failure_simulation and
                    self.status.total_processed >= config.failure_after_n_requests and
                    self.id == 1):  # Kill worker 1 specifically for demo
                self.status.is_alive = False
                print(f"\n💀 [Worker {self.id}] SIMULATED FAILURE after "
                      f"{self.status.total_processed} requests")

            return Response(
                id=request.id,
                result=result,
                latency=latency,
                worker_id=self.id,
                success=True
            )

        except Exception as e:
            self.status.total_failed += 1
            latency = time.time() - start
            return Response(
                id=request.id,
                result="",
                latency=latency,
                worker_id=self.id,
                success=False,
                error=str(e)
            )
        finally:
            self.status.active_connections = max(0, self.status.active_connections - 1)

    def simulate_failure(self):
        """Manually trigger a worker failure."""
        self.status.is_alive = False
        print(f"\n💀 [Worker {self.id}] Manually failed")

    def revive(self):
        """Revive a dead worker."""
        self.status.is_alive = True
        print(f"\n✅ [Worker {self.id}] Revived")
EOF

# ── rag/indexer.py ──────────────────────────────────────────
cat > rag/indexer.py << 'EOF'
# rag/indexer.py
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# Knowledge base — distributed computing + AI topics
DOCUMENTS = [
    "Load balancing distributes network traffic across multiple servers to ensure no single server becomes overwhelmed.",
    "Round robin load balancing cycles through a list of servers sequentially, distributing requests evenly.",
    "Least connections load balancing routes traffic to the server with the fewest active connections.",
    "Fault tolerance is the ability of a system to continue operating even when some components fail.",
    "A distributed system is a system whose components are located on different networked computers.",
    "GPU clusters are groups of GPUs connected together to perform parallel computation tasks.",
    "LLM inference refers to the process of running a trained language model to generate text responses.",
    "RAG (Retrieval-Augmented Generation) enhances LLM responses by retrieving relevant documents first.",
    "ChromaDB is an open-source vector database designed for storing and querying embeddings.",
    "Sentence transformers convert text into dense vector representations called embeddings.",
    "Asyncio is Python's built-in library for writing concurrent code using the async/await syntax.",
    "A heartbeat mechanism periodically checks if nodes in a distributed system are still alive.",
    "Task reassignment occurs when a failed node's pending tasks are moved to healthy nodes.",
    "Throughput measures the number of requests a system can handle per unit of time.",
    "Latency is the time delay between a request being made and the response being received.",
    "Vector databases store data as high-dimensional vectors and support similarity search.",
    "CUDA is NVIDIA's parallel computing platform that enables GPU-accelerated computation.",
    "The master node in a distributed system coordinates task distribution to worker nodes.",
    "Worker nodes in a GPU cluster receive tasks, execute them, and return results to the master.",
    "Concurrent requests are multiple requests being processed simultaneously by a system.",
    "Horizontal scaling adds more machines to handle increased load in a distributed system.",
    "A scheduler decides when and where to run tasks in a distributed computing environment.",
    "Connection pooling maintains a pool of reusable connections to reduce connection overhead.",
    "P95 latency means 95% of requests complete within that time — a key performance metric.",
    "The GIL (Global Interpreter Lock) in Python prevents true parallel CPU-bound threading.",
    "Ollama is a tool for running large language models locally on your own hardware.",
    "llama3.2 is a compact but capable LLM model from Meta that runs efficiently on consumer hardware.",
    "Node failure detection can be implemented using periodic health checks or heartbeat signals.",
    "Load-aware routing selects workers based on both current load and historical response times.",
    "A semaphore limits the number of concurrent operations to prevent resource exhaustion.",
]

_client = None
_collection = None
_model = None


def get_chroma_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
    return _client


def get_collection():
    global _collection, _model
    client = get_chroma_client()

    if _model is None:
        print("📦 Loading embedding model (first time only)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")

    _collection = client.get_or_create_collection(
        name="distributed_llm_kb",
        metadata={"hnsw:space": "cosine"}
    )

    # Only index if empty
    if _collection.count() == 0:
        print("📚 Indexing knowledge base into ChromaDB...")
        embeddings = _model.encode(DOCUMENTS).tolist()
        _collection.add(
            documents=DOCUMENTS,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(DOCUMENTS))]
        )
        print(f"✅ Indexed {len(DOCUMENTS)} documents")
    else:
        print(f"✅ ChromaDB already has {_collection.count()} documents")

    return _collection, _model
EOF

# ── rag/retriever.py ────────────────────────────────────────
cat > rag/retriever.py << 'EOF'
# rag/retriever.py
import asyncio
from rag.indexer import get_collection


async def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve the top_k most relevant documents for a query.
    Returns them as a single context string.
    """
    loop = asyncio.get_event_loop()

    def _query():
        collection, model = get_collection()
        query_embedding = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        docs = results["documents"][0]
        return "\n".join(f"- {doc}" for doc in docs)

    # Run in thread pool to avoid blocking event loop
    context = await loop.run_in_executor(None, _query)
    return context
EOF

# ── llm/inference.py ────────────────────────────────────────
cat > llm/inference.py << 'EOF'
# llm/inference.py
import aiohttp
import json
from common.config import config


async def run_llm(query: str, context: str) -> str:
    """
    Send a prompt to Ollama's local REST API and return the response.
    Model: llama3.2:1b running at localhost:11434
    """
    prompt = f"""You are a helpful assistant. Use the context below to answer concisely.

Context:
{context}

Question: {query}

Answer in 2-3 sentences:"""

    payload = {
        "model": config.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 150,   # limit tokens for speed
        }
    }

    url = f"{config.ollama_base_url}/api/generate"

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Ollama API error: {resp.status}")
            data = await resp.json()
            return data.get("response", "").strip()
EOF

# ── master/scheduler.py ─────────────────────────────────────
cat > master/scheduler.py << 'EOF'
# master/scheduler.py
import asyncio
from typing import List
from common.models import Request, Response
from common.config import config
from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from metrics.collector import MetricsCollector


class Scheduler:
    def __init__(self, load_balancer: LoadBalancer, workers: List[GPUWorker],
                 collector: MetricsCollector):
        self.lb = load_balancer
        self.workers = workers
        self.collector = collector
        self._heartbeat_task = None

    async def start(self):
        """Start the heartbeat monitoring loop."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self):
        """Stop the heartbeat loop."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

    async def handle_request(self, request: Request) -> Response:
        """Dispatch a request to a worker with fault-tolerant retry."""
        max_retries = len(self.workers)

        for attempt in range(max_retries):
            worker = await self.lb.get_next_worker(self.workers)

            if worker is None:
                return Response(
                    id=request.id,
                    result="",
                    latency=0,
                    worker_id=-1,
                    success=False,
                    error="No alive workers available"
                )

            try:
                response = await worker.process(request)
                if response.success:
                    self.collector.record_success(worker.id, response.latency)
                else:
                    self.collector.record_failure(worker.id)
                return response

            except RuntimeError as e:
                # Worker died mid-request — retry with another worker
                print(f"\n⚠️  [Scheduler] Worker {worker.id} failed on request "
                      f"{request.id}, retrying (attempt {attempt + 1})...")
                self.collector.record_failure(worker.id)
                continue

        return Response(
            id=request.id,
            result="",
            latency=0,
            worker_id=-1,
            success=False,
            error="All retry attempts exhausted"
        )

    async def _heartbeat_loop(self):
        """Periodically check all workers and report dead ones."""
        while True:
            await asyncio.sleep(config.heartbeat_interval)
            for worker in self.workers:
                if not worker.status.is_alive:
                    self.collector.mark_worker_dead(worker.id)
EOF

# ── client/load_generator.py ────────────────────────────────
cat > client/load_generator.py << 'EOF'
# client/load_generator.py
import asyncio
import time
from typing import List
from common.models import Request
from master.scheduler import Scheduler
from metrics.collector import MetricsCollector

SAMPLE_QUERIES = [
    "What is load balancing and why is it important?",
    "How does round robin scheduling work?",
    "Explain fault tolerance in distributed systems.",
    "What is the difference between latency and throughput?",
    "How do GPU clusters handle parallel inference?",
    "What is RAG and how does it improve LLM responses?",
    "How does the least connections algorithm work?",
    "What happens when a node fails in a distributed system?",
    "Explain the role of a master node in a cluster.",
    "What is ChromaDB used for?",
]


async def simulate_user(user_id: int, scheduler: Scheduler, collector: MetricsCollector):
    """A single simulated user sending one request."""
    query = SAMPLE_QUERIES[user_id % len(SAMPLE_QUERIES)]
    request = Request(id=user_id, query=query)

    response = await scheduler.handle_request(request)
    collector.record_request_done()

    return response


async def run_load_test(scheduler: Scheduler, collector: MetricsCollector,
                        num_users: int = 100) -> List:
    """
    Simulate num_users concurrent users using asyncio.gather.
    Returns list of all responses.
    """
    print(f"\n🚀 Starting load test with {num_users} concurrent users...")
    collector.start_timer()

    tasks = [
        simulate_user(i, scheduler, collector)
        for i in range(num_users)
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    collector.stop_timer()

    return responses
EOF

# ── metrics/collector.py ────────────────────────────────────
cat > metrics/collector.py << 'EOF'
# metrics/collector.py
import time
import threading
from typing import Dict, List


class MetricsCollector:
    def __init__(self, num_workers: int):
        self._lock = threading.Lock()
        self.num_workers = num_workers

        # Counters
        self.total_requests = 0
        self.completed = 0
        self.failed = 0
        self.latencies: List[float] = []

        # Per-worker
        self.worker_counts: Dict[int, int] = {i: 0 for i in range(num_workers)}
        self.worker_failures: Dict[int, int] = {i: 0 for i in range(num_workers)}
        self.dead_workers = set()

        # Timing
        self._start_time = None
        self._end_time = None

    def start_timer(self):
        self._start_time = time.time()

    def stop_timer(self):
        self._end_time = time.time()

    def record_success(self, worker_id: int, latency: float):
        with self._lock:
            self.completed += 1
            self.latencies.append(latency)
            if worker_id in self.worker_counts:
                self.worker_counts[worker_id] += 1

    def record_failure(self, worker_id: int):
        with self._lock:
            self.failed += 1
            if worker_id in self.worker_failures:
                self.worker_failures[worker_id] += 1

    def record_request_done(self):
        with self._lock:
            self.total_requests += 1

    def mark_worker_dead(self, worker_id: int):
        with self._lock:
            self.dead_workers.add(worker_id)

    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def throughput(self) -> float:
        if self._start_time is None or self._end_time is None:
            return 0.0
        elapsed = self._end_time - self._start_time
        return self.completed / elapsed if elapsed > 0 else 0.0

    def get_summary(self) -> dict:
        return {
            "total": self.total_requests,
            "completed": self.completed,
            "failed": self.failed,
            "avg_latency": round(self.avg_latency, 3),
            "p95_latency": round(self.p95_latency, 3),
            "throughput": round(self.throughput, 2),
            "worker_counts": self.worker_counts,
            "dead_workers": list(self.dead_workers),
        }
EOF

# ── metrics/dashboard.py ────────────────────────────────────
cat > metrics/dashboard.py << 'EOF'
# metrics/dashboard.py
import psutil
import time
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from typing import List
from workers.gpu_worker import GPUWorker
from metrics.collector import MetricsCollector

console = Console()


def get_gpu_stats() -> str:
    """Try to get GPU stats via nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            util, mem_used, mem_total, temp = parts
            return f"GPU: {util}%  VRAM: {mem_used}MB/{mem_total}MB  Temp: {temp}°C"
    except Exception:
        pass
    return "GPU: N/A (nvidia-smi not available)"


def build_table(workers: List[GPUWorker], collector: MetricsCollector,
                strategy: str) -> Table:
    table = Table(box=box.ROUNDED, title=f"[bold cyan]DISTRIBUTED LLM SYSTEM[/bold cyan]",
                  title_justify="center")

    table.add_column("Worker", style="bold")
    table.add_column("Status")
    table.add_column("Active Conn", justify="center")
    table.add_column("Processed", justify="center")
    table.add_column("Avg Latency", justify="center")
    table.add_column("Failed", justify="center")

    for w in workers:
        status = "[green]🟢 alive[/green]" if w.status.is_alive else "[red]🔴 dead[/red]"
        table.add_row(
            f"Worker-{w.id}",
            status,
            str(w.status.active_connections),
            str(w.status.total_processed),
            f"{w.status.avg_latency:.2f}s",
            str(w.status.total_failed),
        )

    return table


def print_summary(workers: List[GPUWorker], collector: MetricsCollector, strategy: str):
    """Print final summary after load test completes."""
    summary = collector.get_summary()

    console.print("\n")
    console.rule("[bold cyan]FINAL RESULTS[/bold cyan]")

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Strategy", strategy)
    table.add_row("Total Requests", str(summary["total"]))
    table.add_row("Completed", f"[green]{summary['completed']}[/green]")
    table.add_row("Failed", f"[red]{summary['failed']}[/red]")
    table.add_row("Avg Latency", f"{summary['avg_latency']}s")
    table.add_row("P95 Latency", f"{summary['p95_latency']}s")
    table.add_row("Throughput", f"{summary['throughput']} req/s")
    table.add_row("Dead Workers", str(summary["dead_workers"]))

    console.print(table)

    console.print("\n[bold]Per-Worker Breakdown:[/bold]")
    for worker_id, count in summary["worker_counts"].items():
        bar = "█" * min(count, 50)
        console.print(f"  Worker-{worker_id}: {bar} ({count})")

    console.print(f"\n[dim]{get_gpu_stats()}[/dim]")
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    console.print(f"[dim]CPU: {cpu}%  RAM: {ram.used // (1024**3)}GB / "
                  f"{ram.total // (1024**3)}GB[/dim]")
EOF

# ── tests/test_load_balancer.py ─────────────────────────────
cat > tests/test_load_balancer.py << 'EOF'
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
EOF

# ── tests/test_fault_tolerance.py ──────────────────────────
cat > tests/test_fault_tolerance.py << 'EOF'
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
EOF

# ── tests/test_rag.py ───────────────────────────────────────
cat > tests/test_rag.py << 'EOF'
# tests/test_rag.py
import asyncio
import pytest
from rag.indexer import get_collection
from rag.retriever import retrieve_context


def test_collection_initializes():
    collection, model = get_collection()
    assert collection.count() > 0


@pytest.mark.asyncio
async def test_retriever_returns_context():
    context = await retrieve_context("What is load balancing?")
    assert isinstance(context, str)
    assert len(context) > 0
    assert "load" in context.lower() or "balanc" in context.lower()


@pytest.mark.asyncio
async def test_retriever_relevance():
    context = await retrieve_context("How does fault tolerance work?")
    assert "fault" in context.lower() or "fail" in context.lower()
EOF

# ── main.py ─────────────────────────────────────────────────
cat > main.py << 'EOF'
# main.py
import asyncio
import argparse
import csv
import os
from common.config import config
from workers.gpu_worker import GPUWorker
from lb.round_robin import RoundRobinBalancer
from lb.least_connections import LeastConnectionsBalancer
from lb.load_aware import LoadAwareBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test
from metrics.collector import MetricsCollector
from metrics.dashboard import print_summary
from rag.indexer import get_collection


STRATEGIES = {
    "round_robin": RoundRobinBalancer,
    "least_connections": LeastConnectionsBalancer,
    "load_aware": LoadAwareBalancer,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed LLM Load Balancer")
    parser.add_argument("--strategy", choices=STRATEGIES.keys(),
                        default=config.lb_strategy)
    parser.add_argument("--users", type=int, default=config.num_users)
    parser.add_argument("--workers", type=int, default=config.num_workers)
    parser.add_argument("--no-fault", action="store_true",
                        help="Disable fault simulation")
    parser.add_argument("--save-results", action="store_true",
                        help="Save results to CSV")
    return parser.parse_args()


async def main():
    args = parse_args()

    print("=" * 60)
    print("  DISTRIBUTED LLM LOAD BALANCER")
    print(f"  Strategy : {args.strategy}")
    print(f"  Workers  : {args.workers}")
    print(f"  Users    : {args.users}")
    print(f"  Model    : {config.ollama_model}")
    print("=" * 60)

    # Override config with CLI args
    config.worker_failure_simulation = not args.no_fault

    # Step 1: Initialize RAG
    print("\n📚 Initializing RAG knowledge base...")
    get_collection()

    # Step 2: Create workers
    workers = [GPUWorker(i) for i in range(args.workers)]

    # Step 3: Create load balancer
    lb = STRATEGIES[args.strategy]()

    # Step 4: Create metrics collector
    collector = MetricsCollector(num_workers=args.workers)

    # Step 5: Create scheduler
    scheduler = Scheduler(lb, workers, collector)
    await scheduler.start()

    # Step 6: Run load test
    responses = await run_load_test(scheduler, collector, num_users=args.users)

    # Step 7: Stop scheduler
    await scheduler.stop()

    # Step 8: Print summary
    print_summary(workers, collector, args.strategy)

    # Step 9: Save results
    if args.save_results:
        os.makedirs("results", exist_ok=True)
        filename = f"results/{args.strategy}_{args.users}users_{args.workers}workers.csv"
        summary = collector.get_summary()
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in summary.items():
                writer.writerow([k, v])
        print(f"\n💾 Results saved to {filename}")


if __name__ == "__main__":
    asyncio.run(main())
EOF

# ── .env ────────────────────────────────────────────────────
cat > .env << 'EOF'
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
NUM_WORKERS=4
NUM_USERS=20
LB_STRATEGY=round_robin
WORKER_FAILURE_SIMULATION=true
FAILURE_AFTER_N_REQUESTS=50
HEARTBEAT_INTERVAL=2.0
EOF

# ── .env.example ────────────────────────────────────────────
cat > .env.example << 'EOF'
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
NUM_WORKERS=4
NUM_USERS=20
LB_STRATEGY=round_robin
WORKER_FAILURE_SIMULATION=true
FAILURE_AFTER_N_REQUESTS=50
HEARTBEAT_INTERVAL=2.0
EOF

# ── .gitignore ──────────────────────────────────────────────
cat > .gitignore << 'EOF'
.env
__pycache__/
*.pyc
*.pyo
.venv/
chroma_db/
results/
.pytest_cache/
*.egg-info/
dist/
build/
EOF

# ── pyproject.toml ──────────────────────────────────────────
cat > pyproject.toml << 'EOF'
[project]
name = "distributed-llm-load-balancer"
version = "0.1.0"
description = "Distributed LLM inference with load balancing, RAG, and fault tolerance"
requires-python = ">=3.10"
dependencies = [
    "chromadb>=0.4.0",
    "sentence-transformers>=2.2.0",
    "pydantic-settings>=2.0.0",
    "rich>=13.0.0",
    "aiohttp>=3.9.0",
    "psutil>=5.9.0",
    "pytest>=7.0.0",
    "pytest-asyncio>=0.23.0",
    "python-dotenv>=1.0.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
EOF

# ── README.md ───────────────────────────────────────────────
cat > README.md << 'EOF'
# Distributed LLM Load Balancer

CSE354 Distributed Computing Project — Ain Shams University

## Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Start Ollama
ollama serve &
ollama pull llama3.2:1b

# 4. Run
uv run python main.py --strategy round_robin --users 10 --workers 4
```

## Strategies
- `round_robin`
- `least_connections`
- `load_aware`

## Testing
```bash
uv run pytest tests/ -v
```
EOF

echo ""
echo "========================================"
echo "  ✅ Project structure created!"
echo "========================================"
echo ""
echo "📁 Structure:"
find . -type f | sort | sed 's|./||' | sed 's|^|   |'
echo ""
echo "Next steps:"
echo "  1. cd $PROJECT_NAME"
echo "  2. curl -LsSf https://astral.sh/uv/install.sh | sh"
echo "  3. uv sync"
echo "  4. ollama serve & (in another terminal)"
echo "  5. Follow phase guides in order"
echo ""
