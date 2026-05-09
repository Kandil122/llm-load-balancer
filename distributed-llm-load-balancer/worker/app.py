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
    print(f"[Worker {WORKER_ID}] Ready")


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
        import traceback
        traceback.print_exc()
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
