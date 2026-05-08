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
    await scheduler.close()


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
    scheduler = HttpScheduler.instance()
    session = await scheduler.get_session()
    
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
