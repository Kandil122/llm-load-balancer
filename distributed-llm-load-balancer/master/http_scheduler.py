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
    _session = None

    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._index = 0
        self._lock = asyncio.Lock()
        self._alive = {i: True for i in range(len(WORKER_URLS))}

    async def get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def handle(self, request_id: int, query: str) -> dict:
        max_retries = len(WORKER_URLS)
        session = await self.get_session()
        
        for attempt in range(max_retries):
            worker_idx = await self._next_alive()
            if worker_idx is None:
                return {"success": False, "error": "No alive workers"}
            
            url = WORKER_URLS[worker_idx]
            try:
                async with session.post(
                    f"{url}/process",
                    json={"id": request_id, "query": query}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("success"):
                            return data
                    
                    # Non-200 = worker is unhealthy
                    self._alive[worker_idx] = False
                    print(f"[Scheduler] worker-{worker_idx} returned {resp.status}, retrying...")
            except Exception as e:
                # Network error = container is truly down
                self._alive[worker_idx] = False
                print(f"[Scheduler] worker-{worker_idx} unreachable ({type(e).__name__}), retrying...")
        
        return {"success": False, "error": "All retry attempts exhausted"}

    async def _next_alive(self):
        """Round-robin over alive workers."""
        async with self._lock:
            alive = [i for i, ok in self._alive.items() if ok]
            if not alive:
                return None
            chosen = alive[self._index % len(alive)]
            self._index = (self._index + 1) % len(alive)
            return chosen

    async def heartbeat_loop(self):
        """Poll /health on every worker every 2 seconds."""
        while True:
            await asyncio.sleep(config.heartbeat_interval)
            session = await self.get_session()
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
