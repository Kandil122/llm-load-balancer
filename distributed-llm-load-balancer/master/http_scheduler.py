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
    def instance(cls, urls=None):
        if not cls._instance:
            cls._instance = cls(urls)
        return cls._instance

    def __init__(self, urls=None):
        self._index = 0
        self._lock = asyncio.Lock()
        
        # Default to internal Docker URLs if none provided
        self.worker_urls = urls or [
            f"http://worker-{i}:{8001 + i}"
            for i in range(config.num_workers)
        ]
        self._alive = {i: True for i in range(len(self.worker_urls))}

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
        max_retries = len(self.worker_urls)
        session = await self.get_session()
        
        for attempt in range(max_retries):
            worker_idx = await self._next_alive()
            if worker_idx is None:
                return {"success": False, "error": "No alive workers"}
            
            url = self.worker_urls[worker_idx]
            try:
                async with session.post(
                    f"{url}/process",
                    json={"id": request_id, "query": query}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("success"):
                            return data
                    
                    if self._alive[worker_idx]:
                        print(f"[Scheduler] {url} returned {resp.status}, retrying...")
                    self._alive[worker_idx] = False
            except Exception as e:
                if self._alive[worker_idx]:
                    print(f"[Scheduler] {url} unreachable ({type(e).__name__}), retrying...")
                self._alive[worker_idx] = False
        
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
        """Poll /health on all workers in parallel for fast recovery."""
        while True:
            await asyncio.sleep(config.heartbeat_interval)
            session = await self.get_session()
            
            async def check_worker(i, url):
                try:
                    async with session.get(
                        f"{url}/health",
                        timeout=aiohttp.ClientTimeout(total=1)
                    ) as r:
                        data = await r.json()
                        was_dead = not self._alive[i]
                        self._alive[i] = data.get("alive", False)
                        if was_dead and self._alive[i]:
                            print(f"✨ [Scheduler] {url} has RECOVERED and is now online.")
                except Exception:
                    self._alive[i] = False

            # Run all checks at once
            await asyncio.gather(*[check_worker(i, url) for i, url in enumerate(self.worker_urls)])
