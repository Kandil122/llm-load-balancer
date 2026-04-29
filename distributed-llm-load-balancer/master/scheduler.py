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
