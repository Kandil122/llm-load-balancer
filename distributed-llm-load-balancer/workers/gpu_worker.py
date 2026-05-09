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
            print(f"\n❌ [Worker {self.id}] Error processing request {request.id}: {e}")
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
