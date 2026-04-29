# Phase 4 — Ollama LLM Integration
> `llm/inference.py`

---

## What You're Building

The bridge between your distributed system and the local `llama3.2:1b` model
running in Ollama. Every worker calls this to generate a real AI response
using the RAG context retrieved in Phase 3.

---

## How It Works

```
run_llm(query, context)
    │
    ▼
Build prompt:
  "You are a helpful assistant.
   Context: {3 retrieved documents}
   Question: {user query}
   Answer in 2-3 sentences:"
    │
    ▼
HTTP POST → http://localhost:11434/api/generate
  payload: { model: "llama3.2:1b", prompt: "...", stream: false }
    │
    ▼
Ollama loads model into GTX 960M VRAM (4GB)
Runs inference on GPU
    │
    ▼
Returns: { "response": "generated text..." }
    │
    ▼
Return response.text to worker
```

---

## GPU Monitoring During Tests

Open a second terminal and watch your GPU live while running tests:

```bash
# Terminal 2 — keep this open during all Phase 4 tests
watch -n 1 nvidia-smi
```

You should see GPU-Util jump from ~0% to 60–90% during inference.
This is what you'll record for your report and show in the demo video.

---

## ✅ Phase 4 Tests

### Test 1 — Single LLM call (no RAG)

```bash
uv run python -c "
import asyncio
import time
from llm.inference import run_llm

async def test():
    print('Sending single request to llama3.2:1b...')
    print('(Watch nvidia-smi in another terminal)')
    print()

    start = time.time()
    result = await run_llm(
        query='What is load balancing?',
        context='Load balancing distributes requests across multiple servers.'
    )
    elapsed = time.time() - start

    print(f'Response ({elapsed:.2f}s):')
    print(f'  {result}')
    print()
    assert len(result) > 10, 'Response too short — model may not be working'
    print('✅ Single LLM call works correctly')

asyncio.run(test())
"
```

Expected output:
```
Sending single request to llama3.2:1b...
(Watch nvidia-smi in another terminal)

Response (3.42s):
  Load balancing is a technique that distributes incoming network
  traffic across multiple servers to prevent any single server from
  becoming overloaded, ensuring high availability and reliability.

✅ Single LLM call works correctly
```

---

### Test 2 — Full RAG → LLM pipeline (end to end)

```bash
uv run python -c "
import asyncio
import time
from rag.retriever import retrieve_context
from llm.inference import run_llm

async def test():
    query = 'How does fault tolerance work in distributed systems?'

    print(f'Query: {query}')
    print()

    # Step 1: RAG
    print('Step 1: Retrieving context from ChromaDB...')
    context = await retrieve_context(query)
    print(f'Context retrieved ({len(context)} chars):')
    print(context)
    print()

    # Step 2: LLM
    print('Step 2: Sending to llama3.2:1b...')
    start = time.time()
    result = await run_llm(query, context)
    elapsed = time.time() - start

    print(f'LLM Response ({elapsed:.2f}s):')
    print(f'  {result}')
    print()
    print('✅ Full RAG → LLM pipeline works end to end')

asyncio.run(test())
"
```

---

### Test 3 — Measure real latency

```bash
uv run python -c "
import asyncio
import time
from llm.inference import run_llm
from rag.retriever import retrieve_context

async def test():
    queries = [
        'What is round robin load balancing?',
        'Explain GPU cluster task distribution.',
        'How does a heartbeat mechanism work?',
    ]

    latencies = []
    for query in queries:
        context = await retrieve_context(query)
        start = time.time()
        result = await run_llm(query, context)
        elapsed = time.time() - start
        latencies.append(elapsed)
        print(f'  [{elapsed:.2f}s] Q: {query[:45]}...')
        print(f'         A: {result[:80]}...')
        print()

    avg = sum(latencies) / len(latencies)
    print(f'Average latency: {avg:.2f}s')
    print(f'Min latency    : {min(latencies):.2f}s')
    print(f'Max latency    : {max(latencies):.2f}s')
    print()
    print('📊 Record these numbers in your report!')
    print('✅ Latency measurement complete')

asyncio.run(test())
"
```

---

### Test 4 — Worker uses LLM correctly

```bash
uv run python -c "
import asyncio
from workers.gpu_worker import GPUWorker
from common.models import Request

async def test():
    worker = GPUWorker(0)
    request = Request(id=1, query='What is a master node in a distributed system?')

    print(f'Processing request {request.id} through Worker {worker.id}...')
    print('(This goes through RAG + LLM)')
    print()

    response = await worker.process(request)

    print(f'Response ID     : {response.id}')
    print(f'Worker ID       : {response.worker_id}')
    print(f'Success         : {response.success}')
    print(f'Latency         : {response.latency:.2f}s')
    print(f'Result preview  : {response.result[:100]}...')
    print()

    assert response.success == True
    assert response.worker_id == 0
    assert response.latency > 0
    print('✅ Worker processes request correctly through full pipeline')

asyncio.run(test())
"
```

---

### Test 5 — Two workers in parallel

```bash
uv run python -c "
import asyncio
import time
from workers.gpu_worker import GPUWorker
from common.models import Request

async def test():
    workers = [GPUWorker(0), GPUWorker(1)]
    requests = [
        Request(id=0, query='What is load balancing?'),
        Request(id=1, query='How does fault tolerance work?'),
    ]

    print('Running 2 workers in parallel...')
    start = time.time()

    responses = await asyncio.gather(
        workers[0].process(requests[0]),
        workers[1].process(requests[1]),
    )

    elapsed = time.time() - start

    for r in responses:
        print(f'  Worker {r.worker_id} | Request {r.id} | '
              f'Latency {r.latency:.2f}s | Success {r.success}')

    print(f'Total wall time: {elapsed:.2f}s')
    print()
    print('✅ Workers process requests in parallel')

asyncio.run(test())
"
```

> **Note:** On a GTX 960M with a single GPU, both workers share the same hardware.
> True parallelism is limited by the GPU — one inference runs at a time.
> For the report, note this is a simulation; in a real cluster each worker
> would have its own dedicated GPU.

---

### Test 6 — Verify GPU is being used

```bash
# Run this while Test 3 or Test 4 is running in another terminal
nvidia-smi

# Or log it to file while tests run:
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu \
           --format=csv --loop=1 > gpu_log.csv &

# Run a test
uv run python -c "
import asyncio
from llm.inference import run_llm
from rag.retriever import retrieve_context

async def test():
    for i in range(3):
        ctx = await retrieve_context('explain distributed computing')
        r = await run_llm('explain distributed computing', ctx)
        print(f'Request {i+1} done')

asyncio.run(test())
"

# Stop logging
kill %1
cat gpu_log.csv
```

---

## ✅ Phase 4 Complete Checklist

- [ ] Single LLM call returns real text response
- [ ] Response is relevant to the query (not garbage)
- [ ] Latency measured and recorded (expect 2–8s on GTX 960M)
- [ ] Full RAG → LLM pipeline works end to end
- [ ] Worker's `process()` method correctly calls RAG then LLM
- [ ] `nvidia-smi` shows GPU utilization during inference
- [ ] Two workers run in parallel without errors

---

## Numbers to Record for Report

During Test 3, note down:
```
Single request latency (no load): ____s
Average across 3 queries        : ____s
GPU utilization during inference : ____%
VRAM usage during inference      : ____MB / 4096MB
GPU temperature during inference : ____°C
```

---

## Common Issues

**`Connection refused` on `localhost:11434`**
```bash
# Ollama is not running — start it:
ollama serve
```

**Response is empty string**
The model may have timed out. Increase timeout in `llm/inference.py`:
```python
timeout=aiohttp.ClientTimeout(total=180)  # 3 minutes
```

**Response is very slow (>30 seconds)**
llama3.2:1b may be running on CPU. Check:
```bash
nvidia-smi  # GPU-Util should be >0% during inference
```
If GPU-Util stays at 0%, Ollama is not using your GPU.
Fix:
```bash
# Check Ollama sees your GPU
ollama run llama3.2:1b "hi" --verbose
# Look for: "GPU layers: 16" in output
```

**`aiohttp.ClientError`**
Make sure Ollama is running and the model is pulled:
```bash
ollama list  # should show llama3.2:1b
```

---

➡️ **Next: Phase 5 — Metrics & Dashboard**
