# Phase 1 — Foundation
> `common/models.py` · `common/config.py`

---

## What You're Building

The shared data structures and configuration that every other module imports.
Nothing runs without this — build it first, test it before moving on.

---

## Step 1 — Run the setup script

```bash
bash setup_project.sh
cd distributed-llm-load-balancer
```

## Step 2 — Create a virtual environment and install Phase 1 dependencies

```bash
# Create and activate a normal Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install only the dependencies needed for Phase 1
python -m pip install -r requirements-phase1.txt
```

Expected output:
```
Successfully installed ...
```

## Step 3 — Verify Ollama is running with llama3.2:1b

```bash
# Terminal 1 — leave this open the whole project
ollama serve

# Terminal 2 — verify model is available
ollama list
```

Expected output:
```
NAME            ID              SIZE    MODIFIED
llama3.2:1b     ...             1.3 GB  ...
```

If llama3.2:1b is not listed:
```bash
ollama pull llama3.2:1b
```

---

## Files to Verify

Open `common/models.py` and confirm these three dataclasses exist:

```python
@dataclass
class Request:
    id: int
    query: str
    timestamp: float   # auto-filled with time.time()

@dataclass
class Response:
    id: int
    result: str
    latency: float
    worker_id: int
    success: bool
    error: Optional[str]

@dataclass
class WorkerStatus:
    id: int
    is_alive: bool
    active_connections: int
    avg_latency: float
    total_processed: int
    ...
```

Open `common/config.py` and confirm it reads from `.env`:
```python
class Settings(BaseSettings):
    ollama_base_url: str
    ollama_model: str       # should be "llama3.2:1b"
    num_workers: int
    num_users: int
    ...
```

---

## ✅ Phase 1 Tests

### Test 1 — Import models with no errors

```bash
python3 -c "
from common.models import Request, Response, WorkerStatus
import time

# Create a request
req = Request(id=1, query='What is load balancing?')
print(f'Request created: id={req.id}, query={req.query}')
print(f'Timestamp auto-filled: {req.timestamp > 0}')

# Create a response
res = Response(id=1, result='Load balancing distributes traffic.', latency=0.42, worker_id=2, success=True)
print(f'Response created: latency={res.latency}s, worker={res.worker_id}')

# Create worker status
ws = WorkerStatus(id=0)
ws.update_latency(1.2)
ws.update_latency(0.8)
print(f'WorkerStatus avg_latency: {ws.avg_latency}s')
print(f'WorkerStatus load_score: {ws.load_score}')
print()
print('✅ All models import and work correctly')
"
```

Expected output:
```
Request created: id=1, query=What is load balancing?
Timestamp auto-filled: True
Response created: latency=0.42s, worker=2
WorkerStatus avg_latency: 1.0s
WorkerStatus load_score: 0.0
✅ All models import and work correctly
```

---

### Test 2 — Config loads from .env

```bash
python3 -c "
from common.config import config
print(f'Ollama URL   : {config.ollama_base_url}')
print(f'Model        : {config.ollama_model}')
print(f'Num Workers  : {config.num_workers}')
print(f'Num Users    : {config.num_users}')
print(f'LB Strategy  : {config.lb_strategy}')
print(f'Fault Sim    : {config.worker_failure_simulation}')
print()
print('✅ Config loaded from .env correctly')
"
```

Expected output:
```
Ollama URL   : http://localhost:11434
Model        : llama3.2:1b
Num Workers  : 4
Num Users    : 20
LB Strategy  : round_robin
Fault Sim    : True
✅ Config loaded from .env correctly
```

---

### Test 3 — Ollama API reachable

```bash
python3 -c "
import asyncio
import aiohttp

async def check_ollama():
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get('http://localhost:11434/api/tags') as r:
                data = await r.json()
                models = [m['name'] for m in data.get('models', [])]
                print(f'Ollama is running. Available models: {models}')
                if any('llama3.2' in m for m in models):
                    print('✅ llama3.2:1b is available')
                else:
                    print('❌ llama3.2:1b not found — run: ollama pull llama3.2:1b')
    except Exception as e:
        print(f'❌ Ollama not reachable: {e}')
        print('   Make sure ollama serve is running in another terminal')

asyncio.run(check_ollama())
"
```

---

## ✅ Phase 1 Complete Checklist

- [ ] `.venv` is activated
- [ ] `python -m pip install -r requirements-phase1.txt` ran with no errors
- [ ] `common/models.py` imports cleanly
- [ ] `common/config.py` reads `.env` values correctly
- [ ] `.env` has `OLLAMA_MODEL=llama3.2:1b`
- [ ] Ollama is running and `llama3.2:1b` is available
- [ ] All three tests above pass

---

## Common Issues

**`ModuleNotFoundError: pydantic_settings`**
```bash
source .venv/bin/activate
python -m pip install -r requirements-phase1.txt
```

**`Config shows wrong model name`**
Check your `.env` file — make sure it says `OLLAMA_MODEL=llama3.2:1b` not `gemma2:2b`

**`Ollama not reachable`**
```bash
# Make sure this is running in a separate terminal
ollama serve
```

---

➡️ **Next: Phase 2 — Workers + Load Balancer**