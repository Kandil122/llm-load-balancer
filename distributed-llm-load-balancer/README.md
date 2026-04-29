# Distributed LLM Load Balancer

CSE354 Distributed Computing Project — Ain Shams University

## Quick Start

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Phase 1 dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements-phase1.txt

# 3. Start Ollama
ollama serve &
ollama pull llama3.2:1b

# 4. Run
python main.py --strategy round_robin --users 10 --workers 4
```

## Strategies
- `round_robin`
- `least_connections`
- `load_aware`

## Testing
```bash
pytest tests/ -v
```
