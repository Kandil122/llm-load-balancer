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
