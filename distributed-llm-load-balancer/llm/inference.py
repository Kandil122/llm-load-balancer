# llm/inference.py
import aiohttp
import json
from common.config import config


async def run_llm(query: str, context: str) -> str:
    """
    Send a prompt to Ollama's local REST API and return the response.
    Model: gemma3:270m running at localhost:11434
    """
    prompt = f"""You are a helpful assistant. Use the context below to answer concisely.

Context:
{context}

Question: {query}

Answer in 2-3 sentences:"""

    payload = {
        "model": config.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 150,   # limit tokens for speed
        }
    }

    url = f"{config.ollama_base_url}/api/generate"

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Ollama API error: {resp.status}")
            data = await resp.json()
            return data.get("response", "").strip()
