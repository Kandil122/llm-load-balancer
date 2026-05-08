# rag/retriever.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.set_default_device("cpu")

import asyncio
from rag.indexer import get_collection

# Initialize ONCE globally (IMPORTANT FIX)
collection, model = get_collection()


async def retrieve_context(query: str, top_k: int = 3) -> str:
    loop = asyncio.get_event_loop()

    def _query():
        query_embedding = model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        docs = results["documents"][0]
        return "\n".join(f"- {doc}" for doc in docs)

    context = await loop.run_in_executor(None, _query)
    return context
