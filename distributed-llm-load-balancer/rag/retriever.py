# rag/retriever.py
import asyncio
from rag.indexer import get_collection


async def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve the top_k most relevant documents for a query.
    Returns them as a single context string.
    """
    loop = asyncio.get_event_loop()

    def _query():
        collection, model = get_collection()
        query_embedding = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        docs = results["documents"][0]
        return "\n".join(f"- {doc}" for doc in docs)

    # Run in thread pool to avoid blocking event loop
    context = await loop.run_in_executor(None, _query)
    return context
