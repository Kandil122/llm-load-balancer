# tests/test_rag.py
import asyncio
import pytest
from rag.indexer import get_collection
from rag.retriever import retrieve_context


def test_collection_initializes():
    collection, model = get_collection()
    assert collection.count() > 0


@pytest.mark.asyncio
async def test_retriever_returns_context():
    context = await retrieve_context("What is load balancing?")
    assert isinstance(context, str)
    assert len(context) > 0
    assert "load" in context.lower() or "balanc" in context.lower()


@pytest.mark.asyncio
async def test_retriever_relevance():
    context = await retrieve_context("How does fault tolerance work?")
    assert "fault" in context.lower() or "fail" in context.lower()
