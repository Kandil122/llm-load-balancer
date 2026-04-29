# Phase 3 — RAG Pipeline
> `rag/indexer.py` · `rag/retriever.py`

---

## What You're Building

A local Retrieval-Augmented Generation pipeline that:
1. Embeds 30 knowledge base documents into ChromaDB on startup
2. For every incoming query, finds the 3 most relevant documents
3. Returns them as a context string passed to the LLM

This is what makes your system give informed answers instead of just raw LLM guesses.

---

## How RAG Works in This Project

```
Startup (once):
  30 documents
      │
      ▼ sentence-transformers (all-MiniLM-L6-v2)
  30 embeddings (384-dimensional vectors)
      │
      ▼
  ChromaDB (stored on disk at ./chroma_db)

Per request:
  user query: "What is load balancing?"
      │
      ▼ sentence-transformers (same model)
  query embedding (384-dimensional vector)
      │
      ▼ cosine similarity search in ChromaDB
  top-3 matching documents
      │
      ▼
  context string → passed to Ollama prompt
```

---

## First Run Warning

The first time you run Phase 3 tests, `sentence-transformers` will download
the `all-MiniLM-L6-v2` model (~90MB). This happens once and is cached locally.
You will see a progress bar — this is normal.

---

## ✅ Phase 3 Tests

### Test 1 — Index the knowledge base

```bash
uv run python -c "
from rag.indexer import get_collection

print('Initializing ChromaDB and indexing documents...')
collection, model = get_collection()

count = collection.count()
print(f'Documents in ChromaDB: {count}')
assert count == 30, f'Expected 30 documents, got {count}'
print()
print('✅ Knowledge base indexed successfully')
"
```

Expected output (first run):
```
📦 Loading embedding model (first time only)...
✅ Embedding model loaded
📚 Indexing knowledge base into ChromaDB...
✅ Indexed 30 documents
Documents in ChromaDB: 30
✅ Knowledge base indexed successfully
```

Expected output (subsequent runs):
```
✅ ChromaDB already has 30 documents
Documents in ChromaDB: 30
✅ Knowledge base indexed successfully
```

---

### Test 2 — ChromaDB persists between runs

```bash
# Run twice — second run should NOT re-index
uv run python -c "from rag.indexer import get_collection; get_collection()"
uv run python -c "from rag.indexer import get_collection; get_collection()"
```

Second run should print:
```
✅ ChromaDB already has 30 documents
```
Not:
```
📚 Indexing knowledge base into ChromaDB...
```

---

### Test 3 — Retriever returns relevant context

```bash
uv run python -c "
import asyncio
from rag.retriever import retrieve_context

async def test():
    query = 'What is load balancing?'
    print(f'Query: {query}')
    print()

    context = await retrieve_context(query)
    print('Retrieved context:')
    print(context)
    print()

    assert isinstance(context, str)
    assert len(context) > 50
    assert 'load' in context.lower() or 'balanc' in context.lower()
    print('✅ Retriever returns relevant context')

asyncio.run(test())
"
```

Expected output:
```
Query: What is load balancing?

Retrieved context:
- Load balancing distributes network traffic across multiple servers...
- Round robin load balancing cycles through a list of servers...
- Least connections load balancing routes traffic to the server...

✅ Retriever returns relevant context
```

---

### Test 4 — Retriever is topic-aware

```bash
uv run python -c "
import asyncio
from rag.retriever import retrieve_context

async def test():
    queries = [
        ('fault tolerance', ['fault', 'fail', 'node']),
        ('GPU inference', ['gpu', 'cuda', 'inference']),
        ('vector database', ['vector', 'chromadb', 'embedding']),
    ]

    for query, expected_keywords in queries:
        context = await retrieve_context(query)
        found = any(kw in context.lower() for kw in expected_keywords)
        status = '✅' if found else '⚠️ '
        print(f'{status} Query: \"{query}\" → relevant context: {found}')

asyncio.run(test())
"
```

Expected output:
```
✅ Query: "fault tolerance" → relevant context: True
✅ Query: "GPU inference" → relevant context: True
✅ Query: "vector database" → relevant context: True
```

---

### Test 5 — Retriever is non-blocking (runs in thread pool)

```bash
uv run python -c "
import asyncio
import time
from rag.retriever import retrieve_context

async def test():
    # Run 5 retrievals concurrently — should not block event loop
    start = time.time()

    tasks = [
        retrieve_context(q) for q in [
            'load balancing',
            'fault tolerance',
            'GPU computing',
            'RAG pipeline',
            'distributed systems',
        ]
    ]

    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    print(f'5 concurrent retrievals completed in {elapsed:.2f}s')
    print(f'All returned context: {all(len(r) > 0 for r in results)}')
    print()
    print('✅ Retriever handles concurrent queries correctly')

asyncio.run(test())
"
```

Expected output:
```
5 concurrent retrievals completed in ~0.3s
All returned context: True
✅ Retriever handles concurrent queries correctly
```

---

### Test 6 — Run pytest for RAG

```bash
uv run pytest tests/test_rag.py -v
```

Expected output:
```
PASSED tests/test_rag.py::test_collection_initializes
PASSED tests/test_rag.py::test_retriever_returns_context
PASSED tests/test_rag.py::test_retriever_relevance

3 passed in Xs
```

---

### Test 7 — Full RAG pipeline output inspection

```bash
uv run python -c "
import asyncio
from rag.retriever import retrieve_context

async def test():
    test_queries = [
        'How does round robin work?',
        'What happens when a node fails?',
        'Explain async programming in Python',
        'What is P95 latency?',
    ]

    for query in test_queries:
        context = await retrieve_context(query, top_k=3)
        lines = context.strip().split('\n')
        print(f'Q: {query}')
        print(f'   Retrieved {len(lines)} documents')
        print(f'   First doc: {lines[0][:70]}...')
        print()

asyncio.run(test())
"
```

---

## ✅ Phase 3 Complete Checklist

- [ ] `all-MiniLM-L6-v2` model downloaded and cached
- [ ] ChromaDB created at `./chroma_db/`
- [ ] All 30 documents indexed on first run
- [ ] Second run does NOT re-index (idempotent)
- [ ] Retriever returns 3 relevant documents per query
- [ ] Retrieved context is topically relevant to the query
- [ ] 5 concurrent retrievals run without blocking
- [ ] All 3 pytest tests pass

---

## Common Issues

**`sentence_transformers` download is slow**
This is a one-time 90MB download. Run Test 1 and wait — it caches after.

**`ChromaDB` error on re-index**
If you see a duplicate ID error, delete and recreate the DB:
```bash
rm -rf chroma_db/
uv run python -c "from rag.indexer import get_collection; get_collection()"
```

**Context not relevant to query**
The `all-MiniLM-L6-v2` model is tuned for semantic similarity.
If results seem wrong, check that your query uses keywords that appear
conceptually in the 30 knowledge base documents in `rag/indexer.py`.

**`run_in_executor` error**
Make sure you call `retrieve_context()` inside an `async` function using `await`.

---

➡️ **Next: Phase 4 — Ollama LLM Integration**
