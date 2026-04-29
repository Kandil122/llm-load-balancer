# rag/indexer.py
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# Knowledge base — distributed computing + AI topics
DOCUMENTS = [
    "Load balancing distributes network traffic across multiple servers to ensure no single server becomes overwhelmed.",
    "Round robin load balancing cycles through a list of servers sequentially, distributing requests evenly.",
    "Least connections load balancing routes traffic to the server with the fewest active connections.",
    "Fault tolerance is the ability of a system to continue operating even when some components fail.",
    "A distributed system is a system whose components are located on different networked computers.",
    "GPU clusters are groups of GPUs connected together to perform parallel computation tasks.",
    "LLM inference refers to the process of running a trained language model to generate text responses.",
    "RAG (Retrieval-Augmented Generation) enhances LLM responses by retrieving relevant documents first.",
    "ChromaDB is an open-source vector database designed for storing and querying embeddings.",
    "Sentence transformers convert text into dense vector representations called embeddings.",
    "Asyncio is Python's built-in library for writing concurrent code using the async/await syntax.",
    "A heartbeat mechanism periodically checks if nodes in a distributed system are still alive.",
    "Task reassignment occurs when a failed node's pending tasks are moved to healthy nodes.",
    "Throughput measures the number of requests a system can handle per unit of time.",
    "Latency is the time delay between a request being made and the response being received.",
    "Vector databases store data as high-dimensional vectors and support similarity search.",
    "CUDA is NVIDIA's parallel computing platform that enables GPU-accelerated computation.",
    "The master node in a distributed system coordinates task distribution to worker nodes.",
    "Worker nodes in a GPU cluster receive tasks, execute them, and return results to the master.",
    "Concurrent requests are multiple requests being processed simultaneously by a system.",
    "Horizontal scaling adds more machines to handle increased load in a distributed system.",
    "A scheduler decides when and where to run tasks in a distributed computing environment.",
    "Connection pooling maintains a pool of reusable connections to reduce connection overhead.",
    "P95 latency means 95% of requests complete within that time — a key performance metric.",
    "The GIL (Global Interpreter Lock) in Python prevents true parallel CPU-bound threading.",
    "Ollama is a tool for running large language models locally on your own hardware.",
    "llama3.2 is a compact but capable LLM model from Meta that runs efficiently on consumer hardware.",
    "Node failure detection can be implemented using periodic health checks or heartbeat signals.",
    "Load-aware routing selects workers based on both current load and historical response times.",
    "A semaphore limits the number of concurrent operations to prevent resource exhaustion.",
]

_client = None
_collection = None
_model = None


def get_chroma_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
    return _client


def get_collection():
    global _collection, _model
    client = get_chroma_client()

    if _model is None:
        print("📦 Loading embedding model (first time only)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")

    _collection = client.get_or_create_collection(
        name="distributed_llm_kb",
        metadata={"hnsw:space": "cosine"}
    )

    # Only index if empty
    if _collection.count() == 0:
        print("📚 Indexing knowledge base into ChromaDB...")
        embeddings = _model.encode(DOCUMENTS).tolist()
        _collection.add(
            documents=DOCUMENTS,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(DOCUMENTS))]
        )
        print(f"✅ Indexed {len(DOCUMENTS)} documents")
    else:
        print(f"✅ ChromaDB already has {_collection.count()} documents")

    return _collection, _model
