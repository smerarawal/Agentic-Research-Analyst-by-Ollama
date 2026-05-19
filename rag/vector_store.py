"""
RAG vector store using ChromaDB with local Ollama embeddings.
Requires: ollama pull nomic-embed-text
"""

import uuid
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


# Local embedding model — no OpenAI key needed
embedding_fn = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text"
)

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="research_memory",
    embedding_function=embedding_fn
)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 30) -> list:
    """
    Split text into overlapping chunks.
    Reduced from 500 to 300 words to stay within llama3 context window.
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def add_documents(documents: list) -> None:
    """Chunk and embed a list of text documents into ChromaDB."""

    for doc in documents:
        chunks = chunk_text(doc)
        for chunk in chunks:
            collection.add(
                documents=[chunk],
                ids=[str(uuid.uuid4())]
            )


def retrieve(query: str, n_results: int = 2) -> dict:
    """Retrieve the most semantically relevant chunks for a query."""

    count = collection.count()

    if count == 0:
        return {"documents": [["No context available yet."]]}

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, count)
    )

    return results


def get_relevant_context(query: str) -> str:
    """
    Returns top 2 most relevant chunks joined as a single string.
    Reduced from 3 to 2 to keep prompt within llama3 context window.
    """

    results = retrieve(query, n_results=2)
    chunks  = results.get("documents", [[]])[0]
    return "\n\n".join(chunks) if chunks else ""
