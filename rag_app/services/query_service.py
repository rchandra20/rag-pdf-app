# services/query_service.py

from .ingest_service import VECTOR_DB # shared in-memory vector DB
import numpy as np
from mistralai import Mistral
import os

# Initialize embedding client
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("Set MISTRAL_API_KEY in environment variables")

EMBEDDING_MODEL = "mistral-embed"
client = Mistral(api_key=api_key)


def embed_text(text: str):
    """Generate embedding vector for a single text string."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, inputs=[text])
    return response.data[0].embedding  # 1024-dim vector


def query_vector_store(question: str, top_k: int = 3):
    """
    Returns the top_k most relevant text chunks for the question.
    Uses cosine similarity over the in-memory VECTOR_DB.
    """
    if not VECTOR_DB:
        return {"answer": "No documents have been ingested yet."}

    # Embed the question
    query_embedding = embed_text(question)

    # Compute cosine similarity against all stored chunks
    results = []
    for chunk, chunk_emb in VECTOR_DB:
        sim = np.dot(query_embedding, chunk_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb))
        results.append((chunk, sim))

    # Sort by similarity and pick top_k
    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:top_k]

    # Format the output
    return {
        "question": question,
        "matches_found": len(top_results),
        "top_chunks": [{"text": r[0], "similarity": r[1]} for r in top_results]
    }