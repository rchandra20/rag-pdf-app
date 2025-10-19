# Logic to handle ingestion of PDFs and processing into chunks

import io
from typing import List
from PyPDF2 import PdfReader

import os
from mistralai import Mistral

# Initialize Mistral client and embedding model
api_key = os.environ.get("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY is not set in environment variables.")

client = Mistral(api_key=api_key)
EMBEDDING_MODEL = "mistral-embed"

VECTOR_DB = []  # simple in-memory vector store

## Driver function
def process_ingestion(filenames: List[str], file_bytes_list: List[bytes]):
    """Main function to orchestrate PDF ingestion."""
    results = []

    for name, file_bytes in zip(filenames, file_bytes_list):
        print(f"Processing file: {name}")

        text = extract_text_from_pdf(file_bytes)
        chunks = chunk_text(text)
        add_chunks_to_vector_store(chunks)
        # store_chunks(chunks, name)

        results.append({
            "filename": name,
            "num_chunks": len(chunks)
        })

    return results

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text content from a PDF file (in memory)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    print(f"Extracted Text: {text}")
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Splits text into overlapping chunks for later embedding/search."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])

    print(f"Chunks created: {chunks}")

    return chunks

def add_chunks_to_vector_store(chunks):
    """
    Store each text chunk along with its embedding vector.
    
    Args:
        chunks (List[str]): Original text chunks
        embeddings_batch_response: EmbeddingResponse from Mistral
    """
    embeddings_batch_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs=chunks
    )
    print("Embeddings Output: {}".format(embeddings_batch_response))

    for chunk, data_obj in zip(chunks, embeddings_batch_response.data):
        VECTOR_DB.append((chunk, data_obj.embedding))

    print(f"VECTOR_DB: {VECTOR_DB}")