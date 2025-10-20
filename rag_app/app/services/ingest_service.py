# Logic to handle ingestion of PDFs and breaking into text chunks

import io
import re
import time
import json
import os
from typing import List
from PyPDF2 import PdfReader

VECTOR_STORE_FILE = "vector_store.json"
VECTOR_STORE = []  # simple custom in-memory vector store

def load_vector_store():
    """Load vector store from JSON file."""
    if os.path.exists(VECTOR_STORE_FILE):
        try:
            with open(VECTOR_STORE_FILE, 'r') as f:
                loaded_store = json.load(f)
                print(f"Loaded vector store with {len(loaded_store)} entries from {VECTOR_STORE_FILE}")
                return loaded_store
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return []
    return []

def save_vector_store():
    """Save vector store to JSON file."""
    try:
        with open(VECTOR_STORE_FILE, 'w') as f:
            json.dump(VECTOR_STORE, f)
        print(f"Vector store saved with {len(VECTOR_STORE)} entries to {VECTOR_STORE_FILE}")
    except Exception as e:
        print(f"Error saving vector store: {e}")

# Initialize vector store on module import
VECTOR_STORE = load_vector_store()

## Driver Function ##
def ingest_service_main(filenames: List[str], file_bytes_list: List[bytes]):
    """Main function to ingest PDFs and load into vector store."""
    results = []

    for name, file_bytes in zip(filenames, file_bytes_list):
        print(f"Processing file: {name}")

        text = extract_text_from_pdf(file_bytes)
        chunks = chunk_text(text)
        print("File: {}, Number Chunks: {}".format(name,len(chunks)))
        add_chunks_to_vector_store(chunks, name)

        results.append({
            "filename": name,
            "num_chunks": len(chunks)
        })

    return results

## Helper Functions ##
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF while cleaning up whitespace and line breaks.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        # Clean up extra spaces and newlines
        page_text = " ".join(page_text.split())
        text += page_text + "\n\n"  # preserve paragraph breaks
    return text.strip()


def chunk_text(
    text: str,
    max_tokens: int = 200,
    overlap_tokens: int = 50
) -> List[str]:
    """
    Chunk text for RAG pipelines:
    - Keeps sentences intact
    - Respects max token limits
    - Adds overlap for context
    """
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = []
    token_count = 0

    for sentence in sentences:
        sentence_tokens = sentence.split()
        sentence_len = len(sentence_tokens)

        if token_count + sentence_len <= max_tokens:
            current_chunk.extend(sentence_tokens)
            token_count += sentence_len
        else:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            # Start new chunk with overlap
            current_chunk = current_chunk[-overlap_tokens:] + sentence_tokens
            token_count = len(current_chunk)

    # Add any remaining text
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Optionally filter out tiny chunks
    # chunks = [c for c in chunks if len(c.split()) > 10]

    return chunks


def add_chunks_to_vector_store(chunks, file_name):
    from app.main import MISTRAL_CLIENT, EMBEDDING_MODEL
    """
    Store each text chunk along with its embedding vector.
    
    Args:
        chunks (List[str]): Original text chunks
        embeddings_batch_response: EmbeddingResponse from Mistral
    """

    # Set custom batch size to avoid Mistral service capacity limits
    BATCH_SIZE = 5

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"Embedding batch: {batch}")

        # Implement retry logic for rate limits / capacity errors
        MAX_RETRIES = 3
        BACKOFF_FACTOR = 10  # seconds; grows exponentially

        for i in range(MAX_RETRIES):
            try:
                embeddings_response = MISTRAL_CLIENT.embeddings.create(
                    model=EMBEDDING_MODEL,
                    inputs=batch
                )
                break  # success — exit retry loop

            except Exception as e:
                # Retry only for rate-limit or service capacity errors
                if "429" in str(e) or "capacity" in str(e).lower():
                    wait = BACKOFF_FACTOR * (2 ** i)
                    print(f"Rate limit hit — retrying in {wait} seconds (attempt {i+1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                else:
                    # Other errors should fail fast
                    raise

        else:
            # Executed if all retries fail
            raise RuntimeError(f"Failed to get embeddings for {file_name} after multiple retries.")
        
        # Append each chunk with its embedding and metadata
        for chunk_text, data_obj in zip(batch, embeddings_response.data):
            VECTOR_STORE.append({
                "chunk": chunk_text,
                "embedding": data_obj.embedding,
                "source_file": file_name
            })
            print(f"Stored chunk embedding: {chunk_text[:50]}...")
    
    # Save the vector store after all chunks are processed
    save_vector_store()