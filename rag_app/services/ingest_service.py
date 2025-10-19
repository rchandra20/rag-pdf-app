# Logic to handle ingestion of PDFs and processing into chunks

import io
from typing import List
from PyPDF2 import PdfReader

def process_ingestion(filenames: List[str], file_bytes_list: List[bytes]):
    """Main function to orchestrate PDF ingestion."""
    results = []

    for name, file_bytes in zip(filenames, file_bytes_list):
        print(f"Processing file: {name}")

        text = extract_text_from_pdf(file_bytes)
        chunks = chunk_text(text)
        store_chunks(chunks, name)

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
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Splits text into overlapping chunks for later embedding/search."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


def store_chunks(chunks: List[str], source_name: str):
    """Saves chunks to your storage backend (stub for now)."""
    print(f"Storing {len(chunks)} chunks for {source_name}")
    # Example:
    # vector_store.add_texts(chunks, metadata={"source": source_name})
    # For now, this is just a stub.
    return True
