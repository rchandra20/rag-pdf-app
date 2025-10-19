# FastAPI app

from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import io
from PyPDF2 import PdfReader

app = FastAPI(title="RAG Pipeline")

@app.post("/ingest")
async def ingest_endpoint(files: List[UploadFile] = File(...)):
    """Upload 1+ PDFs for ingestion"""

    # Fill in function

    return {"status": "success", "ingested_docs": len(files)}

@app.post("/query")
async def query_endpoint(question: str = Form(...)):
    """Query on the ingested PDFs"""

    # Fill in function

    return {"answer": f"'{question}' not found in the documents."}
