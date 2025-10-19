# FastAPI route triggered on /ingest endpoint

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app.services.ingest_service import process_ingestion

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

@router.post("/")
async def ingest_files(files: List[UploadFile] = File(...)):
    """Upload and process multiple files"""
    print("Hit ingest endpoint")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    try:
        file_bytes = [await f.read() for f in files]
        filenames = [f.filename for f in files]
        result = process_ingestion(filenames, file_bytes)
        return {"status": "success", "processed_files": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
