# FastAPI route triggered on /query endpoint

from fastapi import APIRouter, Form, HTTPException
from app.services.query_service import query_service_main

router = APIRouter(prefix="/query", tags=["Querying"])

@router.post("/")
async def query_endpoint(question: str = Form(...)):
    """
    Query the ingested document chunks with a user question.
    """
    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        answer = query_service_main(question)
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
