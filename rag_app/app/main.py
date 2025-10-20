# FastAPI app

from dotenv import load_dotenv
import os
from mistralai import Mistral
from fastapi import FastAPI
from app.routers import ingest, query

# Load environment variables
load_dotenv()

# Initialize Mistral client and models
API_KEY = os.environ.get("MISTRAL_API_KEY")

MISTRAL_CLIENT = Mistral(api_key=API_KEY)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
LANGUAGE_MODEL = os.environ.get("LANGUAGE_MODEL")

# Instantiate FastAPI app
app = FastAPI(title="RAG App")

# Include 2 API Endpoints: Ingest and Query
app.include_router(ingest.router)
app.include_router(query.router)

# Initialize vector store on startup
from app.services.ingest_service import VECTOR_STORE
print(f"FastAPI app started with {len(VECTOR_STORE)} entries in vector store")
