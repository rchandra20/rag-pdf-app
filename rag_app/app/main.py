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
if not API_KEY:
    raise RuntimeError("MISTRAL_API_KEY not set in environment variables.")

MISTRAL_CLIENT = Mistral(api_key=API_KEY)
EMBEDDING_MODEL = "mistral-embed"
LANGUAGE_MODEL = "mistral-small-2503"

# Instantiate FastAPI app
app = FastAPI(title="RAG App")

# Include 2 API Endpoints: Ingest and Query
app.include_router(ingest.router)
app.include_router(query.router)
