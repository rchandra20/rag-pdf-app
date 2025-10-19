# FastAPI app

from fastapi import FastAPI
from dotenv import load_dotenv
from routers import ingest, query

# Load environment variables from .env file
load_dotenv()

# Instantiate FastAPI app
app = FastAPI(title="RAG Pipeline")

# Include 2 API Endpoints: Ingest and Query
app.include_router(ingest.router)
app.include_router(query.router)
