# FastAPI app

import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


from fastapi import FastAPI
from app.routers import ingest, query


# Instantiate FastAPI app
app = FastAPI(title="RAG Pipeline")

# Include 2 API Endpoints: Ingest and Query
app.include_router(ingest.router)
app.include_router(query.router)
