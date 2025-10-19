# FastAPI app

from fastapi import FastAPI
from routers import ingest, query

app = FastAPI(title="RAG Pipeline")

# 2 API Endpoints: Ingest and Query
app.include_router(ingest.router)
app.include_router(query.router)
