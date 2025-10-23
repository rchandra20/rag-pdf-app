import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

# Setup
load_dotenv()

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG App", layout="wide")
st.title("RAG App")

# Sidebar — File Upload Section
st.sidebar.header("PDF File Upload")

uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select one or more PDF files to ingest into the vector store"
)

# ---- Ingest Button ----
if uploaded_files:
    st.sidebar.success(f"Selected {len(uploaded_files)} file(s).")

    if st.sidebar.button("Ingest Files", type="primary"):
        with st.spinner("Processing files..."):
            try:
                files = [
                    ("files", (f.name, f.getvalue(), "application/pdf"))
                    for f in uploaded_files
                ]

                response = requests.post(f"{FASTAPI_URL}/ingest/", files=files)

                if response.status_code == 200:
                    st.sidebar.success("Files ingested successfully!")
                else:
                    st.sidebar.error(f"Error: {response.text}")

            except requests.exceptions.ConnectionError:
                st.sidebar.error(
                    "Cannot connect to FastAPI server. Make sure it's running at http://localhost:8000"
                )
            except Exception as e:
                st.sidebar.error(f"Unexpected error: {str(e)}")

# Sidebar — Vector Store Status
st.sidebar.markdown("### Ingested Files")

vector_store_path = "vector_store.json"

if not os.path.exists(vector_store_path):
    st.sidebar.info("No files ingested yet.")
else:
    try:
        with open(vector_store_path, "r") as f:
            content = f.read().strip()

        if not content:
            st.sidebar.info("Vector store is empty. Ingest files to begin.")
        else:
            try:
                vector_data = json.loads(content)
                if isinstance(vector_data, list) and len(vector_data) > 0:
                    sources = sorted(set(item.get("source_file", "Unknown") for item in vector_data))
                    st.sidebar.success(f"{len(vector_data)} chunks from {len(sources)} files:")
                    for source in sources:
                        count = sum(1 for item in vector_data if item.get("source_file") == source)
                        st.sidebar.write(f"- {source} ({count} chunks)")
                else:
                    st.sidebar.info("Vector store exists but contains no valid data.")
            except json.JSONDecodeError:
                st.sidebar.warning("Vector store file is corrupted or not valid JSON.")
    except Exception as e:
        st.sidebar.error(f"Error reading vector store: {str(e)}")

# Query Interface
st.header("Query Your Knowledge Base")

with st.form(key="query_form"):
    query_text = st.text_area(
        "Enter your question:",
        placeholder="Ask a question about your uploaded documents...",
        height=100,
        help="Type your question and click Submit to search the knowledge base"
    )
    submitted = st.form_submit_button("Submit Query", type="primary")

# ---- Query Handling ----
if submitted:
    query_text = (query_text or "").strip()

    if not query_text:
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Searching knowledge base..."):
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/query/",
                    data={"question": query_text}
                )

                if response.status_code == 200:
                    result = response.json()

                    st.subheader("Answer:")
                    st.write(result.get("answer", "No answer could be generated. Try a different question."))

                    if "retrieved_chunks" in result:
                        st.info(f"Retrieved chunk IDs: {result['retrieved_chunks']}")
                else:
                    st.error(f"Error: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI server. Make sure it's running at http://localhost:8000")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

# Connection Status
st.markdown("---")
st.markdown("Connection Status")

try:
    response = requests.get(f"{FASTAPI_URL}/docs", timeout=2)
    if response.status_code == 200:
        st.success("Connected to FastAPI server.")
    else:
        st.error("FastAPI server not responding properly.")
except requests.exceptions.ConnectionError:
    st.error("Cannot connect to FastAPI server.")
    st.markdown("To start the server, run:")
    st.code("cd rag_app && uvicorn app.main:app --reload")
except Exception as e:
    st.error(f"Connection error: {str(e)}")