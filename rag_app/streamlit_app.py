import streamlit as st
import requests
import json
from typing import List

# FastAPI base URL
FASTAPI_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Pipeline App",
    layout="wide"
)

st.title("RAG Pipeline App")
st.markdown("Upload PDFs and query your knowledge base!")

# Sidebar for file upload
st.sidebar.header("File Upload")
st.sidebar.markdown("Upload PDF files to add to your knowledge base")

# Show ingested files
st.sidebar.markdown("### Ingested Files")
try:
    with open("vector_store.json", "r") as f:
        vector_data = json.load(f)
        if vector_data:
            # Get unique source files
            sources = list(set([item["source_file"] for item in vector_data]))
            st.sidebar.success(f"{len(vector_data)} chunks from {len(sources)} files")
            for source in sources:
                count = len([item for item in vector_data if item["source_file"] == source])
                st.sidebar.write(f"• {source} ({count} chunks)")
        else:
            st.sidebar.info("No files ingested yet")
except FileNotFoundError:
    st.sidebar.info("No files ingested yet")
except Exception as e:
    st.sidebar.error("Error reading vector store")

uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type=['pdf'],
    accept_multiple_files=True,
    help="Select one or more PDF files to ingest into the vector store"
)

if uploaded_files:
    st.sidebar.success(f"Selected {len(uploaded_files)} file(s)")
    
    if st.sidebar.button("Ingest Files", type="primary"):
        if uploaded_files:
            with st.spinner("Processing files..."):
                try:
                    # Prepare files for API
                    files = []
                    filenames = []
                    
                    for uploaded_file in uploaded_files:
                        files.append(("files", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")))
                        filenames.append(uploaded_file.name)
                    
                    # Make request to ingest endpoint
                    response = requests.post(
                        f"{FASTAPI_URL}/ingest/",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.sidebar.success("Files ingested successfully!")
                        
                        # Display results
                        with st.expander("Ingestion Results"):
                            if "processed_files" in result:
                                for file_result in result["processed_files"]:
                                    st.write(f"**{file_result['filename']}**: {file_result['num_chunks']} chunks processed")
                    else:
                        st.sidebar.error(f"Error: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.sidebar.error("Cannot connect to FastAPI server. Make sure it's running on http://localhost:8000")
                except Exception as e:
                    st.sidebar.error(f"Error: {str(e)}")

# Main query interface
st.header("Query Your Knowledge Base")

# Query input
query_text = st.text_area(
    "Enter your question:",
    placeholder="Ask a question about your uploaded documents...",
    height=100,
    help="Type your question and click Submit to search the knowledge base"
)

col1, col2 = st.columns([1, 4])

with col1:
    submit_button = st.button("Submit Query", type="primary")

# Display results
if submit_button and query_text.strip():
    with st.spinner("Searching knowledge base..."):
        try:
            # Make request to query endpoint
            response = requests.post(
                f"{FASTAPI_URL}/query/",
                data={"question": query_text.strip()}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display the answer
                st.subheader("Answer:")
                st.write(result.get("answer", "No answer generated"))
                
                # Display metadata if available
                if "sources" in result and result["sources"]:
                    st.subheader("Sources:")
                    sources = list(set(result["sources"]))  # Remove duplicates
                    for source in sources:
                        st.write(f"• {source}")
                
                if "retrieved_chunks" in result:
                    st.info(f"Retrieved {result['retrieved_chunks']} relevant chunks")
                    
            else:
                st.error(f"Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI server. Make sure it's running on http://localhost:8000")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif submit_button and not query_text.strip():
    st.warning("Please enter a question before submitting.")

# Instructions
st.markdown("---")
st.markdown("### Instructions:")
st.markdown("""
1. **Upload Files**: Use the sidebar to upload PDF files to your knowledge base
2. **Ingest**: Click "Ingest Files" to process and add them to the vector store
3. **Query**: Type your question in the text area and click "Submit Query"
4. **View Results**: Get answers based on your uploaded documents

**Note**: Make sure your FastAPI server is running on `http://localhost:8000`
""")

# Status indicator
st.markdown("---")
st.markdown("### Connection Status")

try:
    response = requests.get(f"{FASTAPI_URL}/docs", timeout=2)
    if response.status_code == 200:
        st.success("Connected to FastAPI server")
    else:
        st.error("FastAPI server not responding properly")
except requests.exceptions.ConnectionError:
    st.error("Cannot connect to FastAPI server")
    st.markdown("**To start the server, run:**")
    st.code("cd rag_app && uvicorn app.main:app --reload")
except Exception as e:
    st.error(f"Connection error: {str(e)}")
