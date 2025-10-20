#!/bin/bash

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

# Extract port numbers from URLs
FASTAPI_PORT=$(echo $FASTAPI_URL | sed 's/.*://')
STREAMLIT_PORT=$(echo $STREAMLIT_URL | sed 's/.*://')

echo "Starting RAG App..."
echo "FastAPI will run on: $FASTAPI_URL"
echo "Streamlit will run on: $STREAMLIT_URL"
echo "Press Ctrl+C to stop both servers"
echo ""

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup INT

# Start FastAPI server in background
cd rag_app
echo "Starting FastAPI server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port $FASTAPI_PORT &
FASTAPI_PID=$!

# Wait a moment for FastAPI to start
sleep 3

# Start Streamlit app in background
echo "Starting Streamlit app..."
streamlit run streamlit_app.py --server.port $STREAMLIT_PORT &
STREAMLIT_PID=$!

# Wait for user to interrupt
wait