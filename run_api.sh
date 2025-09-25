#!/bin/bash

# Startup script for Prescription RAG API
# This script sets up environment variables and starts the FastAPI server

echo "🚀 Starting Prescription RAG API Server"

# Default environment variables (can be overridden)
export API_HOST=${API_HOST:-"0.0.0.0"}
export API_PORT=${API_PORT:-"8081"}
export API_RELOAD=${API_RELOAD:-"false"}
export LOG_LEVEL=${LOG_LEVEL:-"info"}

# RAG system configuration
export OLLAMA_HOST=${OLLAMA_HOST:-"http://localhost:11434"}
export OLLAMA_MODEL=${OLLAMA_MODEL:-"llama3.1:8b"}
export EMBEDDING_MODEL=${EMBEDDING_MODEL:-"nomic-embed-text"}
export LLM_TEMPERATURE=${LLM_TEMPERATURE:-"0.1"}
export CHROMA_PERSIST_DIR=${CHROMA_PERSIST_DIR:-"./chroma_capstone_db_new_small"}

# Optional: Set specific collections to initialize
# export RAG_COLLECTIONS="Tomato,Potato,Rice,Wheat,Corn"

echo "Environment Configuration:"
echo "  API_HOST: $API_HOST"
echo "  API_PORT: $API_PORT"
echo "  OLLAMA_HOST: $OLLAMA_HOST"
echo "  OLLAMA_MODEL: $OLLAMA_MODEL"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  No virtual environment found. It's recommended to use one."
fi

# Install dependencies if requirements.txt has changed
if [ requirements.txt -nt "last_install.txt" ]; then
    echo "📥 Installing/updating dependencies..."
    pip install -r requirements.txt
    touch last_install.txt
fi

# Check if Ollama is running
echo "🔍 Checking Ollama connection..."
if curl -s "$OLLAMA_HOST/api/tags" > /dev/null; then
    echo "✅ Ollama is accessible at $OLLAMA_HOST"
else
    echo "⚠️  Warning: Cannot connect to Ollama at $OLLAMA_HOST"
    echo "   Make sure Ollama is running and accessible"
fi

# Check if ChromaDB is running (if using HTTP client)
if curl -s "http://localhost:8000/api/v1/heartbeat" > /dev/null 2>&1; then
    echo "✅ ChromaDB is accessible at http://localhost:8000"
else
    echo "⚠️  Warning: Cannot connect to ChromaDB at http://localhost:8000"
    echo "   The system will attempt to use persistent directory instead"
fi

echo ""
echo "🌐 Starting API server..."
echo "   Access the API at: http://$API_HOST:$API_PORT"
echo "   Interactive docs: http://$API_HOST:$API_PORT/docs"
echo ""

# Start the server
python -m api.start_server


