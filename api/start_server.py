#!/usr/bin/env python3
"""
Startup script for the Prescription RAG API server.

This script provides an easy way to start the FastAPI server with environment variable configuration.
"""

import os
import sys
import uvicorn
import dotenv

# Add parent directory to path to import the api module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dotenv.load_dotenv()

if __name__ == "__main__":
    # Configuration from environment variables with defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8081"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    # RAG system configuration
    print("🚀 Starting Prescription RAG API Server")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Reload: {reload}")
    print(f"   Log Level: {log_level}")
    print("")
    
    # Environment variable info
    print("📊 RAG Configuration:")
    print(f"   OLLAMA_HOST: {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")
    print(f"   OLLAMA_MODEL: {os.getenv('OLLAMA_MODEL', 'llama3.1:8b')}")
    print(f"   EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')}")
    print(f"   LLM_TEMPERATURE: {os.getenv('LLM_TEMPERATURE', '0.1')}")
    print(f"   RAG_COLLECTIONS: {os.getenv('RAG_COLLECTIONS', 'default collections')}")
    print(f"   CHROMA_PERSIST_DIR: {os.getenv('CHROMA_PERSIST_DIR', './chroma_capstone_db_new_small')}")
    print("")
    print("🌐 API will be available at:")
    print(f"   - Swagger UI: http://{host}:{port}/docs")
    print(f"   - ReDoc: http://{host}:{port}/redoc") 
    print(f"   - Health Check: http://{host}:{port}/health")
    print("")
    
    try:
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)


