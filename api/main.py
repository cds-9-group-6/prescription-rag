"""FastAPI server for the Prescription RAG system."""

import logging
import os
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import our models
from .models import (
    QueryRequest,
    QueryWithMetricsRequest,
    QueryResponse,
    QueryWithSourcesResponse,
    CollectionsInfoResponse,
    AvailableCollectionsResponse,
    ErrorResponse,
    HealthResponse,
    CollectionInfo
)

# Import the OllamaRag class
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rag.rag_with_ollama import OllamaRag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Prescription RAG API",
    description="RAG-based Agricultural Advisory System API using Ollama",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system: Optional[OllamaRag] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global rag_system
    try:
        logger.info("🚀 Starting Prescription RAG API...")
        
        # Get configuration from environment variables
        llm_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_capstone_db_new_small")
        
        # Initialize collections (can be configured via environment variable)
        collections_env = os.getenv("RAG_COLLECTIONS", "")
        if collections_env:
            collections_to_init = [c.strip() for c in collections_env.split(",")]
        else:
            collections_to_init = None  # Use default collections
        
        logger.info(f"Initializing RAG system with LLM: {llm_name}")
        logger.info(f"Temperature: {temperature}, Embedding: {embedding_model}")
        logger.info(f"Collections: {collections_to_init or 'default'}")
        
        # Initialize RAG system
        rag_system = OllamaRag(
            llm_name=llm_name,
            temperature=temperature,
            embedding_model=embedding_model,
            collections_to_init=collections_to_init,
            persist_directory=persist_directory
        )
        
        logger.info("✅ RAG system initialized successfully")
        logger.info(f"Available collections: {rag_system.get_available_collections()}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        raise RuntimeError(f"Failed to initialize RAG system: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down Prescription RAG API...")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not rag_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    available_collections = rag_system.get_available_collections()
    return HealthResponse(
        status="healthy",
        available_collections=available_collections,
        total_collections=len(available_collections),
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    """Run a basic query against the RAG system."""
    if not rag_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    start_time = time.time()
    
    try:
        # Determine collection that would be used
        if request.plant_type and request.plant_type in rag_system.chroma_databases:
            collection_used = request.plant_type
        else:
            collection_used = rag_system._detect_plant_type(request.query)
        
        # Run the query
        answer = rag_system.run_query(
            query_request=request.query,
            plant_type=request.plant_type,
            season=request.season,
            location=request.location,
            disease=request.disease
        )
        
        query_time = time.time() - start_time
        
        return QueryResponse(
            answer=answer,
            collection_used=collection_used,
            query_time=query_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@app.post("/query/metrics", response_model=QueryResponse)
async def run_query_with_metrics(request: QueryWithMetricsRequest):
    """Run a query with comprehensive metrics logging."""
    if not rag_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    start_time = time.time()
    
    try:
        # Determine collection that would be used
        if request.plant_type and request.plant_type in rag_system.chroma_databases:
            collection_used = request.plant_type
        else:
            collection_used = rag_system._detect_plant_type(request.query)
        
        # Run the query with metrics (pass None for mlflow_manager since we're not using it in API)
        answer = rag_system.run_query_with_metrics(
            query_request=request.query,
            plant_type=request.plant_type,
            season=request.season,
            location=request.location,
            disease=request.disease,
            mlflow_manager=None,  # Could be configured later if needed
            session_id=request.session_id
        )
        
        query_time = time.time() - start_time
        
        return QueryResponse(
            answer=answer,
            collection_used=collection_used,
            query_time=query_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Query with metrics failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query with metrics failed: {str(e)}"
        )


@app.post("/query/sources", response_model=QueryWithSourcesResponse)
async def query_with_sources(request: QueryRequest):
    """Run a query and return both answer and source documents."""
    if not rag_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    start_time = time.time()
    
    try:
        # Determine collection that would be used
        if request.plant_type and request.plant_type in rag_system.chroma_databases:
            collection_used = request.plant_type
        else:
            collection_used = rag_system._detect_plant_type(request.query)
        
        # Run the query with sources
        result = rag_system.query_with_sources(
            query_request=request.query,
            plant_type=request.plant_type,
            season=request.season,
            location=request.location,
            disease=request.disease
        )
        
        query_time = time.time() - start_time
        
        # Process source documents for API response
        source_documents = []
        for doc in result.get("source_documents", []):
            source_documents.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return QueryWithSourcesResponse(
            result=result["result"],
            source_documents=source_documents,
            collection_used=collection_used,
            query_time=query_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Query with sources failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query with sources failed: {str(e)}"
        )


@app.get("/collections/info", response_model=CollectionsInfoResponse)
async def get_collection_info():
    """Get information about all initialized collections."""
    if not rag_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    try:
        info = rag_system.get_collection_info()
        
        # Convert to Pydantic models
        collections = {}
        for name, details in info.items():
            collections[name] = CollectionInfo(**details)
        
        return CollectionsInfoResponse(
            collections=collections,
            total_collections=len(collections)
        )
        
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collection info: {str(e)}"
        )


@app.get("/collections", response_model=AvailableCollectionsResponse)
async def get_available_collections():
    """Get list of available collection names."""
    if not rag_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    try:
        collections = rag_system.get_available_collections()
        
        return AvailableCollectionsResponse(
            collections=collections,
            total_collections=len(collections)
        )
        
    except Exception as e:
        logger.error(f"Failed to get available collections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available collections: {str(e)}"
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Prescription RAG API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
        "available_endpoints": [
            "POST /query - Basic query",
            "POST /query/metrics - Query with metrics", 
            "POST /query/sources - Query with source documents",
            "GET /collections - List available collections",
            "GET /collections/info - Detailed collection information",
            "GET /health - Health check"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8081"))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )
