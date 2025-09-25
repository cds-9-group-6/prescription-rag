"""Pydantic models for the Prescription RAG API."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for basic query operations."""
    query: str = Field(..., description="The query text to search for", min_length=1)
    plant_type: Optional[str] = Field(None, description="Optional explicit plant type (overrides auto-detection)")
    season: Optional[str] = Field(None, description="Optional season filter (Summer, Winter, Kharif, Rabi, etc.)")
    location: Optional[str] = Field(None, description="Optional location filter (State/District name)")
    disease: Optional[str] = Field(None, description="Optional disease name filter")


class QueryWithMetricsRequest(QueryRequest):
    """Request model for queries with metrics logging."""
    session_id: Optional[str] = Field("unknown", description="Session ID for tracking")


class QueryResponse(BaseModel):
    """Response model for basic query operations."""
    answer: str = Field(..., description="The generated answer")
    collection_used: str = Field(..., description="The ChromaDB collection used for the query")
    query_time: Optional[float] = Field(None, description="Query execution time in seconds")
    success: bool = Field(..., description="Whether the query was successful")


class QueryWithSourcesResponse(BaseModel):
    """Response model for queries that include source documents."""
    result: str = Field(..., description="The generated answer")
    source_documents: List[Dict[str, Any]] = Field(..., description="Source documents used for the answer")
    collection_used: str = Field(..., description="The ChromaDB collection used for the query")
    query_time: Optional[float] = Field(None, description="Query execution time in seconds")
    success: bool = Field(..., description="Whether the query was successful")


class CollectionInfo(BaseModel):
    """Model for collection information."""
    collection_name: str = Field(..., description="Name of the collection")
    persist_directory: Optional[str] = Field(None, description="Directory where the collection is persisted")
    status: str = Field(..., description="Status of the collection (initialized, error, etc.)")
    has_retriever: bool = Field(..., description="Whether a retriever is available for this collection")
    error: Optional[str] = Field(None, description="Error message if collection initialization failed")


class CollectionsInfoResponse(BaseModel):
    """Response model for collection info."""
    collections: Dict[str, CollectionInfo] = Field(..., description="Information about all collections")
    total_collections: int = Field(..., description="Total number of collections")


class AvailableCollectionsResponse(BaseModel):
    """Response model for available collections list."""
    collections: List[str] = Field(..., description="List of available collection names")
    total_collections: int = Field(..., description="Total number of available collections")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    success: bool = Field(False, description="Always false for error responses")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    available_collections: List[str] = Field(..., description="List of available collections")
    total_collections: int = Field(..., description="Total number of collections")
    timestamp: str = Field(..., description="Current timestamp")


