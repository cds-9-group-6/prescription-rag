# Prescription RAG API

A FastAPI-based REST API for the Agricultural Advisory RAG (Retrieval-Augmented Generation) system using Ollama.

## Features

- 🌱 **Plant-specific querying**: Automatic plant type detection and collection routing
- 🔍 **Advanced filtering**: Season, location, and disease-based metadata filtering  
- 📊 **Comprehensive metrics**: Optional metrics logging for performance monitoring
- 📝 **Source documents**: Return source documents along with answers
- 🏥 **Health monitoring**: Built-in health check and collection status endpoints
- 🚀 **High performance**: Pre-initialized embeddings and retrievers
- 📖 **Auto-documentation**: Interactive Swagger UI and ReDoc documentation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama

Make sure Ollama is running with the required models:

```bash
# Start Ollama
ollama serve

# Pull required models
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 3. Start ChromaDB (Optional)

If using HTTP ChromaDB client:

```bash
# Start ChromaDB server
docker run -p 8000:8000 chromadb/chroma
```

### 4. Start the API Server

```bash
# Using the convenience script
./run_api.sh

# Or directly with Python
python -m api.start_server

# Or with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8081
```

The API will be available at:
- **API Base**: `http://localhost:8081`
- **Interactive Docs**: `http://localhost:8081/docs`
- **ReDoc**: `http://localhost:8081/redoc`
- **Health Check**: `http://localhost:8081/health`

## API Endpoints

### Core Query Endpoints

#### `POST /query`
Basic query functionality with automatic plant type detection.

**Request:**
```json
{
  "query": "What are common diseases in tomatoes?",
  "plant_type": "Tomato",  // optional
  "season": "Summer",      // optional
  "location": "Karnataka", // optional
  "disease": "Blight"      // optional
}
```

**Response:**
```json
{
  "answer": "Common diseases in tomatoes include...",
  "collection_used": "Tomato",
  "query_time": 2.34,
  "success": true
}
```

#### `POST /query/metrics`
Query with comprehensive metrics logging.

**Request:** Same as `/query` plus:
```json
{
  "session_id": "user-session-123"  // optional
}
```

**Response:** Same as `/query`

#### `POST /query/sources`
Query that returns both answer and source documents.

**Request:** Same as `/query`

**Response:**
```json
{
  "result": "Treatment for tomato blight includes...",
  "source_documents": [
    {
      "page_content": "Document content...",
      "metadata": {
        "DistrictName": "Bangalore",
        "StateName": "Karnataka",
        "Disease": "Blight",
        "Season_English": "Summer"
      }
    }
  ],
  "collection_used": "Tomato",
  "query_time": 2.45,
  "success": true
}
```

#### `POST /query/treatment`
Get structured treatment recommendations in JSON format with immediate treatment, weekly plans, and medicine recommendations.

**Request:** Same as `/query`

**Response:**
```json
{
  "treatment": {
    "diagnosis": {
      "disease_name": "Tomato Blight",
      "symptoms": ["brown spots on leaves", "wilting", "leaf yellowing"],
      "severity": "moderate",
      "affected_parts": ["leaves", "stems"]
    },
    "immediate_treatment": {
      "actions": ["Remove infected leaves", "Improve drainage", "Apply copper fungicide"],
      "emergency_measures": ["Isolate affected plants"],
      "timeline": "immediate"
    },
    "weekly_treatment_plan": {
      "week_1": {
        "actions": ["Apply copper sulfate spray", "Monitor daily"],
        "monitoring": "Check for new spots and spreading",
        "expected_results": "Reduction in new spot formation"
      }
    },
    "medicine_recommendations": {
      "primary_treatment": {
        "medicine_name": "Copper Sulfate",
        "active_ingredient": "Copper sulfate pentahydrate",
        "dosage": "2-3 grams per liter",
        "application_method": "Foliar spray",
        "frequency": "Every 7-10 days",
        "duration": "3-4 weeks",
        "precautions": ["Avoid spraying in hot sun", "Use protective equipment"]
      }
    },
    "prevention": {
      "cultural_practices": ["Crop rotation", "Proper spacing"],
      "crop_management": ["Regular pruning", "Drip irrigation"],
      "environmental_controls": ["Good air circulation"],
      "monitoring_schedule": "Weekly during growing season"
    },
    "additional_notes": {
      "weather_considerations": "Avoid spraying during rain",
      "crop_stage_specific": "More critical during flowering",
      "regional_considerations": "Monitor humidity in coastal areas",
      "follow_up": "Contact extension officer if no improvement in 2 weeks"
    }
  },
  "collection_used": "Tomato",
  "query_time": 3.2,
  "success": true,
  "parsing_success": true
}
```

### Collection Management Endpoints

#### `GET /collections`
Get list of available collection names.

**Response:**
```json
{
  "collections": ["Tomato", "Potato", "Rice", "Wheat", "Corn"],
  "total_collections": 5
}
```

#### `GET /collections/info`
Get detailed information about all collections.

**Response:**
```json
{
  "collections": {
    "Tomato": {
      "collection_name": "Tomato",
      "persist_directory": "./chroma_capstone_db_new_small",
      "status": "initialized",
      "has_retriever": true
    }
  },
  "total_collections": 1
}
```

### System Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "available_collections": ["Tomato", "Potato", "Rice"],
  "total_collections": 3,
  "timestamp": "2024-03-15T10:30:00Z"
}
```

#### `GET /`
Root endpoint with API information.

## Configuration

Configure the API using environment variables:

### API Server Configuration
```bash
export API_HOST="0.0.0.0"          # API server host
export API_PORT="8081"             # API server port  
export API_RELOAD="false"          # Enable auto-reload (development)
export LOG_LEVEL="info"            # Logging level
```

### RAG System Configuration
```bash
export OLLAMA_HOST="http://localhost:11434"  # Ollama server URL
export OLLAMA_MODEL="llama3.1:8b"           # LLM model name
export EMBEDDING_MODEL="nomic-embed-text"    # Embedding model
export LLM_TEMPERATURE="0.1"                 # LLM temperature
export CHROMA_PERSIST_DIR="./chroma_db"      # ChromaDB directory
export RAG_COLLECTIONS="Tomato,Potato,Rice" # Collections to initialize
```

## Example Usage

### Python Client Example

```python
import requests
import json

# API base URL
API_BASE = "http://localhost:8081"

# Basic query
def query_rag(query, plant_type=None, season=None):
    response = requests.post(f"{API_BASE}/query", json={
        "query": query,
        "plant_type": plant_type,
        "season": season
    })
    return response.json()

# Query with sources  
def query_with_sources(query):
    response = requests.post(f"{API_BASE}/query/sources", json={
        "query": query
    })
    return response.json()

# Example usage
result = query_rag("How to treat tomato blight?", plant_type="Tomato", season="Summer")
print(f"Answer: {result['answer']}")

sources_result = query_with_sources("Potato disease management")
print(f"Answer: {sources_result['result']}")
print(f"Sources: {len(sources_result['source_documents'])} documents")
```

### cURL Examples

```bash
# Health check
curl -X GET "http://localhost:8081/health"

# Basic query
curl -X POST "http://localhost:8081/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What causes tomato leaf curl?",
    "plant_type": "Tomato",
    "season": "Summer"
  }'

# Query with sources
curl -X POST "http://localhost:8081/query/sources" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Best practices for potato farming",
    "location": "Punjab"
  }'

# Get available collections
curl -X GET "http://localhost:8081/collections"
```

## Supported Plant Types

The system currently supports the following plant types:
- Tomato, Potato, Rice, Wheat, Corn
- Apple, Coconut, Paddy (Dhan)  
- And more based on your ChromaDB collections

Plant types are automatically detected from queries or can be explicitly specified.

## Metadata Filtering

The API supports advanced filtering based on:
- **Season**: Summer, Winter, Kharif, Rabi, Monsoon
- **Location**: State or District names
- **Disease**: Specific disease names
- **Plant Type**: Explicit plant type specification

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `422`: Validation Error (invalid request data)
- `500`: Internal Server Error
- `503`: Service Unavailable (RAG system not initialized)

Error responses include detailed error information:
```json
{
  "error": "Query failed",
  "detail": "Collection not found: InvalidCollection",
  "success": false
}
```

## Development

### Running in Development Mode

```bash
# Enable auto-reload
export API_RELOAD="true"
./run_api.sh
```

### Testing

```bash
# Run basic functionality test
python api/test_endpoints.py

# Test /query/metrics endpoint specifically
./test_query_metrics.sh

# Quick metrics endpoint test
./quick_test_metrics.sh

# Test structured treatment endpoint
./test_structured_treatment.py
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8081
CMD ["python", "-m", "api.start_server"]
```

## Performance Considerations

- **Startup Time**: Collections are pre-initialized for better query performance
- **Memory Usage**: Multiple collections are loaded in memory
- **Concurrent Requests**: FastAPI handles concurrent requests efficiently
- **Caching**: Consider implementing caching for frequently asked questions

## Monitoring and Observability

- Health check endpoint for monitoring
- Comprehensive logging with structured format
- Optional metrics integration (MLflow support in metrics endpoint)
- Query timing and collection usage tracking

## License

This project is part of the Prescription RAG system for agricultural advisory services.

