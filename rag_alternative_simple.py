"""
Alternative RAG implementation without sentence-transformers
Uses OpenAI embeddings or simpler embedding approaches to avoid TensorFlow/PyArrow conflicts
"""

import os
import logging
from typing import Dict, List, Optional
import chromadb
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

# Try multiple embedding providers (fallback chain)
EMBEDDING_PROVIDER = None
EMBEDDING_INSTANCE = None

def get_embeddings():
    """Get embeddings provider with fallback options (no API keys required)"""
    global EMBEDDING_PROVIDER, EMBEDDING_INSTANCE
    
    if EMBEDDING_INSTANCE is not None:
        return EMBEDDING_INSTANCE
    
    # Option 1: Try Ollama embeddings (best choice - no API key, local)
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        EMBEDDING_INSTANCE = OllamaEmbeddings(
            model="nomic-embed-text",  # Good general-purpose embedding model
            base_url=ollama_host
        )
        EMBEDDING_PROVIDER = "ollama"
        logging.info("✅ Using Ollama embeddings (local, no API key required)")
        return EMBEDDING_INSTANCE
    except Exception as e:
        logging.warning(f"Ollama embeddings failed: {e}")
    
    # Option 2: Try GPT4All embeddings (local, no API key)
    try:
        from langchain_community.embeddings import GPT4AllEmbeddings
        EMBEDDING_INSTANCE = GPT4AllEmbeddings()
        EMBEDDING_PROVIDER = "gpt4all"
        logging.info("✅ Using GPT4All embeddings (local, no API key required)")
        return EMBEDDING_INSTANCE
    except Exception as e:
        logging.warning(f"GPT4All embeddings failed: {e}")
    
    # Option 3: Try simple transformers approach (bypass sentence-transformers)
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        EMBEDDING_INSTANCE = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        EMBEDDING_PROVIDER = "huggingface_simple"
        logging.info("✅ Using simple HuggingFace embeddings (CPU only)")
        return EMBEDDING_INSTANCE
    except Exception as e:
        logging.warning(f"HuggingFace embeddings failed: {e}")
    
    # Option 4: Fake embeddings for development/testing (last resort)
    try:
        from langchain_community.embeddings import FakeEmbeddings
        EMBEDDING_INSTANCE = FakeEmbeddings(size=384)
        EMBEDDING_PROVIDER = "fake"
        logging.warning("🚨 Using fake embeddings - for development only!")
        return EMBEDDING_INSTANCE
    except ImportError:
        pass
    
    raise RuntimeError("No embedding provider available. Make sure Ollama is running or install alternative embedding libraries.")

class SimpleOllamaRag:
    """
    Simplified RAG system that avoids sentence-transformers and PyArrow conflicts
    """
    
    SUPPORTED_PLANT_TYPES = {
        'tomato': 'Tomato',
        'potato': 'Potato',
        'rice': 'Rice',
        'wheat': 'Wheat',
        'corn': 'Corn',
    }
    
    DEFAULT_COLLECTIONS = ['Tomato', 'Potato', 'Rice']
    
    prompt_template = """
    You are an agricultural assistant specialized in answering questions about plant diseases.
    Your task is to provide answers strictly based on the provided context when possible.

    Guidelines for answering:
    1. If a relevant answer is available in the context, use that with minimal changes.
    2. If the answer is not available in the context, rely on your agricultural knowledge.
    3. Provide practical, actionable advice for farmers.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    OUTPUT:
    """

    def __init__(self, 
                 llm_name: str, 
                 temperature: float = 0.1,
                 collections_to_init: Optional[List[str]] = None):
        """Initialize simplified RAG system"""
        
        logging.info("🌱 Initializing Simple RAG System (avoiding problematic dependencies)...")
        
        # Initialize LLM
        self._initialize_llm(llm_name, temperature)
        
        # Initialize prompt template
        self.PROMPT = PromptTemplate(
            template=self.prompt_template, input_variables=["context", "question"]
        )
        self.chain_type_kwargs = {"prompt": self.PROMPT}
        
        # Get embeddings (with fallback chain)
        self.embedding = get_embeddings()
        
        # Initialize collections
        self.collections_to_init = collections_to_init or self.DEFAULT_COLLECTIONS
        self.chroma_databases: Dict[str, Chroma] = {}
        self.retrievers: Dict[str, RetrievalQA] = {}
        
        self._initialize_collections()
        
        logging.info(f"✅ Simple RAG system initialized with {EMBEDDING_PROVIDER} embeddings")

    def _initialize_llm(self, llm_name: str, temperature: float):
        """Initialize LLM"""
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", llm_name),
            temperature=temperature,
            base_url=ollama_host,
        )

    def _initialize_collections(self):
        """Initialize ChromaDB collections"""
        try:
            chroma_client = chromadb.HttpClient(host="localhost", port=8000)
        except Exception:
            # Fallback to persistent client
            chroma_client = chromadb.PersistentClient(path="./chroma_capstone_db_new_small")
        
        for collection_name in self.collections_to_init:
            try:
                logging.info(f"🔧 Initializing collection: {collection_name}")
                
                chroma_db = Chroma(
                    client=chroma_client,
                    embedding_function=self.embedding,
                    collection_name=collection_name,
                )
                
                chroma_retriever = chroma_db.as_retriever(
                    search_type="similarity",  # Simpler than MMR
                    search_kwargs={"k": 5}
                )
                
                retrieval_qa = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=chroma_retriever,
                    input_key="query",
                    return_source_documents=True,
                    chain_type_kwargs=self.chain_type_kwargs,
                )
                
                self.chroma_databases[collection_name] = chroma_db
                self.retrievers[collection_name] = retrieval_qa
                
                logging.info(f"✅ Collection {collection_name} initialized")
                
            except Exception as e:
                logging.error(f"❌ Failed to initialize collection {collection_name}: {e}")
                continue

    def run_query(self, query_request: str, **kwargs) -> str:
        """Simple query without complex filtering"""
        try:
            # Use first available collection as default
            collection_name = list(self.retrievers.keys())[0] if self.retrievers else None
            if not collection_name:
                return "Error: No collections available"
            
            retrieval_qa = self.retrievers[collection_name]
            result = retrieval_qa.invoke({"query": query_request})
            
            return result.get("result", "No answer found")
            
        except Exception as e:
            logging.error(f"Query failed: {e}")
            return f"Error processing query: {str(e)}"

    def run_query_with_metrics(self, query_request: str, **kwargs) -> str:
        """Simple query with basic metrics (for MLflow compatibility)"""
        import time
        start_time = time.time()
        
        try:
            result = self.run_query(query_request, **kwargs)
            
            # Log basic metrics if MLflow manager available
            mlflow_manager = kwargs.get('mlflow_manager')
            if mlflow_manager and mlflow_manager.is_available():
                try:
                    import mlflow
                    processing_time = time.time() - start_time
                    mlflow.log_metric("simple_rag_query_time", processing_time)
                    mlflow.log_param("simple_rag_embedding_provider", EMBEDDING_PROVIDER)
                    mlflow.log_metric("simple_rag_success", 1)
                except:
                    pass
            
            return result
            
        except Exception as e:
            # Log error metrics
            mlflow_manager = kwargs.get('mlflow_manager')
            if mlflow_manager and mlflow_manager.is_available():
                try:
                    import mlflow
                    mlflow.log_metric("simple_rag_success", 0)
                    session_id = kwargs.get('session_id', 'unknown')
                    mlflow_manager.log_error(session_id, "simple_rag_error", str(e))
                except:
                    pass
            
            return f"Error: {str(e)}"

# Alias for backward compatibility
OllamaRag = SimpleOllamaRag
