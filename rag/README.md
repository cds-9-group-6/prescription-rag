# RAG System Implementation

This folder contains code for RAG implemetation. 

## 📂 Files in this Repository  

### 1. `rag_with_ollama_api.py`

    - Contains the code to run the RAG system as a **FastAPI service**
    - The **LLM, prompt, database, and retriever** are defined to run the RAG application

### 2. `rag_with_ollama.py`

    -  Implements the **`OllamaRag` class** for the RAG pipeline. 
    -  Just like `rag_with_ollama_api.py`, it contains code under the class in implementation of the RAG system

### 3 `rag_with_ollama_mod.py`

    -  Also implements the **`OllamaRag` class**. 
    -  Similar to `rag_with_ollama.py`, but with a **modified metadata schema** for disease names.  


## How the RAG System Works

### Database

    - The system uses a **ChromaDB vector store** at: ./chroma_capstone_db_new_reduced_hugging_face

    - Embeddings were generated using:  
    **`multi-qa-MiniLM-L6-cos-v1`** 

    - **collection_name**: this has to be passed to access the database for different crops. The list of options are ['Apple', 'Coconut', 'Paddy_Dhan', 'Potato','Tomato']

    - The database was built with the notebook:
    **`Embedding_creations_with_384_dimensions.ipynb`**  
    (found under `sasya-arogya-data-engineering/RAG-data-engineering`). 
    
### Retiver 

    - The Retriver logic uses the following metadata

    1. **`filter`**: These are the metadata filters for refined retrieval:
    - `StateName`  
    - `disease_name` 

    2. **llm**: Model of choice for answering the **user question**

    3. **Query** – The actual **user question**.

    - The retriever performs a similarity search against the database.  
    - Retrieved documents + a custom prompt are passed to the LLM to generate the final answer.  
    