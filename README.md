# Retrieval-Augmented Generation (RAG)

This repository contains files on running RAG (Retrieval-Augmented Generation).

**Notebooks**
  - Rag_huggingface.ipynb → Run RAG using Hugging Face (no API key required)

**Tests**
  - test_metadata_filtering_example.py → Test metadata inputs for the retriever

  - test_chromadb.py → Test document retrieval from a ChromaDB collection

**APIs & Scripts**
  - run_api.sh → Start the Prescription RAG API Server

  - rag_with_ollama_fastapi.py → Run the RAG FastAPI app with Ollama

**Docker**
  - Dockerfile.ollama → Build an Ollama model image

  - Dockerfile.chroma_with_data_e5large → Build a ChromaDB image

**Folders**
  - rag/ → Contains Ragas_implementation

  - docker/ → Reference implementations (Ollama + ChromaDB + FastAPI)

  - api/ → FastAPI-based REST API for Agricultural Advisory RAG

---

## 📂 Files in this Repository  

  - ✅ To run this notebook:  

### 1. `Rag_huggingface.ipynb`  
  - Requires the **`chroma_capstone_db_new_reduced_hugging_face`** vector database.  
  - Simply run the notebook to start the RAG pipeline.  

### 2. Running with Ollama on your system
  - cd into the rag folder
  - run the Fast-api app: rag_with_ollama_fastapi.py
  - Requires the **`chroma_capstone_db_new_reduced_hugging_face`** vector database.  
  
Two files are required, which support the running of the ollama RAG program files: The requirements.txt(Requirements file) and   chroma_capstone_db_new_reduced_hugging_face 

---
## Setup Instructions

### 1. Prepare the Folders

  - Create a folder on your local machine.

  - Place the vector database chroma_capstone_db_new_reduced_hugging_face, requirements.txt and file of your choice rag_with_ollama_fastapi.py or   Rag_huggingface.ipynb

### 2. Set Up a Virtual Environment

  - Ensure you have Python 3.10+ installed.

  - Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On macOS/Linux
    venv\Scripts\activate      # On Windows
    ``` 


### 3. Install Dependencies

  - From the project’s parent folder, run:

    ```bash
    pip install -r requirements.txt
    ```


### 4. Run RAG

  - You can now run either of the following:

  ```bash
  python rag_with_ollama_fastapi.py
  ```
  or open in Jupyter:

  ```bash
  jupyter notebook Rag_huggingface.ipynb
  ```
---




## <TODO> How to call the rag independently when hosted on OpenShift


## <TODO> How to call the rag independently when hosted on local machine

## <TODO> How to setup the local environment for development

