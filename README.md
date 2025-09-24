# Retrieval-Augmented Generation (RAG)

This repository contains files on running RAG (Retrieval-Augmented Generation).

It contains files such as:

- **Rag_with_groq.ipynb** for running the rag notebook using Groq api

- **Rag_huggingface.ipynb** for running the rag notebook using Hugging face.

- **test_metadata_filtering_example.py** to test the various metadata inputs for the retiever system.

- **test_chromadb.py** to test the retiever on obtaining documents for a given collection.

- **run_api.sh** to run the "🚀Starting Prescription RAG API Server".

- **Dockerfile.ollama** Docker file to build the ollama image of the models of your choice, to be run as a container.

- **Dockerfile.chroma_with_data_e5large** Docker file of the Chroma db image, to be run as a container

It contains folders such as:

  - **rag** which contains the Ragas_implementation folder
  - **docker** Refernce folder containing full docker implementation of connecting ollama with chroma db and test_chromadb fast api app
  - **api** A FastAPI-based REST API for the Agricultural Advisory RAG (Retrieval-Augmented Generation) system using Ollama.

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

  - Dockerfile.chroma_with_data_e5large → Build a ChromaDB image with embeddings

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

### 1. Prepare the Vector Database

  - Create a folder on your local machine.

  - Place the four ChromaDB files inside this folder.

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

