# Retrieval-Augmented Generation (RAG)

## 📂 Files in this Repository  

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
  - rag/ → Contains Ragas_implementation and rag_with_ollama_api.py file

  - docker/ → Reference implementations (Ollama + ChromaDB + FastAPI)

  - api/ → FastAPI-based REST API for Agricultural Advisory RAG

  - data/ → Reference folder which contains data folder of crops that has been embedded for chroma db

---

### 1. `Rag_huggingface.ipynb`  
  - ✅ To run the notebook:  

  - install the requirements.txt(Requirements file)
  - Requires the **`chroma_capstone_db_new_reduced_hugging_face`** vector database.
  - Run the notebook for the RAG.  

### 2. Running with Ollama on your system
   - ✅ To run the python file: 

   - Create the folder for RAG
   - install ollama and pull the model of your choice
   - install the requirements.txt(Requirements file)
   - Requires the **`chroma_capstone_db_new_reduced_hugging_face`** vector database.
   - run the Fast-api app: rag_with_ollama_fastapi.py (the file is located under RAG of the same repository)


The vector database in two ways:
### 1. Using Docker

  ```bash
  docker pull amit1994/chromadb_small_huggingface
  ```
### 2. From Google Drive

  [Google Drive Folder](https://drive.google.com/drive/u/0/folders/1vM6zUKWw-AhbEef4_KbRSJwc80vloXcf). 
  - The file name is chroma_capstone_db_new_reduced_hugging_face.zip

---
## Setup Instructions

### 1. Prepare the Folders

  - Create a folder on your local machine.

  - Place the vector database chroma_capstone_db_new_reduced_hugging_face, requirements.txt and files choice rag_with_ollama_fastapi.py, Rag_huggingface.ipynb

### 2. Set Up a Virtual Environment

  - Ensure Python 3.10+ is installed.

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



## commands to deploy in local

1. Ollama with llama3.1:8b
ollama is running as a native app on macbook

2. Chromadb

```bash
podman run -d --name chromadb  -p 8000:8000 quay.io/rajivranjan/chromadb-with-data-arm64:v1
```

3. Prescription

> note the .env is cruicial to make everytihng work end to end

```bash
# if cache dir exists in local laptop
podman run -it  --name prescription  --env-file=.env -v ~/.cache/huggingface:/root/.cache/huggingface:Z -p 8081:8081 quay.io/rajivranjan/prescription:arm64-v1.1

# if cache dir doesn't exists
podman run -it  --name prescription  --env-file=.env -v huggingface-cache:/root/.cache/huggingface -p 8081:8081 quay.io/rajivranjan/prescription:arm64-v1.1
```
