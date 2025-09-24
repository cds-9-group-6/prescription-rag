import pandas as pd
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import uvicorn
import os

chroma_client = chromadb.HttpClient(host="localhost", port=8000)


llm = ChatOllama(
            temperature=1, 
            model_name="llama-3.1-8b",
            max_tokens=600,
            top_p=0.90,
        #     frequency_penalty=1,
        #     presence_penalty=1,
    )

prompt_template = """
You are an agricultural assistant specialized in answering questions about plant diseases.  
Your task is to provide answers strictly based on the provided context when possible.  

Each document contains the following fields:  
- DistrictName  
- StateName  
- Season_English  
- Month  
- Disease  
- QueryText  
- KccAns (this is the official response section from source documents)

Guidelines for answering:
1. If a relevant answer is available in KccAns, use that with minimal changes.
2. Use DistrictName, StateName, Season_English, Month, and Disease only to help interpret the question and select the correct KccAns, but **do not include these details in the final answer unless the question explicitly asks for them**.  
3. If the answer is not available in the context, then rely on your own agricultural knowledge to provide the best possible answer.  
4. Do not invent or assume information when KccAns is present; only fall back to your own knowledge when the context has no suitable answer.  

CONTEXT:
{context}

QUESTION:
{question}

OUTPUT:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

embedding_name = "multi-qa-MiniLM-L6-cos-v1"


embedding = HuggingFaceEmbeddings(model_name=embedding_name,model_kwargs={"device": "cuda"})
chroma_db = Chroma(
    client=chroma_client,
    # persist_directory="./chroma_capstone_db_new",
    persist_directory="./chroma_capstone_db_new_reduced_hugging_face",
    embedding_function=embedding,
    collection_name="Tomato"  # Specify which collection to load
)

# chroma_client = chromadb.HttpClient(host="chroma", port = 8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))

chroma_retriever = chroma_db.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k":12})

# chroma_retriever.get_relevant_documents(question)

h_retrieval_QA1 = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=chroma_retriever,
    input_key="query",
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

app = FastAPI()


class Queryrequest(BaseModel):

    query:str

@app.get("/")
def root():
    return {"message": "Hello, World"}


@app.post("/ask")
def run_query(request:Queryrequest):

    answer = h_retrieval_QA1.invoke({"query": request.query})["result"]

    return answer

# if __name__ == "__main__":

#     uvicorn.run("rag_with_ollama:app", host="0.0.0.0", port=5050, reload=True)