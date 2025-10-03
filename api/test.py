

# import pandas as pd
import logging
# import uvicorn
import os
from typing import Dict, List, Optional

import chromadb
# from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
# Use Ollama embeddings - no API key required, uses your existing Ollama setup
from langchain.embeddings import HuggingFaceEmbeddings

# Alternative options (commented out for reference):
# from langchain_community.embeddings import GPT4AllEmbeddings  # Local GPT4All
# from langchain_community.embeddings import FakeEmbeddings     # For testing only
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


chroma_host = os.getenv("CHROMA_HOST", "localhost")
chroma_port = os.getenv("CHROMA_PORT", 8000)

print("chroma_host",chroma_host)
print("chroma_port",chroma_port)
