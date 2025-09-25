import chromadb
import chromadb.utils.embedding_functions as embedding_functions
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM  # Current import for Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import json


# chroma_client = chromadb.HttpClient(host='0.0.0.0', port=8000)
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-large-instruct"
)

langchain_embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large-instruct",
    model_kwargs={"device": "mps"}  # Change to "mps" or "cuda" if you have GPU
)

collection = chroma_client.get_collection(name="Apple", embedding_function=embedding_func)

def chroma_heartbeat():
    print("ChromaDB heartbeat:", chroma_client.heartbeat())
    chroma_client.heartbeat()
    print("ChromaDB client connected successfully.")

def chroma_list_collections():
    print("Available collections:", chroma_client.list_collections())

chroma_heartbeat()
chroma_list_collections()

# print("Peek:", collection.peek())

def chroma_get_embeddings():
    results = collection.get(limit=1, include=["embeddings"])
    if results["embeddings"] is not None and len(results["embeddings"]) > 0:
        embedding_dimension = len(results["embeddings"][0])
        print(f"\n\nEmbedding dimension: {embedding_dimension}")

    metadata_results = collection.get(include=["metadatas", "documents", "embeddings"])

    print("\n=== Collection Metadata Information ===")
    print(f"Total items in collection: {collection.count()}")

    if metadata_results["metadatas"]:
        print(f"Number of items retrieved: {len(metadata_results['metadatas'])}")
        
        # Print first few metadata entries
        print("\nFirst 5 metadata entries:")
        for i, metadata in enumerate(metadata_results["metadatas"][:5]):
            print(f"Item {i+1}: {metadata}")
        
        # Get unique metadata keys and values
        print("\n=== Metadata Analysis ===")
        all_metadata_keys = set()
        metadata_values = {}
        
        for metadata in metadata_results["metadatas"]:
            if metadata:  # Check if metadata is not None
                for key, value in metadata.items():
                    all_metadata_keys.add(key)
                    if key not in metadata_values:
                        metadata_values[key] = set()
                    metadata_values[key].add(value)
        
        print(f"Metadata keys found: {list(all_metadata_keys)}")
        
        for key, values in metadata_values.items():
            print(f"'{key}' values: {list(values)}")
            print(f"  - Unique {key} count: {len(values)}")

    else:
        print("No metadata found in the collection")



# Run a query (this will use whatever embeddings were stored in that collection)
print("\n=== Running Query ===")
try:
    query_text = "What are common diseases in Apple and how to treat them?\nStateName:HIMACHAL PRADESH"
    results = collection.query(
        # query_texts="How to treat apple diseases?",  # list of query strings
        query_texts=query_text,  # new query
        n_results=3  # number of most similar results
    )
    print("Query results:", results)
    
    # Convert results to pretty JSON format
    print("\n=== Pretty JSON Results ===")
    import json
    
    # Create a more readable structure
    formatted_results = {
        "query": query_text,
        "total_results": len(results.get("documents", [[]])[0]) if results.get("documents") else 0,
        "results": []
    }
    
    # Extract and format each result
    if results.get("documents") and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            result_item = {
                "rank": i + 1,
                "document": results["documents"][0][i] if results.get("documents") else None,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") and len(results["metadatas"][0]) > i else None,
                "distance": results["distances"][0][i] if results.get("distances") and len(results["distances"][0]) > i else None,
                "id": results["ids"][0][i] if results.get("ids") and len(results["ids"][0]) > i else None
            }
            formatted_results["results"].append(result_item)
    
    # Print pretty JSON
    print(json.dumps(formatted_results, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Query failed: {e}")

# embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name="intfloat/multilingual-e5-large-instruct",
#     # If you have a GPU, you can specify the device, otherwise it defaults to CPU
#     # device="mps" 
# )

# embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct",model_kwargs={"device": "mps"})
# embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name="intfloat/multilingual-e5-large-instruct"
# )

# Expected Collections in the database:
# Paddy_Dhan
# Tomato
# try:
#     collection_name = "Tomato"
#     collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_func)
#     collection.count()
#     print(f"Collection '{collection_name}' accessed successfully.")
#     print(f"Number of items in the collection: {collection.count()}")

#     # 4. Perform a query.
#     # results = collection.query(query_texts=["tomato diseases"], n_results=5)

#     # print("\nQuery Results:")
#     # print(results)

# except ValueError as e:
#     print(f"\nError getting collection: {e}")
#     print(
#         "Please make sure the collection name is correct and it exists in the database."
#     )
#     print("Check the 'Available collections' list printed above.")



# Initialize LangChain Chroma wrapper
chroma_db = Chroma(
    client=chroma_client,
    collection_name="Apple",
    embedding_function=langchain_embeddings
)

# Initialize Ollama LLM (Updated import and syntax)
llm = OllamaLLM(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.7,
)

# Create system prompt template (Updated approach)
system_prompt = (
    "You are an agricultural expert assistant. "
    "Use the following context to answer questions about plant diseases, treatments, and agricultural practices.\n\n"
    "Context: {context}\n\n"
    "Please provide detailed, helpful answers based on the context provided. "
    "If the context doesn't contain enough information, say so and provide what information you can."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


# Create document processing chain (Current approach - replaces RetrievalQA)
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Option 1: Similarity Search with new chain approach
print("\n=== Option 1: Similarity Search (Current LangChain API) ===")

similarity_retriever = chroma_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

similarity_rag_chain = create_retrieval_chain(similarity_retriever, question_answer_chain)

# Option 2: MMR Search with new chain approach
print("\n=== Option 2: MMR Search (Current LangChain API) ===")

mmr_retriever = chroma_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

mmr_rag_chain = create_retrieval_chain(mmr_retriever, question_answer_chain)

# Test query
query_text = "What are common diseases in Apple and how to treat them?\nStateName:HIMACHAL PRADESH"

print(f"\nQuery: {query_text}")

########################################################
# Test Similarity Search
print("\n" + "="*60)
print("SIMILARITY SEARCH RESULTS (Current API)")
print("="*60)

try:
    similarity_result = similarity_rag_chain.invoke({"input": query_text})
    
    print("LLM Answer:")
    print(similarity_result["answer"])
    
    print(f"\nSource Documents ({len(similarity_result['context'])}):")
    for i, doc in enumerate(similarity_result["context"], 1):
        print(f"\nDocument {i}:")
        print(f"Content: {doc.page_content[:200]}...")
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")
            
except Exception as e:
    print(f"Similarity search failed: {e}")

########################################################
# Test MMR Search
print("\n" + "="*60)
print("MMR SEARCH RESULTS")
print("="*60)

try:
    mmr_result = mmr_rag_chain.invoke({"input": query_text})
    
    print("LLM Answer:")
    print(mmr_result["answer"])
    
    print(f"\nSource Documents ({len(mmr_result['context'])}):")
    for i, doc in enumerate(mmr_result["context"], 1):
        print(f"\nDocument {i}:")
        print(f"Content: {doc.page_content[:200]}...")
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"Metadata: {doc.metadata}")
            
except Exception as e:
    print(f"MMR search failed: {e}")

########################################################
