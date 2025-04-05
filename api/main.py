import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict
from tqdm import tqdm

app = FastAPI()

# Configuration
COLLECTION_NAME = "notion_docs"
DATA_DIR = "data/notion"
VECTOR_SIZE = 4096  # Size for llama3 embeddings

# Initialize embedding model and client
embedding_model = OllamaEmbeddings(model="llama3:8b")
client = QdrantClient(host="localhost", port=6333)

# Load and split documents
def load_and_split_documents():
    all_docs = []
    if not os.path.exists(DATA_DIR):
        print(f"Warning: {DATA_DIR} does not exist. Skipping document loading.")
        return []
        
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt") or filename.endswith(".md"):
            path = os.path.join(DATA_DIR, filename)
            loader = TextLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['title'] = filename
            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)
    return split_docs

# Initialize or get the vector store
def initialize_vector_store():
    # Create collection if it doesn't exist
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

    # Create the vector store with the existing collection
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )
    return vector_store

# Ingest documents into the vector store
def ingest_documents(force_reingest=False):
    if force_reingest:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted collection '{COLLECTION_NAME}'")
        except Exception:
            pass
        
    vector_store = initialize_vector_store()
    
    # Only load and ingest if we need to
    collection_info = client.get_collection(COLLECTION_NAME)
    if collection_info.points_count == 0 or force_reingest:
        documents = load_and_split_documents()
        if documents:
            print(f"Ingesting {len(documents)} document chunks...")
            vector_store.add_documents(documents)
            print(f"Completed ingestion of {len(documents)} chunks.")
        else:
            print("No documents found to ingest.")
    else:
        print(f"Collection already contains {collection_info.points_count} documents. Skipping ingestion.")
    
    return vector_store

# Initialize the vector store on startup
qdrant = ingest_documents(force_reingest=False)

# Query schema
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# Search endpoint
@app.post("/query")
async def query_api(request: QueryRequest):
    results = qdrant.similarity_search(request.query, k=request.top_k)
    return {
        "query": request.query,
        "results": [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in results
        ]
    }

# Optional: Add an endpoint to force reingestion
@app.post("/reingest")
async def reingest_api():
    global qdrant
    qdrant = ingest_documents(force_reingest=True)
    return {"status": "success", "message": "Documents reingested successfully"}

# For running the ingest script directly
if __name__ == "__main__":
    ingest_documents(force_reingest=True)