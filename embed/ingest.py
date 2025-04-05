import os
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configuration
COLLECTION_NAME = "notion_docs"
DATA_DIR = "data/notion"
VECTOR_SIZE = 4096  # Size for llama3 embeddings

# Initialize embedding model
embedding_model = OllamaEmbeddings(model="llama3:8b")

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

# Ingest documents into Qdrant
def ingest(force_reingest=True):
    client = QdrantClient(host="localhost", port=6333)
    
    # Delete collection if force_reingest is True
    if force_reingest:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted collection '{COLLECTION_NAME}'")
        except Exception:
            # Collection might not exist yet
            pass
    
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
    
    # Load the documents
    documents = load_and_split_documents()
    if not documents:
        print("No documents found to ingest.")
        return
    
    # Create vector store and add documents
    print(f"Ingesting {len(documents)} document chunks...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )
    vector_store.add_documents(documents)
    print(f"Successfully ingested {len(documents)} chunks.")

if __name__ == "__main__":
    ingest(force_reingest=True)
