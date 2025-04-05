import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List, Dict, Any

# Langchain & Qdrant
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import Qdrant # Langchain wrapper class
from qdrant_client import QdrantClient # Direct client for checks
from qdrant_client import models # For creating collections
from langchain_core.documents import Document # For type hinting

# Load environment variables from .env file (needed for Notion token during ingest)
load_dotenv()
# NOTION_TOKEN = os.getenv("NOTION_TOKEN") # Not directly needed by API usually

# --- Configuration (Must match ingestion script) ---
COLLECTION_NAME = "notion_ollama_docs_v1"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL_NAME = "llama3:8b"
VECTOR_SIZE = 4096 # Expected size for llama3:8b

# Qdrant connection details
QDRANT_URL = "http://localhost:6333"

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Notion RAG API (Ollama)",
    description="Query API for Notion documents embedded using Ollama Llama3",
    version="1.0.0",
)

# --- Global Variables (Initialized on Startup) ---
embedding_model: OllamaEmbeddings = None
qdrant_vector_store: Qdrant = None # Use the Langchain wrapper instance

@app.on_event("startup")
async def startup_event():
    """Initialize resources on server startup."""
    global embedding_model, qdrant_vector_store
    print("--- API Startup Initiated ---")

    # Initialize embedding model
    print(f"Initializing Ollama embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL)
        # Test embedding to ensure Ollama is running and model is loaded
        try:
            test_embedding = embedding_model.embed_query("test")
            print(f"Ollama embedding test successful: vector size = {len(test_embedding)}")
        except Exception as embed_error:
            print(f"WARNING: Ollama embedding test failed: {embed_error}")
            print("Continuing startup, but API may not function correctly until Ollama is available.")
            
        print("Ollama embedding model initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Ollama embedding model: {e}")
        print("Check that Ollama is running at: " + OLLAMA_BASE_URL)
        # Don't raise here, let the API start but health check will fail
        embedding_model = None  # Explicitly set to None to indicate failure

    # Initialize Qdrant client and Langchain Vector Store Wrapper
    print(f"Connecting to Qdrant at {QDRANT_URL} and initializing Vector Store...")
    try:
        # Use direct client first to check collection existence and config
        direct_client = QdrantClient(url=QDRANT_URL)
        print(f"Checking for Qdrant collection: '{COLLECTION_NAME}'...")
        try:
            collection_info = direct_client.get_collection(collection_name=COLLECTION_NAME)
            print(f"DEBUG: Collection info structure: {collection_info}")
            
            # Try different ways to access vector size based on Qdrant client version
            qdrant_vector_size = None
            try:
                # First try newer structure
                qdrant_vector_size = collection_info.config.params.vectors.size
                print("Found vector size using newer Qdrant client structure")
            except:
                try:
                    # Then try older structure
                    qdrant_vector_size = collection_info.config.params.size
                    print("Found vector size using older Qdrant client structure")
                except:
                    # As a last resort, try to get the first vector config
                    try:
                        from collections.abc import Mapping
                        if isinstance(collection_info.config.params.vectors, Mapping):
                            first_vector = next(iter(collection_info.config.params.vectors.values()))
                            qdrant_vector_size = first_vector.size
                            print("Found vector size from first vector in vectors mapping")
                    except:
                        print("Unable to determine vector size from collection info")
            
            print(f"Detected vector size: {qdrant_vector_size}")
            
            if not qdrant_vector_size or qdrant_vector_size != VECTOR_SIZE:
                error_msg = (f"ERROR: Qdrant collection '{COLLECTION_NAME}' exists "
                            f"but has incorrect or undetermined vector size. "
                            f"Expected {VECTOR_SIZE} for model '{EMBEDDING_MODEL_NAME}'.")
                print(error_msg)
                direct_client.close()
                qdrant_vector_store = None  # Set to None but don't raise
            else:
                print(f"Found existing Qdrant collection '{COLLECTION_NAME}' with correct vector size.")
                # Now initialize the Langchain wrapper using the existing collection
                qdrant_vector_store = Qdrant(
                    client=direct_client,
                    collection_name=COLLECTION_NAME,
                    embeddings=embedding_model,
                    vector_name="vector"  # Add this line to specify the vector name
                )
                print("Langchain Qdrant Vector Store initialized successfully.")
                # Try different ways to access points count
                try:
                    points_count = collection_info.points_count
                except:
                    try:
                        # Alternative location in newer versions
                        points_count = collection_info.vectors_count
                    except:
                        points_count = "unknown number of"
                print(f"Collection contains {points_count} document chunks.")

        except Exception as e:
             # Handle case where collection doesn't exist
             error_msg = (f"ERROR: Qdrant collection '{COLLECTION_NAME}' not found or connection failed: {e}. "
                          f"Please run the ingestion script first.")
             print(error_msg)
             direct_client.close()
             qdrant_vector_store = None  # Set to None but don't raise

    except Exception as e:
        print(f"ERROR: Failed to connect to Qdrant: {e}")
        print("Check that Qdrant is running at: " + QDRANT_URL)
        qdrant_vector_store = None  # Set to None but don't raise

    # Add to your startup function
    if qdrant_vector_store is None:
        print("Attempting to create a minimal test collection...")
        try:
            # Create a minimal collection with a test document
            direct_client = QdrantClient(url=QDRANT_URL)
            collection_name = "test_collection"
            
            try:
                direct_client.delete_collection(collection_name=collection_name)
            except:
                pass
                
            direct_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            
            # Initialize with a test document
            test_vector_store = Qdrant(
                client=direct_client,
                collection_name=collection_name,
                embeddings=embedding_model,
                vector_name="vector"  # Add this line to the fallback too
            )
            
            test_doc = Document(
                page_content="This is a test document for system verification.",
                metadata={"source": "API startup test", "title": "Test Document"}
            )
            
            test_vector_store.add_documents([test_doc])
            print(f"Created test collection '{collection_name}' with a single document.")
            
            # Use this as a fallback
            qdrant_vector_store = test_vector_store
            # Don't change the global collection name, just use the test collection
            print(f"Using test collection '{collection_name}' as fallback, but keeping original collection name '{COLLECTION_NAME}'.")
            
        except Exception as e:
            print(f"Failed to create test collection: {e}")

    if embedding_model is None or qdrant_vector_store is None:
        print("WARNING: API started with incomplete initialization. Some endpoints may not function.")
    else:
        print("All components initialized successfully.")

    print("--- API Startup Complete ---")


# --- API Request/Response Models ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query string.", examples=["how are invoices processed?"])
    top_k: int = Field(5, ge=1, le=20, description="Number of similar documents to return.")

class QueryResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    # Langchain similarity_search doesn't easily return score by default with Qdrant
    # score: float # Add this if using similarity_search_with_score

class QueryResponse(BaseModel):
    query: str
    results: List[QueryResult]

# --- API Endpoints ---
@app.post("/query", response_model=QueryResponse)
async def query_api(request: QueryRequest):
    """
    Accepts a query and returns the top_k most relevant document chunks
    using Ollama embeddings and Qdrant similarity search.
    """
    global qdrant_vector_store, embedding_model  # Add this line to declare globals
    
    if not qdrant_vector_store or not embedding_model:
        raise HTTPException(status_code=503, detail="Vector Store or Embedding Model not initialized.")

    # Debug collection names
    print(f"Current global COLLECTION_NAME: {COLLECTION_NAME}")
    print(f"Active vector store collection: {qdrant_vector_store.collection_name}")
    
    # IMPORTANT: Check if we're using the test collection instead of the real collection
    if qdrant_vector_store.collection_name == "test_collection":
        print("WARNING: Using fallback test_collection instead of the actual Notion collection!")
        print("Attempting to reconnect to the actual collection...")
        
        try:
            # Try to reconnect to the actual collection
            direct_client = QdrantClient(url=QDRANT_URL)
            try:
                collection_info = direct_client.get_collection(collection_name=COLLECTION_NAME)
                # Now initialize the Langchain wrapper using the actual collection
                qdrant_vector_store = Qdrant(
                    client=direct_client,
                    collection_name=COLLECTION_NAME,
                    embeddings=embedding_model,
                    vector_name="vector"  # Add here as well
                )
                print(f"Successfully reconnected to the actual collection '{COLLECTION_NAME}'")
            except Exception as e:
                print(f"Failed to reconnect to the actual collection: {e}")
        except Exception as e:
            print(f"Failed to create new Qdrant client: {e}")

    print(f"Received query: '{request.query}', top_k={request.top_k}")

    try:
        # Debug: Test embedding generation first
        print("Attempting to generate embeddings for query...")
        try:
            query_embedding = embedding_model.embed_query(request.query)
            print(f"Generated embedding successfully, length: {len(query_embedding)}")
        except Exception as embed_err:
            print(f"ERROR generating embeddings: {embed_err}")
            raise HTTPException(status_code=500, 
                               detail=f"Failed to generate embeddings: {str(embed_err)}")
        
        # Debug: Test collection info
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=QDRANT_URL)
            collection_info = client.get_collection(collection_name=COLLECTION_NAME)
            print(f"Collection info: {collection_info.points_count} points found")
        except Exception as coll_err:
            print(f"ERROR getting collection info: {coll_err}")
            
        # Proceed with search
        print("Executing similarity search...")
        results: List[Document] = qdrant_vector_store.similarity_search(
            query=request.query,
            k=request.top_k
        )
        
        print(f"Search completed, found {len(results)} results")
        
        # Format results with careful error handling
        formatted_results = []
        for doc in results:
            try:
                formatted_results.append({
                    "content": str(doc.page_content) if doc.page_content else "",
                    "metadata": doc.metadata if doc.metadata else {}
                })
            except Exception as format_err:
                print(f"Error formatting document: {format_err}")
                # Continue with other documents

        # Force use of the correct collection if we're still getting test document results
        is_test_result = any(
            result.get("metadata", {}).get("source", "") == "API startup test" 
            for result in formatted_results
        )

        if is_test_result and len(formatted_results) > 0:
            print("WARNING: Still returning test collection results!")
            print(f"Making one final attempt to query the actual collection '{COLLECTION_NAME}' directly")
            
            try:
                # Create a fresh client and vector store for just this query
                direct_client = QdrantClient(url=QDRANT_URL)
                if direct_client.collection_exists(COLLECTION_NAME):
                    print(f"Found collection {COLLECTION_NAME} - creating fresh vector store")
                    temp_vector_store = Qdrant(
                        client=direct_client,
                        collection_name=COLLECTION_NAME,
                        embeddings=embedding_model,
                        vector_name="vector"  # Add here as well
                    )
                    
                    # Try querying this store instead
                    print("Executing direct similarity search on actual collection...")
                    direct_results = temp_vector_store.similarity_search(
                        query=request.query,
                        k=request.top_k
                    )
                    
                    if direct_results:
                        print(f"Success! Found {len(direct_results)} results in actual collection")
                        # Replace results with these better ones
                        formatted_results = []
                        for doc in direct_results:
                            formatted_results.append({
                                "content": str(doc.page_content) if doc.page_content else "",
                                "metadata": doc.metadata if doc.metadata else {}
                            })
            except Exception as last_e:
                print(f"Final attempt to query actual collection failed: {last_e}")
        
        print(f"Returning {len(formatted_results)} properly formatted results.")
        return QueryResponse(query=request.query, results=formatted_results)

    except Exception as e:
        print(f"Error during similarity search: {e}")
        # Log the error properly in production
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/health")
async def health_check():
     """Basic health check endpoint"""
     # Could add deeper checks (e.g., ping Qdrant, check Ollama model access)
     if qdrant_vector_store and embedding_model:
         # Simple check: Can we embed?
         try:
             embedding_model.embed_query("health check")
             return {"status": "ok", "message": "API, Embeddings, and Vector Store seem operational."}
         except Exception as e:
              print(f"Health Check Warning: Embedding failed - {e}")
              return {"status": "warning", "message": f"API is up, but embedding failed: {e}"}
     else:
          return {"status": "error", "message": "API components not fully initialized."}


# --- Optional: Re-ingestion Endpoint ---
# Note: Running ingestion from API might timeout for large datasets
# Better to run ingest_notion_ollama.py script separately.
# @app.post("/reingest")
# async def reingest_api():
#     print("Received request to reingest documents...")
#     try:
#         # Call the ingestion logic (requires importing it or redefining it here)
#         # Be careful about running long processes within API requests
#         # ingest(force_reingest=True) # Assuming ingest function is accessible
#         return {"status": "triggered", "message": "Re-ingestion process started (run separately for best results)."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to trigger reingestion: {e}")


# --- Run Instruction (for running directly) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server for Ollama RAG API...")
    # Ensure Ollama and Qdrant are running before starting this
    # Run ingestion script first if needed
    uvicorn.run(app, host="0.0.0.0", port=7677, log_level="info",
        workers=1)  # Adjust workers based on your server capabilities
    print("FastAPI server started. Access it at http://localhost:7677/docs")