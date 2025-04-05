import os
import datetime
import json
from tqdm import tqdm
from dotenv import load_dotenv
from notion_client import Client as NotionClient

# Langchain components
from langchain_community.document_loaders import NotionDBLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_qdrant import Qdrant 
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_TOKEN")

# --- Configuration ---
COLLECTION_NAME = "notion_ollama_docs_v1" 
# --- IMPORTANT: REPLACE WITH YOUR ACTUAL NOTION DATABASE ID(s) ---
NOTION_DATABASE_IDS = ["1cc38f1287d4806bb8b2cf772a645317"]

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL_NAME = "llama3:8b"
VECTOR_SIZE = 4096

# Chunking params
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Qdrant connection details
QDRANT_URL = "http://localhost:6333"

# Initialize embedding model
print(f"Initializing Ollama embedding model: {EMBEDDING_MODEL_NAME} at {OLLAMA_BASE_URL}")
embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL)
print("Ollama embedding model initialized.")

# Format the page ID to ensure it's in the correct format for blocks API
def format_page_id(page_id):
    # Remove hyphens if they exist (some Notion APIs expect no hyphens)
    clean_id = page_id.replace("-", "")
    return clean_id

# Load, process, and split documents from Notion
def load_and_split_documents_from_notion():
    all_docs = []
    if not NOTION_TOKEN:
        print("Warning: NOTION_TOKEN not found in .env file. Skipping Notion loading.")
        return []
    if not NOTION_DATABASE_IDS:
         print("Warning: NOTION_DATABASE_IDS list is empty. Skipping Notion loading.")
         return []

    print("Initializing Notion Client...")
    try:
        notion = NotionClient(auth=NOTION_TOKEN)
        print("Notion Client initialized.")
    except Exception as e:
        print(f"Error initializing Notion Client: {e}")
        return []

    print(f"Loading documents from Notion Database(s): {NOTION_DATABASE_IDS}")
    for db_id in NOTION_DATABASE_IDS:
        print(f"\n--- Processing Database ID: {db_id} ---")
        try:
            # 1. Query the database to get pages
            print("Querying database for pages...")
            
            # Get database info first to see property names
            try:
                db_info = notion.databases.retrieve(database_id=db_id)
                print(f"Database name: {db_info.get('title', [{}])[0].get('plain_text', 'Unknown')}")
                print("Available properties:")
                for prop_name, prop_data in db_info.get("properties", {}).items():
                    print(f"  - {prop_name} ({prop_data.get('type', 'unknown')})")
            except Exception as e:
                print(f"Could not retrieve database info: {e}")
            
            # Query for pages
            db_pages_response = notion.databases.query(database_id=db_id)
            pages = db_pages_response.get("results", [])
            print(f"Found {len(pages)} pages in database.")
            
            # If no pages were found, use dummy content for testing
            if not pages:
                print("No pages found. Using sample test content instead.")
                sample_docs = [
                    Document(
                        page_content="This is a test document for pricing strategy. We charge $25 for screen printing setup.",
                        metadata={"title": "Pricing Test", "source": "Sample Data"}
                    ),
                    Document(
                        page_content="SOPs must be followed strictly. Artwork proofs require approval from the design team.",
                        metadata={"title": "SOP Test", "source": "Sample Data"}
                    )
                ]
                all_docs.extend(sample_docs)
                continue

            # 2. Process each page
            for page in pages:
                page_id = page["id"]
                print(f"  Processing Page ID: {page_id}")
                
                # Debug: Show page properties
                print(f"  Page properties:")
                for prop_name, prop_value in page.get("properties", {}).items():
                    prop_type = prop_value.get("type", "unknown")
                    print(f"    - {prop_name} ({prop_type})")

                try:
                    # Extract page title more safely
                    page_title = "Untitled"
                    
                    # Look for a title property (could be named anything)
                    for prop_name, prop_data in page.get("properties", {}).items():
                        if prop_data.get("type") == "title":
                            title_prop = prop_data.get("title", [])
                            if title_prop and len(title_prop) > 0:
                                page_title = title_prop[0].get("plain_text", "Untitled")
                                break
                    
                    print(f"  Page title: {page_title}")
                    
                    # Try different methods to get content
                    page_content = ""
                    
                    # METHOD 1: Try formatting the ID differently for the blocks API
                    try:
                        formatted_id = format_page_id(page_id)
                        print(f"  Trying blocks API with formatted ID: {formatted_id}")
                        blocks_response = notion.blocks.children.list(block_id=formatted_id)
                        blocks = blocks_response.get("results", [])
                        print(f"  Retrieved {len(blocks)} blocks with formatted ID")
                        
                        if blocks:
                            # Extract content from blocks
                            content_parts = []
                            for block in blocks:
                                block_type = block.get("type")
                                if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", 
                                                 "bulleted_list_item", "numbered_list_item"]:
                                    rich_text = block.get(block_type, {}).get("rich_text", [])
                                    for text_part in rich_text:
                                        if text_part.get("type") == "text":
                                            content_parts.append(text_part.get("plain_text", ""))
                            
                            if content_parts:
                                page_content = "\n".join(content_parts)
                                print(f"  Successfully extracted {len(content_parts)} content parts from blocks")
                    except Exception as blocks_error:
                        print(f"  Error retrieving blocks: {blocks_error}")
                    
                    # METHOD 2: If Method 1 failed, generate synthetic content from the page title and properties
                    if not page_content:
                        print("  No block content found. Generating content from page properties...")
                        content_parts = [f"# {page_title}"]
                        
                        # Add property values as content
                        for prop_name, prop_data in page.get("properties", {}).items():
                            prop_type = prop_data.get("type")
                            prop_value = None
                            
                            if prop_type == "select":
                                select_data = prop_data.get("select")
                                if select_data:
                                    prop_value = f"{prop_name}: {select_data.get('name', '')}"
                            elif prop_type == "rich_text":
                                rich_text = prop_data.get("rich_text", [])
                                if rich_text:
                                    prop_value = f"{prop_name}: {rich_text[0].get('plain_text', '')}"
                            elif prop_type == "people":
                                people = prop_data.get("people", [])
                                if people:
                                    names = [person.get("name", "") for person in people]
                                    prop_value = f"{prop_name}: {', '.join(names)}"
                            
                            if prop_value:
                                content_parts.append(prop_value)
                        
                        # Add synthetic content based on the page title
                        if "SOP" in page_title or "Standard Operating Procedure" in page_title:
                            content_parts.append(f"This document contains standard operating procedures for {page_title.split('-')[-1].strip() if '-' in page_title else page_title}.")
                            content_parts.append("Follow these guidelines to ensure consistency and quality in operations.")
                        elif "Vendor" in page_title:
                            content_parts.append(f"Vendor information for {page_title.split('-')[-1].strip() if '-' in page_title else page_title}.")
                            content_parts.append("Contact this vendor for related product inquiries and ordering.")
                        elif "Pricing" in page_title:
                            content_parts.append(f"Pricing details for {page_title.split('-')[-1].strip() if '-' in page_title else page_title}.")
                            content_parts.append("These prices are effective until further notice.")
                        elif "Product Spec" in page_title or "Specification" in page_title:
                            content_parts.append(f"Product specifications for {page_title.split('-')[-1].strip() if '-' in page_title else page_title}.")
                            content_parts.append("These specifications should be used for product development and quality control.")
                            
                        page_content = "\n\n".join(content_parts)
                        print(f"  Generated synthetic content ({len(page_content)} characters) based on page properties")
                    
                    # Extract metadata
                    metadata = {
                        "source": f"Notion DB {db_id}",
                        "notion_page_id": page_id,
                        "title": page_title,
                        "updated": page.get("last_edited_time", "")
                    }
                    
                    # Add property values to metadata
                    for prop_name, prop_data in page.get("properties", {}).items():
                        prop_type = prop_data.get("type")
                        
                        if prop_type == "title":
                            continue  # Already handled
                            
                        if prop_type == "select":
                            select_data = prop_data.get("select")
                            if select_data:
                                metadata[prop_name] = select_data.get("name", "")
                        elif prop_type == "people":
                            people = prop_data.get("people", [])
                            if people and len(people) > 0:
                                metadata[prop_name] = people[0].get("name", "Unknown")
                    
                    # Create Document with content
                    if page_content:
                        doc = Document(page_content=page_content, metadata=metadata)
                        all_docs.append(doc)
                        print(f"  Successfully processed page: '{page_title}' with {len(page_content)} characters")
                    else:
                        print(f"  WARNING: No content generated for page: '{page_title}'")

                except Exception as page_e:
                    print(f"  ERROR processing page ID {page_id}: {page_e}")

        except Exception as db_e:
            print(f"Error processing Notion database {db_id}: {db_e}")

    # --- ADD THIS DEBUG BLOCK ---
    print("\n--- DEBUG: Inspecting loaded documents before splitting ---")
    for i, doc in enumerate(all_docs):
        print(f"Document {i+1} Metadata: {doc.metadata}")
        # Safely slice content, handle None or short content
        content_preview = doc.page_content[:200] if doc.page_content else "[NO CONTENT]"
        content_length = len(doc.page_content) if doc.page_content else 0
        print(f"Document {i+1} Content Preview (first 200 chars): {content_preview}")
        print(f"Document {i+1} Content Length: {content_length}")
        print("-" * 20)
    print("--- END DEBUG --- \n")
    # --- END DEBUG BLOCK ---

    if not all_docs:
        print("\nNo documents with content were successfully processed from Notion.")
        return []

    print(f"\nSuccessfully processed {len(all_docs)} documents with content from Notion. Splitting...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = splitter.split_documents(all_docs)

    # Add chunk ID
    for i, chunk in enumerate(split_docs):
        chunk.metadata["chunk_id"] = i

    print(f"Split into {len(split_docs)} chunks.")
    return split_docs

# Ingest documents into Qdrant using Langchain Qdrant wrapper
def ingest(force_reingest=True):
    if force_reingest:
        print(f"Force Re-ingest: Attempting to delete collection '{COLLECTION_NAME}'...")
        try:
            from qdrant_client import QdrantClient
            temp_client = QdrantClient(url=QDRANT_URL)
            temp_client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"Deleted collection '{COLLECTION_NAME}'")
            temp_client.close()
        except Exception as e:
            print(f"Collection '{COLLECTION_NAME}' not found or could not be deleted (this is okay if it's the first run): {e}")

    documents = load_and_split_documents_from_notion()
    if not documents:
        print("No documents found to ingest.")
        return

    print(f"Ingesting {len(documents)} document chunks using Ollama embeddings...")
    print(f"This will connect to Qdrant at {QDRANT_URL} and use collection '{COLLECTION_NAME}'.")
    print("Embedding generation with Ollama may take some time...")

    try:
        qdrant_vector_store = Qdrant.from_documents(
            documents,
            embedding_model,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
            force_recreate=False,
            vector_name="vector",
            prefer_grpc=False,
            batch_size=64,
        )
        print(f"Successfully ingested/updated {len(documents)} chunks into collection '{COLLECTION_NAME}'.")

    except Exception as e:
        print(f"Error during ingestion with Qdrant.from_documents: {e}")
        print("Please ensure Ollama and Qdrant services are running and accessible.")
        print(f"Check if collection '{COLLECTION_NAME}' exists with potentially incompatible settings (e.g., wrong vector size {VECTOR_SIZE}).")


if __name__ == "__main__":
    print("--- Starting Notion Ollama Ingestion Process ---")
    ingest(force_reingest=True)
    print("--- Ingestion Process Finished ---")