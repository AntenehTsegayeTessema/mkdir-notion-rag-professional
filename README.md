Markdown

# Notion RAG Backend with Ollama & Qdrant (Practice Project)

This project is a practice implementation of a Retrieval-Augmented Generation (RAG) backend system. It aims to connect to a Notion database, index its content (currently metadata + synthetic content) using Ollama embeddings, store vectors in Qdrant, and provide a simple FastAPI interface for semantic search.

**Current Status:** Development / Practice. The pipeline successfully connects to Notion, extracts metadata, generates **synthetic content** (as actual page block content retrieval is currently failing - returning "0 blocks"), chunks, embeds using Ollama, stores in Qdrant, and serves results via API. **The indexed content is NOT the full text from Notion page bodies.**

## Features

* Connects to specified Notion Database(s) via the Notion API.
* Extracts page metadata (Title, Category, Owner, Last Edited Time, etc.) from Notion properties.
* Generates synthetic page content based on metadata as a fallback when block retrieval fails.
* Chunks document content using `RecursiveCharacterTextSplitter`.
* Generates vector embeddings locally using Ollama (specifically `llama3:8b`).
* Stores document chunks, metadata, and embeddings in a Qdrant vector database (running via Docker).
* Provides a FastAPI backend with:
    * `/query` (POST): Accepts a text query and returns the `top_k` most relevant document chunks based on semantic similarity.
    * `/health` (GET): Basic health check for the API and its components.

## Tech Stack

* **Python:** 3.11+
* **Backend Framework:** FastAPI
* **Vector Database:** Qdrant (running in Docker)
* **Embeddings:** Ollama (`llama3:8b` model) via `langchain-ollama`
* **Notion Integration:** `notion-client`, `langchain-community` (`NotionDBLoader` - currently used for metadata only)
* **Orchestration/Helpers:** Langchain (`langchain-qdrant`, `langchain-text-splitters`)
* **Environment:** `python-dotenv`, `uvicorn`

## Project Structure (Example)

.
├── api/
│   ├── init.py        # Ensures 'api' is treated as a package
│   └── main_ollama_api.py # FastAPI application
├── embed/
│   └── ingest_notion_ollama.py # Ingestion script
├── .env                   # Stores API keys/tokens (GITIGNORED!)
├── .gitignore
├── requirements.txt
├── README.md              # This file
└── qdrant_storage/        # Qdrant data persistence (GITIGNORED!)


## Setup and Installation

**Prerequisites:**

* Python 3.11+ and Pip
* Docker Desktop (or Docker service) installed and running
* Git installed
* Ollama installed and running ([ollama.com](https://ollama.com/))
* Ollama `llama3:8b` model pulled: `ollama pull llama3:8b`

**Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Notion:**
    * Go to [www.notion.so/my-integrations](https://www.notion.so/my-integrations) and create a new **Internal Integration**.
    * Give it a name (e.g., "Ollama RAG Integration").
    * Ensure it has **"Read content"** capability enabled.
    * Copy the **Internal Integration Token** (starts with `secret_...`).
    * Create or choose a **Notion Database** to use as your knowledge source.
    * Add properties (columns) to your database that match the names expected by the script (or modify the script): `Name` (Title type), `Category` (Select or Text type), `Owner` (Person or Text type), `Last edited time` (Last edited time type).
    * Add content (pages with text in the body) to this database. *(Note: Content extraction is currently failing, but pages need to exist).*
    * Get the **Database ID** from the database URL.
    * **Share** the Notion Database with the Integration you created ("Can view" permission).

5.  **Configure Environment Variables:**
    * Create a file named `.env` in the project root directory.
    * Add your Notion token:
        ```dotenv
        NOTION_TOKEN="secret_YOUR_NOTION_INTEGRATION_TOKEN_HERE"
        ```

6.  **Configure Ingestion Script:**
    * Open `embed/ingest_notion_ollama.py`.
    * Find the `NOTION_DATABASE_IDS` list.
    * Replace the placeholder ID with your actual Notion Database ID:
        ```python
        NOTION_DATABASE_IDS = ["YOUR_ACTUAL_NOTION_DATABASE_ID"]
        ```
    * Verify that the property names used in the metadata extraction section match your Notion Database column names.

7.  **Start Qdrant:**
    * Make sure Docker is running.
    * Run the following command in your terminal (from the project root):
        ```bash
        # Remove old container if it exists
        docker stop qdrant_ollama || true && docker rm qdrant_ollama || true

        # Run new container
        docker run --name qdrant_ollama -d -p 6333:6333 -p 6334:6334 \
            -v "$(pwd)/qdrant_storage:/qdrant/storage" \
            qdrant/qdrant
        ```
    * Verify it's running by checking `docker ps` or visiting `http://localhost:6333/dashboard`.

8.  **Start Ollama:**
    * Ensure the Ollama application or service is running locally.

## Running the Application

1.  **Run Ingestion (Required First):**
    * Make sure Qdrant and Ollama are running.
    * Make sure your virtual environment is active.
    * Run the ingestion script from the project root directory:
        ```bash
        python embed/ingest_notion_ollama.py
        ```
    * Monitor the output. It should connect to Notion, generate synthetic content (due to the current block loading issue), chunk it, embed using Ollama (this may take time), and store it in Qdrant. Check the Qdrant dashboard to verify.

2.  **Run API Server:**
    * Make sure Qdrant and Ollama are running.
    * Make sure your virtual environment is active.
    * Run the Uvicorn server from the project root directory:
        ```bash
        uvicorn api.main_ollama_api:app --reload --port 7677
        ```
    * The API will be available at `http://127.0.0.1:7677`.

## Usage

* **API Documentation (Swagger):** Access `http://127.0.0.1:7677/docs` in your browser.
* **Query Endpoint:** Send POST requests to `/query`.

    **Example using `curl`:**
    ```bash
    curl -X POST "[http://127.0.0.1:7677/query](https://www.google.com/search?q=http://127.0.0.1:7677/query)" \
    -H "Content-Type: application/json" \
    -d '{"query": "Details about screen printing setup fees", "top_k": 3}'
    ```
    *(Note: Query results will be based on the synthetic content derived from titles/properties, not the full page text).*

* **Health Check:** Access `http://127.0.0.1:7677/health` in your browser or via GET request.

## Known Issues & Limitations

* **Notion Block Content Loading Failure:** The primary issue is that the script currently fails to retrieve block content (`Found 0 blocks`) from Notion pages using `notion.blocks.children.list`.
* **Synthetic Content:** As a workaround, the ingestion script generates basic, synthetic content from page metadata. This significantly limits the knowledge base for the RAG system. Querying works primarily on titles and properties.
* **Local Only:** This setup runs locally and is not configured for cloud deployment.
* **Basic Implementation:** Lacks advanced features like fine-tuning, robust error handling, monitoring, security, or a user interface.

## Future Work / TODO

* **[HIGH PRIORITY]** Resolve the Notion block content loading issue (Investigate permissions, page structures, `notion-client` interactions).
* Implement proper re-embedding strategy for updated Notion content.
* Explore adding relevance scores to API results.
* Implement Phase 2 goals (LLM integration for Q&A).
* Add API authentication.
* Containerize the FastAPI application for easier deployment.
