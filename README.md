# ResearchPal

ResearchPal is a local RAG-based research assistant built for the `LOG6951A` lab project. It supports:

- ingesting documents from URLs, PDFs, TXT files, and HTML files
- recursive chunking
- local embeddings with Ollama
- storage in ChromaDB
- retrieval with cosine similarity and MMR through LangChain
- optional query rewriting
- a minimal chat UI

## Project Structure

### `src/`

- `build_db.py`  
  Builds or rebuilds the local ChromaDB collection from the configured document sources.

- `chat_ui.py`  
  Minimal desktop chat interface with query input, chat history, and source ingestion buttons.

- `chunker.py`  
  Handles recursive chunking for URLs, PDFs, TXT files, and HTML files.

- `conversation_handler.py`  
  Coordinates ingestion, question answering, and rolling conversation history across turns.

- `embeddings.py`  
  Generates embeddings for chunks using the local Ollama embedding model.

- `ingester.py`  
  Runs the ingestion pipeline: detect source type, chunk content, embed chunks, and store them in ChromaDB.

- `observability.py`  
  Contains optional LangSmith tracing helpers.

- `rag_pipeline.py`  
  Implements the RAG answering pipeline, including retrieval, optional query rewriting, prompting, and LLM generation.

- `retriever.py`  
  Implements retrieval from ChromaDB, including cosine similarity and MMR through LangChain.

- `vector_store.py`  
  Wraps ChromaDB storage and retrieval operations for chunk records and embeddings.

### `src/test_scripts/`

- `file_intake_test.py`  
  Teacher-friendly local intake test for URLs and files placed in `src/test_scripts/test_files`.

- `query_rewriting_compare.py`  
  Compares bad queries with and without query rewriting.

- `retrieval_method_compare.py`  
  Compares cosine similarity and MMR across representative test queries.

## Setup

### Linux

Run these commands from the project root:

```bash
python3 -m pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama pull embeddinggemma
ollama pull llama3.2
```

### Windows

Run these commands from a PowerShell terminal opened in the project root:

```powershell
python -m pip install -r requirements.txt
```

Install Ollama for Windows from:

https://ollama.com/download/windows

Then run:

```powershell
ollama pull embeddinggemma
ollama pull llama3.2
```

## Running Tests

### Linux

```bash
python3 src/test_scripts/file_intake_test.py
python3 src/test_scripts/query_rewriting_compare.py
python3 src/test_scripts/retrieval_method_compare.py
```

### Windows

```powershell
python src\test_scripts\file_intake_test.py
python src\test_scripts\query_rewriting_compare.py
python src\test_scripts\retrieval_method_compare.py
```

## Rebuild The Database

### Linux

```bash
python3 src/build_db.py
```

### Windows

```powershell
python src\build_db.py
```

## Launch The UI

### Linux

```bash
python3 src/chat_ui.py
```

### Windows

```powershell
python src\chat_ui.py
```

## Optional LangSmith Setup

If you want tracing enabled, create a `.env` file in the project root based on `.env.example`:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=researchpal
```

Then install:

```bash
pip install python-dotenv langsmith
```

## Notes

- Ollama must be running for embeddings and local LLM generation to work.
- The chat UI assumes your ChromaDB is already populated.
- Query rewriting uses the same local Ollama LLM as the answer-generation pipeline.

---

Made with the help of ChatGPT.
