Linux:

Do these commands in a terminal open into -/researchpal$ order to setup:

python3 -m pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama pull embeddinggemma
ollama pull llama3.2

These commands will run the tests:
python3 src/test_scripts/file_intake_test.py
python3 src/test_scripts/query_rewriting_compare.py
python3 src/test_scripts/retrieval_method_compare.py

This command will rebuild the database:
python3 src/build_db.py

This command will launch the ui:
python3 src/chat_ui.py

Windows:

Do these commands in a PowerShell terminal opened into C:\...\researchpal in order to setup:

python -m pip install -r requirements.txt

Download and install Ollama for Windows from:
https://ollama.com/download/windows

Then run:
ollama pull embeddinggemma
ollama pull llama3.2

These commands will run the tests:
python src\test_scripts\file_intake_test.py
python src\test_scripts\query_rewriting_compare.py
python src\test_scripts\retrieval_method_compare.py

This command will rebuild the database:
python src\build_db.py

This command will launch the ui:
python src\chat_ui.py

Note: Data only has two files. Most of the files in the database come from online sources, mostly wikipedia. All the links are in SOURCES in build_db.py. If you wish to add more urls files or folders, you can add them in data and SOURCES or do so through the UI.

Short descriptions of the files in src:

build_db.py
Builds or rebuilds the local ChromaDB collection from the selected document sources.

chat_ui.py
Provides the minimal desktop chat interface, including query input, chat history, and source ingestion buttons.

chunker.py
Handles document chunking for URLs, PDFs, TXT files, and HTML files using recursive chunking.

conversation_handler.py
Coordinates ingestion, question answering, and rolling conversation history across turns.

embeddings.py
Generates embeddings for chunks using the local Ollama embedding model.

ingester.py
Runs the ingestion pipeline: detect source type, chunk content, embed chunks, and store them in ChromaDB.

observability.py
Contains the optional LangSmith tracing helpers used for observability.

rag_pipeline.py
Implements the RAG answering pipeline, including retrieval, optional query rewriting, prompting, and LLM generation.

retriever.py
Implements retrieval from ChromaDB, including cosine similarity and MMR retrieval through LangChain.

vector_store.py
Wraps ChromaDB storage and retrieval operations for chunk records and embeddings.

