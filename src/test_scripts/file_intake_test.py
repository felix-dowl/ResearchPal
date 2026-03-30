import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conversation_handler import ConversationHandler


# Teacher-facing test inputs:
# - add any URLs to SOURCES
# - add any local .pdf/.txt/.html files to the test_files folder
SOURCES = [
    "https://en.wikipedia.org/wiki/Led_Zeppelin",
]

# Preview behavior:
# - K = 3 prints the first 3 chunks for each ingested source
# - K = -1 prints all chunks for each ingested source
K = 3


def print_chunk_preview(source: str, chunks: list) -> None:
    print(f"\nSource: {source}")
    print(f"Chunk count: {len(chunks)}")

    preview_chunks = chunks if K == -1 else chunks[:K]
    for index, chunk in enumerate(preview_chunks, start=1):
        print(f"\nChunk {index}:")
        print(f"Metadata: {chunk.metadata}")
        print(chunk.text[:500])

    if K != -1 and len(chunks) > K:
        print(f"\n... {len(chunks) - K} more chunks not shown.")

    print("\n" + "=" * 80)


handler = ConversationHandler()
test_files = Path(__file__).resolve().parent / "test_files"

for source in SOURCES:
    print(f"Ingesting source: {source}")
    source_type, chunks = handler.ingester.chunk_source(source)
    print_chunk_preview(source, chunks)
    result = handler.ingest(source)
    print(f"Stored {result.stored_count} chunks from {source_type} source.")

local_files = handler.ingester.list_supported_files(str(test_files))
print(f"\nIngesting local files from: {test_files}")

for file_path in local_files:
    print(f"Ingesting file: {file_path}")
    source_type, chunks = handler.ingester.chunk_source(file_path)
    print_chunk_preview(file_path, chunks)
    result = handler.ingest(file_path)
    print(f"Stored {result.stored_count} chunks from {source_type} source.")

print("\nDone.")
