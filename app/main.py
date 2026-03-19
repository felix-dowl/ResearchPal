from ingester import DocumentIngester
from retriever import ChromaRetriever


ingester = DocumentIngester()
result = ingester.ingest("https://en.wikipedia.org/wiki/Led_Zeppelin")
retriever = ChromaRetriever()
query = "Led Zeppelin was formed in London in 1968."
retrieved_chunks = retriever.retrieve(query, k=1)
retrieved_chunks_mmr = retriever.retrieve_mmr(query, k=1)

print(f"Ingested source: {result.source}")
print(f"Source type: {result.source_type}")
print(f"Stored chunks: {result.stored_count}\n")

if retrieved_chunks:
    top_chunk = retrieved_chunks[0]
    print(f"Query: {query}\n")
    print("Top similarity chunk:")
    print(top_chunk.page_content)
    print(f"\nMetadata: {top_chunk.metadata}")

if retrieved_chunks_mmr:
    top_mmr_chunk = retrieved_chunks_mmr[0]
    print("\nTop MMR chunk:")
    print(top_mmr_chunk.page_content)
    print(f"\nMetadata: {top_mmr_chunk.metadata}")
