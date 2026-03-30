import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retriever import ChromaRetriever


load_dotenv(PROJECT_ROOT / ".env")


# Retrieval parameters for the comparison harness.
# k=4 keeps the result set small enough to inspect manually while still showing redundancy/diversity.
# fetch_k=10 gives MMR a larger candidate pool to diversify from before it picks the final top-k.
# lambda_mult=0.5 gives MMR an even relevance/diversity balance for a first comparison.
K = 4
FETCH_K = 10
LAMBDA_MULT = 0.5

TEST_QUERIES = [
    "Who were the members of Led Zeppelin?",
    "What are some of the most influential rock bands of all time?",
    "Which bands in the corpus are associated with heavy metal?",
    "Which artist in the corpus is best known as the lead singer of Queen?",
    "Which bands from the corpus are closely connected to grunge or alternative rock?",
]


def print_similarity_results(retriever: ChromaRetriever, query: str) -> None:
    results = retriever.retrieve_with_scores(query, k=K)
    print("Similarity results:")
    for rank, (document, score) in enumerate(results, start=1):
        metadata = document.metadata
        print(f"  Rank {rank} | score={score}")
        print(
            f"  Source: {metadata.get('source')} | "
            f"title: {metadata.get('title')} | "
            f"document_type: {metadata.get('document_type')}"
        )
        print(f"  Text: {document.page_content[:250]}")
        print()


def print_mmr_results(retriever: ChromaRetriever, query: str) -> None:
    results = retriever.retrieve_mmr(
        query,
        k=K,
        fetch_k=FETCH_K,
        lambda_mult=LAMBDA_MULT,
    )
    unique_sources = len({document.metadata.get("source") for document in results})

    print(f"MMR results: (unique sources in top-{K}: {unique_sources})")
    for rank, document in enumerate(results, start=1):
        metadata = document.metadata
        print(f"  Rank {rank}")
        print(
            f"  Source: {metadata.get('source')} | "
            f"title: {metadata.get('title')} | "
            f"document_type: {metadata.get('document_type')}"
        )
        print(f"  Text: {document.page_content[:250]}")
        print()

retriever = ChromaRetriever()
stored_chunk_count = retriever.vector_store._collection.count()
print(f"Using existing ChromaDB collection with {stored_chunk_count} stored chunks.")
print(
    f"Parameters: k={K}, fetch_k={FETCH_K}, lambda_mult={LAMBDA_MULT}. "
    "Use the similarity scores and the source diversity in MMR to discuss tradeoffs in the report.\n"
)

for index, query in enumerate(TEST_QUERIES, start=1):
    print(f"Query {index}: {query}")
    print("-" * 80)
    print_similarity_results(retriever, query)
    print_mmr_results(retriever, query)
    print("=" * 80)
    print()
