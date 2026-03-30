import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conversation_handler import ConversationHandler


load_dotenv(PROJECT_ROOT / ".env")


TEST_QUERIES = [
    "the drummer guy?",
    "lead zeplin best stuff",
    "that band from seattle with kurt",
]

METHOD = "mmr"
K = 4
FETCH_K = 10
LAMBDA_MULT = 0.5


def print_response_block(title: str, rewritten_query: str, answer: str) -> None:
    print(title)
    print("-" * len(title))
    print(f"Rewritten query: {rewritten_query}")
    print("Answer:")
    print(answer)
    print()


handler = ConversationHandler()

for index, query in enumerate(TEST_QUERIES, start=1):
    print("=" * 100)
    print(f"Query {index}: {query}")
    print("=" * 100)

    handler.clear_history()
    plain_response = handler.ask(
        query,
        rewrite_query=False,
        method=METHOD,
        k=K,
        fetch_k=FETCH_K,
        lambda_mult=LAMBDA_MULT,
    )
    print_response_block(
        "Without query rewriting",
        plain_response.rewritten_query,
        plain_response.answer,
    )

    handler.clear_history()
    rewritten_response = handler.ask(
        query,
        rewrite_query=True,
        method=METHOD,
        k=K,
        fetch_k=FETCH_K,
        lambda_mult=LAMBDA_MULT,
    )
    print_response_block(
        "With query rewriting",
        rewritten_response.rewritten_query,
        rewritten_response.answer,
    )
