from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ingester import BatchIngestionResult, DocumentIngester, IngestionResult
from observability import maybe_trace, traceable
from rag_pipeline import RAGPipeline, RAGResponse


@dataclass
class ConversationTurn:
    question: str
    answer: str
    citations: list[dict[str, str | int]]


class ConversationHandler:
    def __init__(
        self,
        pipeline: RAGPipeline | None = None,
        ingester: DocumentIngester | None = None,
        max_history: int = 20,
    ) -> None:
        self.ingester = ingester or DocumentIngester()
        self.pipeline = pipeline or RAGPipeline()
        self.history: deque[ConversationTurn] = deque(maxlen=max_history)

    @traceable(name="conversation_ingest")
    def ingest(self, source: str) -> IngestionResult | BatchIngestionResult:
        with maybe_trace("researchpal-ingestion"):
            return self.ingester.ingest(source)

    @traceable(name="conversation_ask")
    def ask(
        self,
        question: str,
        *,
        rewrite_query: bool = False,
        method: str = "similarity",
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> RAGResponse:
        with maybe_trace("researchpal-conversation"):
            response = self.pipeline.answer(
                question,
                conversation_history=self.get_history(),
                rewrite_query=rewrite_query,
                method=method,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
            )
            self.history.append(
                ConversationTurn(
                    question=response.question,
                    answer=response.answer,
                    citations=response.citations,
                )
            )
            return response

    def get_history(self) -> list[ConversationTurn]:
        return list(self.history)

    def clear_history(self) -> None:
        self.history.clear()
