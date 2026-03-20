from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from retriever import ChromaRetriever


@dataclass
class RAGResponse:
    question: str
    answer: str
    citations: list[dict[str, str | int]]


class RAGPipeline:
    def __init__(
        self,
        retriever: ChromaRetriever | None = None,
        model: str = "llama3.2",
    ) -> None:
        self.retriever = retriever or ChromaRetriever()
        self.llm = ChatOllama(model=model)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are ResearchPal, a factual research assistant. Your job is to help the user answer any questions they have based on a factual basis from the provided sources. "
                        "1. Answer only from the provided context. "
                        "2. Be transparent when the context is insufficient. "
                        "3. At the end of the answer, cite the individual documents used in your answer by using the provided source metadata."
                    ),
                ),
                (
                    "human",
                    (
                        "Question:\n{question}\n\n"
                        "Conversation history:\n{conversation_history}\n\n"
                        "Context:\n{context}\n\n"
                        "Write a concise answer grounded in the context only. Include citations for your answer."
                    ),
                ),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def build_context(self, documents) -> str:
        context_parts = []

        for i, document in enumerate(documents, start=1):
            metadata = document.metadata
            source = metadata.get("source", "unknown")
            title = metadata.get("title", "unknown")
            page = metadata.get("page")
            page_text = f", page {page}" if page is not None else ""
            context_parts.append(
                f"[{i}] Source: {source} | Title: {title}{page_text}\n{document.page_content}"
            )

        return "\n\n".join(context_parts)

    def build_conversation_history(self, conversation_history: list[Any] | None) -> str:
        if not conversation_history:
            return "No prior conversation."

        history_parts = []
        for turn in conversation_history:
            history_parts.append(f"User: {getattr(turn, 'question', '')}")
            history_parts.append(f"Assistant: {getattr(turn, 'answer', '')}")

        return "\n".join(history_parts)

    def build_citations(self, documents) -> list[dict[str, str | int]]:
        citations = []

        for i, document in enumerate(documents, start=1):
            metadata = document.metadata
            citation: dict[str, str | int] = {
                "index": i,
                "source": str(metadata.get("source", "unknown")),
                "title": str(metadata.get("title", "unknown")),
            }

            if metadata.get("page") is not None:
                citation["page"] = int(metadata["page"])

            citations.append(citation)

        return citations

    def answer(
        self,
        question: str,
        *,
        conversation_history: list[Any] | None = None,
        method: str = "similarity",
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> RAGResponse:
        if method == "mmr":
            documents = self.retriever.retrieve_mmr(
                question,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
            )
        else:
            documents = self.retriever.retrieve(question, k=k)

        context = self.build_context(documents)
        history = self.build_conversation_history(conversation_history)
        answer = self.chain.invoke(
            {
                "question": question,
                "conversation_history": history,
                "context": context,
            }
        )

        return RAGResponse(
            question=question,
            answer=answer,
            citations=self.build_citations(documents),
        )
