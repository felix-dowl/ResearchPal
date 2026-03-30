from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from observability import maybe_trace, traceable
from retriever import ChromaRetriever


@dataclass
class RAGResponse:
    question: str
    rewritten_query: str
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
        self.system_prompt_text = (
            "You are ResearchPal, a factual research assistant. Your job is to help the user answer any questions they have based on a factual basis from the provided sources. "
            "1. Answer only from the provided context. "
            "2. Be transparent when the context is insufficient. "
            "3. Never cite with bare numbers like [1] or [2]. "
            "4. Every citation must include the full source metadata exactly as provided in the context, including source and title, and page when available. "
            "5. End each answer with a 'Sources used:' section listing the full references you relied on."
        )
        self.user_prompt_text = (
            "Question:\n{question}\n\n"
            "Conversation history:\n{conversation_history}\n\n"
            "Context:\n{context}\n\n"
            "Write a concise answer grounded in the context only. "
            "When citing, use full references such as 'Source: <source> | Title: <title> | Page: <page>' rather than numeric markers."
        )
        self.query_rewrite_system_prompt = (
            "You rewrite user questions into clearer retrieval queries for a RAG system focused on rock and roll music. "
            "Expand ambiguous references, keep the original intent, and add useful specificity when needed. "
            "Return only the rewritten query."
        )
        self.query_rewrite_user_prompt = (
            "Original question:\n{question}\n\n"
            "Conversation history:\n{conversation_history}\n\n"
            "Rewrite this into a concise but clearer retrieval query."
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.system_prompt_text,
                ),
                (
                    "human",
                    self.user_prompt_text,
                ),
            ]
        )
        self.query_rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.query_rewrite_system_prompt),
                ("human", self.query_rewrite_user_prompt),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.query_rewrite_chain = self.query_rewrite_prompt | self.llm | StrOutputParser()

    @traceable(name="build_context")
    def build_context(self, documents) -> str:
        context_parts = []

        for i, document in enumerate(documents, start=1):
            metadata = document.metadata
            source = metadata.get("source", "unknown")
            title = metadata.get("title", "unknown")
            page = metadata.get("page")
            page_text = f", page {page}" if page is not None else ""
            context_parts.append(
                f"Reference {i}: Source: {source} | Title: {title}{page_text}\n{document.page_content}"
            )

        return "\n\n".join(context_parts)

    @traceable(name="build_conversation_history")
    def build_conversation_history(self, conversation_history: list[Any] | None) -> str:
        if not conversation_history:
            return "No prior conversation."

        history_parts = []
        for turn in conversation_history:
            history_parts.append(f"User: {getattr(turn, 'question', '')}")
            history_parts.append(f"Assistant: {getattr(turn, 'answer', '')}")

        return "\n".join(history_parts)

    @traceable(name="build_citations")
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

    @traceable(name="serialize_retrieved_chunks")
    def serialize_documents(self, documents) -> list[dict[str, Any]]:
        serialized_documents = []

        for i, document in enumerate(documents, start=1):
            metadata = document.metadata
            serialized_documents.append(
                {
                    "index": i,
                    "source": metadata.get("source", "unknown"),
                    "title": metadata.get("title", "unknown"),
                    "page": metadata.get("page"),
                    "document_type": metadata.get("document_type"),
                    "text": document.page_content,
                }
            )

        return serialized_documents

    @traceable(name="generate_llm_answer")
    def generate_answer(
        self,
        *,
        question: str,
        conversation_history: str,
        context: str,
        system_parameters: dict[str, Any],
    ) -> str:
        return self.chain.invoke(
            {
                "question": question,
                "conversation_history": conversation_history,
                "context": context,
                "system_parameters": system_parameters,
            }
        )

    @traceable(name="rewrite_query")
    def rewrite_query(
        self,
        *,
        question: str,
        conversation_history: str,
    ) -> str:
        rewritten_query = self.query_rewrite_chain.invoke(
            {
                "question": question,
                "conversation_history": conversation_history,
            }
        ).strip()

        return rewritten_query or question

    @traceable(name="rag_answer")
    def answer(
        self,
        question: str,
        *,
        conversation_history: list[Any] | None = None,
        rewrite_query: bool = False,
        method: str = "similarity",
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> RAGResponse:
        with maybe_trace("researchpal-rag"):
            history = self.build_conversation_history(conversation_history)
            retrieval_query = (
                self.rewrite_query(
                    question=question,
                    conversation_history=history,
                )
                if rewrite_query
                else question
            )

            if method == "mmr":
                documents = self.retriever.retrieve_mmr(
                    retrieval_query,
                    k=k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                )
            else:
                documents = self.retriever.retrieve(retrieval_query, k=k)

            self.serialize_documents(documents)
            context = self.build_context(documents)
            answer = self.generate_answer(
                question=question,
                conversation_history=history,
                context=context,
                system_parameters={
                    "query_rewriting_enabled": rewrite_query,
                    "retrieval_query": retrieval_query,
                    "retrieval_method": method,
                    "k": k,
                    "fetch_k": fetch_k,
                    "lambda_mult": lambda_mult,
                    "llm_model": self.llm.model,
                    "system_prompt": self.system_prompt_text,
                },
            )

            return RAGResponse(
                question=question,
                rewritten_query=retrieval_query,
                answer=answer,
                citations=self.build_citations(documents),
            )
