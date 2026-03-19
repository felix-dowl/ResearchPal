from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from embeddings import OllamaEmbedder


class LangChainOllamaEmbeddings(Embeddings):
    def __init__(self, embedder: OllamaEmbedder | None = None) -> None:
        self.embedder = embedder or OllamaEmbedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed_text(text)


class ChromaRetriever:
    def __init__(
        self,
        collection_name: str = "researchpal_chunks",
        persist_directory: str = "./chroma_db",
        embedder: OllamaEmbedder | None = None,
    ) -> None:
        self.embedding_function = LangChainOllamaEmbeddings(embedder)
        self.vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embedding_function,
        )

    def retrieve(
        self,
        query: str,
        k: int = 4,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_metadata,
        )

    def retrieve_mmr(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        return self.vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter_metadata,
        )

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_metadata,
        )

    def as_retriever(
        self,
        method: str = "similarity",
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter_metadata: dict[str, Any] | None = None,
    ):
        search_kwargs: dict[str, Any] = {"k": k}

        if method == "mmr":
            search_kwargs["fetch_k"] = fetch_k
            search_kwargs["lambda_mult"] = lambda_mult

        if filter_metadata is not None:
            search_kwargs["filter"] = filter_metadata

        return self.vector_store.as_retriever(
            search_type=method,
            search_kwargs=search_kwargs,
        )


class ChromaSimilarityRetriever(ChromaRetriever):
    pass
