from __future__ import annotations

from typing import Any

import chromadb

from chunker import Chunk


class ChromaStore:
    def __init__(
        self,
        collection_name: str = "researchpal_chunks",
        persist_directory: str = "./chroma_db",
    ) -> None:
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def chunk_to_metadata(self, chunk: Chunk) -> dict[str, str | int | float | bool]:
        metadata: dict[str, str | int | float | bool] = {
            "source": chunk.source,
            "token_count": chunk.token_count,
            "ingested_at": chunk.ingested_at,
        }

        if chunk.title is not None:
            metadata["title"] = chunk.title

        if chunk.page is not None:
            metadata["page"] = chunk.page

        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value

        return metadata

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks must match the number of embeddings.")

        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[self.chunk_to_metadata(chunk) for chunk in chunks],
            embeddings=embeddings,
        )

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, Any]:
        if not chunk_ids:
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": [],
            }

        results = self.collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas", "embeddings"],
        )
        embeddings = results.get("embeddings")

        if embeddings is not None:
            embeddings = embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)

        return {
            "ids": results.get("ids", []),
            "documents": results.get("documents", []),
            "metadatas": results.get("metadatas", []),
            "embeddings": embeddings or [],
        }
