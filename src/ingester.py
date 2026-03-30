from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from chunker import BaseChunker, HTMLChunker, PDFChunker, TextChunker
from embeddings import OllamaEmbedder
from vector_store import ChromaStore


@dataclass
class IngestionResult:
    source: str
    source_type: str
    chunk_count: int
    stored_count: int
    collection_name: str


@dataclass
class BatchIngestionResult:
    results: list[IngestionResult]

    @property
    def source_count(self) -> int:
        return len(self.results)

    @property
    def chunk_count(self) -> int:
        return sum(result.chunk_count for result in self.results)

    @property
    def stored_count(self) -> int:
        return sum(result.stored_count for result in self.results)


class DocumentIngester:
    def __init__(
        self,
        base_chunker: BaseChunker | None = None,
        embedder: OllamaEmbedder | None = None,
        store: ChromaStore | None = None,
    ) -> None:
        self.base_chunker = base_chunker or BaseChunker()
        self.embedder = embedder or OllamaEmbedder()
        self.store = store or ChromaStore()
        self.text_chunker = TextChunker(self.base_chunker)
        self.html_chunker = HTMLChunker(self.base_chunker)
        self.pdf_chunker = PDFChunker(self.base_chunker)
        self.supported_file_types = {".pdf", ".txt", ".html", ".htm"}

    def detect_source_type(self, source: str) -> str:
        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return "url"

        path = Path(source)
        if not path.exists():
            raise ValueError(
                "Unsupported source or missing file. Provide an existing .pdf/.txt/.html path, a folder, or an http/https URL."
            )

        if path.is_dir():
            return "folder"

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return "pdf"

        if suffix == ".txt":
            return "txt"

        if suffix in {".html", ".htm"}:
            return "html"

        raise ValueError(
            "Unsupported source. Provide an http/https URL, a folder, or a .pdf/.txt/.html file."
        )

    def list_supported_files(self, folder_path: str) -> list[str]:
        folder = Path(folder_path)
        return sorted(
            str(path)
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in self.supported_file_types
        )

    def chunk_source(
        self,
        source: str,
        *,
        title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, list]:
        source_type = self.detect_source_type(source)
        source_metadata = {
            **(extra_metadata or {}),
            "document_type": source_type,
        }

        if source_type == "url":
            chunks = self.html_chunker.chunk_url(
                source,
                title=title,
                extra_metadata=source_metadata,
            )
        elif source_type == "folder":
            raise ValueError("Folders must be ingested with ingest() or ingest_folder(), not chunk_source().")
        elif source_type == "html":
            chunks = self.html_chunker.chunk_html_file(
                source,
                title=title,
                extra_metadata=source_metadata,
            )
        elif source_type == "pdf":
            chunks = self.pdf_chunker.chunk_pdf(
                source,
                title=title,
                extra_metadata=source_metadata,
            )
        else:
            chunks = self.text_chunker.chunk_txt(
                source,
                title=title,
                extra_metadata=source_metadata,
            )

        return source_type, chunks

    def ingest(
        self,
        source: str,
        *,
        title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult | BatchIngestionResult:
        source_type = self.detect_source_type(source)

        if source_type == "folder":
            return self.ingest_folder(source, extra_metadata=extra_metadata)

        source_type, chunks = self.chunk_source(
            source,
            title=title,
            extra_metadata=extra_metadata,
        )
        embeddings = self.embedder.embed_chunks(chunks)
        self.store.upsert_chunks(chunks, embeddings)

        return IngestionResult(
            source=source,
            source_type=source_type,
            chunk_count=len(chunks),
            stored_count=len(chunks),
            collection_name=self.store.collection.name,
        )

    def ingest_many(
        self,
        sources: list[str],
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> BatchIngestionResult:
        results = [
            self.ingest(source, extra_metadata=extra_metadata)
            for source in sources
        ]
        return BatchIngestionResult(results=results)

    def ingest_folder(
        self,
        folder_path: str,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> BatchIngestionResult:
        sources = self.list_supported_files(folder_path)

        if not sources:
            raise ValueError("No supported files were found in the provided folder.")

        return self.ingest_many(sources, extra_metadata=extra_metadata)
