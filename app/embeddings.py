from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen

from chunker import Chunk


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "embeddinggemma",
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._post_json(
            "/api/embed",
            {
                "model": self.model,
                "input": texts,
            },
        )
        embeddings = response.get("embeddings")

        if not isinstance(embeddings, list):
            raise ValueError("Ollama response did not include a valid 'embeddings' list.")

        return embeddings

    def embed_text(self, text: str) -> list[float]:
        embeddings = self.embed_texts([text])

        if not embeddings:
            raise ValueError("Ollama returned no embedding for the provided text.")

        return embeddings[0]

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        return self.embed_texts([chunk.text for chunk in chunks])
