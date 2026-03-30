from __future__ import annotations
import html
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any
from urllib.request import Request, urlopen
import tiktoken
import fitz


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: str
    token_count: int
    ingested_at: str
    # Optional chunk metadata
    page: int | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseChunker:
    def __init__(
        self,
        max_tokens: int = 300,
        min_tokens: int = 100,
        min_side_chars: int = 40,
        encoding_name: str = "cl100k_base",
        separator_order: list[str] | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.min_side_chars = min_side_chars
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.separator_order = separator_order or [
            "paragraph",
            "newline",
            "sentence",
            "comma",
        ]

    # ----------------------------
    # Core token / cleanup methods
    # ----------------------------

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def light_clean(self, text: str) -> str:
        text = text.strip()

        # Fix hyphenated line breaks: "exam-\nple" -> "example"
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Remove spaces around newlines
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)

        # Collapse repeated spaces/tabs, but keep newlines
        text = re.sub(r"[ \t]+", " ", text)

        # Collapse 3+ newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def finalize_chunk_text(self, text: str) -> str:
        # Replace single newlines with spaces
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse repeated spaces/tabs
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    # ----------------------------
    # Split helpers
    # ----------------------------

    def find_separator_positions(self, text: str) -> dict[str, list[int]]:
        positions = {
            "paragraph": [],
            "newline": [],
            "sentence": [],
            "comma": [],
        }

        for match in re.finditer(r"\n\n+", text):
            positions["paragraph"].append(match.start())

        for match in re.finditer(r"(?<!\n)\n(?!\n)", text):
            positions["newline"].append(match.start())

        for match in re.finditer(r"(?<=[.!?])\s+", text):
            positions["sentence"].append(match.start())

        for match in re.finditer(r",\s+", text):
            positions["comma"].append(match.start() + 1)

        return positions

    def choose_best_middle_split(
        self,
        text: str,
        positions: list[int],
    ) -> int | None:
        if not positions:
            return None

        mid = len(text) / 2
        valid_positions = []

        for pos in positions:
            left = text[:pos].strip()
            right = text[pos:].strip()

            if len(left) >= self.min_side_chars and len(right) >= self.min_side_chars:
                valid_positions.append(pos)

        if not valid_positions:
            return None

        return min(valid_positions, key=lambda p: abs(p - mid))

    def hard_split(self, text: str) -> tuple[str, str]:
        mid = len(text) // 2

        left_space = text.rfind(" ", 0, mid)
        right_space = text.find(" ", mid)

        if left_space == -1 and right_space == -1:
            return text[:mid].strip(), text[mid:].strip()

        if left_space == -1:
            split_pos = right_space
        elif right_space == -1:
            split_pos = left_space
        else:
            split_pos = left_space if (mid - left_space) <= (right_space - mid) else right_space

        return text[:split_pos].strip(), text[split_pos:].strip()

    # ----------------------------
    # Recursive splitting
    # ----------------------------

    def recursive_split(self, text: str) -> list[str]:
        text = self.light_clean(text)

        if not text:
            return []

        if self.count_tokens(text) <= self.max_tokens:
            return [self.finalize_chunk_text(text)]

        all_positions = self.find_separator_positions(text)

        for sep_type in self.separator_order:
            split_pos = self.choose_best_middle_split(text, all_positions[sep_type])

            if split_pos is not None:
                left = text[:split_pos].strip()
                right = text[split_pos:].strip()

                if left and right:
                    return self.recursive_split(left) + self.recursive_split(right)

        left, right = self.hard_split(text)

        if not left or not right:
            return [self.finalize_chunk_text(text)]

        return self.recursive_split(left) + self.recursive_split(right)

    # ----------------------------
    # Small chunk merging
    # ----------------------------

    def build_chunk_metadata(
        self,
        metadata: dict[str, Any] | None = None,
        *,
        title: str | None = None,
        ingested_at: str,
    ) -> dict[str, Any]:
        return {
            **(metadata or {}),
            "title": title,
            "ingested_at": ingested_at,
        }

    def create_chunk(
        self,
        *,
        text: str,
        source: str,
        chunk_index: int,
        metadata: dict[str, Any],
        title: str | None,
        ingested_at: str,
    ) -> Chunk:
        return Chunk(
            text=text,
            source=source,
            chunk_id=f"{source}_c{chunk_index}",
            token_count=self.count_tokens(text),
            title=title,
            ingested_at=ingested_at,
            metadata=metadata.copy(),
        )

    def merge_chunk_pair(self, left: Chunk, right: Chunk) -> Chunk:
        merged_text = f"{left.text} {right.text}".strip()
        merged_text = self.finalize_chunk_text(merged_text)

        return replace(
            left,
            text=merged_text,
            token_count=self.count_tokens(merged_text),
            metadata=left.metadata.copy(),
        )

    def merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        chunks = chunks.copy()
        i = 0

        while i < len(chunks):
            current = chunks[i]

            if current.token_count >= self.min_tokens:
                i += 1
                continue

            prev_exists = i > 0
            next_exists = i < len(chunks) - 1

            if not prev_exists and not next_exists:
                break

            prev_merged_tokens = None
            next_merged_tokens = None

            if prev_exists:
                prev_merged_tokens = self.count_tokens(chunks[i - 1].text + " " + current.text)

            if next_exists:
                next_merged_tokens = self.count_tokens(current.text + " " + chunks[i + 1].text)

            if next_exists and next_merged_tokens is not None and next_merged_tokens <= self.max_tokens:
                chunks[i + 1] = self.merge_chunk_pair(current, chunks[i + 1])
                del chunks[i]
                continue

            if prev_exists and prev_merged_tokens is not None and prev_merged_tokens <= self.max_tokens:
                chunks[i - 1] = self.merge_chunk_pair(chunks[i - 1], current)
                del chunks[i]
                i -= 1
                continue

            i += 1

        for chunk in chunks:
            chunk.token_count = self.count_tokens(chunk.text)

        return chunks

    # ----------------------------
    # Public chunking entrypoints
    # ----------------------------

    def chunk_text(
        self,
        text: str,
        source: str = "unknown",
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> list[Chunk]:
        ingested_at = datetime.now().isoformat(timespec="seconds")
        chunk_metadata = self.build_chunk_metadata(
            metadata,
            title=title,
            ingested_at=ingested_at,
        )
        pieces = self.recursive_split(text)
        chunks = [
            self.create_chunk(
                text=piece,
                source=source,
                chunk_index=i,
                metadata=chunk_metadata,
                title=title,
                ingested_at=ingested_at,
            )
            for i, piece in enumerate(pieces)
        ]

        chunks = self.merge_small_chunks(chunks)
        return self.reassign_chunk_ids(chunks)

    def reassign_chunk_ids(self, chunks: list[Chunk]) -> list[Chunk]:
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{chunk.source}_c{i}"
            chunk.token_count = self.count_tokens(chunk.text)

        return chunks


class TextChunker:
    def __init__(self, base_chunker: BaseChunker) -> None:
        self.base_chunker = base_chunker

    def chunk_txt(
        self,
        txt_path: str,
        title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()

        source = txt_path
        title = title or txt_path

        return self.base_chunker.chunk_text(
            text=text,
            source=source,
            metadata=extra_metadata,
            title=title,
        )


class HTMLChunker:
    def __init__(self, base_chunker: BaseChunker) -> None:
        self.base_chunker = base_chunker

    def fetch_html(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response:
            return response.read().decode("utf-8", errors="replace")

    def extract_main_text(self, html_content: str) -> tuple[str, str | None]:
        title_match = re.search(
            r"<title[^>]*>\s*(.*?)\s*</title>",
            html_content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        page_title = html.unescape(title_match.group(1).strip()) if title_match else None

        cleaned_html = re.sub(
            r"<(script|style|noscript|svg)[^>]*>.*?</\1>",
            "",
            html_content,
            flags=re.IGNORECASE | re.DOTALL,
        )

        main_match = re.search(
            r"<(main|article)[^>]*>(.*?)</\1>",
            cleaned_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        content_html = main_match.group(2) if main_match else cleaned_html
        content_html = re.sub(
            r"<(nav|header|footer|aside|form)[^>]*>.*?</\1>",
            "",
            content_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        content_html = re.sub(r"<br\s*/?>", "\n", content_html, flags=re.IGNORECASE)
        content_html = re.sub(r"</(p|div|section|li|h[1-6])>", "\n\n", content_html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", content_html)
        text = html.unescape(text)

        return text, page_title

    def chunk_url(
        self,
        url: str,
        title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        html_content = self.fetch_html(url)
        text, extracted_title = self.extract_main_text(html_content)

        return self.base_chunker.chunk_text(
            text=text,
            source=url,
            metadata=extra_metadata,
            title=title or extracted_title or url,
        )

    def chunk_html(
        self,
        html_content: str,
        source: str = "unknown.html",
        title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        text, extracted_title = self.extract_main_text(html_content)

        return self.base_chunker.chunk_text(
            text=text,
            source=source,
            metadata=extra_metadata,
            title=title or extracted_title or source,
        )

    def chunk_html_file(
        self,
        html_path: str,
        title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        with open(html_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        return self.chunk_html(
            html_content,
            source=html_path,
            title=title or html_path,
            extra_metadata=extra_metadata,
        )


class PDFChunker:
    def __init__(self, text_chunker: BaseChunker) -> None:
        self.text_chunker = text_chunker

    def apply_pdf_metadata(
        self,
        chunk: Chunk,
        *,
        page_number: int,
        chunk_index: int,
        title: str,
        ingested_at: str,
    ) -> None:
        chunk.chunk_id = f"{chunk.source}_p{page_number}_c{chunk_index}"
        chunk.page = page_number
        chunk.title = title
        chunk.ingested_at = ingested_at
        chunk.metadata["page"] = page_number

    def extract_pdf_blocks(self, pdf_path: str) -> list[dict]:
        doc = fitz.open(pdf_path)
        results = []

        for page_idx, page in enumerate(doc):
            blocks = page.get_text("blocks")
            page_blocks = []

            for block in blocks:
                text = block[4]
                cleaned = text.strip()
                if cleaned:
                    page_blocks.append({
                        "text": cleaned,
                    })

            results.append({
                "page_number": page_idx + 1,
                "blocks": page_blocks,
            })

        doc.close()
        return results

    def chunk_pdf(
        self,
        pdf_path: str,
        title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        extra_metadata = extra_metadata or {}
        ingested_at = datetime.now().isoformat(timespec="seconds")
        source = pdf_path
        title = title or pdf_path

        extracted_pages = self.extract_pdf_blocks(pdf_path)
        final_chunks: list[Chunk] = []

        for page in extracted_pages:
            page_number = page["page_number"]
            page_chunks: list[Chunk] = []

            for block in page["blocks"]:
                page_chunks.extend(
                    self.text_chunker.chunk_text(
                        text=block["text"],
                        source=source,
                        metadata=extra_metadata,
                        title=title,
                    )
                )

            page_chunks = self.text_chunker.merge_small_chunks(page_chunks)

            for chunk_idx, chunk in enumerate(page_chunks):
                self.apply_pdf_metadata(
                    chunk,
                    page_number=page_number,
                    chunk_index=chunk_idx,
                    title=title,
                    ingested_at=ingested_at,
                )
                final_chunks.append(chunk)

        return final_chunks
