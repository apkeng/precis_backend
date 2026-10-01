"""Turns extracted page text into overlapping chunks suitable for embedding.

Character-based (not tokenizer-based) on purpose: it keeps the pipeline
dependency-free and model-agnostic, at the cost of chunk sizes being
approximate token counts rather than exact ones.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page_number: int  # 1-indexed page the chunk starts on


def _split_page(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start += step
    return pieces


def chunk_pages(pages: list[str], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk each page independently so every chunk stays attributable to one page."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        for piece in _split_page(page_text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(text=piece, chunk_index=len(chunks), page_number=page_number)
            )
    return chunks
