"""PDF text extraction. Pure-Python (pypdf) - no external service or paid OCR API."""
from __future__ import annotations

import io

from pypdf import PdfReader


class EmptyPdfError(ValueError):
    """Raised when a PDF has no extractable text (e.g. scanned image-only pages)."""


def extract_text_by_page(pdf_bytes: bytes) -> list[str]:
    """Return a list of extracted text, one entry per page (0-indexed)."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]

    if not any(page.strip() for page in pages):
        raise EmptyPdfError(
            "No extractable text found in this PDF. It may be a scanned "
            "image-only document, which this pipeline does not OCR."
        )
    return pages
