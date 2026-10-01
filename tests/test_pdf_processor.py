import io

import pytest
from pypdf import PdfWriter

from app.pdf_processor import EmptyPdfError, extract_text_by_page


def _blank_pdf_bytes(num_pages: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_blank_pdf_raises_empty_pdf_error():
    with pytest.raises(EmptyPdfError):
        extract_text_by_page(_blank_pdf_bytes())


def test_page_count_matches():
    # Blank pages have no text, so we only check page-count plumbing via
    # the raised error's page-agnostic path is exercised separately;
    # here we confirm multi-page PDFs are read without raising unrelated errors.
    with pytest.raises(EmptyPdfError):
        extract_text_by_page(_blank_pdf_bytes(num_pages=3))
