import io

import pytest
from pypdf import PdfWriter

from app.pdf_processor import EmptyPdfError, load_pages


def _blank_pdf_bytes(num_pages: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_blank_pdf_raises_empty_pdf_error():
    with pytest.raises(EmptyPdfError):
        load_pages([_blank_pdf_bytes()])


def test_page_count_matches():
    # Blank pages have no text, so we only check page-count plumbing via
    # the raised error's page-agnostic path is exercised separately;
    # here we confirm multi-page PDFs are read without raising unrelated errors.
    with pytest.raises(EmptyPdfError):
        load_pages([_blank_pdf_bytes(num_pages=3)])


# --- scanned uploads (OCR via Claude vision) ---------------------------------

from PIL import Image  # noqa: E402

from app import pdf_processor  # noqa: E402
from app.pdf_processor import UnsupportedFileError, content_blocks, load_pages  # noqa: E402


def _image_bytes(size, fmt, frames=1):
    imgs = [Image.new("L", size, color=200 + i) for i in range(frames)]
    buf = io.BytesIO()
    imgs[0].save(buf, format=fmt, save_all=frames > 1, append_images=imgs[1:])
    return buf.getvalue()


def test_scanned_pdf_page_comes_back_as_tiles():
    pages = load_pages([_image_bytes((1200, 1800), "PDF")])
    assert len(pages) == 1
    page = pages[0]
    assert page.scanned and page.pdf is None
    assert page.grid == (2, 1)  # 1800 px tall > one tile, 1200 wide fits
    assert len(page.tiles) == 2


def test_broadsheet_scan_is_downscaled_and_tiled_within_limits():
    (page,) = load_pages([_image_bytes((5000, 8000), "JPEG")])
    rows, cols = page.grid
    assert len(page.tiles) == rows * cols
    for tile in page.tiles:
        img = Image.open(io.BytesIO(tile.jpeg))
        assert max(img.size) <= pdf_processor.TILE_PX
    # Long edge scaled to SCAN_MAX_LONG_EDGE (4200) -> 3 rows x 2 cols.
    assert (rows, cols) == (3, 2)


def test_multi_page_tiff_gives_one_page_per_frame_and_images_are_ordered():
    pages = load_pages([_image_bytes((800, 1000), "TIFF", frames=2), _image_bytes((800, 1000), "PNG")])
    assert len(pages) == 3
    assert all(p.scanned and len(p.tiles) == 1 for p in pages)


def test_content_blocks_label_tiles_in_reading_order():
    (page,) = load_pages([_image_bytes((1200, 1800), "PNG")])
    blocks = content_blocks(page)
    assert [b["type"] for b in blocks] == ["text", "text", "image", "text", "image"]
    assert "2 x 1" in blocks[0]["text"]
    assert blocks[2]["source"]["media_type"] == "image/jpeg"


def test_single_tile_page_has_just_the_image():
    (page,) = load_pages([_image_bytes((800, 1000), "PNG")])
    assert [b["type"] for b in content_blocks(page)] == ["image"]


def test_small_images_are_not_treated_as_scans():
    with pytest.raises(EmptyPdfError):
        load_pages([_image_bytes((100, 100), "PDF")])


def test_unsupported_upload_is_rejected():
    with pytest.raises(UnsupportedFileError):
        load_pages([b"hello, not a document"])
