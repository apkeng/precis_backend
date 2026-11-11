"""Turning uploads (PDFs and scanned images) into pages for the pipeline.

A page with a usable text layer is read with pypdf, as before. A page
without one - a scanned PDF, or a photo/scan uploaded as JPEG, PNG, WEBP or
TIFF - comes back as images, and Claude reads it (vision) in the same call
that segments or transcribes it. No local OCR engine or extra paid OCR API.

Claude downsamples any image whose long edge is over ~1568 px, which would
make newsprint unreadable on a full scanned page. So a scan is cut into
overlapping tiles of at most that size and sent as one ordered set of images.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field

from PIL import Image, ImageOps
from pypdf import PdfReader, PdfWriter

# Fewer characters than this in a page's text layer (and an image on the
# page) means it's a scan; real e-paper pages have thousands.
SCANNED_PAGE_MAX_CHARS = 200
# Images smaller than this on both sides are logos/icons, not scans.
MIN_SCAN_SIDE_PX = 400
# Scans are first downscaled to this long edge (~180 dpi for a broadsheet,
# plenty for print), then cut into tiles Claude reads at full resolution.
SCAN_MAX_LONG_EDGE = 4200
TILE_PX = 1568
TILE_OVERLAP_PX = 120
JPEG_QUALITY = 85

_PDF_MAGIC = b"%PDF"
_IMAGE_MAGICS = (b"\xff\xd8\xff", b"\x89PNG", b"II*\x00", b"MM\x00*", b"GIF8")


class EmptyPdfError(ValueError):
    """Raised when an upload has neither text nor anything readable as a scan."""


class UnsupportedFileError(ValueError):
    """Raised for an upload that is neither a PDF nor a supported image."""


@dataclass
class ScanTile:
    row: int
    col: int
    jpeg: bytes


@dataclass
class Page:
    text: str = ""
    # Set when the page has to be read from its image (OCR).
    tiles: list[ScanTile] = field(default_factory=list)
    grid: tuple[int, int] = (0, 0)  # rows, cols
    # Fallback for a scanned PDF page whose image(s) pypdf can't hand over
    # cleanly (several strips, JBIG2...): the one-page PDF itself, which
    # Claude reads as a document.
    pdf: bytes | None = None

    @property
    def scanned(self) -> bool:
        return bool(self.tiles or self.pdf)


def is_supported_upload(data: bytes) -> bool:
    return data.startswith(_PDF_MAGIC) or _is_image(data)


def _is_image(data: bytes) -> bool:
    return data.startswith(_IMAGE_MAGICS) or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")


def load_pages(uploads: list[bytes]) -> list[Page]:
    """Pages of every upload, in order. A PDF gives one page per PDF page;
    an image gives one page per frame (multi-page TIFFs have several)."""
    pages: list[Page] = []
    for data in uploads:
        if data.startswith(_PDF_MAGIC):
            pages.extend(_pdf_pages(data))
        elif _is_image(data):
            pages.extend(_image_pages(data))
        else:
            raise UnsupportedFileError("Upload must be a PDF or an image (JPEG, PNG, WEBP, TIFF).")
    if not any(page.text.strip() or page.scanned for page in pages):
        raise EmptyPdfError(
            "No text or scanned content found in this upload - every page is blank."
        )
    return pages


def extract_text_by_page(pdf_bytes: bytes) -> list[str]:
    """Text layer only, one entry per page (no OCR)."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [page.extract_text() or "" for page in reader.pages]


def _pdf_pages(pdf_bytes: bytes) -> list[Page]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for pdf_page in reader.pages:
        text = pdf_page.extract_text() or ""
        if len(text.strip()) >= SCANNED_PAGE_MAX_CHARS:
            pages.append(Page(text=text))
            continue
        scan = _page_scan_image(pdf_page)
        if scan is None:
            pages.append(Page(text=text))  # blank, or a page of only small graphics
        elif scan == "whole-page":
            pages.append(Page(text=text, pdf=_single_page_pdf(pdf_page)))
        else:
            pages.append(_tiled_page(scan, text))
    return pages


def _page_scan_image(pdf_page) -> Image.Image | str | None:
    """The page's scan image, "whole-page" when it has scan-sized images that
    can't be used directly, or None when it has none."""
    try:
        images = list(pdf_page.images)
    except Exception:  # noqa: BLE001 - unreadable image streams: let Claude render the page
        return "whole-page"
    big = []
    for image_file in images:
        try:
            img = image_file.image
        except Exception:  # noqa: BLE001 - e.g. JBIG2, which Pillow can't decode
            return "whole-page"
        if img is not None and min(img.size) >= MIN_SCAN_SIDE_PX:
            big.append(img)
    if not big:
        return None
    if len(big) > 1:
        return "whole-page"  # scanner stored the page in strips/layers
    return big[0]


def _single_page_pdf(pdf_page) -> bytes:
    writer = PdfWriter()
    writer.add_page(pdf_page)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _image_pages(data: bytes) -> list[Page]:
    try:
        img = Image.open(io.BytesIO(data))
        frames = []
        for index in range(getattr(img, "n_frames", 1)):
            img.seek(index)
            frames.append(img.copy())
    except Exception as exc:  # noqa: BLE001 - corrupt or unsupported image
        raise UnsupportedFileError(f"Could not read the uploaded image: {exc}") from exc
    return [_tiled_page(ImageOps.exif_transpose(frame)) for frame in frames]


def _tiled_page(img: Image.Image, text: str = "") -> Page:
    img = img.convert("L" if img.mode in ("1", "L", "LA", "I;16") else "RGB")
    scale = SCAN_MAX_LONG_EDGE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)

    cols, xs = _tile_starts(img.width)
    rows, ys = _tile_starts(img.height)
    tiles = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            tile = img.crop((x, y, min(x + TILE_PX, img.width), min(y + TILE_PX, img.height)))
            buf = io.BytesIO()
            tile.save(buf, format="JPEG", quality=JPEG_QUALITY)
            tiles.append(ScanTile(row=r, col=c, jpeg=buf.getvalue()))
    return Page(text=text, tiles=tiles, grid=(rows, cols))


def _tile_starts(length: int) -> tuple[int, list[int]]:
    if length <= TILE_PX:
        return 1, [0]
    count = math.ceil((length - TILE_OVERLAP_PX) / (TILE_PX - TILE_OVERLAP_PX))
    step = (length - TILE_PX) / (count - 1)
    return count, [round(i * step) for i in range(count)]


def content_blocks(page: Page) -> list[dict]:
    """Claude message content blocks that show a scanned page."""
    if page.pdf is not None:
        return [{
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf",
                       "data": base64.b64encode(page.pdf).decode()},
        }]
    rows, cols = page.grid
    blocks: list[dict] = []
    if len(page.tiles) > 1:
        blocks.append({"type": "text", "text": (
            f"The scanned page is cut into {rows} x {cols} overlapping tiles "
            "(rows top to bottom, columns left to right). Text in an overlap "
            "strip appears on two tiles - read it once."
        )})
    for tile in page.tiles:
        if len(page.tiles) > 1:
            blocks.append({"type": "text", "text": f"Tile row {tile.row + 1}, column {tile.col + 1}:"})
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": base64.b64encode(tile.jpeg).decode()},
        })
    return blocks
