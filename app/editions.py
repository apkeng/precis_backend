"""Whole-edition ingestion: an e-paper PDF (every page of one day's paper)
is split into articles, and each article becomes a clipping that goes
through the normal tag-and-match step (app/ingestion.py).

PDF text comes out in column order with several stories per page, so Claude
does the splitting: one call per page returns each article's headline and
cleaned text, and labels what kind of item it is. Only news and opinion
pieces are ingested - adverts, indexes, captions, sport and entertainment
are dropped here, before any per-article tagging cost. A story continued
onto another page becomes two clippings, which the event matcher then
joins into one event.
"""
from __future__ import annotations

from app import llm, store
from app.ingestion import process_clipping
from app.pdf_processor import extract_text_by_page
from app.text_cleanup import clean_extracted_text

INGESTED_KINDS = ("news", "opinion")
MIN_ARTICLE_CHARS = 300

SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "articles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "headline": {"type": "string"},
                    "kind": {"type": "string", "enum": ["news", "opinion", "other"]},
                    "text": {"type": "string"},
                },
                "required": ["headline", "kind", "text"],
            },
        }
    },
    "required": ["articles"],
}

SEGMENT_SYSTEM_PROMPT = """You split one page of a newspaper's e-paper into \
its separate articles. The page text was extracted from a PDF in column \
order, so line breaks fall mid-sentence and pieces of different items can \
sit next to each other: a front-page index, photo captions, "Continued on \
page N" jumps, bylines, adverts, weather boxes.

Return every article on the page, each with:
- headline: the article's own headline (not a section label or kicker).
- kind: "news" for reporting on public affairs - politics, government, law \
and courts, economy and business, international affairs, science, \
environment, society; "opinion" for editorials, columns and op-eds; \
"other" for sport, entertainment, lifestyle, adverts, indexes and anything else.
- text: the article's full text exactly as printed, with line breaks inside \
sentences joined, paragraphs separated by blank lines, and captions, \
bylines and "Continued on" markers removed. Do not summarise, shorten or \
reword it. For "other" items, text may be empty."""


def process_edition(edition_id: str, pdf_bytes: bytes) -> None:
    """Runs as a background task after the upload response has been sent."""
    edition = store.get_edition(edition_id)
    try:
        pages = extract_text_by_page(pdf_bytes)
    except Exception as exc:  # noqa: BLE001 - surfaced through the edition's status
        store.update_edition(edition_id, status="failed", error=str(exc))
        return
    store.update_edition(edition_id, pages=len(pages))

    page_errors = []
    for page_number, raw in enumerate(pages, start=1):
        try:
            _ingest_page(edition, page_number, clean_extracted_text(raw))
        except Exception as exc:  # noqa: BLE001 - one bad page shouldn't stop the edition
            page_errors.append(f"page {page_number}: {exc}")
        store.update_edition(edition_id, pages_done=page_number)

    store.update_edition(
        edition_id,
        status="ready" if len(page_errors) < len(pages) else "failed",
        error="; ".join(page_errors) or None,
    )


def _ingest_page(edition: store.Edition, page_number: int, text: str) -> None:
    if not text.strip():
        return
    result = llm.generate_json(
        SEGMENT_SYSTEM_PROMPT,
        f"{edition.paper}, {edition.date}, page {page_number}:\n\n{text}",
        SEGMENT_SCHEMA,
        effort="low",
    )
    for article in result["articles"]:
        if article["kind"] not in INGESTED_KINDS or len(article["text"]) < MIN_ARTICLE_CHARS:
            continue
        clipping = store.create_clipping(
            edition.paper,
            edition.date,
            page_number,
            article["headline"],
            article["text"],
            edition_id=edition.id,
        )
        # Sequential, so later articles can join events earlier ones created.
        process_clipping(clipping.id)
