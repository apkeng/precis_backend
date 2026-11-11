"""Reading a scanned page's text with Claude vision (for single clippings).

Whole editions don't use this: their scanned pages go straight into the
segmenting call in app/editions.py, which reads and splits in one step.
"""
from __future__ import annotations

from app import llm
from app.pdf_processor import Page, content_blocks

TRANSCRIBE_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}

TRANSCRIBE_SYSTEM_PROMPT = """You transcribe a scanned newspaper clipping. \
Return its article text exactly as printed: join line breaks inside \
sentences and words hyphenated across lines, separate paragraphs with blank \
lines, and leave out captions, bylines, page furniture and "Continued on" \
markers. Do not summarise, shorten, correct or reword. Where a word is \
illegible, give your best reading rather than a placeholder. If the image \
has no readable text, return an empty string."""


def page_text(page: Page) -> str:
    """The page's text: its text layer, or a transcription of its scan."""
    if not page.scanned:
        return page.text
    result = llm.generate_json(
        TRANSCRIBE_SYSTEM_PROMPT,
        "Transcribe the clipping above.",
        TRANSCRIBE_SCHEMA,
        effort="low",
        attachments=content_blocks(page),
    )
    return result["text"]
