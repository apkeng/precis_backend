"""Write each event's Claude outputs - analysis, how papers framed it, and
its quiz question pool - as soon as ingestion has filed its clippings, so
readers only ever read cached results instead of waiting on Claude.

Runs at the end of an edition (or after a single clipping), over the events
that ingestion touched. Anything already cached is skipped, and a failure
on one event only means that event is generated on first view instead.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from app import analysis, quiz, store
from app.config import settings
from app.topics import QUESTION_FORMATS

log = logging.getLogger(__name__)


def prepare_event(event_id: str) -> None:
    event = store.get_event(event_id)
    if event.clipping_count == 0:
        return
    analysis.event_analysis(event)
    analysis.framing(event)
    quiz.ensure_pool(event, QUESTION_FORMATS)


def prepare_events(event_ids: list[str]) -> None:
    """Prepare `event_ids` (deduplicated), PREGENERATE_WORKERS at a time."""
    if not settings.pregenerate:
        return
    ids = [i for i in dict.fromkeys(event_ids) if i]
    if not ids:
        return

    def run(event_id: str) -> None:
        try:
            prepare_event(event_id)
        except Exception:  # noqa: BLE001 - the event falls back to on-view generation
            log.exception("Pre-generating event %s failed", event_id)

    with ThreadPoolExecutor(max_workers=max(1, settings.pregenerate_workers)) as pool:
        list(pool.map(run, ids))


def prepare_edition(edition_id: str) -> None:
    prepare_events([c["event_id"] for c in store.edition_clippings(edition_id) if c["status"] == "ready"])
