"""Clipping ingestion: chunk + embed locally, then one Claude call decides
which syllabus topics the clipping touches and which event it reports on.

Events are how the app is organised - several papers covering the same
story over several days are one event. A new clipping is compared with the
nearest existing events from the surrounding days (by vector similarity) and
Claude picks one of them or starts a new event. Clippings that touch none of
the 14 topics are kept as `off_syllabus` and never surface in the app.
"""
from __future__ import annotations

from datetime import date, timedelta

from app import llm, store, vector_store
from app.chunking import chunk_pages
from app.config import settings
from app.topics import TOPIC_IDS, describe_topics

TAG_SYSTEM_PROMPT = f"""You file newspaper clippings for a civil-services \
exam preparation app. For each clipping, decide:

1. Whether it is relevant to at least one of these syllabus topics, and which \
ones (use the ids):
{describe_topics()}

2. Whether it reports on the same underlying event as one of the candidate \
events listed - the same happening, decision, verdict or phenomenon, possibly \
covered by a different paper or on a later day. A follow-up on the same story \
counts as the same event; a different story on a related theme does not.

If it is a new event, write a neutral headline (under 15 words) and a \
one-sentence dek summarising it. If it matches a candidate, still fill the \
headline and dek, but they are ignored. Use only what the clipping says."""


def _tag_schema(candidate_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "relevant": {"type": "boolean"},
            "topics": {"type": "array", "items": {"type": "string", "enum": TOPIC_IDS}},
            "match": {"type": "string", "enum": [*candidate_ids, "new"]},
            "headline": {"type": "string"},
            "dek": {"type": "string"},
        },
        "required": ["relevant", "topics", "match", "headline", "dek"],
    }


def _tag_prompt(clipping: store.Clipping, candidates: list[store.Event]) -> str:
    listed = "\n".join(f"- {e.id} ({e.date}): {e.title}. {e.dek}" for e in candidates) or "(none)"
    return (
        f"Candidate events:\n{listed}\n\n"
        f"Clipping - {clipping.paper}, {clipping.date}, page {clipping.page or '?'}:\n"
        f"{clipping.title}\n\n{clipping.text[: settings.excerpt_chars]}"
    )


def _window(iso_date: str) -> tuple[str, str]:
    d = date.fromisoformat(iso_date)
    span = timedelta(days=settings.event_match_window_days)
    return (d - span).isoformat(), (d + span).isoformat()


def process_clipping(clipping_id: str) -> None:
    """Runs as a background task after the upload response has been sent."""
    try:
        clipping = store.get_clipping(clipping_id)
        chunks = chunk_pages([clipping.text], settings.chunk_size, settings.chunk_overlap)
        vector_store.add_clipping_chunks(
            clipping.id, clipping.paper, clipping.date, clipping.page, chunks
        )

        date_from, date_to = _window(clipping.date)
        candidate_ids = vector_store.nearest_events(
            f"{clipping.title}\n{clipping.text[:1000]}",
            settings.event_match_candidates,
            date_from,
            date_to,
        )
        candidates = store.get_events(candidate_ids)
        tag = llm.generate_json(
            TAG_SYSTEM_PROMPT,
            _tag_prompt(clipping, candidates),
            _tag_schema([e.id for e in candidates]),
            effort="low",
        )

        topics = [t for t in dict.fromkeys(tag["topics"]) if t in TOPIC_IDS]
        if not tag["relevant"] or not topics:
            store.set_clipping_status(clipping.id, "off_syllabus")
            return

        if tag["match"] in candidate_ids:
            event_id = tag["match"]
        else:
            event = store.create_event(tag["headline"], tag["dek"], clipping.date, topics)
            vector_store.upsert_event(event.id, event.title, event.dek, event.date)
            event_id = event.id

        store.attach_clipping(clipping.id, event_id, topics)
        vector_store.set_clipping_event(clipping.id, event_id)
    except Exception as exc:  # noqa: BLE001 - last resort, nothing else can surface this
        store.set_clipping_status(clipping_id, "failed", str(exc))
