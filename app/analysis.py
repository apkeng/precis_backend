"""Claude-written views of an event, grounded in its clippings: the event
analysis, how each paper framed it, answers to "Ask the sources", and the
related-event links behind the story analysis screen.

Sources are numbered [1]..[n] in the order `store.event_clippings` returns
them, and every citation Claude writes refers to that numbering - the
frontend highlights source n when [n] is clicked.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from app import llm, store, vector_store
from app.config import settings
from app.topics import TOPICS_BY_ID

GROUNDING_RULES = """Use ONLY the numbered sources below. Do not use outside \
knowledge and do not guess. Cite the sources behind each claim inline as [n]. \
If the sources don't support something, leave it out rather than inventing it."""

_CITE = re.compile(r"\s*\[(\d+)\]")


def _sources_block(clippings: list[store.Clipping]) -> str:
    return "\n\n".join(
        f"[{i}] {c.paper}, {c.date}, p.{c.page or '?'} - {c.title}\n{c.text[: settings.excerpt_chars]}"
        for i, c in enumerate(clippings, start=1)
    )


def _drop_bad_cites(text: str, source_count: int) -> str:
    """Remove citations that point at a source number that doesn't exist."""
    return _CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= source_count else "", text)


# --- Event analysis ----------------------------------------------------------

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "what_happened": {"type": "string"},
        "why_it_matters": {"type": "string"},
        "prelims_facts": {"type": "array", "items": {"type": "string"}},
        "gs_paper": {"type": "string"},
        "mains_question": {"type": "string"},
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"date": {"type": "string"}, "text": {"type": "string"}},
                "required": ["date", "text"],
            },
        },
    },
    "required": [
        "what_happened", "why_it_matters", "prelims_facts", "gs_paper", "mains_question", "timeline",
    ],
}

ANALYSIS_SYSTEM_PROMPT = f"""You write neutral study notes on a news event \
for civil-services aspirants (UPSC CSE). {GROUNDING_RULES}

Fields:
- what_happened: 2-3 sentences, with [n] citations.
- why_it_matters: 2-3 sentences on significance for governance, economy, \
society or the environment, with [n] citations.
- prelims_facts: 3-5 crisp, exam-style facts stated or clearly implied by the \
sources (articles, acts, bodies, numbers). No citations in this field.
- gs_paper: the most relevant paper and theme, e.g. "GS II · Federalism".
- mains_question: one Mains-style question this event could prompt. No \
citations, no surrounding quotes.
- timeline: the dated steps the sources mention, oldest first, with short \
date labels like "26 Sep" or "Mar 2026"."""


def event_analysis(event: store.Event) -> dict:
    cached = event.caches.get("analysis")
    if cached:
        return cached
    clippings = store.event_clippings(event.id)
    result = llm.generate_json(
        ANALYSIS_SYSTEM_PROMPT,
        f"Event: {event.title}\n\nSources:\n{_sources_block(clippings)}",
        ANALYSIS_SCHEMA,
    )
    for key in ("what_happened", "why_it_matters"):
        result[key] = _drop_bad_cites(result[key], len(clippings))
    store.set_event_cache(event.id, "analysis", result)
    return result


# --- How papers framed it ----------------------------------------------------

TONES = ["Explanatory", "Critical", "Neutral"]

FRAMING_SYSTEM_PROMPT = """For each newspaper, describe in one short sentence \
the angle its coverage of this event takes, and classify its tone as \
Explanatory (mainly explains), Critical (questions or criticises) or Neutral \
(reports without comment). Base this ONLY on the clippings given."""


def framing(event: store.Event) -> list[dict]:
    cached = event.caches.get("framing")
    if cached is not None:
        return cached
    clippings = store.event_clippings(event.id)
    papers = sorted({c.paper for c in clippings})
    if not papers:
        return []
    schema = {
        "type": "object",
        "properties": {
            "papers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "paper": {"type": "string", "enum": papers},
                        "angle": {"type": "string"},
                        "tone": {"type": "string", "enum": TONES},
                    },
                    "required": ["paper", "angle", "tone"],
                },
            }
        },
        "required": ["papers"],
    }
    result = llm.generate_json(
        FRAMING_SYSTEM_PROMPT,
        f"Event: {event.title}\n\nClippings:\n{_sources_block(clippings)}",
        schema,
        effort="low",
    )
    counts = {p: sum(c.paper == p for c in clippings) for p in papers}
    seen: set[str] = set()
    out = []
    for item in result["papers"]:
        if item["paper"] in seen:
            continue
        seen.add(item["paper"])
        out.append({**item, "clipping_count": counts[item["paper"]]})
    store.set_event_cache(event.id, "framing", out)
    return out


# --- Ask the sources ---------------------------------------------------------

ASK_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "found_in_sources": {"type": "boolean"}},
    "required": ["answer", "found_in_sources"],
}

ASK_SYSTEM_PROMPT = f"""You answer a civil-services aspirant's question about \
a news event. {GROUNDING_RULES}

Keep the answer to 2-4 sentences. If the sources don't answer the question, \
set found_in_sources to false and say briefly what they do and don't cover."""


def ask(event: store.Event, question: str, date_from: str, date_to: str) -> dict:
    """Retrieve passages from clippings in the period (this event's first),
    then answer only from them. Returns the answer and the numbered sources
    its [n] citations refer to."""
    query = f"{event.title}\n{question}"
    hits = vector_store.query_clippings(query, settings.default_top_k // 2 or 1, event_id=event.id)
    hits += vector_store.query_clippings(query, settings.default_top_k, date_from, date_to)

    # Group passages by clipping so each source number is one clipping.
    passages: dict[str, list[str]] = {}
    for hit in hits:
        texts = passages.setdefault(hit["clipping_id"], [])
        if hit["text"] not in texts:
            texts.append(hit["text"])
    clippings = [store.get_clipping(cid) for cid in passages]
    block = "\n\n".join(
        f"[{i}] {c.paper}, {c.date} - {c.title}\n" + "\n...\n".join(passages[c.id])
        for i, c in enumerate(clippings, start=1)
    ) or "(No clippings in this period matched the question.)"

    result = llm.generate_json(
        ASK_SYSTEM_PROMPT, f"Event: {event.title}\n\nSources:\n{block}\n\nQuestion: {question}", ASK_SCHEMA
    )
    return {
        "question": question,
        "answer": _drop_bad_cites(result["answer"], len(clippings)),
        "found_in_sources": result["found_in_sources"],
        "sources": [
            {"n": i, "paper": c.paper, "date": c.date, "title": c.title}
            for i, c in enumerate(clippings, start=1)
        ],
    }


# --- Related events (story analysis) -----------------------------------------

LINK_SYSTEM_PROMPT = """You map how news stories connect, for civil-services \
aspirants. For each candidate event, decide whether it is DIRECTLY linked to \
the main story - a cause, consequence, the same policy/legal process, the \
same actors acting on the same issue, or a concrete dependency - rather than \
merely sharing a broad theme. Give a one-line reason (under 15 words) for \
every candidate; for unlinked ones, the reason may be empty."""


def _check_links(event: store.Event, candidates: list[store.Event]) -> None:
    """Ask Claude about every candidate whose link to `event` isn't cached."""
    known = store.get_links(event.id, [c.id for c in candidates])
    unknown = [c for c in candidates if c.id not in known]
    if not unknown:
        return
    schema = {
        "type": "object",
        "properties": {
            "links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "string", "enum": [c.id for c in unknown]},
                        "linked": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["event_id", "linked", "reason"],
                },
            }
        },
        "required": ["links"],
    }
    listed = "\n".join(f"- {c.id} ({c.date}): {c.title}. {c.dek}" for c in unknown)
    result = llm.generate_json(
        LINK_SYSTEM_PROMPT,
        f"Main story ({event.date}): {event.title}. {event.dek}\n\nCandidates:\n{listed}",
        schema,
        effort="low",
    )
    answered = set()
    for link in result["links"]:
        if link["event_id"] in answered:
            continue
        answered.add(link["event_id"])
        store.set_link(event.id, link["event_id"], link["linked"], link["reason"])
    for c in unknown:  # anything Claude skipped counts as not linked
        if c.id not in answered:
            store.set_link(event.id, c.id, False, "")


def related_events(event: store.Event, date_from: str, date_to: str) -> dict:
    """Events related to `event`: direct links (Claude-judged, from the most
    similar events at any date) and same-theme events (shared topic, in the
    period). Direct links outside the period are counted, not listed."""
    candidate_ids = vector_store.nearest_events(
        f"{event.title}\n{event.dek}", settings.trace_candidates, exclude=event.id
    )
    candidates = store.get_events(candidate_ids)
    candidates = [c for c in candidates if c.clipping_count > 0]
    _check_links(event, candidates)
    links = store.get_links(event.id, [c.id for c in candidates])

    direct = [(c, links[c.id][1]) for c in candidates if links.get(c.id, (False, ""))[0]]
    in_range = [(c, r) for c, r in direct if date_from <= c.date <= date_to]
    outside = [c for c, _ in direct if not date_from <= c.date <= date_to]

    related = [{"event": c, "kind": "direct", "reason": r} for c, r in in_range]
    seen = {event.id, *(c.id for c, _ in in_range)}
    for other in store.list_events(date_from, date_to, event.topics):
        if other.id in seen:
            continue
        shared = next(t for t in event.topics if t in other.topics)
        related.append(
            {"event": other, "kind": "theme", "reason": f"Shares the {TOPICS_BY_ID[shared].name} thread"}
        )

    outside_range = None
    if outside:
        dates = [date_from, date_to, *(c.date for c in outside)]
        outside_range = {"from": min(dates), "to": max(dates)}
    return {"related": related, "outside_count": len(outside), "outside_range": outside_range}


def coverage_days(event: store.Event, related: list[dict], date_from: str, date_to: str) -> list[dict]:
    """Day-by-day clipping counts for the story and its related events, capped
    at TRACE_MAX_DAYS days from the start of the period."""
    start, end = date.fromisoformat(date_from), date.fromisoformat(date_to)
    end = min(end, start + timedelta(days=settings.trace_max_days - 1))
    rel_ids = [r["event"].id for r in related]
    counts = store.daily_counts([event.id, *rel_ids], start.isoformat(), end.isoformat())
    days = []
    d = start
    while d <= end:
        iso = d.isoformat()
        days.append(
            {
                "date": iso,
                "story": counts.get(event.id, {}).get(iso, 0),
                "related": sum(counts.get(rid, {}).get(iso, 0) for rid in rel_ids),
            }
        )
        d += timedelta(days=1)
    return days
