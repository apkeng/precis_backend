"""Quiz drafting: Claude writes UPSC-style questions only from the clippings
of events in the chosen period, each tied back to the clipping it came from.

When the period doesn't have enough material, the quiz comes back shorter
than requested rather than padded - Claude is told to stop when it runs out
of groundable questions, and anything that fails the checks below is dropped.
"""
from __future__ import annotations

import uuid

from app import llm, store
from app.config import settings
from app.topics import TOPICS_BY_ID

OPTION_COUNT = 4

FORMAT_GUIDE = {
    "MCQ": "single correct answer from four options; statements must be empty",
    "Statement": 'two or three numbered statements in "statements"; options are '
    'combinations like "1 only", "2 only", "Both 1 and 2", "Neither 1 nor 2"',
    "Assertion–Reason": 'stem is "Assertion (A): ...\\nReason (R): ..."; options are the four '
    "standard A/R choices; statements must be empty",
    "Match": 'items to pair go in "statements" as lines like "A. X — 1. Y"; options are '
    'pairings like "A-2, B-3, C-1"',
}

QUIZ_SYSTEM_PROMPT = """You write UPSC Civil Services Prelims-style practice \
questions from newspaper clippings. Every question must be answerable from \
the clipping it cites (source_ref) - its facts, numbers, bodies, articles or \
the concepts it explicitly explains. Never rely on outside knowledge for the \
correct answer, and never write two questions testing the same fact.

Write at most the requested number of questions. If the material only \
supports fewer good questions, write fewer - do not pad. Spread questions \
across events and the allowed formats. Each question has exactly four \
options, one of them correct (answer_index is 0-3), and a one- or \
two-sentence explanation that says why the answer is right and, where useful, \
why a tempting option is wrong.

Formats:
{formats}

Difficulty: {difficulty}."""

DIFFICULTY_GUIDE = {
    "Easy": "direct recall of facts stated in the clipping",
    "Mixed": "a mix of direct recall and questions needing the reader to connect two facts",
    "Hard": "close distractors and questions needing the reader to connect facts or rule out plausible statements",
}


class NoMaterialError(ValueError):
    """No event in the selection has clippings to draw questions from."""


def _schema(event_refs: list[str], source_refs: list[str], topics: list[str], formats: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "event_ref": {"type": "string", "enum": event_refs},
                        "source_ref": {"type": "string", "enum": source_refs},
                        "topic": {"type": "string", "enum": topics},
                        "type": {"type": "string", "enum": formats},
                        "stem": {"type": "string"},
                        "statements": {"type": "array", "items": {"type": "string"}},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "answer_index": {"type": "integer"},
                        "explanation": {"type": "string"},
                    },
                    "required": [
                        "event_ref", "source_ref", "topic", "type", "stem",
                        "statements", "options", "answer_index", "explanation",
                    ],
                },
            }
        },
        "required": ["questions"],
    }


def generate(
    events: list[store.Event],
    topics: list[str],
    formats: list[str],
    length: int,
    difficulty: str,
) -> list[dict]:
    """Draft up to `length` questions from `events`, limited to `topics` and
    `formats`. Returns question dicts ready for the API."""
    # Most-covered events first: they carry the most groundable material.
    events = sorted(events, key=lambda e: e.clipping_count, reverse=True)[: settings.quiz_max_events]
    blocks, event_by_ref, source_by_ref, event_sources = [], {}, {}, {}
    for i, event in enumerate(events, start=1):
        clippings = store.event_clippings(event.id)[: settings.quiz_clippings_per_event]
        if not clippings:
            continue
        eref = f"E{i}"
        event_by_ref[eref] = event
        lines = [f"Event {eref} ({event.date}): {event.title}", f"Topics: {', '.join(event.topics)}"]
        for c in clippings:
            sref = f"S{len(source_by_ref) + 1}"
            source_by_ref[sref] = c
            event_sources.setdefault(eref, set()).add(sref)
            lines.append(f"[{sref}] {c.paper}, {c.date} - {c.title}\n{c.text[: settings.excerpt_chars]}")
        blocks.append("\n".join(lines))
    if not event_by_ref:
        raise NoMaterialError("No clippings in this selection to draw questions from.")

    topic_list = [t for t in topics if t in TOPICS_BY_ID]
    system = QUIZ_SYSTEM_PROMPT.format(
        formats="\n".join(f"- {f}: {FORMAT_GUIDE[f]}" for f in formats),
        difficulty=DIFFICULTY_GUIDE.get(difficulty, DIFFICULTY_GUIDE["Mixed"]),
    )
    prompt = (
        f"Write up to {length} questions.\n"
        f"Allowed topics (tag each question with one that its event carries): "
        f"{', '.join(f'{t} ({TOPICS_BY_ID[t].name})' for t in topic_list)}\n\n"
        + "\n\n".join(blocks)
    )
    result = llm.generate_json(
        system,
        prompt,
        _schema(list(event_by_ref), list(source_by_ref), topic_list, formats),
        effort="high",
    )

    questions, seen_stems = [], set()
    for q in result["questions"]:
        event = event_by_ref[q["event_ref"]]
        if q["source_ref"] not in event_sources[q["event_ref"]]:
            continue  # cites another event's clipping
        if q["topic"] not in event.topics:
            continue
        if len(q["options"]) != OPTION_COUNT or not 0 <= q["answer_index"] < OPTION_COUNT:
            continue
        if q["stem"] in seen_stems:
            continue
        seen_stems.add(q["stem"])
        clipping = source_by_ref[q["source_ref"]]
        questions.append(
            {
                "id": uuid.uuid4().hex[:12],
                "event_id": event.id,
                "event_title": event.title,
                "topic": q["topic"],
                "type": q["type"],
                "stem": q["stem"],
                "statements": q["statements"],
                "options": q["options"],
                "answer": q["answer_index"],
                "explanation": q["explanation"],
                "source": {"paper": clipping.paper, "date": clipping.date, "title": clipping.title},
            }
        )
        if len(questions) == length:
            break
    return questions


def event_quiz(event: store.Event, formats: list[str]) -> list[dict]:
    """The short "Practice this event" quiz, cached until a new clipping joins."""
    cached = event.caches.get("quiz")
    if cached is not None:
        return cached
    questions = generate([event], event.topics, formats, 3, "Mixed")
    store.set_event_cache(event.id, "quiz", questions)
    return questions
