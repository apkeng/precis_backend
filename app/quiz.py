"""Quizzes: Claude writes UPSC-style questions only from an event's
clippings, each tied back to the clipping it came from.

Every event has a question pool, written once (right after ingestion, see
app/pregenerate.py, or on first use) and cached with the event until a new
clipping joins it. Quizzes are assembled from these pools, so a reader's
custom quiz usually costs no Claude call. Events from before pre-generation
(or whose pre-generation failed) get their pool drafted the first time a
quiz needs them - only as many as the quiz is short of - and kept.

When the selection doesn't have enough pooled questions, the quiz comes back
shorter than requested rather than padded. Claude is told not to pad a pool
either, and anything that fails the checks below is dropped.
"""
from __future__ import annotations

import math
import random
import uuid
from concurrent.futures import ThreadPoolExecutor

from app import llm, store
from app.config import settings
from app.topics import QUESTION_FORMATS, TOPICS_BY_ID

OPTION_COUNT = 4
POOL_DIFFICULTIES = ["Easy", "Hard"]
EVENT_QUIZ_LENGTH = 3

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
questions from newspaper clippings about one news event. Every question must \
be answerable from the clipping it cites (source_ref) - its facts, numbers, \
bodies, articles or the concepts it explicitly explains. Never rely on \
outside knowledge for the correct answer, and never write two questions \
testing the same fact.

Write at most the requested number of questions. If the material only \
supports fewer good questions, write fewer - do not pad. Spread questions \
across the formats below. Each question has exactly four options, one of \
them correct (answer_index is 0-3), and a one- or two-sentence explanation \
that says why the answer is right and, where useful, why a tempting option \
is wrong.

Formats:
{formats}

Label each question's difficulty, and aim for a mix of both:
- Easy: direct recall of a fact stated in the clipping.
- Hard: close distractors, or the reader must connect facts or rule out \
plausible statements."""


class NoMaterialError(ValueError):
    """Nothing in the selection to draw questions from."""


def _schema(source_refs: list[str], topics: list[str], formats: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_ref": {"type": "string", "enum": source_refs},
                        "topic": {"type": "string", "enum": topics},
                        "type": {"type": "string", "enum": formats},
                        "difficulty": {"type": "string", "enum": POOL_DIFFICULTIES},
                        "stem": {"type": "string"},
                        "statements": {"type": "array", "items": {"type": "string"}},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "answer_index": {"type": "integer"},
                        "explanation": {"type": "string"},
                    },
                    "required": [
                        "source_ref", "topic", "type", "difficulty", "stem",
                        "statements", "options", "answer_index", "explanation",
                    ],
                },
            }
        },
        "required": ["questions"],
    }


def generate_pool(event: store.Event, formats: list[str]) -> list[dict]:
    """Draft up to QUIZ_POOL_SIZE questions from `event`'s clippings, in any of
    `formats`. Returns question dicts ready for the API."""
    clippings = store.event_clippings(event.id)[: settings.quiz_clippings_per_event]
    if not clippings:
        return []
    source_by_ref = {f"S{i}": c for i, c in enumerate(clippings, start=1)}
    sources = "\n\n".join(
        f"[{ref}] {c.paper}, {c.date} - {c.title}\n{c.text[: settings.excerpt_chars]}"
        for ref, c in source_by_ref.items()
    )
    system = QUIZ_SYSTEM_PROMPT.format(formats="\n".join(f"- {f}: {FORMAT_GUIDE[f]}" for f in formats))
    prompt = (
        f"Write up to {settings.quiz_pool_size} questions.\n"
        f"Topics (tag each question with one): "
        f"{', '.join(f'{t} ({TOPICS_BY_ID[t].name})' for t in event.topics if t in TOPICS_BY_ID)}\n\n"
        f"Event ({event.date}): {event.title}\n\n{sources}"
    )
    result = llm.generate_json(
        system, prompt, _schema(list(source_by_ref), event.topics, formats), effort="high"
    )

    questions, seen_stems = [], set()
    for q in result["questions"]:
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
                "difficulty": q["difficulty"],
                "stem": q["stem"],
                "statements": q["statements"],
                "options": q["options"],
                "answer": q["answer_index"],
                "explanation": q["explanation"],
                "source": {"paper": clipping.paper, "date": clipping.date, "title": clipping.title},
            }
        )
        if len(questions) == settings.quiz_pool_size:
            break
    return questions


def ensure_pool(event: store.Event, formats: list[str]) -> list[dict]:
    """The event's question pool, drafted and cached on first use."""
    cached = event.caches.get("quiz")
    if cached is not None:
        return cached
    pool = generate_pool(event, formats)
    store.set_event_cache(event.id, "quiz", pool)
    return pool


def event_quiz(event: store.Event, formats: list[str]) -> list[dict]:
    """The short "Practice this event" quiz: the first few pooled questions,
    one per format where the pool allows."""
    pool = ensure_pool(event, formats)
    picked, types = [], set()
    for q in pool:
        if q["type"] not in types:
            picked.append(q)
            types.add(q["type"])
    picked += [q for q in pool if q not in picked]
    return picked[:EVENT_QUIZ_LENGTH]


def assemble(
    events: list[store.Event],
    topics: list[str],
    formats: list[str],
    length: int,
    difficulty: str,
    rng: random.Random | None = None,
) -> list[dict]:
    """Up to `length` pooled questions from `events`, limited to `topics`,
    `formats` and `difficulty` ("Mixed" takes both). Taken one event at a
    time, most-covered events first, so the quiz spreads across stories."""
    rng = rng or random.Random()
    wanted = set(topics)
    pools = []
    for event in sorted(events, key=lambda e: e.clipping_count, reverse=True):
        pool = [
            q for q in event.caches.get("quiz") or []
            if q["topic"] in wanted
            and q["type"] in formats
            and (difficulty not in POOL_DIFFICULTIES or q.get("difficulty") == difficulty)
        ]
        rng.shuffle(pool)
        if pool:
            pools.append(pool)

    questions = []
    while pools and len(questions) < length:
        for pool in pools:
            questions.append(pool.pop())
            if len(questions) == length:
                break
        pools = [p for p in pools if p]
    return questions


def build(
    events: list[store.Event],
    topics: list[str],
    formats: list[str],
    length: int,
    difficulty: str,
    rng: random.Random | None = None,
) -> list[dict]:
    """A custom quiz: `assemble` from the existing pools, and if that falls
    short, first draft pools (in parallel) for enough of the events that
    have none - most-covered first, at most QUIZ_FALLBACK_EVENTS."""
    if not events:
        raise NoMaterialError("No clippings in this selection to draw questions from.")
    questions = assemble(events, topics, formats, length, difficulty, rng)
    if len(questions) >= length:
        return questions

    wanted = set(topics)
    unpooled = sorted(
        (e for e in events if e.caches.get("quiz") is None and wanted.intersection(e.topics)),
        key=lambda e: e.clipping_count,
        reverse=True,
    )
    # A pool spans every format and both difficulties, so only part of it
    # fits this quiz.
    share = len(formats) / len(QUESTION_FORMATS) * (0.5 if difficulty in POOL_DIFFICULTIES else 1)
    per_event = max(1.0, settings.quiz_pool_size * share)
    needed = math.ceil((length - len(questions)) / per_event)
    drafting = unpooled[: min(needed, settings.quiz_fallback_events)]
    if not drafting:
        return questions

    def draft(event: store.Event) -> Exception | None:
        try:
            event.caches["quiz"] = ensure_pool(event, QUESTION_FORMATS)
        except Exception as exc:  # noqa: BLE001 - one failed event shouldn't sink the quiz
            return exc
        return None

    with ThreadPoolExecutor(max_workers=len(drafting)) as pool:
        errors = [e for e in pool.map(draft, drafting) if e is not None]
    questions = assemble(events, topics, formats, length, difficulty, rng)
    if not questions and errors:
        raise errors[0]
    return questions
