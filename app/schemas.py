"""Pydantic request/response models. JSON field names are camelCase, which
is what the Précis frontend's TypeScript types expect; Python code uses the
snake_case attribute names."""
from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from app.topics import DIFFICULTIES, QUESTION_FORMATS, TOPIC_IDS


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


def _iso(value: str) -> str:
    Date.fromisoformat(value)  # raises ValueError -> 422
    return value


# --- Reference data ----------------------------------------------------------


class TopicOut(CamelModel):
    id: str
    name: str
    short: str
    group: str


class TopicsResponse(CamelModel):
    topics: list[TopicOut]
    formats: list[str]
    difficulties: list[str]


class StatsResponse(CamelModel):
    papers: int
    clippings_this_month: int
    refreshed_at: str | None


# --- Clippings ---------------------------------------------------------------


class ClippingIn(CamelModel):
    paper: str = Field(..., min_length=1)
    date: str = Field(..., description="Publication date, YYYY-MM-DD.")
    page: int | None = Field(default=None, ge=1)
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, description="Full text of the clipping.")

    _check_date = field_validator("date")(_iso)


class ClippingOut(CamelModel):
    id: str
    paper: str
    date: str
    page: int | None
    title: str
    status: str
    error: str | None
    event_id: str | None


class EditionOut(CamelModel):
    id: str
    paper: str
    date: str
    pages: int
    pages_done: int
    status: str  # processing | ready | failed
    error: str | None
    clippings: dict[str, int] = Field(
        default_factory=dict, description="Clippings by status: ready, off_syllabus, failed, processing."
    )
    events: int = 0


# --- Events ------------------------------------------------------------------


class EventSummary(CamelModel):
    id: str
    date: str
    title: str
    dek: str
    topics: list[str]
    clipping_count: int
    paper_count: int


class EventListResponse(CamelModel):
    events: list[EventSummary]
    clipping_count: int
    topic_counts: dict[str, int]


class SourceOut(CamelModel):
    n: int
    id: str
    paper: str
    date: str
    page: int | None
    title: str
    excerpt: str


class TimelineItem(CamelModel):
    date: str
    text: str


class AnalysisOut(CamelModel):
    what_happened: str
    why_it_matters: str
    prelims_facts: list[str]
    gs_paper: str
    mains_question: str
    timeline: list[TimelineItem]


class EventDetail(EventSummary):
    sources: list[SourceOut]
    analysis: AnalysisOut


class AskIn(CamelModel):
    question: str = Field(..., min_length=1, max_length=500)
    date_from: str = Field(..., alias="from")
    date_to: str = Field(..., alias="to")

    _check_dates = field_validator("date_from", "date_to")(_iso)


class AskSource(CamelModel):
    n: int
    paper: str
    date: str
    title: str


class AskResponse(CamelModel):
    question: str
    answer: str
    found_in_sources: bool
    sources: list[AskSource]


class RelatedOut(CamelModel):
    event: EventSummary
    kind: str  # "direct" | "theme"
    reason: str


class CoverageDay(CamelModel):
    date: str
    story: int
    related: int


class FramingOut(CamelModel):
    paper: str
    angle: str
    tone: str
    clipping_count: int


class DateRange(CamelModel):
    date_from: str = Field(..., alias="from")
    date_to: str = Field(..., alias="to")


class TraceResponse(CamelModel):
    event: EventSummary
    days: list[CoverageDay]
    related: list[RelatedOut]
    outside_count: int
    outside_range: DateRange | None
    framing: list[FramingOut]


# --- Quizzes -----------------------------------------------------------------


class QuizSource(CamelModel):
    paper: str
    date: str
    title: str


class QuestionOut(CamelModel):
    id: str
    event_id: str
    event_title: str
    topic: str
    type: str
    stem: str
    statements: list[str]
    options: list[str]
    answer: int
    explanation: str
    source: QuizSource


class QuizIn(CamelModel):
    date_from: str = Field(..., alias="from")
    date_to: str = Field(..., alias="to")
    topics: list[str] = Field(..., min_length=1)
    formats: list[str] = Field(default_factory=lambda: list(QUESTION_FORMATS), min_length=1)
    length: int = Field(default=10, ge=1, le=30)
    difficulty: str = "Mixed"
    event_ids: list[str] | None = Field(
        default=None, description="Limit to these events (e.g. a story and its related events)."
    )

    _check_dates = field_validator("date_from", "date_to")(_iso)

    @field_validator("topics")
    @classmethod
    def _known_topics(cls, value: list[str]) -> list[str]:
        unknown = set(value) - set(TOPIC_IDS)
        if unknown:
            raise ValueError(f"Unknown topics: {', '.join(sorted(unknown))}")
        return value

    @field_validator("formats")
    @classmethod
    def _known_formats(cls, value: list[str]) -> list[str]:
        unknown = set(value) - set(QUESTION_FORMATS)
        if unknown:
            raise ValueError(f"Unknown formats: {', '.join(sorted(unknown))}")
        return value

    @field_validator("difficulty")
    @classmethod
    def _known_difficulty(cls, value: str) -> str:
        if value not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {', '.join(DIFFICULTIES)}")
        return value


class QuizResponse(CamelModel):
    questions: list[QuestionOut]
    requested: int
