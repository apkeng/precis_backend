"""FastAPI app for Précis: ingest newspaper clippings into events, then serve
event analysis, story tracing, "ask the sources" and quizzes - all grounded
in the clippings of the period the reader picked."""
from __future__ import annotations

import re
from datetime import date

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import analysis, quiz, store
from app.config import settings
from app.ingestion import process_clipping
from app.llm import LLMGenerationError, SchemaValidationError
from app.pdf_processor import extract_text_by_page
from app.schemas import (
    AskIn,
    AskResponse,
    ClippingIn,
    ClippingOut,
    EventDetail,
    EventListResponse,
    EventSummary,
    QuizIn,
    QuizResponse,
    StatsResponse,
    TopicsResponse,
    TraceResponse,
)
from app.topics import DIFFICULTIES, QUESTION_FORMATS, TOPIC_IDS, TOPICS

app = FastAPI(
    title="Précis backend",
    description=(
        "Newspaper clippings grouped into syllabus-tagged events, with "
        "Claude-written analysis and quizzes grounded in those clippings."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(LLMGenerationError)
@app.exception_handler(SchemaValidationError)
def llm_error_handler(_request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(ValueError)
def value_error_handler(_request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def require_ingest_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.ingest_api_key and x_api_key != settings.ingest_api_key:
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key.")


def _date_param(value: str, name: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{name} must be YYYY-MM-DD.") from exc
    return value


def _event_or_404(event_id: str) -> store.Event:
    try:
        return store.get_event(event_id)
    except store.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No event with id {event_id!r}.") from exc


def _summary(e: store.Event) -> EventSummary:
    return EventSummary(
        id=e.id,
        date=e.date,
        title=e.title,
        dek=e.dek,
        topics=e.topics,
        clipping_count=e.clipping_count,
        paper_count=e.paper_count,
    )


def _excerpt(text: str, limit: int = 280) -> str:
    """The clipping's opening sentence or two, cut at a word boundary."""
    text = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = sentences[0]
    if len(sentences) > 1 and len(out) + len(sentences[1]) < limit:
        out += " " + sentences[1]
    if len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0] + "…"
    return out


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/topics", response_model=TopicsResponse)
def topics() -> TopicsResponse:
    return TopicsResponse(
        topics=[{"id": t.id, "name": t.name, "short": t.short, "group": t.group} for t in TOPICS],
        formats=QUESTION_FORMATS,
        difficulties=DIFFICULTIES,
    )


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    return StatsResponse(**store.index_stats(date.today().isoformat()[:7]))


# --- Ingestion ---------------------------------------------------------------


def _clipping_out(c: store.Clipping) -> ClippingOut:
    return ClippingOut(
        id=c.id, paper=c.paper, date=c.date, page=c.page, title=c.title,
        status=c.status, error=c.error, event_id=c.event_id,
    )


@app.post("/clippings", response_model=ClippingOut, status_code=202, dependencies=[Depends(require_ingest_key)])
def ingest_clipping(body: ClippingIn, background_tasks: BackgroundTasks) -> ClippingOut:
    clipping = store.create_clipping(body.paper, body.date, body.page, body.title, body.text)
    background_tasks.add_task(process_clipping, clipping.id)
    return _clipping_out(clipping)


def _ingest_pdf(clipping_id: str, pdf_bytes: bytes) -> None:
    try:
        text = "\n\n".join(extract_text_by_page(pdf_bytes))
    except Exception as exc:  # noqa: BLE001 - surfaced through the clipping's status
        store.set_clipping_status(clipping_id, "failed", str(exc))
        return
    store.set_clipping_text(clipping_id, text)
    process_clipping(clipping_id)


@app.post(
    "/clippings/pdf", response_model=ClippingOut, status_code=202, dependencies=[Depends(require_ingest_key)]
)
async def ingest_clipping_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Clipping PDF with a text layer."),
    paper: str = Form(..., min_length=1),
    date_: str = Form(..., alias="date"),
    title: str = Form(..., min_length=1),
    page: int | None = Form(default=None),
) -> ClippingOut:
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")
    _date_param(date_, "date")
    pdf_bytes = await file.read()
    clipping = store.create_clipping(paper, date_, page, title, "")
    background_tasks.add_task(_ingest_pdf, clipping.id, pdf_bytes)
    return _clipping_out(clipping)


@app.get("/clippings/{clipping_id}", response_model=ClippingOut)
def get_clipping(clipping_id: str) -> ClippingOut:
    try:
        return _clipping_out(store.get_clipping(clipping_id))
    except store.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No clipping with id {clipping_id!r}.") from exc


# --- Reading -----------------------------------------------------------------


@app.get("/events", response_model=EventListResponse)
def list_events(
    date_from: str = Query(..., alias="from"),
    date_to: str = Query(..., alias="to"),
    topics: str | None = Query(
        default=None, description="Comma-separated topic ids; omit for all, empty for none."
    ),
) -> EventListResponse:
    _date_param(date_from, "from")
    _date_param(date_to, "to")
    in_range = store.list_events(date_from, date_to)
    wanted = None if topics is None else {t for t in topics.split(",") if t}
    events = in_range if wanted is None else [e for e in in_range if wanted.intersection(e.topics)]
    return EventListResponse(
        events=[_summary(e) for e in events],
        clipping_count=sum(e.clipping_count for e in events),
        topic_counts={t: sum(t in e.topics for e in in_range) for t in TOPIC_IDS},
    )


@app.get("/events/{event_id}", response_model=EventDetail)
def get_event(event_id: str) -> EventDetail:
    event = _event_or_404(event_id)
    clippings = store.event_clippings(event.id)
    return EventDetail(
        **_summary(event).model_dump(),
        sources=[
            {"n": i, "id": c.id, "paper": c.paper, "date": c.date, "page": c.page,
             "title": c.title, "excerpt": _excerpt(c.text)}
            for i, c in enumerate(clippings, start=1)
        ],
        analysis=analysis.event_analysis(event),
    )


@app.post("/events/{event_id}/ask", response_model=AskResponse)
def ask(event_id: str, body: AskIn) -> AskResponse:
    event = _event_or_404(event_id)
    return AskResponse(**analysis.ask(event, body.question, body.date_from, body.date_to))


@app.get("/events/{event_id}/trace", response_model=TraceResponse)
def trace(
    event_id: str,
    date_from: str = Query(..., alias="from"),
    date_to: str = Query(..., alias="to"),
) -> TraceResponse:
    _date_param(date_from, "from")
    _date_param(date_to, "to")
    event = _event_or_404(event_id)
    rel = analysis.related_events(event, date_from, date_to)
    return TraceResponse(
        event=_summary(event),
        days=analysis.coverage_days(event, rel["related"], date_from, date_to),
        related=[{**r, "event": _summary(r["event"])} for r in rel["related"]],
        outside_count=rel["outside_count"],
        outside_range=rel["outside_range"],
        framing=analysis.framing(event),
    )


# --- Quizzes -----------------------------------------------------------------


@app.get("/events/{event_id}/quiz", response_model=QuizResponse)
def event_quiz(event_id: str) -> QuizResponse:
    event = _event_or_404(event_id)
    questions = quiz.event_quiz(event, QUESTION_FORMATS)
    return QuizResponse(questions=questions, requested=3)


@app.post("/quizzes", response_model=QuizResponse)
def create_quiz(body: QuizIn) -> QuizResponse:
    events = store.list_events(body.date_from, body.date_to, body.topics)
    if body.event_ids is not None:
        wanted = set(body.event_ids)
        events = [e for e in events if e.id in wanted]
    if not events:
        raise HTTPException(status_code=409, detail="No events in this period match the selected topics.")
    try:
        questions = quiz.generate(events, body.topics, body.formats, body.length, body.difficulty)
    except quiz.NoMaterialError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return QuizResponse(questions=questions, requested=body.length)
