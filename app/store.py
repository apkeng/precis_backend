"""Structured records in SQLite: clippings, the events they're grouped into,
and Claude's cached per-event outputs.

Chroma (app/vector_store.py) only holds what needs similarity search -
clipping chunks and one vector per event. Everything relational (which
clipping belongs to which event, date-range listings, cached analyses)
lives here, so there's still no separate database server to run.

Dates are ISO `YYYY-MM-DD` strings, which sort and compare correctly as text.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.config import settings

# Columns added after a table first shipped: CREATE TABLE IF NOT EXISTS
# leaves an existing table alone, so these are added to older databases.
_ADDED_COLUMNS = [("editions", "uploaded_by", "TEXT")]

# Per-event caches, cleared whenever a new clipping joins the event so the
# next read regenerates them from the fuller set of sources.
EVENT_CACHE_FIELDS = ("analysis", "framing", "quiz")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    dek TEXT NOT NULL,
    date TEXT NOT NULL,
    topics TEXT NOT NULL,
    created_at TEXT NOT NULL,
    analysis TEXT,
    framing TEXT,
    quiz TEXT
);
CREATE INDEX IF NOT EXISTS events_date ON events(date);

CREATE TABLE IF NOT EXISTS clippings (
    id TEXT PRIMARY KEY,
    paper TEXT NOT NULL,
    date TEXT NOT NULL,
    page INTEGER,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    event_id TEXT REFERENCES events(id),
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    edition_id TEXT REFERENCES editions(id)
);
CREATE INDEX IF NOT EXISTS clippings_event ON clippings(event_id);
CREATE INDEX IF NOT EXISTS clippings_date ON clippings(date);

CREATE TABLE IF NOT EXISTS editions (
    id TEXT PRIMARY KEY,
    paper TEXT NOT NULL,
    date TEXT NOT NULL,
    pages INTEGER NOT NULL DEFAULT 0,
    pages_done INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    uploaded_by TEXT
);
CREATE INDEX IF NOT EXISTS editions_created ON editions(created_at);

CREATE TABLE IF NOT EXISTS event_links (
    a TEXT NOT NULL,
    b TEXT NOT NULL,
    linked INTEGER NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (a, b)
);
"""


class NotFoundError(KeyError):
    pass


@dataclass
class Clipping:
    id: str
    paper: str
    date: str
    page: int | None
    title: str
    text: str
    event_id: str | None
    status: str
    error: str | None
    created_at: str
    edition_id: str | None = None


@dataclass
class Edition:
    id: str
    paper: str
    date: str
    pages: int
    pages_done: int
    status: str
    error: str | None
    created_at: str
    uploaded_by: str | None = None


@dataclass
class Event:
    id: str
    title: str
    dek: str
    date: str
    topics: list[str]
    created_at: str
    clipping_count: int = 0
    paper_count: int = 0
    caches: dict = field(default_factory=dict)


def _db_path() -> str:
    if settings.sqlite_path:
        return settings.sqlite_path
    return os.path.join(os.path.dirname(os.path.abspath(settings.chroma_persist_dir)), "precis.db")


_initialized_paths: set[str] = set()


@contextmanager
def _conn():
    path = _db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        if path not in _initialized_paths:
            conn.executescript(_SCHEMA)
            for table, column, kind in _ADDED_COLUMNS:
                existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
            _initialized_paths.add(path)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clipping(row: sqlite3.Row) -> Clipping:
    return Clipping(**{k: row[k] for k in row.keys()})


def _event(row: sqlite3.Row) -> Event:
    keys = row.keys()
    caches = {f: json.loads(row[f]) for f in EVENT_CACHE_FIELDS if f in keys and row[f]}
    return Event(
        id=row["id"],
        title=row["title"],
        dek=row["dek"],
        date=row["date"],
        topics=json.loads(row["topics"]),
        created_at=row["created_at"],
        clipping_count=row["clipping_count"] if "clipping_count" in keys else 0,
        paper_count=row["paper_count"] if "paper_count" in keys else 0,
        caches=caches,
    )


# --- Clippings ---------------------------------------------------------------


def create_clipping(
    paper: str, date: str, page: int | None, title: str, text: str, edition_id: str | None = None
) -> Clipping:
    clipping = Clipping(
        id=uuid.uuid4().hex,
        paper=paper,
        date=date,
        page=page,
        title=title,
        text=text,
        event_id=None,
        status="processing",
        error=None,
        created_at=_now(),
        edition_id=edition_id,
    )
    with _conn() as c:
        c.execute(
            "INSERT INTO clippings (id, paper, date, page, title, text, status, created_at, edition_id) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (clipping.id, paper, date, page, title, text, "processing", clipping.created_at, edition_id),
        )
    return clipping


def set_clipping_text(clipping_id: str, text: str) -> None:
    with _conn() as c:
        c.execute("UPDATE clippings SET text=? WHERE id=?", (text, clipping_id))


def set_clipping_status(clipping_id: str, status: str, error: str | None = None) -> None:
    with _conn() as c:
        c.execute("UPDATE clippings SET status=?, error=? WHERE id=?", (status, error, clipping_id))


def get_clipping(clipping_id: str) -> Clipping:
    with _conn() as c:
        row = c.execute("SELECT * FROM clippings WHERE id=?", (clipping_id,)).fetchone()
    if row is None:
        raise NotFoundError(clipping_id)
    return _clipping(row)


def attach_clipping(clipping_id: str, event_id: str, topics: list[str]) -> None:
    """Join a clipping to an event, widen the event's topics, and drop the
    event's cached outputs so they're regenerated with the new source."""
    with _conn() as c:
        row = c.execute("SELECT topics FROM events WHERE id=?", (event_id,)).fetchone()
        if row is None:
            raise NotFoundError(event_id)
        merged = list(dict.fromkeys(json.loads(row["topics"]) + topics))
        resets = ", ".join(f"{f}=NULL" for f in EVENT_CACHE_FIELDS)
        c.execute(f"UPDATE events SET topics=?, {resets} WHERE id=?", (json.dumps(merged), event_id))
        c.execute(
            "UPDATE clippings SET event_id=?, status='ready', error=NULL WHERE id=?",
            (event_id, clipping_id),
        )


def event_clippings(event_id: str) -> list[Clipping]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM clippings WHERE event_id=? AND status='ready' ORDER BY date, paper, page",
            (event_id,),
        ).fetchall()
    return [_clipping(r) for r in rows]


def daily_counts(event_ids: list[str], date_from: str, date_to: str) -> dict[str, dict[str, int]]:
    """{event_id: {date: clipping count}} for clippings dated within the range."""
    if not event_ids:
        return {}
    marks = ",".join("?" * len(event_ids))
    with _conn() as c:
        rows = c.execute(
            f"SELECT event_id, date, COUNT(*) AS n FROM clippings "
            f"WHERE status='ready' AND event_id IN ({marks}) AND date BETWEEN ? AND ? "
            f"GROUP BY event_id, date",
            (*event_ids, date_from, date_to),
        ).fetchall()
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        out.setdefault(r["event_id"], {})[r["date"]] = r["n"]
    return out


# --- Editions ----------------------------------------------------------------


def create_edition(paper: str, date: str, uploaded_by: str | None = None) -> Edition:
    edition = Edition(
        id=uuid.uuid4().hex, paper=paper, date=date, pages=0, pages_done=0,
        status="processing", error=None, created_at=_now(), uploaded_by=uploaded_by,
    )
    with _conn() as c:
        c.execute(
            "INSERT INTO editions (id, paper, date, status, created_at, uploaded_by) VALUES (?,?,?,?,?,?)",
            (edition.id, paper, date, "processing", edition.created_at, uploaded_by),
        )
    return edition


def list_editions(limit: int = 50) -> list[Edition]:
    """Most recently uploaded first."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM editions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [Edition(**{k: r[k] for k in r.keys()}) for r in rows]


def edition_clippings(edition_id: str) -> list[dict]:
    """An edition's articles in page order, with the event each was filed under."""
    with _conn() as c:
        rows = c.execute(
            "SELECT c.id, c.page, c.title, c.status, c.error, c.event_id, e.title AS event_title "
            "FROM clippings c LEFT JOIN events e ON e.id = c.event_id "
            "WHERE c.edition_id=? ORDER BY c.page, c.created_at",
            (edition_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_edition(edition_id: str) -> Edition:
    with _conn() as c:
        row = c.execute("SELECT * FROM editions WHERE id=?", (edition_id,)).fetchone()
    if row is None:
        raise NotFoundError(edition_id)
    return Edition(**{k: row[k] for k in row.keys()})


def known_papers() -> list[str]:
    with _conn() as c:
        rows = c.execute("SELECT DISTINCT paper FROM clippings UNION SELECT DISTINCT paper FROM editions")
        return sorted(r[0] for r in rows)


def update_edition(edition_id: str, **fields) -> None:
    allowed = {"pages", "pages_done", "status", "error"}
    if not fields or set(fields) - allowed:
        raise ValueError(f"Can only update {sorted(allowed)}")
    assignments = ", ".join(f"{k}=?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE editions SET {assignments} WHERE id=?", (*fields.values(), edition_id))


def edition_progress(edition_id: str) -> dict:
    """Clipping counts by status, and how many distinct events they formed."""
    with _conn() as c:
        rows = c.execute(
            "SELECT status, COUNT(*) AS n FROM clippings WHERE edition_id=? GROUP BY status", (edition_id,)
        ).fetchall()
        events = c.execute(
            "SELECT COUNT(DISTINCT event_id) FROM clippings WHERE edition_id=? AND status='ready'",
            (edition_id,),
        ).fetchone()[0]
    return {"by_status": {r["status"]: r["n"] for r in rows}, "events": events}


# --- Events ------------------------------------------------------------------

_EVENT_SELECT = """
SELECT e.*, COUNT(c.id) AS clipping_count, COUNT(DISTINCT c.paper) AS paper_count
FROM events e LEFT JOIN clippings c ON c.event_id = e.id AND c.status = 'ready'
"""


def create_event(title: str, dek: str, date: str, topics: list[str]) -> Event:
    event = Event(id=uuid.uuid4().hex, title=title, dek=dek, date=date, topics=topics, created_at=_now())
    with _conn() as c:
        c.execute(
            "INSERT INTO events (id, title, dek, date, topics, created_at) VALUES (?,?,?,?,?,?)",
            (event.id, title, dek, date, json.dumps(topics), event.created_at),
        )
    return event


def get_event(event_id: str) -> Event:
    with _conn() as c:
        row = c.execute(_EVENT_SELECT + " WHERE e.id=? GROUP BY e.id", (event_id,)).fetchone()
    if row is None:
        raise NotFoundError(event_id)
    return _event(row)


def get_events(event_ids: list[str]) -> list[Event]:
    if not event_ids:
        return []
    marks = ",".join("?" * len(event_ids))
    with _conn() as c:
        rows = c.execute(
            _EVENT_SELECT + f" WHERE e.id IN ({marks}) GROUP BY e.id", tuple(event_ids)
        ).fetchall()
    return [_event(r) for r in rows]


def list_events(date_from: str, date_to: str, topics: list[str] | None = None) -> list[Event]:
    """Events dated within the range, newest first, optionally limited to
    events tagged with at least one of `topics`. Events whose clippings all
    failed or are still processing are left out."""
    with _conn() as c:
        rows = c.execute(
            _EVENT_SELECT + " WHERE e.date BETWEEN ? AND ? GROUP BY e.id "
            "HAVING clipping_count > 0 ORDER BY e.date DESC, clipping_count DESC",
            (date_from, date_to),
        ).fetchall()
    events = [_event(r) for r in rows]
    if topics is not None:
        wanted = set(topics)
        events = [e for e in events if wanted.intersection(e.topics)]
    return events


def set_event_cache(event_id: str, name: str, value) -> None:
    if name not in EVENT_CACHE_FIELDS:
        raise ValueError(f"Unknown event cache {name!r}")
    with _conn() as c:
        c.execute(f"UPDATE events SET {name}=? WHERE id=?", (json.dumps(value), event_id))


# --- Links between events ----------------------------------------------------


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def get_links(event_id: str, other_ids: list[str]) -> dict[str, tuple[bool, str]]:
    """Cached link verdicts between `event_id` and each of `other_ids`."""
    out: dict[str, tuple[bool, str]] = {}
    with _conn() as c:
        for other in other_ids:
            row = c.execute(
                "SELECT linked, reason FROM event_links WHERE a=? AND b=?", _pair(event_id, other)
            ).fetchone()
            if row is not None:
                out[other] = (bool(row["linked"]), row["reason"])
    return out


def set_link(a: str, b: str, linked: bool, reason: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO event_links VALUES (?,?,?,?)", (*_pair(a, b), int(linked), reason)
        )


# --- Index stats -------------------------------------------------------------


def index_stats(month_prefix: str) -> dict:
    """Papers indexed, clippings dated in the given `YYYY-MM` month, and
    when the newest clipping was ingested."""
    with _conn() as c:
        papers = c.execute("SELECT COUNT(DISTINCT paper) FROM clippings WHERE status='ready'").fetchone()[0]
        month = c.execute(
            "SELECT COUNT(*) FROM clippings WHERE status='ready' AND date LIKE ?", (month_prefix + "%",)
        ).fetchone()[0]
        last = c.execute("SELECT MAX(created_at) FROM clippings WHERE status='ready'").fetchone()[0]
    return {"papers": papers, "clippings_this_month": month, "refreshed_at": last}
