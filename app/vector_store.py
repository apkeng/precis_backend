"""Thin wrapper around a local, on-disk Chroma vector database.

Two collections, both embedded with the same local sentence-transformers
model as Newspaper_ingestion:

  - `clippings`: chunks of every clipping, used to retrieve passages for
    "Ask the sources". Metadata carries the clipping, its event, paper and
    date, so retrieval can be confined to a date range or one event.
  - `events`: one vector per event (title + dek), used to find the event a
    new clipping belongs to and the events related to a story.

Dates are stored as `YYYYMMDD` integers because Chroma's range operators
($gte/$lte) only work on numbers.
"""
from __future__ import annotations

import chromadb

from app.chunking import Chunk
from app.config import settings
from app.embeddings import LocalEmbeddingFunction

CLIPPINGS = "clippings"
EVENTS = "events"

_client = None
_embedding_fn = LocalEmbeddingFunction()


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def _collection(name: str):
    return get_client().get_or_create_collection(
        name=name, embedding_function=_embedding_fn, metadata={"hnsw:space": "cosine"}
    )


def date_int(iso_date: str) -> int:
    return int(iso_date.replace("-", ""))


def _date_where(date_from: str | None, date_to: str | None) -> list[dict]:
    clauses = []
    if date_from:
        clauses.append({"date": {"$gte": date_int(date_from)}})
    if date_to:
        clauses.append({"date": {"$lte": date_int(date_to)}})
    return clauses


def _where(clauses: list[dict]) -> dict | None:
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _query(collection, text: str, n: int, where: dict | None) -> dict:
    n = min(n, collection.count())
    if n <= 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    return collection.query(query_texts=[text], n_results=n, where=where)


# --- Clippings ---------------------------------------------------------------


def add_clipping_chunks(
    clipping_id: str, paper: str, date: str, page: int | None, chunks: list[Chunk]
) -> None:
    if not chunks:
        return
    _collection(CLIPPINGS).add(
        ids=[f"{clipping_id}_{c.chunk_index}" for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "clipping_id": clipping_id,
                "event_id": "",
                "paper": paper,
                "date": date_int(date),
                "page": page or 0,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ],
    )


def set_clipping_event(clipping_id: str, event_id: str) -> None:
    collection = _collection(CLIPPINGS)
    found = collection.get(where={"clipping_id": clipping_id})
    if not found["ids"]:
        return
    collection.update(
        ids=found["ids"],
        metadatas=[{**m, "event_id": event_id} for m in found["metadatas"]],
    )


def query_clippings(
    text: str,
    top_k: int,
    date_from: str | None = None,
    date_to: str | None = None,
    event_id: str | None = None,
) -> list[dict]:
    clauses = _date_where(date_from, date_to)
    if event_id:
        clauses.append({"event_id": event_id})
    result = _query(_collection(CLIPPINGS), text, top_k, _where(clauses))
    return [
        {"text": doc, "clipping_id": meta["clipping_id"], "event_id": meta.get("event_id") or None}
        for doc, meta in zip(result["documents"][0], result["metadatas"][0])
    ]


# --- Events ------------------------------------------------------------------


def upsert_event(event_id: str, title: str, dek: str, date: str) -> None:
    _collection(EVENTS).upsert(
        ids=[event_id], documents=[f"{title}\n{dek}"], metadatas=[{"date": date_int(date)}]
    )


def nearest_events(
    text: str,
    n: int,
    date_from: str | None = None,
    date_to: str | None = None,
    exclude: str | None = None,
) -> list[str]:
    """Ids of the events most similar to `text`, closest first."""
    result = _query(
        _collection(EVENTS), text, n + (1 if exclude else 0), _where(_date_where(date_from, date_to))
    )
    return [i for i in result["ids"][0] if i != exclude][:n]
