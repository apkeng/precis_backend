import re

import pytest
from chromadb import EmbeddingFunction
from fastapi.testclient import TestClient

import app.main as main_module
from app import analysis, ingestion, quiz, store, vector_store
from app.config import settings


class _FakeEmbeddingFunction(EmbeddingFunction):
    """Deterministic, dependency-free stand-in for the real sentence-transformers
    embedder, so tests don't need to download a model. Texts sharing words get
    similar vectors, which is enough for nearest-event lookups."""

    _VOCAB = ["cyclone", "odisha", "governor", "bill", "court", "trade", "rbi", "census"]

    def __init__(self):
        pass

    def __call__(self, input):
        out = []
        for t in input:
            words = t.lower()
            out.append([float(words.count(w)) + 0.01 for w in self._VOCAB])
        return out

    def name(self):
        return "fake"


class FakeClaude:
    """Answers every schema the app sends, recording each call. Tests tweak
    `tag_topics`, `relevant`, `unlinked_words` and `force_new` to steer it."""

    def __init__(self):
        self.calls = []
        self.tag_topics = ["geo"]
        self.relevant = True
        self.unlinked_words: set[str] = set()
        self.force_new = False

    def __call__(self, system, prompt, schema, effort="medium"):
        props = schema["properties"]
        self.calls.append((set(props), prompt))
        if "relevant" in props:
            match = "new" if self.force_new else props["match"]["enum"][0]  # nearest, or "new"
            headline = prompt.split("\n\n")[1].splitlines()[1]
            return {
                "relevant": self.relevant, "topics": self.tag_topics, "match": match,
                "headline": headline, "dek": "Summary of " + headline,
            }
        if "articles" in props:
            page = int(re.search(r"page (\d+):", prompt).group(1))
            body = prompt.split(":\n\n", 1)[1]
            return {"articles": [
                {"headline": f"Page {page} lead story", "kind": "news", "text": body * 20},
                {"headline": "Advert", "kind": "other", "text": ""},
                {"headline": "Brief", "kind": "news", "text": "Too short."},
            ]}
        if "what_happened" in props:
            return {
                "what_happened": "It happened [1]. Also [9].",
                "why_it_matters": "It matters [2].",
                "prelims_facts": ["Fact one."],
                "gs_paper": "GS I · Geography",
                "mains_question": "Discuss.",
                "timeline": [{"date": "28 Sep", "text": "Landfall"}],
            }
        if "papers" in props:
            papers = props["papers"]["items"]["properties"]["paper"]["enum"]
            return {"papers": [{"paper": p, "angle": "An angle", "tone": "Neutral"} for p in papers]}
        if "answer" in props:
            return {"answer": "The sources say so [1].", "found_in_sources": True}
        if "links" in props:
            ids = props["links"]["items"]["properties"]["event_id"]["enum"]
            lines = {i: next(l for l in prompt.splitlines() if l.startswith(f"- {i}")) for i in ids}
            return {"links": [
                {"event_id": i, "linked": not any(w in lines[i] for w in self.unlinked_words), "reason": "Linked"}
                for i in ids
            ]}
        if "questions" in props:
            item = props["questions"]["items"]["properties"]
            erefs = item["event_ref"]["enum"]
            # S-refs are numbered in event order, so E{n}'s first source is the
            # first S after the previous events' sources.
            first_source = {e: s for e, s in re.findall(r"Event (E\d+).*?\[(S\d+)\]", prompt, re.S)}
            topic = item["topic"]["enum"][0]
            qs = []
            for e in erefs:
                qs.append({
                    "event_ref": e, "source_ref": first_source[e], "topic": topic,
                    "type": item["type"]["enum"][0], "stem": f"Question on {e}?", "statements": [],
                    "options": ["a", "b", "c", "d"], "answer_index": 0, "explanation": "Because.",
                })
            # One malformed question, which must be dropped.
            qs.append({**qs[0], "stem": "Bad?", "options": ["a", "b"]})
            return {"questions": qs}
        raise AssertionError(f"Unexpected schema: {sorted(props)}")


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "precis.db"))
    monkeypatch.setattr(settings, "ingest_api_key", "")
    monkeypatch.setattr(vector_store, "_embedding_fn", _FakeEmbeddingFunction())
    vector_store._client = None
    yield
    vector_store._client = None


@pytest.fixture
def fake_claude(monkeypatch):
    fake = FakeClaude()
    for module in (ingestion, analysis, quiz):
        monkeypatch.setattr(module.llm, "generate_json", fake)
    return fake


@pytest.fixture
def client():
    return TestClient(main_module.app)


def ingest(client, **overrides):
    body = {
        "paper": "The Courier", "date": "2026-09-28", "page": 1,
        "title": "Cyclone makes landfall in Odisha",
        "text": "The cyclone crossed the Odisha coast near Paradip. Three lakh people were evacuated.",
        **overrides,
    }
    resp = client.post("/clippings", json=body)
    assert resp.status_code == 202, resp.text
    # TestClient runs background tasks before returning.
    return store.get_clipping(resp.json()["id"])
