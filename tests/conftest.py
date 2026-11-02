import re

import pytest
from chromadb import EmbeddingFunction
from fastapi.testclient import TestClient

import app.main as main_module
from app import analysis, ingestion, quiz, store, vector_store
from app.config import settings


INGEST_KEY = "test-ingest-key"
KEY_HEADERS = {"X-API-Key": INGEST_KEY}


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
        if "links" in props:
            ids = props["links"]["items"]["properties"]["event_id"]["enum"]
            lines = {i: next(l for l in prompt.splitlines() if l.startswith(f"- {i}")) for i in ids}
            return {"links": [
                {"event_id": i, "linked": not any(w in lines[i] for w in self.unlinked_words), "reason": "Linked"}
                for i in ids
            ]}
        if "questions" in props:
            # A pool for one event: every format at both difficulties, plus
            # one malformed question that must be dropped.
            item = props["questions"]["items"]["properties"]
            title = re.search(r"^Event \([^)]*\): (.*)$", prompt, re.M).group(1)
            qs = [
                {
                    "source_ref": item["source_ref"]["enum"][0], "topic": item["topic"]["enum"][0],
                    "type": fmt, "difficulty": level, "stem": f"{level} {fmt} on {title}?",
                    "statements": [], "options": ["a", "b", "c", "d"], "answer_index": 0,
                    "explanation": "Because.",
                }
                for fmt in item["type"]["enum"]
                for level in item["difficulty"]["enum"]
            ]
            qs.insert(0, {**qs[0], "stem": "Bad?", "options": ["a", "b"]})
            return {"questions": qs}
        raise AssertionError(f"Unexpected schema: {sorted(props)}")


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "precis.db"))
    monkeypatch.setattr(settings, "ingest_api_key", INGEST_KEY)
    monkeypatch.setattr(settings, "firebase_project_id", "precis-test")
    monkeypatch.setattr(settings, "admin_email_domain", "apokryfon.com")
    # Off by default so tests see exactly the Claude calls they trigger;
    # tests of pre-generation turn it on.
    monkeypatch.setattr(settings, "pregenerate", False)
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
    resp = client.post("/clippings", json=body, headers=KEY_HEADERS)
    assert resp.status_code == 202, resp.text
    # TestClient runs background tasks before returning.
    return store.get_clipping(resp.json()["id"])


@pytest.fixture
def signed_in(monkeypatch):
    """Accept any bearer token whose text is an email, as that verified user.
    Returns the Authorization headers for an email."""
    from app import auth

    def fake_verify(token):
        if token == "expired":
            raise ValueError("Invalid or expired sign-in token: expired")
        email, _, flag = token.partition("|")
        return {"sub": "uid-" + email, "email": email, "email_verified": flag != "unverified"}

    monkeypatch.setattr(auth, "verify_firebase_token", fake_verify)
    return lambda token: {"Authorization": f"Bearer {token}"}
