from app import analysis, store
from app import quiz as quiz_module
from app.llm import LLMGenerationError
from app.config import settings
from tests.conftest import KEY_HEADERS, ingest

SEP = {"from": "2026-09-01", "to": "2026-09-30"}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_topics_lists_all_fourteen(client):
    body = client.get("/topics").json()
    assert len(body["topics"]) == 14
    assert body["topics"][0] == {
        "id": "natintl", "name": "National & International Events", "short": "Nat. & Intl.", "group": "Prelims",
    }
    assert body["formats"] == ["MCQ", "Statement", "Assertion–Reason", "Match"]


def test_clippings_on_the_same_story_join_one_event(client, fake_claude):
    first = ingest(client)
    second = ingest(client, paper="Daily Ledger", date="2026-09-29", title="Cyclone toll stays low in Odisha")

    assert first.status == second.status == "ready"
    assert first.event_id == second.event_id

    events = client.get("/events", params=SEP).json()
    assert len(events["events"]) == 1
    event = events["events"][0]
    assert event["title"] == "Cyclone makes landfall in Odisha"
    assert event["clippingCount"] == 2
    assert event["paperCount"] == 2
    assert event["topics"] == ["geo"]
    assert events["clippingCount"] == 2
    assert events["topicCounts"]["geo"] == 1


def test_off_syllabus_clipping_creates_no_event(client, fake_claude):
    fake_claude.relevant = False
    clipping = ingest(client, title="Film festival opens")
    assert clipping.status == "off_syllabus"
    assert client.get("/events", params=SEP).json()["events"] == []


def test_failed_tagging_marks_clipping_failed(client, monkeypatch):
    from app import ingestion, llm

    def boom(*args, **kwargs):
        raise llm.LLMGenerationError("Anthropic down")

    monkeypatch.setattr(ingestion.llm, "generate_json", boom)
    clipping = ingest(client)
    assert clipping.status == "failed"
    assert "Anthropic down" in clipping.error


def test_event_list_filters_by_topic_and_date(client, fake_claude):
    fake_claude.force_new = True
    ingest(client)
    fake_claude.tag_topics = ["const", "polgov"]
    ingest(client, date="2026-09-26", title="Court sets limits on Governor's time for Bills")

    def titles(**params):
        return [e["title"] for e in client.get("/events", params={**SEP, **params}).json()["events"]]

    assert titles() == ["Cyclone makes landfall in Odisha", "Court sets limits on Governor's time for Bills"]
    assert titles(topics="const") == ["Court sets limits on Governor's time for Bills"]
    assert titles(topics="") == []
    assert client.get("/events", params={"from": "2026-09-27", "to": "2026-09-30"}).json()["events"][0][
        "title"
    ] == "Cyclone makes landfall in Odisha"
    assert client.get("/events", params={"from": "bad", "to": "2026-09-30"}).status_code == 422


def test_event_detail_has_numbered_sources_and_analysis(client, fake_claude):
    clipping = ingest(client)
    resp = client.get(f"/events/{clipping.event_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"][0]["n"] == 1
    assert body["sources"][0]["excerpt"].startswith("The cyclone crossed")
    # Citation to a non-existent source [9] is stripped; [1] is kept.
    assert body["analysis"]["whatHappened"] == "It happened [1]. Also."
    assert body["analysis"]["timeline"] == [{"date": "28 Sep", "text": "Landfall"}]

    # Second read is served from cache.
    calls = len(fake_claude.calls)
    client.get(f"/events/{clipping.event_id}")
    assert len(fake_claude.calls) == calls


def test_new_clipping_invalidates_cached_analysis(client, fake_claude):
    clipping = ingest(client)
    client.get(f"/events/{clipping.event_id}")
    assert store.get_event(clipping.event_id).caches.get("analysis")
    ingest(client, paper="Metro Times")
    assert not store.get_event(clipping.event_id).caches.get("analysis")


def test_unknown_event_is_404(client):
    assert client.get("/events/nope").status_code == 404


def test_trace_lists_direct_links_theme_events_and_outside_period(client, fake_claude):
    fake_claude.force_new = True
    fake_claude.unlinked_words = {"Earthquake"}
    story = ingest(client)
    # Directly linked, in the period.
    ingest(client, date="2026-09-17", title="Climate talks cite rapid cyclone intensification")
    # Shares a topic only.
    ingest(client, date="2026-09-06", title="Earthquake strikes Assam")
    # Directly linked but before the period.
    ingest(client, date="2026-08-20", title="Cyclone season outlook for Odisha")

    resp = client.get(f"/events/{story.event_id}/trace", params=SEP)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    kinds = {r["event"]["title"]: r["kind"] for r in body["related"]}
    assert kinds["Climate talks cite rapid cyclone intensification"] == "direct"
    assert kinds["Earthquake strikes Assam"] == "theme"
    assert "Cyclone season outlook for Odisha" not in kinds
    assert body["outsideCount"] == 1
    assert body["outsideRange"] == {"from": "2026-08-20", "to": "2026-09-30"}

    assert len(body["days"]) == 30
    by_date = {d["date"]: d for d in body["days"]}
    assert by_date["2026-09-28"]["story"] == 1
    assert by_date["2026-09-17"]["related"] == 1
    assert body["framing"] == [
        {"paper": "The Courier", "angle": "An angle", "tone": "Neutral", "clippingCount": 1}
    ]


def test_quiz_is_assembled_from_event_pools(client, fake_claude, signed_in):
    staff = signed_in("yadu@apokryfon.com")
    fake_claude.force_new = True
    first = ingest(client)
    second = ingest(client, date="2026-09-20", title="Second cyclone forms")
    for c in (first, second):  # writes each event's question pool
        assert client.get(f"/events/{c.event_id}/quiz", headers=staff).status_code == 200

    calls = len(fake_claude.calls)
    body = {**SEP, "topics": ["geo"], "formats": ["MCQ"], "length": 10, "difficulty": "Mixed"}
    resp = client.post("/quizzes", json=body, headers=staff)
    assert resp.status_code == 200, resp.text
    assert len(fake_claude.calls) == calls  # assembled from the pools, no Claude call
    quiz = resp.json()
    assert quiz["requested"] == 10
    # Two events x MCQ at both difficulties; the malformed question was dropped.
    assert len(quiz["questions"]) == 4
    assert all(q["type"] == "MCQ" and q["stem"] != "Bad?" for q in quiz["questions"])
    # Spread across events: the first two come from different events.
    assert quiz["questions"][0]["eventId"] != quiz["questions"][1]["eventId"]
    q = quiz["questions"][0]
    assert q["topic"] == "geo"
    assert q["answer"] == 0
    assert q["source"]["paper"] == "The Courier"

    easy = client.post("/quizzes", json={**body, "difficulty": "Easy"}, headers=staff).json()
    assert [q["stem"].split()[0] for q in easy["questions"]] == ["Easy", "Easy"]
    short = client.post("/quizzes", json={**body, "length": 1}, headers=staff).json()
    assert len(short["questions"]) == 1
    # Only the pooled questions matching format and difficulty: one per event.
    none = client.post("/quizzes", json={**body, "formats": ["Match"], "difficulty": "Hard"}, headers=staff)
    assert [q["type"] for q in none.json()["questions"]] == ["Match", "Match"]


def test_quiz_rejects_bad_input_and_empty_selection(client, fake_claude, signed_in):
    staff = signed_in("yadu@apokryfon.com")
    base = {**SEP, "topics": ["geo"], "formats": ["MCQ"], "length": 5, "difficulty": "Mixed"}
    assert client.post("/quizzes", json={**base, "topics": ["nope"]}, headers=staff).status_code == 422
    assert client.post("/quizzes", json={**base, "formats": ["Essay"]}, headers=staff).status_code == 422
    assert client.post("/quizzes", json={**base, "topics": []}, headers=staff).status_code == 422
    assert client.post("/quizzes", json=base, headers=staff).status_code == 409  # nothing ingested


def test_quiz_drafts_pools_for_events_without_one(client, fake_claude, signed_in, monkeypatch):
    staff = signed_in("yadu@apokryfon.com")
    fake_claude.force_new = True
    events = [
        ingest(client, date=f"2026-09-{day}", title=f"Cyclone number {day}").event_id
        for day in ("10", "11", "12", "13")
    ]
    monkeypatch.setattr(settings, "quiz_fallback_events", 3)

    def pool_calls():
        return sum("questions" in props for props, _ in fake_claude.calls)

    # MCQ only, Mixed: 2 usable questions per pool, so 3 more are 2 pools.
    body = {**SEP, "topics": ["geo"], "formats": ["MCQ"], "length": 3, "difficulty": "Mixed"}
    resp = client.post("/quizzes", json=body, headers=staff)
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["questions"]) == 3
    assert pool_calls() == 2
    pooled = [e for e in events if store.get_event(e).caches.get("quiz") is not None]
    assert len(pooled) == 2  # kept for later quizzes

    # Same quiz again: served from the saved pools.
    client.post("/quizzes", json=body, headers=staff)
    assert pool_calls() == 2
    # A longer one drafts at most QUIZ_FALLBACK_EVENTS more.
    resp = client.post("/quizzes", json={**body, "length": 30}, headers=staff)
    assert len(resp.json()["questions"]) == 8  # 4 events x 2 MCQs
    assert pool_calls() == 4


def test_quiz_fallback_failure_is_an_error_only_with_nothing_to_show(client, fake_claude, signed_in, monkeypatch):
    staff = signed_in("yadu@apokryfon.com")
    ingest(client)

    def overloaded(system, prompt, schema, effort="medium"):
        raise LLMGenerationError("overloaded")

    monkeypatch.setattr(quiz_module.llm, "generate_json", overloaded)
    body = {**SEP, "topics": ["geo"], "formats": ["MCQ"], "length": 5, "difficulty": "Mixed"}
    resp = client.post("/quizzes", json=body, headers=staff)
    assert resp.status_code == 502, resp.text


def test_event_quiz_is_cached(client, fake_claude, signed_in):
    staff = signed_in("yadu@apokryfon.com")
    clipping = ingest(client)
    first = client.get(f"/events/{clipping.event_id}/quiz", headers=staff).json()
    assert first["requested"] == 3
    # One question per format where the pool allows.
    assert len({q["type"] for q in first["questions"]}) == 3
    calls = len(fake_claude.calls)
    second = client.get(f"/events/{clipping.event_id}/quiz", headers=staff).json()
    assert first == second
    assert len(fake_claude.calls) == calls


def test_ingestion_pregenerates_event_outputs(client, fake_claude, monkeypatch):
    monkeypatch.setattr(settings, "pregenerate", True)
    clipping = ingest(client)
    caches = store.get_event(clipping.event_id).caches
    assert set(caches) == {"analysis", "framing", "quiz"}
    assert caches["quiz"]

    # Readers are served from the cache.
    calls = len(fake_claude.calls)
    assert client.get(f"/events/{clipping.event_id}").status_code == 200
    assert len(fake_claude.calls) == calls

    # A new clipping on the event clears and re-writes them.
    ingest(client, paper="Metro Times")
    assert set(store.get_event(clipping.event_id).caches) == {"analysis", "framing", "quiz"}


def test_failed_pregeneration_leaves_event_for_on_view(client, fake_claude, monkeypatch):
    monkeypatch.setattr(settings, "pregenerate", True)
    real = fake_claude.__call__

    def no_analysis(system, prompt, schema, effort="medium"):
        if "what_happened" in schema["properties"]:
            raise RuntimeError("overloaded")
        return real(system, prompt, schema, effort)

    monkeypatch.setattr(analysis.llm, "generate_json", no_analysis)
    clipping = ingest(client)
    assert store.get_clipping(clipping.id).status == "ready"
    assert "analysis" not in store.get_event(clipping.event_id).caches


def test_ingest_needs_the_key(client, fake_claude, monkeypatch):
    body = {"paper": "P", "date": "2026-09-28", "title": "T", "text": "Cyclone text."}
    assert client.post("/clippings", json=body).status_code == 401
    assert client.post("/clippings", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/clippings", json=body, headers=KEY_HEADERS).status_code == 202
    # No key configured: scripts can't ingest at all.
    monkeypatch.setattr(settings, "ingest_api_key", "")
    assert client.post("/clippings", json=body, headers={"X-API-Key": ""}).status_code == 401


def test_pdf_ingest_reads_text(client, fake_claude, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "extract_text_by_page", lambda b: ["Cyclone hits Odisha coast."])
    resp = client.post(
        "/clippings/pdf",
        files={"file": ("c.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"paper": "The Courier", "date": "2026-09-28", "title": "Cyclone"},
        headers=KEY_HEADERS,
    )
    assert resp.status_code == 202, resp.text
    clipping = store.get_clipping(resp.json()["id"])
    assert clipping.text == "Cyclone hits Odisha coast."
    assert clipping.status == "ready"


def test_stats(client, fake_claude):
    ingest(client)
    body = client.get("/stats").json()
    assert body["papers"] == 1
    assert body["refreshedAt"]
