from app import editions, store
from app.text_cleanup import clean_extracted_text


def test_clean_extracted_text_repairs_ligatures_and_hyphenation():
    raw = "/f_ive held; /f_low of /f_irst ﬁles; oﬀ the De-\npartment; well-\nKnown"
    assert clean_extracted_text(raw) == "five held; flow of first files; off the Department; well-\nKnown"


def test_edition_is_split_into_articles_and_ingested(client, fake_claude, monkeypatch):
    fake_claude.force_new = True
    monkeypatch.setattr(
        editions,
        "extract_text_by_page",
        lambda b: ["Monsoon ends 12.6% below normal, says IMD. Cyclone watch.", "", "Court on Governor's Bills."],
    )
    resp = client.post(
        "/editions/pdf",
        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"paper": "The Hindu", "date": "2026-10-01"},
    )
    assert resp.status_code == 202, resp.text
    edition_id = resp.json()["id"]

    body = client.get(f"/editions/{edition_id}").json()
    assert body["status"] == "ready"
    assert body["pages"] == 3
    assert body["pagesDone"] == 3
    # One news article per non-empty page; the advert and the too-short brief are skipped.
    assert body["clippings"] == {"ready": 2}
    assert body["events"] == 2

    events = client.get("/events", params={"from": "2026-10-01", "to": "2026-10-01"}).json()["events"]
    assert sorted(e["title"] for e in events) == ["Page 1 lead story", "Page 3 lead story"]


def test_edition_page_failure_is_recorded_and_others_continue(client, fake_claude, monkeypatch):
    monkeypatch.setattr(editions, "extract_text_by_page", lambda b: ["Page one text.", "Page two text."])
    real = fake_claude.__call__

    def flaky(system, prompt, schema, effort="medium"):
        if "articles" in schema["properties"] and "page 2:" in prompt:
            raise RuntimeError("segmenting failed")
        return real(system, prompt, schema, effort)

    monkeypatch.setattr(editions.llm, "generate_json", flaky)
    resp = client.post(
        "/editions/pdf",
        files={"file": ("paper.pdf", b"%PDF", "application/pdf")},
        data={"paper": "The Hindu", "date": "2026-10-01"},
    )
    edition = store.get_edition(resp.json()["id"])
    assert edition.status == "ready"
    assert edition.error == "page 2: segmenting failed"


def test_unknown_edition_is_404(client):
    assert client.get("/editions/nope").status_code == 404
