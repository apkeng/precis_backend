from app import auth, editions, store
from app.config import settings
from tests.conftest import KEY_HEADERS

PDF = {"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")}
FORM = {"paper": "The Hindu", "date": "2026-10-01"}


def test_internal_email_rule(monkeypatch):
    assert auth.is_internal_email("yadu@apokryfon.com", True)
    assert auth.is_internal_email("Yadu@Apokryfon.COM", True)
    assert not auth.is_internal_email("yadu@apokryfon.com", False)  # unverified
    assert not auth.is_internal_email("someone@gmail.com", True)
    assert not auth.is_internal_email("x@evil-apokryfon.com", True)  # suffix, not domain
    assert not auth.is_internal_email("x@apokryfon.com.evil.io", True)
    assert not auth.is_internal_email(None, True)
    monkeypatch.setattr(settings, "admin_email_domain", "")
    assert not auth.is_internal_email("yadu@apokryfon.com", True)


def test_admin_endpoints_need_an_internal_sign_in(client, signed_in):
    assert client.get("/admin/me").status_code == 401
    assert client.get("/admin/editions", headers=KEY_HEADERS).status_code == 401  # key isn't enough
    assert client.get("/admin/me", headers=signed_in("expired")).status_code == 401
    assert client.get("/admin/me", headers={"Authorization": "Token abc"}).status_code == 401

    outsider = client.get("/admin/me", headers=signed_in("someone@gmail.com"))
    assert outsider.status_code == 403
    assert "someone@gmail.com" in outsider.json()["detail"]
    assert client.get("/admin/me", headers=signed_in("yadu@apokryfon.com|unverified")).status_code == 403

    me = client.get("/admin/me", headers=signed_in("yadu@apokryfon.com"))
    assert me.status_code == 200
    assert me.json() == {"email": "yadu@apokryfon.com"}


def test_sign_in_unconfigured_is_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "firebase_project_id", "")
    resp = client.get("/admin/me", headers={"Authorization": "Bearer something"})
    assert resp.status_code == 401
    assert "FIREBASE_PROJECT_ID" in resp.json()["detail"]


def test_admin_uploads_edition_and_sees_progress_and_articles(client, fake_claude, signed_in, monkeypatch):
    fake_claude.force_new = True
    monkeypatch.setattr(editions, "extract_text_by_page", lambda b: ["Monsoon ends below normal.", "Court ruling on the Governor's Bills."])
    admin = signed_in("yadu@apokryfon.com")

    assert client.post("/editions/pdf", files=PDF, data=FORM, headers=signed_in("a@gmail.com")).status_code == 403
    resp = client.post("/editions/pdf", files=PDF, data=FORM, headers=admin)
    assert resp.status_code == 202, resp.text
    edition_id = resp.json()["id"]
    assert resp.json()["uploadedBy"] == "yadu@apokryfon.com"

    listing = client.get("/admin/editions", headers=admin).json()
    assert [e["id"] for e in listing["editions"]] == [edition_id]
    assert listing["editions"][0]["status"] == "ready"
    assert listing["papers"] == ["The Hindu"]

    articles = client.get(f"/admin/editions/{edition_id}/articles", headers=admin).json()["articles"]
    assert [(a["page"], a["title"], a["status"]) for a in articles] == [
        (1, "Page 1 lead story", "ready"),
        (2, "Page 2 lead story", "ready"),
    ]
    assert articles[0]["eventTitle"] == "Page 1 lead story"
    assert client.get("/admin/editions/nope/articles", headers=admin).status_code == 404


def test_key_uploads_are_attributed_to_the_key(client, fake_claude, monkeypatch):
    monkeypatch.setattr(editions, "extract_text_by_page", lambda b: ["Text."])
    resp = client.post("/editions/pdf", files=PDF, data=FORM, headers=KEY_HEADERS)
    assert store.get_edition(resp.json()["id"]).uploaded_by == "api-key"


def test_old_database_gains_uploaded_by_column(tmp_path, monkeypatch):
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE editions (id TEXT PRIMARY KEY, paper TEXT NOT NULL, date TEXT NOT NULL, "
        "pages INTEGER NOT NULL DEFAULT 0, pages_done INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL, "
        "error TEXT, created_at TEXT NOT NULL)"
    )
    old.execute("INSERT INTO editions VALUES ('e1','P','2026-10-01',1,1,'ready',NULL,'2026-10-01T00:00:00')")
    old.commit()
    old.close()

    monkeypatch.setattr(settings, "sqlite_path", str(path))
    assert store.get_edition("e1").uploaded_by is None
    assert store.create_edition("P", "2026-10-02", "a@apokryfon.com").uploaded_by == "a@apokryfon.com"
