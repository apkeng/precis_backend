from datetime import date, timedelta
from urllib.parse import parse_qs, urlparse

import pytest

from app import mailer, payments, store
from app.config import settings
from tests.conftest import ingest

SEP = {"from": "2026-09-01", "to": "2026-09-30"}
QUIZ = {**SEP, "topics": ["geo"], "formats": ["MCQ"], "length": 5, "difficulty": "Mixed"}


@pytest.fixture
def outbox(monkeypatch):
    """Capture outgoing email instead of calling Resend."""
    sent = []

    def fake_send(to, subject, html, text, images=()):
        sent.append({"to": to, "subject": subject, "html": html, "text": text, "images": list(images)})
        return f"msg-{len(sent)}"

    monkeypatch.setattr(mailer, "send", fake_send)
    monkeypatch.setattr(settings, "upi_vpa", "truesight@okaxis")
    monkeypatch.setattr(settings, "upi_payee_name", "Apokryfon")
    monkeypatch.setattr(settings, "app_url", "https://reader.test")
    return sent


def test_account_needs_a_sign_in_and_is_created_on_first_call(client, signed_in):
    assert client.get("/account").status_code == 401
    assert client.get("/account", headers=signed_in("expired")).status_code == 401

    me = client.get("/account", headers=signed_in("asha@gmail.com"))
    assert me.status_code == 200, me.text
    body = me.json()
    assert body["email"] == "asha@gmail.com"
    assert body["status"] == "new"
    assert body["quizAccess"] is False
    assert body["internal"] is False
    assert body["priceInr"] == 500
    assert len(body["ref"]) == 6
    # Same account (and reference) on the next call.
    assert client.get("/account", headers=signed_in("asha@gmail.com")).json()["ref"] == body["ref"]

    staff = client.get("/account", headers=signed_in("yadu@apokryfon.com")).json()
    assert staff["internal"] is True and staff["quizAccess"] is True


def test_quizzes_need_a_sign_in_and_an_active_subscription(client, fake_claude, signed_in):
    clipping = ingest(client)
    assert client.post("/quizzes", json=QUIZ).status_code == 401
    assert client.get(f"/events/{clipping.event_id}/quiz").status_code == 401

    reader = signed_in("asha@gmail.com")
    assert client.post("/quizzes", json=QUIZ, headers=reader).status_code == 402
    assert client.get(f"/events/{clipping.event_id}/quiz", headers=reader).status_code == 402

    store.update_subscriber("uid-asha@gmail.com", access_until=date.today().isoformat())
    assert client.post("/quizzes", json=QUIZ, headers=reader).status_code == 200
    assert client.get(f"/events/{clipping.event_id}/quiz", headers=reader).status_code == 200

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    store.update_subscriber("uid-asha@gmail.com", access_until=yesterday)
    assert client.post("/quizzes", json=QUIZ, headers=reader).status_code == 402


def test_confirm_payment_emails_a_upi_qr_with_the_reference(client, signed_in, outbox):
    reader = signed_in("asha@gmail.com")
    resp = client.post("/account/confirm-payment", headers=reader)
    assert resp.status_code == 200, resp.text
    account = resp.json()
    assert account["status"] == "payment_pending"
    assert account["paymentConfirmedAt"]

    [mail] = outbox
    assert mail["to"] == "asha@gmail.com"
    assert "₹500" in mail["subject"]
    assert account["ref"] in mail["html"] and "truesight@okaxis" in mail["html"]
    assert 'src="cid:upi-qr"' in mail["html"]
    [qr] = mail["images"]
    assert qr.content_id == "upi-qr" and qr.content.startswith(b"\x89PNG")

    uri = next(line for line in mail["text"].splitlines() if line.startswith("UPI link: ")).removeprefix("UPI link: ")
    params = parse_qs(urlparse(uri).query)
    assert uri.startswith("upi://pay?")
    assert params["pa"] == ["truesight@okaxis"]
    assert params["am"] == ["500.00"]
    assert params["cu"] == ["INR"]
    assert account["ref"] in params["tn"][0]

    # Pressing again straight away doesn't send a second email.
    assert client.post("/account/confirm-payment", headers=reader).status_code == 200
    assert len(outbox) == 1


def test_confirm_payment_refused_without_upi_or_email(client, signed_in, outbox, monkeypatch):
    reader = signed_in("asha@gmail.com")
    monkeypatch.setattr(settings, "upi_vpa", "")
    resp = client.post("/account/confirm-payment", headers=reader)
    assert resp.status_code == 503
    assert "UPI_VPA" in resp.json()["detail"]
    assert client.get("/account", headers=reader).json()["status"] == "new"  # nothing recorded

    monkeypatch.setattr(settings, "upi_vpa", "truesight@okaxis")

    def down(*a, **k):
        raise mailer.MailError("The email service refused the message (403): domain not verified")

    monkeypatch.setattr(mailer, "send", down)
    resp = client.post("/account/confirm-payment", headers=reader)
    assert resp.status_code == 502
    assert "domain not verified" in resp.json()["detail"]
    assert client.get("/account", headers=reader).json()["status"] == "new"


def test_admin_sees_counts_and_activates_with_a_login_email(client, signed_in, outbox):
    admin = signed_in("yadu@apokryfon.com")
    client.get("/account", headers=signed_in("browser@gmail.com"))  # account, never paid
    client.post("/account/confirm-payment", headers=signed_in("asha@gmail.com"))

    assert client.get("/admin/subscribers", headers=signed_in("asha@gmail.com")).status_code == 403
    listing = client.get("/admin/subscribers", headers=admin).json()
    assert listing["counts"] == {"accounts": 2, "paymentConfirmed": 1, "awaitingActivation": 1, "active": 0}
    assert [s["email"] for s in listing["subscribers"]] == ["asha@gmail.com", "browser@gmail.com"]
    uid = listing["subscribers"][0]["uid"]

    # Resending without granting days is refused for a reader with no access.
    assert client.post(f"/admin/subscribers/{uid}/login-email", json={"days": 0}, headers=admin).status_code == 400

    resp = client.post(f"/admin/subscribers/{uid}/login-email", json={}, headers=admin)
    assert resp.status_code == 200, resp.text
    sub = resp.json()
    end = date.today() + timedelta(days=29)
    assert sub["status"] == "active"
    assert sub["accessUntil"] == end.isoformat()
    assert sub["activatedBy"] == "yadu@apokryfon.com"
    assert sub["loginEmailSentAt"]
    login = outbox[-1]
    assert login["to"] == "asha@gmail.com"
    assert "https://reader.test/?signin=1" in login["html"]
    assert f"{end.day} {end:%B %Y}" in login["text"]

    counts = client.get("/admin/subscribers", headers=admin).json()["counts"]
    assert counts == {"accounts": 2, "paymentConfirmed": 1, "awaitingActivation": 0, "active": 1}
    assert client.get("/account", headers=signed_in("asha@gmail.com")).json()["quizAccess"] is True

    # A plain resend keeps the end date; a renewal extends from it.
    assert client.post(f"/admin/subscribers/{uid}/login-email", json={"days": 0}, headers=admin).json()["accessUntil"] == end.isoformat()
    renewed = client.post(f"/admin/subscribers/{uid}/login-email", json={"days": 30}, headers=admin).json()
    assert renewed["accessUntil"] == (end + timedelta(days=30)).isoformat()

    assert client.post("/admin/subscribers/nope/login-email", json={}, headers=admin).status_code == 404


def test_renewal_payment_shows_as_awaiting_while_access_lasts(signed_in):
    s = store.upsert_subscriber("u1", "a@b.c", None)
    today = date.today().isoformat()
    s = store.update_subscriber("u1", activated_at="2026-01-01T00:00:00+00:00", access_until="2999-01-01")
    assert s.status(today) == "active"
    s = store.update_subscriber("u1", payment_confirmed_at="2026-02-01T00:00:00+00:00")
    assert s.status(today) == "payment_pending"
    assert payments.today() == today
