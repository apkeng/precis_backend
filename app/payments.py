"""Paid quiz access, without a payment gateway.

1. A signed-in reader presses "Confirm payment". We email them a UPI QR code
   for the monthly price, with their account reference in the payment note.
2. They pay from any UPI app. Nothing here sees the payment.
3. An admin matches the payment (by reference) and, from the admin panel,
   grants access and sends a login email.
"""
from __future__ import annotations

import html
import io
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote, urlencode

import segno

from app import mailer, store
from app.config import settings

# A second press of "Confirm payment" this soon after the last email doesn't
# send another one.
RESEND_AFTER = timedelta(minutes=5)


class PaymentsNotConfigured(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def today() -> str:
    return date.today().isoformat()


def upi_uri(ref: str) -> str:
    if not settings.upi_vpa:
        raise PaymentsNotConfigured("Payments aren't set up yet (UPI_VPA). Please try again later.")
    params = {
        "pa": settings.upi_vpa,
        "pn": settings.upi_payee_name,
        "am": f"{settings.subscription_price_inr:.2f}",
        "cu": "INR",
        "tn": f"Truesight quiz {ref}",
    }
    # UPI apps expect the "@" in a UPI ID as-is.
    return "upi://pay?" + urlencode(params, quote_via=lambda v, *_: quote(v, safe="@"))


def qr_png(data: str) -> bytes:
    out = io.BytesIO()
    segno.make(data, error="m").save(out, kind="png", scale=8, border=2)
    return out.getvalue()


def _greeting(s: store.Subscriber) -> str:
    return f"Hi {s.name.split()[0]}," if s.name else "Hi,"


def _page(body: str) -> str:
    return (
        '<div style="font-family:Georgia,serif;color:#3b1e1a;max-width:520px;line-height:1.5">'
        f"{body}"
        '<p style="color:#7d5a54;font-size:13px">Questions? Just reply to this email.<br>Truesight support</p>'
        "</div>"
    )


def payment_email(s: store.Subscriber) -> tuple[str, str, str, list[mailer.InlineImage]]:
    uri = upi_uri(s.ref)
    price = f"₹{settings.subscription_price_inr}"
    days = settings.subscription_days
    subject = f"Your Truesight quiz subscription: pay {price} by UPI"
    greeting = html.escape(_greeting(s))
    body = (
        f"<p>{greeting}</p>"
        f"<p>Thanks for subscribing to Truesight's news-analysis quizzes. Access costs <b>{price}</b> "
        f"for {days} days.</p>"
        "<p><b>To pay:</b> scan this QR code with any UPI app (GPay, PhonePe, Paytm, your bank's app).</p>"
        '<p><img src="cid:upi-qr" alt="UPI QR code" width="240" height="240"></p>'
        f"<p>Or pay {price} to UPI ID <b>{html.escape(settings.upi_vpa)}</b> "
        f"({html.escape(settings.upi_payee_name)}) and add the reference "
        f"<b>{s.ref}</b> in the note.</p>"
        "<p><b>What happens next:</b> once we've matched your payment, we'll email you to say your "
        f"access is on. Then sign in at <a href=\"{settings.app_url}\">{settings.app_url}</a> with "
        f"{html.escape(s.email)}.</p>"
    )
    text = (
        f"{_greeting(s)}\n\n"
        f"Thanks for subscribing to Truesight's news-analysis quizzes. Access costs {price} for {days} days.\n\n"
        f"To pay: scan the QR code in this email with any UPI app, or pay {price} to UPI ID "
        f"{settings.upi_vpa} ({settings.upi_payee_name}) with the reference {s.ref} in the note.\n\n"
        f"UPI link: {uri}\n\n"
        "Once we've matched your payment, we'll email you to say your access is on. Then sign in at "
        f"{settings.app_url} with {s.email}.\n\nTruesight support"
    )
    image = mailer.InlineImage(filename=f"truesight-upi-{s.ref}.png", content=qr_png(uri), content_id="upi-qr")
    return subject, _page(body), text, [image]


def confirm_payment(s: store.Subscriber) -> store.Subscriber:
    """Record the reader's confirmation and email them the QR code - unless
    one went out moments ago."""
    now = _now()
    if s.payment_email_sent_at and now - datetime.fromisoformat(s.payment_email_sent_at) < RESEND_AFTER:
        return store.update_subscriber(s.uid, payment_confirmed_at=now.isoformat())
    subject, body, text, images = payment_email(s)  # before any write: may be unconfigured
    mailer.send(s.email, subject, body, text, images)
    return store.update_subscriber(
        s.uid, payment_confirmed_at=now.isoformat(), payment_email_sent_at=now.isoformat()
    )


def login_email(s: store.Subscriber) -> tuple[str, str, str]:
    end = date.fromisoformat(s.access_until) if s.access_until else None
    until = f"{end.day} {end:%B %Y}" if end else ""
    link = f"{settings.app_url}/?signin=1"
    subject = "Your Truesight quiz access is on"
    body = (
        f"<p>{html.escape(_greeting(s))}</p>"
        f"<p>We've received your payment. Quizzes are open to you until <b>{until}</b>.</p>"
        f'<p><a href="{link}" style="display:inline-block;background:#b8463b;color:#fff;'
        'padding:10px 18px;border-radius:6px;text-decoration:none">Sign in and start a quiz</a></p>'
        f"<p>Sign in with <b>{html.escape(s.email)}</b>, the account you subscribed with.</p>"
    )
    text = (
        f"{_greeting(s)}\n\nWe've received your payment. Quizzes are open to you until {until}.\n\n"
        f"Sign in at {link} with {s.email}, the account you subscribed with.\n\nTruesight support"
    )
    return subject, _page(body), text


def grant_and_email(s: store.Subscriber, days: int, admin: str) -> store.Subscriber:
    """Extend access by `days` (from today, or from the end of current access
    if that's later) and send the login email. `days=0` just resends it."""
    fields: dict = {}
    if days:
        start = max(date.today(), date.fromisoformat(s.access_until) + timedelta(days=1)) if s.access_until else date.today()
        fields = {
            "access_until": (start + timedelta(days=days - 1)).isoformat(),
            "activated_at": _now().isoformat(),
            "activated_by": admin,
        }
    elif not s.access_until or s.access_until < today():
        raise ValueError("This reader has no current access; grant days to activate them.")
    preview = store.Subscriber(**{**s.__dict__, **fields})
    subject, body, text = login_email(preview)
    mailer.send(s.email, subject, body, text)
    return store.update_subscriber(s.uid, **fields, login_email_sent_at=_now().isoformat())
