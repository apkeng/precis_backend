"""Outgoing email through Resend's HTTP API."""
from __future__ import annotations

import base64
from dataclasses import dataclass

import requests

from app.config import settings

_RESEND_URL = "https://api.resend.com/emails"


class MailError(RuntimeError):
    pass


@dataclass(frozen=True)
class InlineImage:
    """An image shown in the HTML body as <img src="cid:{content_id}">."""

    filename: str
    content: bytes
    content_id: str


def send(to: str, subject: str, html: str, text: str, images: list[InlineImage] = ()) -> str:
    """Send one email and return Resend's message id, or raise MailError."""
    if not settings.resend_api_key:
        raise MailError("Email isn't configured on this service (RESEND_API_KEY).")
    payload = {
        "from": settings.mail_from,
        "to": [to],
        "subject": subject,
        "html": html,
        "text": text,
        "attachments": [
            {
                "filename": img.filename,
                "content": base64.b64encode(img.content).decode(),
                "content_id": img.content_id,
            }
            for img in images
        ],
    }
    try:
        resp = requests.post(
            _RESEND_URL,
            json=payload,
            headers={"Authorization": f"Bearer {settings.resend_api_key}"},
            timeout=20,
        )
    except requests.RequestException as exc:
        raise MailError(f"Couldn't reach the email service: {exc}") from exc
    if resp.status_code >= 300:
        try:
            detail = resp.json().get("message") or resp.text
        except ValueError:
            detail = resp.text
        raise MailError(f"The email service refused the message ({resp.status_code}): {detail}")
    return resp.json().get("id", "")
