"""Who may ingest: internal users signed in with Google, or scripts holding
the ingest key.

Admins sign in to the Précis admin panel with Google through Firebase
Authentication; the panel sends the Firebase ID token as `Authorization:
Bearer <token>`. A token counts as internal only if its email is verified
and on ADMIN_EMAIL_DOMAIN. Token verification is the same as
Newspaper_ingestion's app/auth.py.

Readers sign in the same way (Google or email + password, any domain) to
use quizzes; see `require_reader` and `require_quiz_access`.

Scripts (scripts/ingest_edition.py, scripts/seed_sample.py) keep using the
`X-API-Key` header with INGEST_API_KEY. With neither a valid token nor the
key, ingestion is refused - there is no anonymous ingest.
"""
from __future__ import annotations

import hmac
import threading
import time
from dataclasses import dataclass
from datetime import date

import requests
from fastapi import Header, HTTPException
from google.auth import exceptions as google_auth_exceptions
from google.auth import jwt as google_jwt

from app import store
from app.config import settings

_CERTS_URL = (
    "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
)
# Google rotates these signing certs every few hours and publishes each one
# well before using it, so an hour-old copy is always still valid.
_CERTS_TTL_SECONDS = 3600

_certs: dict[str, str] = {}
_certs_fetched_at = 0.0
_certs_lock = threading.Lock()


def _signing_certs() -> dict[str, str]:
    global _certs, _certs_fetched_at
    with _certs_lock:
        if not _certs or time.time() - _certs_fetched_at > _CERTS_TTL_SECONDS:
            response = requests.get(_CERTS_URL, timeout=10)
            response.raise_for_status()
            _certs = response.json()
            _certs_fetched_at = time.time()
        return _certs


def verify_firebase_token(token: str) -> dict:
    """Return the claims of a valid Firebase ID token, or raise ValueError."""
    project_id = settings.firebase_project_id
    if not project_id:
        raise ValueError("Sign-in is not configured on this service (FIREBASE_PROJECT_ID).")
    try:
        claims = google_jwt.decode(
            token, certs=_signing_certs(), audience=project_id, clock_skew_in_seconds=10
        )
    except (ValueError, google_auth_exceptions.GoogleAuthError) as exc:
        raise ValueError(f"Invalid or expired sign-in token: {exc}") from exc
    if claims.get("iss") != f"https://securetoken.google.com/{project_id}":
        raise ValueError("Sign-in token was issued for a different project.")
    if not claims.get("sub"):
        raise ValueError("Sign-in token has no user id.")
    return claims


def is_internal_email(email: str | None, verified: bool) -> bool:
    domain = settings.admin_email_domain.lower()
    return bool(email and verified and domain and email.lower().endswith("@" + domain))


@dataclass(frozen=True)
class Caller:
    """Who made an ingest request: an admin's email, or the ingest key."""

    label: str


def _bearer(authorization: str) -> str:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Authorization header must be 'Bearer <token>'.")
    return token.strip()


def _admin_from_token(token: str) -> Caller:
    try:
        claims = verify_firebase_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503, detail="Couldn't reach Google to verify the sign-in token."
        ) from exc
    email = claims.get("email")
    if not is_internal_email(email, bool(claims.get("email_verified"))):
        raise HTTPException(
            status_code=403,
            detail=f"The admin panel is for verified @{settings.admin_email_domain} accounts"
            + (f"; you're signed in as {email}." if email else "."),
        )
    return Caller(label=email)


def require_admin(authorization: str | None = Header(default=None)) -> Caller:
    """FastAPI dependency for admin-panel endpoints: a signed-in internal user."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Sign in with your Google account.")
    return _admin_from_token(_bearer(authorization))


def require_ingest_access(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> Caller:
    """FastAPI dependency for ingestion: an internal user, or the ingest key."""
    if authorization:
        return _admin_from_token(_bearer(authorization))
    key = settings.ingest_api_key
    if key and x_api_key and hmac.compare_digest(x_api_key, key):
        return Caller(label="api-key")
    raise HTTPException(
        status_code=401, detail="Ingestion needs an internal sign-in or a valid X-API-Key."
    )


@dataclass(frozen=True)
class Reader:
    """A signed-in reader of any domain."""

    uid: str
    email: str
    name: str | None
    internal: bool


def require_reader(authorization: str | None = Header(default=None)) -> Reader:
    """FastAPI dependency: anyone signed in with an email address."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Sign in to use quizzes.")
    try:
        claims = verify_firebase_token(_bearer(authorization))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503, detail="Couldn't reach Google to verify the sign-in token."
        ) from exc
    email = claims.get("email")
    if not email:
        raise HTTPException(status_code=401, detail="Sign in with an account that has an email address.")
    return Reader(
        uid=claims["sub"],
        email=email,
        name=claims.get("name"),
        internal=is_internal_email(email, bool(claims.get("email_verified"))),
    )


def require_quiz_access(authorization: str | None = Header(default=None)) -> Reader:
    """FastAPI dependency for quizzes: an internal user, or a reader whose
    paid access hasn't run out. 402 tells the app to show the payment step."""
    reader = require_reader(authorization)
    if reader.internal:
        return reader
    sub = store.upsert_subscriber(reader.uid, reader.email, reader.name)
    # Access already granted counts even while a renewal awaits the admin.
    if not (sub.access_until and sub.access_until >= date.today().isoformat()):
        raise HTTPException(status_code=402, detail="Quizzes need an active subscription.")
    return reader
