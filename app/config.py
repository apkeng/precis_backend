"""Central configuration, all overridable via environment variables.

Same stack as Newspaper_ingestion: embeddings (sentence-transformers) and the
vector store (Chroma) are free and local; Claude does the reading - topic
tagging, event matching, analysis, "ask the sources" and quiz drafting.
"""
import os


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


class Settings:
    # Claude API (Anthropic). Reads credentials from ANTHROPIC_API_KEY /
    # ANTHROPIC_AUTH_TOKEN / `ant auth login` automatically; no key is stored here.
    anthropic_model: str = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
    anthropic_max_tokens: int = _int("ANTHROPIC_MAX_TOKENS", 16000)
    # Only needed for an API key that isn't scoped to a workspace - the API
    # then rejects every request unless it names one via this header.
    anthropic_workspace_id: str = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()

    # Local embedding model (sentence-transformers) - same model as
    # Newspaper_ingestion, downloaded once, then runs on CPU.
    embedding_model_name: str = os.environ.get(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Local, on-disk vector store (Chroma).
    chroma_persist_dir: str = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")

    # Structured records (clippings, events, cached analyses). Empty means a
    # "precis.db" beside the Chroma dir, so it lands on the same disk.
    sqlite_path: str = os.environ.get("SQLITE_PATH", "")

    # Chunking defaults (characters, not tokens).
    chunk_size: int = _int("CHUNK_SIZE", 1500)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 200)

    # Retrieval defaults.
    default_top_k: int = _int("DEFAULT_TOP_K", 8)
    # How many nearby events Claude is shown when deciding whether a new
    # clipping belongs to an existing event, and how far apart (days) they
    # may be.
    event_match_candidates: int = _int("EVENT_MATCH_CANDIDATES", 5)
    event_match_window_days: int = _int("EVENT_MATCH_WINDOW_DAYS", 10)
    # How many candidate events the story-analysis screen asks Claude to
    # check for a direct link.
    trace_candidates: int = _int("TRACE_CANDIDATES", 8)
    # Cap on events (and clippings per event) sent to Claude for one quiz.
    quiz_max_events: int = _int("QUIZ_MAX_EVENTS", 24)
    quiz_clippings_per_event: int = _int("QUIZ_CLIPPINGS_PER_EVENT", 2)
    excerpt_chars: int = _int("EXCERPT_CHARS", 2400)
    trace_max_days: int = _int("TRACE_MAX_DAYS", 120)

    # Shared secret for ingestion scripts (X-API-Key header). Empty means
    # scripts can't ingest; the admin panel still can, with a sign-in.
    ingest_api_key: str = os.environ.get("INGEST_API_KEY", "").strip()

    # Admin panel sign-in (see app/auth.py): the Firebase project whose ID
    # tokens the panel sends - the Précis project by default. Not a secret,
    # it's the `aud` every token from that project carries. Plus the email
    # domain that counts as internal.
    firebase_project_id: str = os.environ.get("FIREBASE_PROJECT_ID", "newsdigest-9fd33").strip()
    admin_email_domain: str = os.environ.get("ADMIN_EMAIL_DOMAIN", "apokryfon.com").strip().lstrip("@")

    # Origins allowed to call this API from a browser (comma-separated).
    # Defaults cover the Précis Vite dev server. Each entry is trimmed and has
    # any trailing slash removed, since a browser's Origin header is always
    # bare scheme+host+port and CORSMiddleware matches it exactly.
    cors_allowed_origins: list[str] = [
        origin.strip().rstrip("/")
        for origin in os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]


settings = Settings()
