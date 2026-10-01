# Précis backend

The RAG service behind Précis, a news-analysis app for
civil-services aspirants. It ingests newspaper clippings, groups them into
**events** tagged with 14 syllabus topics, and serves Claude-written
analysis, story tracing, "ask the sources" answers and quizzes. Every output
is grounded **only** in the clippings of the period the reader picked.

It reuses the stack of
[Newspaper_ingestion](https://github.com/apkeng/Newspaper_ingestion): the
same local embedding model, Chroma as the vector store, and Claude with
schema-enforced JSON output.

## How it works

```
clipping (JSON or PDF) --> chunk --> local embeddings --> Chroma `clippings`
        |
        +--> nearest events (Chroma `events`, ±10 days) --> Claude: topics + same event or new?
                                                                  |
                                                    SQLite: clipping -> event, event topics

reader --> /events?from&to&topics          list events in a period
       --> /events/{id}                    Claude analysis, citing sources [n]   (cached)
       --> /events/{id}/ask                retrieve passages in period -> Claude answer
       --> /events/{id}/trace              related events (Claude-judged links + shared topics),
                                           day-by-day coverage, how each paper framed it
       --> /quizzes                        Claude drafts questions from the period's clippings
```

| Stage | Tool | Cost |
|---|---|---|
| PDF text extraction | `pypdf` | Free |
| Chunking | `app/chunking.py` (from Newspaper_ingestion) | Free |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free, local CPU |
| Vector database | `chromadb` (embedded, on disk) | Free |
| Records & caches | SQLite (stdlib), beside the Chroma dir | Free |
| LLM | Claude API (`claude-opus-5`, same as Newspaper_ingestion) | Paid, per token |

**Events.** When a clipping arrives, its nearest existing events from the
surrounding days are shown to Claude with the clipping. Claude tags the
syllabus topics, then either picks the event the clipping reports on or
starts a new one with a neutral headline. Clippings that touch none of the 14
topics are kept as `off_syllabus` and never appear in the app.

**Caching.** Event analysis, paper framing, the per-event practice quiz and
event-to-event link verdicts are stored after the first request. When a new
clipping joins an event, that event's cached outputs are cleared so they're
regenerated from the fuller set of sources.

**Short quizzes, not padded ones.** Claude is told to write *up to* the
requested number of questions and to stop when the material runs out.
Questions that cite another event's clipping, use a topic the event doesn't
carry, or don't have exactly four options are dropped. The response includes
`requested` so the UI can say the quiz came out shorter than asked.

**Refusal fallbacks.** Calls opt into Claude's server-side fallback
(`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`). If a safety
classifier declines a request, which can happen on hard news such as
attacks or security incidents, the API re-runs it on Anthropic's
recommended fallback model instead of failing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # or `ant auth login` once
uvicorn app.main:app --reload
```

The first call that touches embeddings downloads `all-MiniLM-L6-v2` (~90MB)
from Hugging Face and caches it.

Load the sample clippings (the placeholder stories from the design):

```bash
python scripts/seed_sample.py
```

Or with Docker: `docker compose up --build` (pass `ANTHROPIC_API_KEY`).

## Configuration

All via environment variables (see `app/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_MODEL` | `claude-opus-5` | Claude model |
| `ANTHROPIC_MAX_TOKENS` | `16000` | Max output tokens per call |
| `ANTHROPIC_API_KEY` | *(unset)* | Claude API key (or `ant auth login`) |
| `ANTHROPIC_WORKSPACE_ID` | *(unset)* | Only for keys not scoped to a workspace |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Vector store location |
| `SQLITE_PATH` | `<chroma dir>/../precis.db` | Records and caches |
| `INGEST_API_KEY` | *(unset)* | If set, `POST /clippings*` require `X-API-Key` |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173` | Comma-separated browser origins |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1500` / `200` | Chunking (characters) |
| `DEFAULT_TOP_K` | `8` | Passages retrieved for "ask the sources" |
| `EVENT_MATCH_CANDIDATES` / `EVENT_MATCH_WINDOW_DAYS` | `5` / `10` | Events shown to Claude when filing a clipping |
| `TRACE_CANDIDATES` | `8` | Similar events checked for a direct link |
| `QUIZ_MAX_EVENTS` / `QUIZ_CLIPPINGS_PER_EVENT` | `24` / `2` | Material sent per quiz |
| `EXCERPT_CHARS` | `2400` | Max characters of each clipping sent to Claude |
| `TRACE_MAX_DAYS` | `120` | Longest period the coverage chart covers |

## API

JSON field names are camelCase. Dates are `YYYY-MM-DD`.

| Method & path | Purpose |
|---|---|
| `GET /topics` | The 14 topics (id, name, short label, exam paper), question formats, difficulties |
| `GET /stats` | Papers indexed, clippings this month, last ingest time |
| `POST /clippings` | Ingest `{paper, date, page?, title, text}`. Returns `202` with `status: "processing"` |
| `POST /clippings/pdf` | Same, as multipart (`file`, `paper`, `date`, `title`, `page?`) |
| `GET /clippings/{id}` | Status: `processing` / `ready` / `off_syllabus` / `failed` and its `eventId` |
| `GET /events?from&to&topics=a,b` | Events in the period. Omit `topics` for all, or pass it empty for none. Includes `topicCounts` for the whole period |
| `GET /events/{id}` | Event with numbered `sources` and `analysis` (`whatHappened`, `whyItMatters` with `[n]` citations, `prelimsFacts`, `gsPaper`, `mainsQuestion`, `timeline`) |
| `POST /events/{id}/ask` | `{question, from, to}` → `{answer, foundInSources, sources}` |
| `GET /events/{id}/trace?from&to` | `days` (clippings per day on the story and on related events), `related` (`kind: "direct" \| "theme"` and a reason), `outsideCount` / `outsideRange` (direct links outside the period), `framing` per paper |
| `GET /events/{id}/quiz` | Three-question practice quiz for one event |
| `POST /quizzes` | `{from, to, topics, formats, length, difficulty, eventIds?}` → `{questions, requested}` |

Errors: `404` for an unknown id, `409` when a quiz has no material, `422`
for bad input, `502` when Claude fails or its output doesn't validate.

Example:

```bash
curl -X POST localhost:8000/clippings -H 'Content-Type: application/json' -d '{
  "paper": "The Courier", "date": "2026-09-28", "page": 1,
  "title": "Cyclone makes landfall in Odisha",
  "text": "The storm crossed the coast near Paradip ..."
}'
curl 'localhost:8000/events?from=2026-09-01&to=2026-09-30&topics=geo,envsec'
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a fake embedding function and a fake Claude, so they run
offline with no model download and no API key.

## Known limitations

- Scanned or image-only PDFs aren't OCR'd.
- Event matching only looks ±`EVENT_MATCH_WINDOW_DAYS` around a clipping's
  date, so a story that resurfaces weeks later starts a new event. The trace
  screen still links the two through Claude's direct-link check.
- Ingestion runs as an in-process background task. A restart mid-ingest
  leaves that clipping `processing`; re-post it.
- There is no per-user data. Clippings are shared by every reader, and only
  ingestion is protected (by `INGEST_API_KEY`).
