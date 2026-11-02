"""Write the analysis, framing and quiz question pool for events ingested
before pre-generation existed (or whose pre-generation failed). Optional:
readers' requests generate any of these that are missing on first use; this
just saves them the wait.

    python scripts/pregenerate.py 2026-10-12 2026-10-19            # dry run: counts only
    python scripts/pregenerate.py 2026-10-12 2026-10-19 --write

Runs against the local database directly (start it from the repo root with
the backend's .env loaded), so it works whether or not the server is up.
Every missing output is one Claude call - check the dry run's count first.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import store  # noqa: E402
from app.config import settings  # noqa: E402
from app.pregenerate import prepare_events  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) != 2:
    sys.exit(__doc__)
date_from, date_to = args

events = store.list_events(date_from, date_to)
missing = {f: sum(e.caches.get(f) is None for e in events) for f in store.EVENT_CACHE_FIELDS}
todo = [e.id for e in events if any(e.caches.get(f) is None for f in store.EVENT_CACHE_FIELDS)]
print(f"{len(events)} events from {date_from} to {date_to}; {len(todo)} need work.")
print("Missing (one Claude call each): " + ", ".join(f"{f} {n}" for f, n in missing.items()))

if "--write" not in sys.argv:
    print("Dry run - pass --write to generate them.")
    sys.exit(0)

settings.pregenerate = True
prepare_events(todo)
events = store.get_events(todo)
left = sum(any(e.caches.get(f) is None for f in store.EVENT_CACHE_FIELDS) for e in events)
print(f"Done. {len(todo) - left} events complete, {left} still missing something (failed; see log).")
