"""Load the sample clippings (the placeholder stories from the Précis design)
into a running backend, so the frontend has something to show.

    python scripts/seed_sample.py                       # http://localhost:8000
    python scripts/seed_sample.py https://api.example.com

Uses INGEST_API_KEY from the environment if the backend requires one. Each
clipping is tagged by Claude, so this makes ~24 API calls.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
headers = {"X-API-Key": os.environ["INGEST_API_KEY"]} if os.environ.get("INGEST_API_KEY") else {}
clippings = json.loads((Path(__file__).parent / "sample_clippings.json").read_text())

ids = []
for c in clippings:
    resp = requests.post(f"{base}/clippings", json=c, headers=headers, timeout=30)
    resp.raise_for_status()
    ids.append(resp.json()["id"])
    # Ingestion runs in the background; give each clipping a moment so the
    # next one can be matched to the event it creates.
    while requests.get(f"{base}/clippings/{ids[-1]}", timeout=30).json()["status"] == "processing":
        time.sleep(1)
    print(f"{c['date']}  {c['paper']:<12}  {c['title'][:70]}")

statuses = [requests.get(f"{base}/clippings/{i}", timeout=30).json()["status"] for i in ids]
print({s: statuses.count(s) for s in set(statuses)})
