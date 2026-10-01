"""Upload a whole day's e-paper PDF to a running backend and follow progress.

    python scripts/ingest_edition.py th_international_01_10_2026.pdf "The Hindu" 2026-10-01
    python scripts/ingest_edition.py paper.pdf "The Hindu" 2026-10-01 --api https://api.example.com

Uses INGEST_API_KEY from the environment if the backend requires one. An
18-page edition takes a few minutes: one Claude call per page to split it
into articles, then one per news or opinion article to tag and file it.
"""
from __future__ import annotations

import argparse
import os
import time

import requests

parser = argparse.ArgumentParser()
parser.add_argument("pdf")
parser.add_argument("paper")
parser.add_argument("date", help="YYYY-MM-DD")
parser.add_argument("--api", default="http://localhost:8000")
args = parser.parse_args()

base = args.api.rstrip("/")
headers = {"X-API-Key": os.environ["INGEST_API_KEY"]} if os.environ.get("INGEST_API_KEY") else {}
with open(args.pdf, "rb") as f:
    resp = requests.post(
        f"{base}/editions/pdf",
        files={"file": (os.path.basename(args.pdf), f, "application/pdf")},
        data={"paper": args.paper, "date": args.date},
        headers=headers,
        timeout=120,
    )
resp.raise_for_status()
edition_id = resp.json()["id"]
print(f"Edition {edition_id} uploaded; processing...")

while True:
    e = requests.get(f"{base}/editions/{edition_id}", timeout=30).json()
    print(f"  page {e['pagesDone']}/{e['pages'] or '?'}  clippings {e['clippings']}  events {e['events']}")
    if e["status"] != "processing":
        break
    time.sleep(10)

print(f"Done: {e['status']}" + (f" ({e['error']})" if e["error"] else ""))
