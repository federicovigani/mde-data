#!/usr/bin/env python3
# =============================================================================
# backfill.py — Download missing snapshots from CDN into snapshots/
#
# Runs inside the mde-data repository (checked out by GitHub Actions).
# Fetches every trading day in 2025 and 2026 from jsDelivr CDN.
# Skips files that already exist. Prints a summary at the end.
#
# The GitHub Action commits whatever new files this script writes.
#
# Usage (local):
#   pip install requests
#   python3 backfill.py
#   python3 backfill.py --year 2025
#   python3 backfill.py --year 2026
#   python3 backfill.py --dry-run
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta

import requests

CDN_BASE = "https://cdn.jsdelivr.net/gh/federicovigani/mde-data@main/snapshots/"
OUT_DIR  = "snapshots"
DELAY    = 0.3
TIMEOUT  = 15

# ---------------------------------------------------------------------------
# NYSE market calendar
# ---------------------------------------------------------------------------

HOLIDAYS = {
    # 2025
    date(2025, 1, 1),
    date(2025, 1, 20),
    date(2025, 2, 17),
    date(2025, 4, 18),
    date(2025, 5, 26),
    date(2025, 6, 19),
    date(2025, 7, 4),
    date(2025, 9, 1),
    date(2025, 11, 27),
    date(2025, 12, 25),
    # 2026
    date(2026, 1, 1),
    date(2026, 1, 19),
    date(2026, 2, 16),
    date(2026, 4, 3),
    date(2026, 5, 25),
    date(2026, 6, 19),
    date(2026, 7, 3),
    date(2026, 9, 7),
    date(2026, 11, 26),
    date(2026, 12, 25),
}


def trading_days(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in HOLIDAYS:
            days.append(d)
        d += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# Fetch one snapshot from CDN
# ---------------------------------------------------------------------------

def fetch(snap_date: str) -> dict | None:
    try:
        r = requests.get(f"{CDN_BASE}{snap_date}.json", timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and "meta" in data and "quadrant" in data:
                return data
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(start: date, end: date, dry_run: bool) -> int:
    """Returns number of new files written."""
    today = date.today()
    days  = [d for d in trading_days(start, end) if d <= today]

    os.makedirs(OUT_DIR, exist_ok=True)

    written = 0
    skipped = 0
    missing = []

    print(f"Range: {start} to {end} — {len(days)} trading days to check")
    print()

    for i, d in enumerate(days, 1):
        snap_date = d.isoformat()
        out_path  = os.path.join(OUT_DIR, f"{snap_date}.json")

        if os.path.exists(out_path):
            skipped += 1
            continue

        print(f"[{i:4d}/{len(days)}] {snap_date} ... ", end="", flush=True)

        if dry_run:
            print("dry-run")
            continue

        data = fetch(snap_date)
        time.sleep(DELAY)

        if data is None:
            print("not on CDN")
            missing.append(snap_date)
            continue

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        label = data.get("quadrant", {}).get("label", "?")
        score = data.get("macro_score", {}).get("score", "?")
        print(f"saved  ({label} | score {score})")
        written += 1

    print()
    print(f"Already existed : {skipped}")
    print(f"Newly downloaded: {written}")
    print(f"Not on CDN      : {len(missing)}")
    if missing:
        print(f"Missing dates   : {missing}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",    type=int, choices=[2025, 2026])
    parser.add_argument("--start",   help="YYYY-MM-DD")
    parser.add_argument("--end",     help="YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    today = date.today()

    if args.year == 2025:
        start, end = date(2025, 1, 2), date(2025, 12, 31)
    elif args.year == 2026:
        start, end = date(2026, 1, 2), today
    elif args.start or args.end:
        start = date.fromisoformat(args.start) if args.start else date(2025, 1, 2)
        end   = date.fromisoformat(args.end)   if args.end   else today
    else:
        start, end = date(2025, 1, 2), today

    written = run(start, end, dry_run=args.dry_run)

    # Exit code 0 whether or not anything was written — the Action handles the commit.
    sys.exit(0)


if __name__ == "__main__":
    main()
