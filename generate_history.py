#!/usr/bin/env python3
# =============================================================================
# generate_history.py — Generate historical MDE snapshots from scratch
#
# Downloads all historical data from yfinance + FRED in one pass, then
# slices to each missing trading day and runs the scoring logic.
# No cache file needed. Runs in GitHub Actions from the mde-data repo.
#
# Requires secrets: FRED_API_KEY
# Requires pip: yfinance requests pandas numpy
#
# Usage:
#   python3 generate_history.py                   # all missing dates
#   python3 generate_history.py --year 2025
#   python3 generate_history.py --start 2026-01-01 --end 2026-04-12
#   python3 generate_history.py --dry-run
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FRED_API_KEY   = os.getenv("FRED_API_KEY", "")
SNAPSHOTS_DIR  = "snapshots"

# How far back to download (needed for accurate 10yr VIX percentile)
# 2010 start ensures 2020 dates have a full 10yr lookback window
HISTORY_START  = "2010-01-01"

# Tickers
SPY_TICKER     = "SPY"
VIX_TICKER     = "^VIX"
VIX3M_TICKER   = "^VIX3M"
HYG_TICKER     = "HYG"
LQD_TICKER     = "LQD"

SECTOR_ETFS = [
    "XLK", "XLV", "XLF", "XLE", "XLI",
    "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB",
]

DEEP_VALUE_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "ADBE", "CRM",
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP",
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT",
    "XOM", "CVX", "COP",
    "HD", "MCD", "SBUX", "NKE", "LOW",
    "WMT", "COST", "TGT",
    "BA", "CAT", "MMM", "HON",
    "NEE", "DUK",
]

# FRED series IDs
FRED_HY  = "BAMLH0A0HYM2"   # HY OAS in percent (3.50 = 350 bps)
FRED_IG  = "BAMLC0A0CM"     # IG OAS in percent
FRED_10Y = "DGS10"          # 10yr Treasury yield
FRED_2Y  = "DGS2"           # 2yr Treasury yield

# ---------------------------------------------------------------------------
# NYSE holiday calendar
# ---------------------------------------------------------------------------

HOLIDAYS = {
    date(2015, 1, 1), date(2015, 1, 19), date(2015, 2, 16), date(2015, 4, 3),
    date(2015, 5, 25), date(2015, 7, 3), date(2015, 9, 7), date(2015, 11, 26), date(2015, 12, 25),
    date(2016, 1, 1), date(2016, 1, 18), date(2016, 2, 15), date(2016, 3, 25),
    date(2016, 5, 30), date(2016, 7, 4), date(2016, 9, 5), date(2016, 11, 24), date(2016, 12, 26),
    date(2017, 1, 2), date(2017, 1, 16), date(2017, 2, 20), date(2017, 4, 14),
    date(2017, 5, 29), date(2017, 7, 4), date(2017, 9, 4), date(2017, 11, 23), date(2017, 12, 25),
    date(2018, 1, 1), date(2018, 1, 15), date(2018, 2, 19), date(2018, 3, 30),
    date(2018, 5, 28), date(2018, 7, 4), date(2018, 9, 3), date(2018, 11, 22), date(2018, 12, 5),
    date(2018, 12, 25),
    date(2019, 1, 1), date(2019, 1, 21), date(2019, 2, 18), date(2019, 4, 19),
    date(2019, 5, 27), date(2019, 7, 4), date(2019, 9, 2), date(2019, 11, 28), date(2019, 12, 25),
    date(2020, 1, 1), date(2020, 1, 20), date(2020, 2, 17), date(2020, 4, 10),
    date(2020, 5, 25), date(2020, 7, 3), date(2020, 9, 7), date(2020, 11, 26), date(2020, 12, 25),
    date(2021, 1, 1), date(2021, 1, 18), date(2021, 2, 15), date(2021, 4, 2),
    date(2021, 5, 31), date(2021, 7, 5), date(2021, 9, 6), date(2021, 11, 25), date(2021, 12, 24),
    date(2022, 1, 17), date(2022, 2, 21), date(2022, 4, 15),
    date(2022, 5, 30), date(2022, 6, 20), date(2022, 7, 4), date(2022, 9, 5),
    date(2022, 11, 24), date(2022, 12, 26),
    date(2023, 1, 2), date(2023, 1, 16), date(2023, 2, 20), date(2023, 4, 7),
    date(2023, 5, 29), date(2023, 6, 19), date(2023, 7, 4), date(2023, 9, 4),
    date(2023, 11, 23), date(2023, 12, 25),
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2),
    date(2024, 11, 28), date(2024, 12, 25),
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
}


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in HOLIDAYS


def trading_days(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days


def missing_dates(start: date, end: date, overwrite: bool = False) -> list[date]:
    """Return trading days in range that need a snapshot generated."""
    today = date.today()
    return [
        d for d in trading_days(start, end)
        if d <= today
        and (overwrite or not os.path.exists(os.path.join(SNAPSHOTS_DIR, f"{d.isoformat()}.json")))
    ]


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    print(f"  Downloading prices for {len(tickers)} tickers from {start}...")
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]] if "Close" in raw.columns else raw
    closes.index = pd.to_datetime(closes.index).normalize()
    print(f"  Got {len(closes)} rows, {len(closes.columns)} tickers")
    return closes


def download_fred(series_id: str, start: str) -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":       series_id,
        "api_key":         FRED_API_KEY,
        "file_type":       "json",
        "observation_start": start,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json()["observations"]
    s = pd.Series(
        {o["date"]: float(o["value"]) for o in obs if o["value"] != "."},
        name=series_id,
        dtype=float,
    )
    s.index = pd.to_datetime(s.index)
    return s


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def pct_rank(series: pd.Series, value: float) -> float:
    """What percentile is value within series? 0-100."""
    return float((series <= value).mean() * 100)


def rolling_z(series: pd.Series, window: int, value: float) -> float:
    tail = series.tail(window).dropna()
    if len(tail) < 10:
        return 0.0
    std = tail.std()
    return float((value - tail.mean()) / std) if std > 0 else 0.0


def ret_pct(series: pd.Series, n: int) -> float:
    if len(series) < n + 1:
        return 0.0
    return float((series.iloc[-1] / series.iloc[-(n + 1)] - 1) * 100)


# ---------------------------------------------------------------------------
# Snapshot generator
# ---------------------------------------------------------------------------

def generate_snapshot(
    snap_date: str,
    prices: pd.DataFrame,
    hy_s: pd.Series,
    ig_s: pd.Series,
    y10_s: pd.Series,
    y2_s: pd.Series,
) -> dict | None:
    d = pd.Timestamp(snap_date)

    def asof(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        return s[s.index <= d]

    # Slice all series to as-of date
    spy   = asof(prices[SPY_TICKER])   if SPY_TICKER   in prices.columns else pd.Series(dtype=float)
    vix   = asof(prices[VIX_TICKER])   if VIX_TICKER   in prices.columns else pd.Series(dtype=float)
    vix3m = asof(prices[VIX3M_TICKER]) if VIX3M_TICKER in prices.columns else pd.Series(dtype=float)
    hy    = asof(hy_s)
    ig    = asof(ig_s)
    y10   = asof(y10_s)
    y2    = asof(y2_s)

    if len(spy) < 22 or len(vix) < 22 or len(hy) < 22:
        print(f"    Skipping {snap_date}: insufficient history")
        return None

    # ----- VIX -----
    vix_latest  = float(vix.iloc[-1])
    vix_10y     = vix.tail(252 * 10)
    vix_pct_10y = pct_rank(vix_10y, vix_latest)

    # VIX term structure
    vix3m_latest = float(vix3m.iloc[-1]) if len(vix3m) > 0 else vix_latest
    ts_spread    = round(vix3m_latest - vix_latest, 2)
    ts_inverted  = ts_spread < 0

    # ----- SPY -----
    spy_5d   = ret_pct(spy, 5)
    spy_5d_s = spy.pct_change(5).dropna()
    spy_5d_z = float(
        (spy_5d / 100 - spy_5d_s.tail(504).mean()) / spy_5d_s.tail(504).std()
    ) if len(spy_5d_s) >= 20 and spy_5d_s.tail(504).std() > 0 else 0.0

    # ----- HY OAS -----
    hy_latest   = float(hy.iloc[-1])          # in percent (e.g. 3.50 = 350 bps)
    hy_5yr      = hy.tail(252 * 5)
    hy_pct_5y   = pct_rank(hy_5yr, hy_latest)
    hy_pct_3y   = pct_rank(hy.tail(252 * 3), hy_latest)
    hy_z        = rolling_z(hy_5yr, len(hy_5yr), hy_latest)

    # Panic signal: HY change in bps
    hy_bps   = hy * 100
    hy_c5d   = float(hy_bps.iloc[-1] - hy_bps.iloc[-6])  if len(hy_bps) >= 6  else 0.0
    hy_c10d  = float(hy_bps.iloc[-1] - hy_bps.iloc[-11]) if len(hy_bps) >= 11 else 0.0
    hy_accel = round(abs(hy_c5d) / abs(hy_c10d), 3) if abs(hy_c10d) > 0.5 else None

    # ----- IG OAS -----
    ig_latest = float(ig.iloc[-1]) if len(ig) > 0 else 0.0
    ig_5yr    = ig.tail(252 * 5)
    ig_pct_5y = pct_rank(ig_5yr, ig_latest) if len(ig_5yr) > 10 else 50.0
    ig_z      = rolling_z(ig_5yr, len(ig_5yr), ig_latest) if len(ig_5yr) > 10 else 0.0

    # ----- Yield curve -----
    curve_latest = None
    if len(y10) > 0 and len(y2) > 0:
        # Forward-fill and align on same date
        curve = (y10.reindex(y10.index.union(y2.index)).ffill()
                 - y2.reindex(y10.index.union(y2.index)).ffill())
        curve_latest = round(float(curve.iloc[-1]), 3)

    # ----- Quadrant -----
    equity_stress = (vix_pct_10y >= 70) or (spy_5d_z <= -2.0)
    credit_stress = hy_z >= 1.5

    if equity_stress and credit_stress:
        quad, quad_label = "PANIC",       "PANIC MODE"
    elif equity_stress and not credit_stress:
        quad, quad_label = "DISLOCATION", "EQUITY DISLOCATION"
    elif credit_stress and not equity_stress:
        quad, quad_label = "WARNING",     "STORM COMING"
    elif vix_pct_10y < 30 and abs(hy_z) < 0.5:
        quad, quad_label = "CALM",        "CALM"
    else:
        quad, quad_label = "BAU",         "BUSINESS AS USUAL"

    # ----- Stress score -----
    ts_bonus = 5 if ts_inverted else 0
    raw_score = (
        (vix_pct_10y / 100) * 25
        + ts_bonus
        + (hy_pct_5y / 100) * 30
        + (ig_pct_5y / 100) * 20
    )
    score = min(round(raw_score * (100 / 85), 1), 100.0)

    if score >= 85:   score_label = "CRISIS"
    elif score >= 65: score_label = "DISTRESSED"
    elif score >= 40: score_label = "ELEVATED"
    elif score >= 20: score_label = "CAUTIOUS"
    else:             score_label = "CALM"

    # ----- Panic grade -----
    if quad == "PANIC" and hy_accel is not None:
        if hy_accel < 0.6:   panic_grade = "ENTER"
        elif hy_accel < 1.0: panic_grade = "WATCH"
        else:                 panic_grade = "STAY OUT"
    elif quad == "PANIC":
        panic_grade = "WATCH"
    else:
        panic_grade = ""

    # ----- Breadth (sector ETFs) -----
    sectors_red = 0
    for ticker in SECTOR_ETFS:
        if ticker in prices.columns:
            s = asof(prices[ticker])
            if len(s) >= 6 and ret_pct(s, 5) < 0:
                sectors_red += 1

    # ----- Type B dislocations (sector ETFs) -----
    flagged    = []
    approaching = []
    if equity_stress:
        for ticker in SECTOR_ETFS:
            if ticker not in prices.columns:
                continue
            s = asof(prices[ticker])
            if len(s) < 30:
                continue
            r5d  = ret_pct(s, 5)
            r10d = ret_pct(s, 10)
            r20d = ret_pct(s, 20)
            s5d  = s.pct_change(5).dropna()
            z5d  = rolling_z(s5d.tail(504), len(s5d.tail(504)), float(s5d.iloc[-1])) if len(s5d) >= 20 else 0.0
            high_52w     = float(s.tail(252).max())
            pct_from_high = float((s.iloc[-1] / high_52w - 1) * 100) if high_52w > 0 else 0.0

            # Dislocation score (mirrors engine logic)
            dscore = 0
            if r5d < -3:           dscore += 20
            if r5d < -5:           dscore += 15
            if z5d < -2.0:         dscore += 25
            if z5d < -3.0:         dscore += 15
            if pct_from_high < -10: dscore += 15
            if not credit_stress:  dscore += 10  # credit not confirming = pure dislocation

            entry = {
                "ticker":            ticker,
                "name":              ticker,
                "return_5d":         round(r5d, 2),
                "return_10d":        round(r10d, 2),
                "return_20d":        round(r20d, 2),
                "zscore":            round(z5d, 2),
                "pct_from_high":     round(pct_from_high, 1),
                "dislocation_score": float(dscore),
                "dislocation_type":  "PURE DISLOCATION" if not credit_stress else "MIXED",
            }
            if dscore >= 60:
                flagged.append(entry)
            elif dscore >= 40:
                approaching.append(entry)

    flagged.sort(key=lambda x: x["dislocation_score"], reverse=True)

    # ----- Type C deep value -----
    deep_value = []
    for ticker in DEEP_VALUE_UNIVERSE:
        if ticker not in prices.columns:
            continue
        s = asof(prices[ticker])
        if len(s) < 42:
            continue
        r1d  = float(s.pct_change().iloc[-1] * 100)
        r5d  = ret_pct(s, 5)
        r10d = ret_pct(s, 10)
        r20d = ret_pct(s, 20)
        r40d = ret_pct(s, 40)

        if r5d > -3.0:
            continue  # not enough drawdown

        daily_rets    = s.pct_change().dropna().tail(504)
        z_score       = rolling_z(daily_rets, len(daily_rets), float(s.pct_change().iloc[-1])) if len(daily_rets) >= 20 else 0.0
        high_52w      = float(s.tail(252).max())
        pct_from_high = float((s.iloc[-1] / high_52w - 1) * 100) if high_52w > 0 else 0.0
        low_40        = float(s.tail(40).min())
        pct_vs_40_low = float((s.iloc[-1] / low_40 - 1) * 100) if low_40 > 0 else 0.0
        consec_down   = 0
        for ret in reversed(s.pct_change().dropna().tail(10).tolist()):
            if ret < 0:
                consec_down += 1
            else:
                break

        if z_score < -2.5:
            deep_value.append({
                "ticker":             ticker,
                "name":               ticker,
                "zscore":             round(z_score, 2),
                "return_1d":          round(r1d, 2),
                "return_5d":          round(r5d, 2),
                "return_10d":         round(r10d, 2),
                "return_20d":         round(r20d, 2),
                "return_40d":         round(r40d, 2),
                "pct_from_high":      round(pct_from_high, 1),
                "pct_vs_40d_low":     round(pct_vs_40_low, 1),
                "consecutive_down_days": consec_down,
                "entry_verdict":      "WAIT",
                "reason":             "historical backfill — no LLM verdict",
            })

    deep_value.sort(key=lambda x: x["zscore"])
    deep_value = deep_value[:5]

    return {
        "meta": {
            "date":        snap_date,
            "n_flagged":   len(flagged),
            "n_deep_value": len(deep_value),
            "source":      "historical_backfill",
        },
        "quadrant": {
            "label":    quad_label,
            "quadrant": quad,
            "spy_5d":   round(spy_5d, 2),
            "spy_5d_z": round(spy_5d_z, 2),
        },
        "macro_score": {
            "score": score,
            "label": score_label,
        },
        "panic_signal": {
            "grade":          panic_grade,
            "hy_accel_ratio": hy_accel,
            "hy_c5d_bps":     round(hy_c5d, 1),
            "hy_c10d_bps":    round(hy_c10d, 1),
        },
        "credit": {
            "hy_oas": {
                "latest":  round(hy_latest, 4),   # raw FRED percent (2.79 = 279 bps); display code does *100
                "pct_5y":  round(hy_pct_5y, 1),
                "pct_3y":  round(hy_pct_3y, 1),
                "zscore":  round(hy_z, 2),
            },
            "ig_oas": {
                "latest": round(ig_latest, 4),
                "pct_5y": round(ig_pct_5y, 1),
                "zscore": round(ig_z, 2),
            },
            "curve_10y2y": {
                "latest": curve_latest,
            },
            "_composite": {
                "credit_is_calm":   not credit_stress,
                "credit_stress_z":  round(max(hy_z, ig_z), 2),
            },
        },
        "vix": {
            "vix": {
                "latest":          round(vix_latest, 2),
                "percentile_10y":  round(vix_pct_10y, 1),
            },
            "term_structure": {
                "inverted": ts_inverted,
                "spread":   ts_spread,
            },
        },
        "dislocations": {
            "flagged":     flagged,
            "approaching": approaching,
        },
        "deep_value": deep_value,
        "breadth": {
            "sectors_red":   sectors_red,
            "sectors_total": len(SECTOR_ETFS),
        },
        "fred_staleness": {
            "_summary": {
                "any_stale":   False,
                "max_lag_days": 0,
            }
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",      type=int, choices=list(range(2015, 2030)))
    parser.add_argument("--start",     help="YYYY-MM-DD")
    parser.add_argument("--end",       help="YYYY-MM-DD")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate snapshots that already exist")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    today = date.today()

    if args.year:
        start = date(args.year, 1, 2)
        end   = min(date(args.year, 12, 31), today)
    elif args.start or args.end:
        start = date.fromisoformat(args.start) if args.start else date(2020, 1, 2)
        end   = date.fromisoformat(args.end)   if args.end   else today
    else:
        start = date(2020, 1, 2)
        end   = today

    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    targets = missing_dates(start, end, overwrite=args.overwrite)
    if not targets:
        print("Nothing to generate — all dates already have snapshots.")
        sys.exit(0)

    mode = "OVERWRITE" if args.overwrite else "MISSING ONLY"
    print(f"\n{'='*60}")
    print(f"  MDE Historical Backfill  [{mode}]")
    print(f"  Range    : {start} to {end}")
    print(f"  Targets  : {len(targets)} dates")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("Dry run — first 10 targets:")
        for d in targets[:10]:
            print(f"  {d}")
        if len(targets) > 10:
            print(f"  ... and {len(targets) - 10} more")
        sys.exit(0)

    if not FRED_API_KEY:
        print("ERROR: FRED_API_KEY environment variable not set.")
        sys.exit(1)

    # ---- Download all data once ----
    print("Step 1/3  Download price data from yfinance...")
    all_tickers = [SPY_TICKER, VIX_TICKER, VIX3M_TICKER] + SECTOR_ETFS + DEEP_VALUE_UNIVERSE
    prices = download_prices(all_tickers, HISTORY_START)

    print("\nStep 2/3  Download credit and rate data from FRED...")
    hy_s  = download_fred(FRED_HY,  HISTORY_START)
    ig_s  = download_fred(FRED_IG,  HISTORY_START)
    y10_s = download_fred(FRED_10Y, HISTORY_START)
    y2_s  = download_fred(FRED_2Y,  HISTORY_START)
    print(f"  HY OAS:  {len(hy_s)} obs  last={hy_s.index[-1].date()}")
    print(f"  IG OAS:  {len(ig_s)} obs  last={ig_s.index[-1].date()}")
    print(f"  10yr:    {len(y10_s)} obs")
    print(f"  2yr:     {len(y2_s)} obs")

    # ---- Generate per date ----
    print(f"\nStep 3/3  Generating {len(targets)} snapshots...\n")
    written = 0
    errors  = []

    for i, d in enumerate(targets, 1):
        snap_date = d.isoformat()
        print(f"  [{i:4d}/{len(targets)}] {snap_date} ... ", end="", flush=True)

        try:
            snap = generate_snapshot(snap_date, prices, hy_s, ig_s, y10_s, y2_s)
        except Exception as e:
            print(f"ERROR: {e}")
            errors.append(snap_date)
            continue

        if snap is None:
            print("skipped (insufficient history)")
            continue

        out = os.path.join(SNAPSHOTS_DIR, f"{snap_date}.json")
        with open(out, "w") as f:
            json.dump(snap, f, indent=2)

        label = snap["quadrant"]["label"]
        score = snap["macro_score"]["score"]
        print(f"{label} | {score}")
        written += 1

    print(f"\n{'='*60}")
    print(f"  Written : {written}")
    print(f"  Errors  : {len(errors)}")
    if errors:
        print(f"  Failed  : {errors}")
    print(f"{'='*60}\n")

    # Non-zero exit if any errors, so the Action can flag it
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
