"""
bucket_analysis.py -- Reusable edge/confidence bucket win-rate report.

Answers the recurring question "where are the winning buckets?" for any
sport/market combination, across BOTH data sources:

  1. output/shadow_log_graded.jsonl -- every candidate the pipeline ever
     scored, published or not, once graded. Use this to see whether real
     edge exists just outside the current gatekeeper thresholds, since it
     has far more volume than published picks alone.
  2. output/pick_history.jsonl -- only the picks that were actually
     published. This is the ground truth for "how are we actually doing,"
     but volume is much lower and improves over time.

Buckets are edge_pct x confidence, so noise vs. signal is visible at a
glance. Every bucket also reports n, wins, losses, win_pct, and how far
win_pct sits from the breakeven line for a given odds price (-110 by
default), so a bucket with a great win% but tiny n doesn't get mistaken
for a real edge.

Run as a script:
    python -m core.bucket_analysis --sport WNBA --market player_assists player_rebounds
    python -m core.bucket_analysis --sport MLB --market pitcher_strikeouts --source pick_history
    python -m core.bucket_analysis --sport WNBA --market player_rebounds --edge-bins 0,4,8,1000 --conf-bins 0,65,75,101

Can also be imported and called directly:
    from core.bucket_analysis import run_report
    run_report(sport="WNBA", markets=["player_assists", "player_rebounds"])
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from .grading_utils import read_jsonl

DEFAULT_SHADOW_LOG_PATH = "output/shadow_log_graded.jsonl"
DEFAULT_PICK_HISTORY_PATH = "output/pick_history.jsonl"

DEFAULT_EDGE_BINS = [0, 2, 4, 6, 8, 10, 15, 20, 1000]
DEFAULT_CONF_BINS = [0, 60, 65, 70, 75, 80, 85, 90, 101]

# Breakeven win% needed to profit at a given American odds price.
# -110 (standard juice) needs 52.38%.
DEFAULT_ODDS = -110


def _breakeven_pct(odds: int) -> float:
    if odds < 0:
        return -odds / (-odds + 100) * 100
    return 100 / (odds + 100) * 100


def _market_of(rec: dict[str, Any]) -> Optional[str]:
    """
    shadow_log_graded.jsonl records carry the market in `_market`.
    pick_history.jsonl records carry it in `market`.
    """
    return rec.get("_market") or rec.get("market")


def _load_records(
    source: str,
    sport: str,
    markets: list[str],
    shadow_log_path: str,
    pick_history_path: str,
) -> list[dict[str, Any]]:
    path = shadow_log_path if source == "shadow_log" else pick_history_path
    if not Path(path).exists():
        return []
    recs = read_jsonl(path)
    sport_up = sport.upper()
    market_set = {m.lower() for m in markets} if markets else None
    out = []
    for r in recs:
        if str(r.get("sport", "")).upper() != sport_up:
            continue
        if r.get("actual_result") not in ("win", "loss"):
            continue  # drop pushes/ungraded -- can't compute win rate off them
        mkt = _market_of(r)
        if market_set and (mkt is None or mkt.lower() not in market_set):
            continue
        out.append(r)
    return out


def _bucket_1d(
    recs: list[dict[str, Any]],
    keyfn: Callable[[dict[str, Any]], Optional[float]],
    bins: list[float],
) -> dict[tuple[float, float], list[int]]:
    buckets: dict[tuple[float, float], list[int]] = defaultdict(lambda: [0, 0])  # [wins, n]
    for r in recs:
        v = keyfn(r)
        if v is None:
            continue
        for lo, hi in zip(bins[:-1], bins[1:]):
            if lo <= v < hi:
                buckets[(lo, hi)][1] += 1
                if r["actual_result"] == "win":
                    buckets[(lo, hi)][0] += 1
                break
    return buckets


def _bucket_2d(
    recs: list[dict[str, Any]],
    edge_bins: list[float],
    conf_bins: list[float],
) -> dict[tuple[tuple[float, float], tuple[float, float]], list[int]]:
    edge_ranges = list(zip(edge_bins[:-1], edge_bins[1:]))
    conf_ranges = list(zip(conf_bins[:-1], conf_bins[1:]))
    grid: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for r in recs:
        e, c = r.get("edge_pct"), r.get("confidence")
        if e is None or c is None:
            continue
        eb = next((b for b in edge_ranges if b[0] <= e < b[1]), None)
        cb = next((b for b in conf_ranges if b[0] <= c < b[1]), None)
        if eb is None or cb is None:
            continue
        grid[(eb, cb)][1] += 1
        if r["actual_result"] == "win":
            grid[(eb, cb)][0] += 1
    return grid


def _print_1d(buckets: dict, label: str, breakeven: float, min_n: int) -> None:
    print(f"\n--- {label} ---")
    print(f"{'range':<16}{'n':<6}{'wins':<6}{'win%':<8}{'vs breakeven'}")
    for k in sorted(buckets):
        w, n = buckets[k]
        if n == 0:
            continue
        wp = round(100 * w / n, 1)
        flag = "" if n >= min_n else "  (n<%d, low confidence)" % min_n
        delta = round(wp - breakeven, 1)
        sign = "+" if delta >= 0 else ""
        print(f"{k[0]:>6}-{k[1]:<9}{n:<6}{w:<6}{wp:<8}{sign}{delta}pt{flag}")


def _print_2d(grid: dict, label: str, breakeven: float, min_n: int) -> None:
    print(f"\n--- {label} (edge x confidence) ---")
    print(f"{'edge':<12}{'conf':<12}{'n':<6}{'wins':<6}{'win%':<8}{'vs breakeven'}")
    for k in sorted(grid):
        eb, cb = k
        w, n = grid[k]
        if n == 0:
            continue
        wp = round(100 * w / n, 1)
        flag = "" if n >= min_n else "  (n<%d, low confidence)" % min_n
        delta = round(wp - breakeven, 1)
        sign = "+" if delta >= 0 else ""
        print(f"{str(eb):<12}{str(cb):<12}{n:<6}{w:<6}{wp:<8}{sign}{delta}pt{flag}")


def run_report(
    sport: str,
    markets: Optional[list[str]] = None,
    source: str = "shadow_log",
    edge_bins: Optional[list[float]] = None,
    conf_bins: Optional[list[float]] = None,
    odds: int = DEFAULT_ODDS,
    min_n: int = 20,
    shadow_log_path: str = DEFAULT_SHADOW_LOG_PATH,
    pick_history_path: str = DEFAULT_PICK_HISTORY_PATH,
    per_market: bool = True,
) -> None:
    """
    Print a bucketed win-rate report for `sport` (+ optional `markets`
    filter) pulled from either the shadow log (all candidates, published
    or not -- more volume) or pick_history (published only -- ground
    truth but lower volume).

    `min_n` flags buckets below this sample size as low-confidence so a
    lucky 3-for-3 doesn't get mistaken for an edge.
    """
    edge_bins = edge_bins or DEFAULT_EDGE_BINS
    conf_bins = conf_bins or DEFAULT_CONF_BINS
    markets = markets or []
    breakeven = round(_breakeven_pct(odds), 2)

    recs = _load_records(source, sport, markets, shadow_log_path, pick_history_path)
    mkt_label = "+".join(markets) if markets else "ALL"
    print(f"\n{'='*70}")
    print(f"{sport.upper()} / {mkt_label}  |  source={source}  |  n_graded={len(recs)}"
          f"  |  breakeven={breakeven}% (odds {odds})")
    print(f"{'='*70}")

    if not recs:
        print("No graded records found for this sport/market/source combination.")
        return

    if source == "pick_history":
        # Every record in pick_history.jsonl was published by definition --
        # there's no `published` field on these records to check.
        print(f"published={len(recs)}  (pick_history.jsonl only contains published picks)")
    else:
        published_n = sum(1 for r in recs if r.get("published"))
        print(f"published={published_n}  rejected/unpublished={len(recs) - published_n}")

    _print_1d(_bucket_1d(recs, lambda r: r.get("edge_pct"), edge_bins),
              f"{mkt_label} by EDGE%", breakeven, min_n)
    _print_1d(_bucket_1d(recs, lambda r: r.get("confidence"), conf_bins),
              f"{mkt_label} by CONFIDENCE", breakeven, min_n)
    _print_2d(_bucket_2d(recs, edge_bins, conf_bins),
              f"{mkt_label} combined", breakeven, min_n)

    if per_market and len(markets) > 1:
        for mkt in markets:
            sub = [r for r in recs if (_market_of(r) or "").lower() == mkt.lower()]
            if not sub:
                continue
            print(f"\n{'-'*70}")
            print(f"  breakdown: {mkt}  (n={len(sub)})")
            _print_1d(_bucket_1d(sub, lambda r: r.get("edge_pct"), edge_bins),
                      f"{mkt} by EDGE%", breakeven, min_n)
            _print_1d(_bucket_1d(sub, lambda r: r.get("confidence"), conf_bins),
                      f"{mkt} by CONFIDENCE", breakeven, min_n)


def _parse_bins(s: str) -> list[float]:
    return [float(x) for x in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sport", required=True, help="e.g. WNBA, MLB")
    ap.add_argument("--market", nargs="*", default=[], help="one or more market keys, e.g. player_assists player_rebounds")
    ap.add_argument("--source", choices=["shadow_log", "pick_history"], default="shadow_log")
    ap.add_argument("--edge-bins", type=_parse_bins, default=None, help="comma-separated, e.g. 0,4,8,1000")
    ap.add_argument("--conf-bins", type=_parse_bins, default=None, help="comma-separated, e.g. 0,65,75,101")
    ap.add_argument("--odds", type=int, default=DEFAULT_ODDS, help="American odds price for breakeven calc, default -110")
    ap.add_argument("--min-n", type=int, default=20, help="flag buckets below this sample size")
    ap.add_argument("--shadow-log-path", default=DEFAULT_SHADOW_LOG_PATH)
    ap.add_argument("--pick-history-path", default=DEFAULT_PICK_HISTORY_PATH)
    args = ap.parse_args()

    run_report(
        sport=args.sport,
        markets=args.market,
        source=args.source,
        edge_bins=args.edge_bins,
        conf_bins=args.conf_bins,
        odds=args.odds,
        min_n=args.min_n,
        shadow_log_path=args.shadow_log_path,
        pick_history_path=args.pick_history_path,
    )


if __name__ == "__main__":
    main()
