"""
Generate calibration/summary reports from graded picks in pick_history.jsonl.

Bucket boundaries below mirror the real gatekeeper thresholds in
decision_gatekeeper.py (SPORT_TIER_THRESHOLDS): confidence_score is on a
0-100 scale (not 0-1) and edge_percentage is already a percentage (e.g.
7.79 means 7.79%, not 0.0779). MLB/WNBA share the same tier bar
(Nuke edge>=3.5% conf>=85, Diamond edge>=2.0% conf>=78, Gold edge>=1.0%
conf>=68); NBA is slightly higher. These buckets roughly straddle those
lines so you can see performance just below/above each gate.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

# confidence_score is 0-100
CONFIDENCE_BUCKETS = [
    (0.0, 68.0, "below Gold floor (<68)"),
    (68.0, 78.0, "Gold band (68-78)"),
    (78.0, 85.0, "Diamond band (78-85)"),
    (85.0, 100.01, "Nuke band (85+)"),
]

# edge_percentage is already a percentage, e.g. 3.5 means 3.5%
EDGE_BUCKETS = [
    (0.0, 1.0, "below Gold floor (<1%)"),
    (1.0, 2.0, "Gold band (1-2%)"),
    (2.0, 3.5, "Diamond band (2-3.5%)"),
    (3.5, 999.0, "Nuke band (3.5%+)"),
]


def _bucket(value: float, buckets: list[tuple[float, float, str]]) -> str:
    for lo, hi, label in buckets:
        if lo <= value < hi:
            return label
    return "unbucketed"


def _wlp_line(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    wins = losses = pushes = 0
    profit = 0.0
    staked = 0.0
    for r in records:
        result = r.get("actual_result")
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        elif result == "push":
            pushes += 1
        else:
            continue
        profit += r.get("profit_units") or 0.0
        staked += (r.get("stake_pct_bankroll") or 1.0) / 100.0  # match graders: % bankroll -> fraction
    roi = profit / staked if staked else 0.0
    n = wins + losses + pushes
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_pct": round(wins / (wins + losses), 4) if (wins + losses) else None,
        "profit_units": round(profit, 2),
        "roi": round(roi, 4),
    }


def _group_report(records: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        try:
            groups[key_fn(r)].append(r)
        except (KeyError, TypeError):
            groups["unknown"].append(r)
    return {k: _wlp_line(v) for k, v in groups.items()}


def generate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in records if r.get("actual_result") in ("win", "loss", "push")]

    overall = _wlp_line(graded)
    by_market = _group_report(graded, lambda r: r.get("market", "unknown"))
    # `tier` is computed by decision_gatekeeper.py but never persisted to
    # pick_history.jsonl -- every row will bucket to "unknown" here until
    # that's added upstream. Not a grading bug, out of scope for this fix.
    by_tier = _group_report(graded, lambda r: r.get("tier", "unknown"))  # "Nuke"/"Diamond"/"Gold Standard"
    by_confidence = _group_report(
        graded, lambda r: _bucket(float(r.get("confidence", -1)), CONFIDENCE_BUCKETS)
    )
    by_edge = _group_report(
        graded, lambda r: _bucket(float(r.get("edge_pct", -1)), EDGE_BUCKETS)
    )

    ranked_markets = sorted(
        by_market.items(), key=lambda kv: kv[1]["roi"] if kv[1]["n"] >= 5 else -999, reverse=True
    )
    top_markets = ranked_markets[:5]
    bottom_markets = ranked_markets[-5:]

    return {
        "overall": overall,
        "roi_by_market": by_market,
        "roi_by_confidence_bucket": by_confidence,
        "roi_by_edge_bucket": by_edge,
        "roi_by_tier": by_tier,
        "top_markets": top_markets,
        "bottom_markets": bottom_markets,
        "ungraded_count": len(records) - len(graded),
    }


def print_summary(summary: dict[str, Any]) -> None:
    o = summary["overall"]
    print("=== Overall ===")
    print(
        f"N={o['n']}  W-L-P: {o['wins']}-{o['losses']}-{o['pushes']}  "
        f"Win%={o['win_pct']}  Profit={o['profit_units']}u  ROI={o['roi']:.2%}"
    )

    for section_title, key in (
        ("ROI by Market", "roi_by_market"),
        ("ROI by Confidence Bucket", "roi_by_confidence_bucket"),
        ("ROI by Edge Bucket", "roi_by_edge_bucket"),
        ("ROI by Tier", "roi_by_tier"),
    ):
        print(f"\n=== {section_title} ===")
        for group, stats in summary[key].items():
            print(
                f"{group:>20}: N={stats['n']:>4}  ROI={stats['roi']:.2%}  "
                f"Profit={stats['profit_units']}u  Win%={stats['win_pct']}"
            )

    print("\n=== Top Markets (min 5 picks) ===")
    for name, stats in summary["top_markets"]:
        print(f"{name:>20}: ROI={stats['roi']:.2%}  N={stats['n']}")

    print("\n=== Bottom Markets (min 5 picks) ===")
    for name, stats in summary["bottom_markets"]:
        print(f"{name:>20}: ROI={stats['roi']:.2%}  N={stats['n']}")

    if summary["ungraded_count"]:
        print(f"\n({summary['ungraded_count']} picks still ungraded)")
