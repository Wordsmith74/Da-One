"""
check_threshold_staleness.py
=============================
Standalone sanity check for the hard-coded pitcher_strikeouts Gold-tier
thresholds in core/decision_gatekeeper.py (_MARKET_TIER_THRESHOLDS_BY_SIDE).

Why this exists
----------------
Those thresholds are comments-with-numbers baked in from one-time snapshots
of output/shadow_log_graded.jsonl (e.g. "775-record pool, graded
2026-07-25"). Nothing re-checks them against fresh data, and nothing flags
when a cutoff's live sample size never grew past the tiny n it was
originally fit on -- which is indistinguishable, from inside the code, from
"still validated." Run this against the current graded log before trusting
(or re-deriving) those numbers.

Usage
-----
    python3 check_threshold_staleness.py

Exits non-zero if any tracked cutoff has drifted below breakeven or is
still sitting on a sample too small to trust (n < MIN_TRUSTED_N).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

GRADED_LOG = Path("output/shadow_log_graded.jsonl")
BREAKEVEN_WIN_PCT = 0.5238  # standard -110 breakeven
MIN_TRUSTED_N = 100         # below this, "validated" just means "not yet disproven"

# Mirror of core/decision_gatekeeper.py's _MARKET_TIER_THRESHOLDS_BY_SIDE
# ("pitcher_strikeouts", side) -> (min_edge_pct, min_confidence, claimed_win_pct, claimed_n)
# Update this table if the source thresholds change.
TRACKED_CUTOFFS = {
    ("pitcher_strikeouts", "over"):  (1.0, 60.0, 0.570, 135),
    ("pitcher_strikeouts", "under"): (3.0, 65.0, 0.656, 32),
}


def load_graded(market: str, side: str) -> list[dict]:
    rows = []
    if not GRADED_LOG.exists():
        return rows
    with GRADED_LOG.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("_market") == market and d.get("side") == side:
                rows.append(d)
    return rows


def evaluate(market: str, side: str, min_edge: float, min_conf: float) -> Counter:
    c = Counter()
    for d in load_graded(market, side):
        if d.get("actual_result") == "push":
            continue
        edge = d.get("edge_pct") or 0
        conf = d.get("confidence") or 0
        if edge >= min_edge and conf >= min_conf:
            c[d["actual_result"]] += 1
    return c


def main() -> int:
    problems = 0
    print(f"{'market/side':<28} {'cutoff':<18} {'n':>5} {'win%':>7} {'claimed':>9}  status")
    for (market, side), (min_edge, min_conf, claimed_wr, claimed_n) in TRACKED_CUTOFFS.items():
        c = evaluate(market, side, min_edge, min_conf)
        n = c["win"] + c["loss"]
        wr = c["win"] / n if n else 0.0
        cutoff = f"edge>={min_edge} conf>={min_conf}"
        flags = []
        if n < MIN_TRUSTED_N:
            flags.append(f"n={n} < {MIN_TRUSTED_N} (unvalidated small sample)")
        if wr < BREAKEVEN_WIN_PCT:
            flags.append(f"win% {wr:.1%} below breakeven {BREAKEVEN_WIN_PCT:.1%}")
        if n >= claimed_n and abs(wr - claimed_wr) > 0.05:
            flags.append(f"drifted from claimed {claimed_wr:.1%}")
        status = "OK" if not flags else "FLAG: " + "; ".join(flags)
        if flags:
            problems += 1
        print(f"{market+'/'+side:<28} {cutoff:<18} {n:>5} {wr:>6.1%} {claimed_wr:>8.1%}  {status}")

    if problems:
        print(f"\n{problems} tracked cutoff(s) need review before being trusted live.")
    else:
        print("\nAll tracked cutoffs still within tolerance.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
