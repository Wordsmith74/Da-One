"""Grade NRFI (No Run First Inning) / YRFI (Yes Run First Inning) picks.

The market is always a fixed 0.5 line -- no push is possible, unlike
totals.py's F5/game total grading. `pick["market"]` (normalized) is
"nrfi" or "yrfi"; `actual_first_inning_runs` comes from
core.mlb.get_first_inning_runs().
"""
from __future__ import annotations

from typing import Optional

from .grading_utils import GradeOutcome, settle


def grade(pick: dict, actual_first_inning_runs: Optional[float]) -> Optional[GradeOutcome]:
    """
    pick["market"] (already normalized by market_normalized() upstream, per
    core/historical_grader.py's dispatch) is expected to be exactly "nrfi"
    or "yrfi" -- these are the internal keys game_markets.py._process_nrfi_yrfi
    writes onto the candidate ("market_key") and that market_gate.py /
    market_governance.py already key off of.
    """
    if actual_first_inning_runs is None:
        return None  # e.g. game postponed before the 1st inning completed

    side = (pick.get("market") or pick.get("side") or "").strip().lower()
    odds = pick.get("pick_time_odds")
    stake = (pick.get("stake_pct_bankroll") or 1.0) / 100.0

    ran_scored = actual_first_inning_runs > 0

    if side == "nrfi":
        result = "loss" if ran_scored else "win"
    elif side == "yrfi":
        result = "win" if ran_scored else "loss"
    else:
        return None  # unrecognized side -- don't silently guess

    # See strikeouts.py for why odds can be None -- same handling here.
    if odds is None:
        return GradeOutcome(actual_result=result, actual_stat=actual_first_inning_runs, profit_units=None, roi=None)

    return settle(result, odds, stake, actual_first_inning_runs)
