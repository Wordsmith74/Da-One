"""Grade game totals and F5 (first-5-innings, MLB-specific) totals."""
from __future__ import annotations

from typing import Optional

from .grading_utils import GradeOutcome, settle


def grade(pick: dict, actual_stat: Optional[float]) -> Optional[GradeOutcome]:
    """
    Handles both `market == "game_total"` and `market == "f5_total"` --
    the stat_fetcher call site decides which total to fetch; this function
    just compares actual_stat to the line, same logic either way.
    """
    if actual_stat is None:
        return None  # e.g. game postponed / suspended before completion

    line = pick["pick_time_line"]
    side = pick["side"].lower()
    odds = pick.get("pick_time_odds")
    stake = (pick.get("stake_pct_bankroll") or 1.0) / 100.0  # e.g. 14.94 -> 0.1494 (14.94% of bankroll)

    if actual_stat == line:
        result = "push"
    elif (side == "over" and actual_stat > line) or (side == "under" and actual_stat < line):
        result = "win"
    else:
        result = "loss"

    # See strikeouts.py for why odds can be None -- same handling here.
    if odds is None:
        return GradeOutcome(actual_result=result, actual_stat=actual_stat, profit_units=None, roi=None)

    return settle(result, odds, stake, actual_stat)
