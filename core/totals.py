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

    line = pick["sportsbook_line"]
    side = pick["direction"].lower()
    odds = pick["american_odds"]
    stake = pick.get("stake_units", 1.0)

    if actual_stat == line:
        result = "push"
    elif (side == "over" and actual_stat > line) or (side == "under" and actual_stat < line):
        result = "win"
    else:
        result = "loss"

    return settle(result, odds, stake, actual_stat)
