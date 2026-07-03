"""Grade pitcher strikeout props (MLB)."""
from __future__ import annotations

from typing import Optional

from .grading_utils import GradeOutcome, settle


def grade(pick: dict, actual_stat: Optional[float]) -> Optional[GradeOutcome]:
    """
    pick: expects keys `pick_time_line` (float), `side` ("over"/"under"),
          `pick_time_odds`, `stake_pct_bankroll`.
    actual_stat: strikeouts recorded, or None if unavailable (e.g. pitcher
          didn't play -- caller should treat that as "no grade" / void,
          not push, unless your book's rules say otherwise).
    """
    if actual_stat is None:
        return None  # ungraded -- e.g. rainout, early pull, DFA, etc.

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

    # pick_time_odds is missing on most rows since ~2026-06-30 (upstream
    # capture gap, not a grading bug) -- win/loss/push can still be
    # recorded even when odds are unknown, we just can't settle profit/roi.
    # TODO: confirm against grading_utils.settle()/GradeOutcome that this
    # is the right shape once that file is available -- constructing
    # GradeOutcome directly here is a best guess at its fields.
    if odds is None:
        return GradeOutcome(actual_result=result, actual_stat=actual_stat, profit_units=None, roi=None)

    return settle(result, odds, stake, actual_stat)
