"""Grade pitcher strikeout props (MLB)."""
from __future__ import annotations

from typing import Optional

from ..grading_utils import GradeOutcome, settle


def grade(pick: dict, actual_stat: Optional[float]) -> Optional[GradeOutcome]:
    """
    pick: expects keys `sportsbook_line` (float), `direction` ("over"/"under"),
          `american_odds`, `stake_units`.
    actual_stat: strikeouts recorded, or None if unavailable (e.g. pitcher
          didn't play -- caller should treat that as "no grade" / void,
          not push, unless your book's rules say otherwise).
    """
    if actual_stat is None:
        return None  # ungraded -- e.g. rainout, early pull, DFA, etc.

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
