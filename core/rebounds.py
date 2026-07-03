"""Grade player rebound props (WNBA)."""
from __future__ import annotations

from typing import Optional

from ..grading_utils import GradeOutcome, settle


def grade(pick: dict, actual_stat: Optional[float]) -> Optional[GradeOutcome]:
    if actual_stat is None:
        return None  # e.g. player DNP -- treat as void upstream, not push

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
