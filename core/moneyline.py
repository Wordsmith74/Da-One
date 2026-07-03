"""Grade moneyline picks (MLB and WNBA)."""
from __future__ import annotations

from typing import Optional

from .grading_utils import GradeOutcome, settle


def grade(pick: dict, actual_winner: Optional[str]) -> Optional[GradeOutcome]:
    """
    pick: expects `team` to hold the team name/abbreviation the pick backed
          (per the Bet dataclass -- moneyline picks don't use `direction`
          the way over/under props do).
    actual_winner: winning team's name, or None if the game wasn't decided
          (postponed, suspended and not resumed, etc.).

    NOTE: team-name matching here is a simple case-insensitive substring
    check. TODO: swap in your canonical team-name normalizer if `team`
    stores abbreviations (e.g. "NYY") rather than full names -- otherwise a
    Yankees pick could silently fail to match "New York Yankees" from the
    stat_fetchers API response.
    """
    if actual_winner is None:
        return None

    odds = pick["american_odds"]
    stake = pick.get("stake_units", 1.0)
    picked_team = pick["team"].lower()

    if picked_team in actual_winner.lower() or actual_winner.lower() in picked_team:
        result = "win"
    else:
        result = "loss"

    return settle(result, odds, stake, actual_stat=None)
