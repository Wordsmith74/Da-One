"""Grade moneyline picks (MLB and WNBA)."""
from __future__ import annotations

from typing import Optional

from .grading_utils import GradeOutcome, settle


def grade(pick: dict, actual_winner: Optional[str]) -> Optional[GradeOutcome]:
    """
    pick: expects `team` to hold the team name/abbreviation the pick backed.

    NOTE: unlike the prop graders, `team` was NOT found in the real
    pick_history.jsonl sample I checked (it only contained prop rows, no
    moneyline rows) -- so this field name is UNVERIFIED. If moneyline picks
    actually store the backed side differently (e.g. derived from `side` +
    `matchup` instead of a dedicated `team` field), this will KeyError.
    Send a real moneyline row from pick_history.jsonl to confirm/fix.

    actual_winner: winning team's name, or None if the game wasn't decided
          (postponed, suspended and not resumed, etc.).

    NOTE: team-name matching here is a simple case-insensitive substring
    check. TODO: swap in your canonical team-name normalizer if `team`
    stores abbreviations (e.g. "NYY") rather than full names -- otherwise a
    Yankees pick could silently fail to match "New York Yankees" from the
    stat_fetchers API response. (This also matters for the abbreviated
    "SD@CHC_MLB_..." matchup format seen elsewhere in the data -- home/away
    there come through as bare codes like "SD"/"CHC", not full names.)
    """
    if actual_winner is None:
        return None

    odds = pick.get("pick_time_odds")
    stake = (pick.get("stake_pct_bankroll") or 1.0) / 100.0  # e.g. 14.94 -> 0.1494 (14.94% of bankroll)
    picked_team = pick["team"].lower()

    if picked_team in actual_winner.lower() or actual_winner.lower() in picked_team:
        result = "win"
    else:
        result = "loss"

    # See strikeouts.py for why odds can be None -- same handling here.
    if odds is None:
        return GradeOutcome(actual_result=result, actual_stat=None, profit_units=None, roi=None)

    return settle(result, odds, stake, actual_stat=None)
