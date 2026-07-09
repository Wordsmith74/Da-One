"""Grade MLB full-game run line (spread) picks.

Previously unsupported: fetch_actual_stat() had no branch for `run_line`
at all, so every run_line pick permanently failed grading with
`unsupported_market:run_line` regardless of how long it sat in
pick_history.jsonl. See core/mlb.py's get_run_line_margin() for the
paired raw-fact fetcher.
"""
from __future__ import annotations

from typing import Optional

from .grading_utils import GradeOutcome, settle


def grade(pick: dict, actual_margin: Optional[float]) -> Optional[GradeOutcome]:
    """
    actual_margin: run differential for the picked side (side_runs -
    opponent_runs), as returned by mlb.get_run_line_margin() -- NOT yet
    combined with the line. This function applies pick["pick_time_line"]
    itself, same division of labor as totals.py (fetcher returns a raw
    fact, grader applies the bet's terms).

    pick_time_line is signed from the picked side's perspective, e.g.
    -1.5 for a favorite that must win by 2+ to cover, +1.5 for an
    underdog that covers with a loss by 1 or any win. Covers when
    actual_margin + line > 0.
    """
    if actual_margin is None:
        return None  # postponed/suspended before final, or side unresolved

    line = pick["pick_time_line"]
    odds = pick.get("pick_time_odds")
    stake = (pick.get("stake_pct_bankroll") or 1.0) / 100.0  # e.g. 14.94 -> 0.1494

    covering_margin = actual_margin + line
    if covering_margin == 0:
        result = "push"  # only reachable on whole-number lines; MLB run
                          # lines are almost always .5, kept for safety
    elif covering_margin > 0:
        result = "win"
    else:
        result = "loss"

    # See strikeouts.py for why odds can be None -- same handling here.
    if odds is None:
        return GradeOutcome(actual_result=result, actual_stat=actual_margin, profit_units=None, roi=None)

    return settle(result, odds, stake, actual_margin)
