"""Grade moneyline picks (MLB and WNBA)."""
from __future__ import annotations

import re
from typing import Optional

from .grading_utils import GradeOutcome, settle

# Strips a trailing "_SPORT" and/or "_YYYY-MM-DD" off the home side of a
# matchup string, same shapes core/historical_grader.py's parse_matchup()
# already handles defensively (this module doesn't have access to that
# function's parsed output at the call site, so it re-derives just enough
# to identify the backed team -- see grade()'s docstring for why).
_DATE_SUFFIX_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})$")
_SPORT_SUFFIX_RE = re.compile(r"_(MLB|NBA|WNBA)$", re.IGNORECASE)


def _picked_team_from_matchup(pick: dict) -> Optional[str]:
    """
    Real pick_history.jsonl moneyline rows (confirmed live) have no `team`
    field at all -- the backed side is `side` ("home"/"away") plus the
    `matchup` string, e.g.:
        {"matchup": "GSV@ATL_WNBA_2026-07-04", "side": "home", ...}
    which backs the home team, "ATL".

    Falls back to a `team` field if one is ever present (kept for
    forward-compatibility / defense in depth), then to the away/home slice
    of `matchup` for whichever `side` says. Returns None (not "loss") if
    it can't be determined -- an unparseable/missing side should leave the
    pick ungraded, not silently score it wrong.
    """
    if pick.get("team"):
        return str(pick["team"]).strip().lower()

    side = (pick.get("side") or "").strip().lower()
    matchup = pick.get("matchup") or ""
    if side not in ("home", "away") or "@" not in matchup:
        return None

    away_raw, home_raw = matchup.split("@", 1)
    home_raw = _DATE_SUFFIX_RE.sub("", home_raw)
    home_raw = _SPORT_SUFFIX_RE.sub("", home_raw)

    picked_raw = home_raw if side == "home" else away_raw
    picked = picked_raw.replace("_", " ").strip().lower()
    return picked or None


def grade(pick: dict, actual_winner: Optional[str]) -> Optional[GradeOutcome]:
    """
    pick: backed team is derived via _picked_team_from_matchup() -- see its
    docstring for the real (confirmed-live) schema this handles.

    actual_winner: winning team's name, or None if the game wasn't decided
          (postponed, suspended and not resumed, etc.).

    Team-name matching is a case-insensitive substring check, so a bare
    abbreviation like "ATL" matches a full display name like "Atlanta
    Dream" from either direction. This is intentionally loose (no
    canonical team-name normalizer wired in yet) -- if a false match ever
    turns up in grading_rejects.jsonl / a wrong-looking grade, that's the
    place to tighten it.
    """
    if actual_winner is None:
        return None

    picked_team = _picked_team_from_matchup(pick)
    if picked_team is None:
        return None  # can't determine the backed side -- leave ungraded, don't guess

    odds = pick.get("pick_time_odds")
    stake = (pick.get("stake_pct_bankroll") or 1.0) / 100.0  # e.g. 14.94 -> 0.1494 (14.94% of bankroll)

    actual_lower = actual_winner.lower()
    if picked_team in actual_lower or actual_lower in picked_team:
        result = "win"
    else:
        result = "loss"

    # See strikeouts.py for why odds can be None -- same handling here.
    if odds is None:
        return GradeOutcome(actual_result=result, actual_stat=None, profit_units=None, roi=None)

    return settle(result, odds, stake, actual_stat=None)
