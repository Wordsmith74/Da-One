"""
core/intelligence/umpire_intel.py

Home-plate umpire adjustment for MLB pitcher strikeout props and full-game
totals (Fix #5 — "situational-spot database").

What this actually delivers vs. what it doesn't
-------------------------------------------------
Researched before writing this: there is no free public API for umpire
*zone tendency* (how wide/tight a given umpire's zone runs, K-rate boost,
etc.) — the sites that track this (UmpScorecards and similar) publish it
on their own site/Patreon, not through an API, and the other "umpire
report" sites that exist are subscription content. Scraping a third-party
site without a documented API is fragile and outside what this module
should silently attempt.

What IS free and real: MLB Stats API exposes umpire *assignments* --
which real person is working a given game, home plate specifically --
via the boxscore endpoint's "officials" list and the /jobs?jobType=UMP
crew-assignment endpoint. That's the same no-key MLB Stats API this
codebase already uses everywhere else (pitcher_intel.py, bullpen_intel.py,
get_mlb_game_totals_history in data/fetch.py).

So this module is split into two honest halves:
  1. get_home_plate_umpire() — REAL, fetches the actual assigned umpire's
     name for a game from MLB Stats API. Verify the exact response shape
     against a live call before trusting in production (written without
     network access to confirm the officials[] entry format/labels match
     what's assumed below — same caveat bullpen_intel.py already carries
     for its heuristic sections).
  2. UMPIRE_ZONE_TENDENCY — a lookup table from umpire name to a
     league_mean_adjustment, DELIBERATELY EMPTY at ship time. Filling it
     with invented numbers would be worse than leaving it empty: a
     fabricated "this umpire runs a wide zone" adjustment would silently
     bias the model on made-up data. Populate this table yourself from a
     real source (UmpScorecards, or MLB's own ABS-era CSAA stats once
     public) on whatever cadence you refresh other reference data at --
     see the docstring on the table below for the exact format expected.

Until the table is populated, get_umpire_intel() always returns a zero
adjustment — this module is currently pure infrastructure, not yet a
live signal. That's an honest state, not a bug.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

import requests

logger = logging.getLogger("betting_bot")

STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"

# ---------------------------------------------------------------------------
# Zone tendency lookup — POPULATE FROM A REAL SOURCE, DO NOT INVENT VALUES.
#
# Format: "Full Umpire Name" -> league_mean_adjustment (float, runs for
# totals / fractional strikeouts for K props -- same unit pitcher_intel.py
# and bullpen_intel.py already use, so it composes additively with them).
# A positive adjustment = tighter zone / hitter-friendly (raises the total,
# lowers expected Ks). A negative adjustment = wider zone / pitcher-friendly
# (lowers the total, raises expected Ks). Pick the sign convention to match
# whatever source you're pulling from -- UmpScorecards' "zone size" is
# already expressed relative to 1.000 league average, so
# adj = (1.000 - zone_size) * SOME_SCALE is a reasonable way to derive
# this table from their public scorecards without needing an API.
#
# Example format (NOT real data -- delete once you've populated this for
# real, this is here only to show the expected shape):
#   "Example Ump": -0.3,
# ---------------------------------------------------------------------------
UMPIRE_ZONE_TENDENCY: dict[str, float] = {}

# Below this many umpire-identified games in the table, don't trust it to
# mean anything even if someone starts populating it with a handful of
# entries -- matches the spirit of MIN_SAMPLE_SIZE gates used elsewhere in
# this codebase (threshold_optimizer.py, probability_calibrator.py).
MIN_TABLE_ENTRIES_TO_TRUST = 15


@dataclass
class UmpireIntelFactor:
    umpire_name: Optional[str] = None
    league_mean_adjustment: float = 0.0
    factor_text: str = ""


def get_home_plate_umpire(game_pk: int) -> Optional[str]:
    """
    Look up the assigned home-plate umpire's name for a specific game via
    MLB Stats API's boxscore endpoint.

    game_pk: MLB Stats API's own game primary key (NOT the balldontlie or
    Odds-API game id used elsewhere in this codebase -- callers need to
    resolve gamePk first, e.g. via the /schedule endpoint the rest of this
    codebase already calls for real game-total history).

    Returns the umpire's full name, or None on any failure (missing data,
    network error, unexpected response shape) -- never guesses.

    NOTE: written without network access to confirm the exact response
    shape. Per MLB Stats API's documented boxscore structure, "officials"
    is a list of {"official": {"id", "fullName"}, "officialType": str}
    dicts, where officialType is expected to be "Home Plate" for the HP
    umpire -- spot-check this against a live response before trusting it
    in production, same caveat bullpen_intel.py already documents for its
    own heuristic sections.
    """
    try:
        r = requests.get(f"{STATSAPI_BASE}/game/{game_pk}/boxscore", timeout=10)
        r.raise_for_status()
        officials = r.json().get("officials", [])
        for o in officials:
            official_type = (o.get("officialType") or "").strip().lower()
            if official_type == "home plate":
                name = (o.get("official") or {}).get("fullName")
                return name
        logger.debug(
            f"[umpire_intel] No 'Home Plate' official found in boxscore "
            f"officials for gamePk={game_pk} (found types: "
            f"{[o.get('officialType') for o in officials]})"
        )
        return None
    except Exception as exc:
        logger.debug(f"[umpire_intel] boxscore fetch failed for gamePk={game_pk}: {exc}")
        return None


def resolve_game_pk(home_abbr: str, away_abbr: str, game_date: date) -> Optional[int]:
    """
    Resolve MLB Stats API's own gamePk for a matchup/date, so callers that
    only have team abbreviations (the convention every other intelligence
    module in this codebase uses -- pitcher_intel, bullpen_intel) don't
    need to separately track the Odds-API game id vs. the MLB Stats API id.

    Matches on the home team abbreviation appearing in the schedule for
    that date; returns None if zero or more than one plausible match is
    found (ambiguous -- e.g. a doubleheader -- rather than guessing which
    game).
    """
    try:
        r = requests.get(
            f"{STATSAPI_BASE}/schedule",
            params={"sportId": 1, "date": game_date.isoformat()},
            timeout=10,
        )
        r.raise_for_status()
        candidates = []
        for day in r.json().get("dates", []):
            for g in day.get("games", []):
                home_team = (g.get("teams", {}).get("home", {}).get("team", {}) or {})
                away_team = (g.get("teams", {}).get("away", {}).get("team", {}) or {})
                home_abbrev = (home_team.get("abbreviation") or "").upper()
                away_abbrev = (away_team.get("abbreviation") or "").upper()
                if home_abbrev == home_abbr.upper() and away_abbrev == away_abbr.upper():
                    candidates.append(g.get("gamePk"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            logger.debug(
                f"[umpire_intel] Ambiguous schedule match for {away_abbr}@{home_abbr} "
                f"on {game_date} ({len(candidates)} games -- doubleheader?) -- skipping."
            )
        return None
    except Exception as exc:
        logger.debug(f"[umpire_intel] schedule lookup failed for {away_abbr}@{home_abbr}: {exc}")
        return None


def get_umpire_intel_for_matchup(
    home_abbr: str, away_abbr: str, game_date: Optional[date] = None,
) -> UmpireIntelFactor:
    """
    Convenience wrapper matching pitcher_intel.get_pitcher_intel() /
    bullpen_intel.get_bullpen_intel()'s (home_abbr, away_abbr, game_date)
    calling convention -- resolves gamePk internally so callers don't need
    to plumb it through separately. Never raises; zero adjustment on any
    failure at any step.
    """
    if game_date is None:
        game_date = date.today()
    game_pk = resolve_game_pk(home_abbr, away_abbr, game_date)
    return get_umpire_intel(game_pk)


def get_umpire_intel(game_pk: Optional[int]) -> UmpireIntelFactor:
    """
    Public entry point, mirrors get_pitcher_intel()/get_bullpen_intel()'s
    calling convention: never raises, always returns a usable factor with
    zero adjustment on any failure or missing data.

    Currently returns zero adjustment for every umpire until
    UMPIRE_ZONE_TENDENCY is populated from a real source (see module
    docstring) -- the umpire_name field is still filled in for real when
    resolvable, so the factor_text at least surfaces "who's behind the
    plate today" even before the tendency table has real numbers in it.
    """
    if game_pk is None:
        return UmpireIntelFactor()

    name = get_home_plate_umpire(game_pk)
    if name is None:
        return UmpireIntelFactor()

    if len(UMPIRE_ZONE_TENDENCY) < MIN_TABLE_ENTRIES_TO_TRUST:
        # Table not populated (or not populated enough) yet -- identify the
        # umpire for visibility, but don't act on an untrusted/empty table.
        return UmpireIntelFactor(
            umpire_name=name,
            league_mean_adjustment=0.0,
            factor_text=f"HP umpire: {name} (no zone-tendency data loaded)",
        )

    adj = UMPIRE_ZONE_TENDENCY.get(name)
    if adj is None:
        return UmpireIntelFactor(
            umpire_name=name,
            league_mean_adjustment=0.0,
            factor_text=f"HP umpire: {name} (no profile in tendency table)",
        )

    direction = "hitter-friendly" if adj > 0 else "pitcher-friendly"
    return UmpireIntelFactor(
        umpire_name=name,
        league_mean_adjustment=adj,
        factor_text=f"HP umpire: {name} ({direction}, {adj:+.2f} adj)",
    )
