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
     league_mean_adjustment. AS OF THIS UPDATE, this is no longer
     hardcoded empty: core/intelligence/umpire_zone_compute.py self-computes
     it from real Statcast called-ball/called-strike data joined to Bill
     Petti's public umpire-ID file (see that module's docstring for the
     full derivation and its own honesty caveats). This module lazily loads
     that computed table on first use via _get_zone_tendency_table() rather
     than eagerly at import time, since building it (when the cache is
     stale) triggers a real, possibly-slow Statcast pull -- import time is
     the wrong place for that.

If umpire_zone_compute's table is empty (pybaseball not installed, network
unavailable, or the Petti join failed), get_umpire_intel() still returns a
zero adjustment exactly as before -- degrading to the prior honest-infra
state rather than raising, matching this module's original fail-safe
contract.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

import requests

from core.mlb_team_ids import MLB_TEAM_IDS

logger = logging.getLogger("betting_bot")

STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"

# ---------------------------------------------------------------------------
# Zone tendency lookup.
#
# Format: "Full Umpire Name" -> league_mean_adjustment (float, runs for
# totals / fractional strikeouts for K props -- same unit pitcher_intel.py
# and bullpen_intel.py already use, so it composes additively with them).
# A positive adjustment = tighter zone / hitter-friendly (raises the total,
# lowers expected Ks). A negative adjustment = wider zone / pitcher-friendly
# (lowers the total, raises expected Ks).
#
# Populated lazily from core.intelligence.umpire_zone_compute's self-
# computed, cached table (see module docstring above) -- NOT hardcoded here
# anymore. Falls back to an empty dict (same as before) if that module
# can't produce real data, so behavior degrades honestly rather than
# fabricating values.
# ---------------------------------------------------------------------------
_ZONE_TABLE_CACHE: Optional[dict[str, float]] = None


def _get_zone_tendency_table() -> dict[str, float]:
    """
    Lazily loads and flattens umpire_zone_compute's
    {name: {"league_mean_adjustment": float, ...}} table down to the
    {name: float} shape this module's own get_umpire_intel() consumes.
    Cached at process level after first successful load so repeated calls
    within one pipeline run don't re-touch disk/network.
    """
    global _ZONE_TABLE_CACHE
    if _ZONE_TABLE_CACHE is not None:
        return _ZONE_TABLE_CACHE

    try:
        from core.intelligence.umpire_zone_compute import load_or_build_umpire_zone_tendency_table
        raw = load_or_build_umpire_zone_tendency_table()
        flattened = {
            name: entry.get("league_mean_adjustment", 0.0)
            for name, entry in raw.items()
            if "league_mean_adjustment" in entry
        }
    except Exception as exc:
        logger.debug(f"[umpire_intel] zone-tendency table load failed: {exc!r}")
        flattened = {}

    _ZONE_TABLE_CACHE = flattened
    return flattened


# Backwards-compatible module attribute (some callers/tests may still refer
# to UMPIRE_ZONE_TENDENCY directly) -- kept as a live property-like lookup
# by resolving it once at first access via get_umpire_intel() below rather
# than at import time. If you need the raw dict directly, call
# _get_zone_tendency_table() instead of relying on this name being populated
# at import.
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

    Matches on team id (via MLB_TEAM_IDS) appearing in the schedule for
    that date; returns None if zero or more than one plausible match is
    found (ambiguous -- e.g. a doubleheader -- rather than guessing which
    game).

    NOTE: this used to match on team.abbreviation without requesting
    hydrate=team -- the schedule endpoint's default team sub-object is
    just {id, name, link} without that hydrate, so abbreviation was always
    "" and this always returned None. Matching on id (present either way,
    and not a fragile string-format dependency) fixes that; see the
    identical bug/fix in data/mlb_probable_pitchers.get_probable_pitchers.
    """
    home_id = MLB_TEAM_IDS.get(home_abbr.upper())
    away_id = MLB_TEAM_IDS.get(away_abbr.upper())
    if home_id is None or away_id is None:
        logger.debug(
            f"[umpire_intel] unrecognized abbreviation(s) home={home_abbr!r} "
            f"away={away_abbr!r} -- not in MLB_TEAM_IDS."
        )
        return None

    try:
        r = requests.get(
            f"{STATSAPI_BASE}/schedule",
            params={"sportId": 1, "date": game_date.isoformat(), "hydrate": "team"},
            timeout=10,
        )
        r.raise_for_status()
        candidates = []
        for day in r.json().get("dates", []):
            for g in day.get("games", []):
                home_team = (g.get("teams", {}).get("home", {}).get("team", {}) or {})
                away_team = (g.get("teams", {}).get("away", {}).get("team", {}) or {})
                if home_team.get("id") == home_id and away_team.get("id") == away_id:
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

    Zone-tendency table is loaded lazily from
    core.intelligence.umpire_zone_compute (see module docstring) -- returns
    zero adjustment if that table is empty or too small to trust, same
    fail-safe posture as before this was wired up. The umpire_name field is
    still filled in for real whenever resolvable, so the factor_text at
    least surfaces "who's behind the plate today" even with no tendency
    data loaded.
    """
    if game_pk is None:
        return UmpireIntelFactor()

    name = get_home_plate_umpire(game_pk)
    if name is None:
        return UmpireIntelFactor()

    table = _get_zone_tendency_table()
    if len(table) < MIN_TABLE_ENTRIES_TO_TRUST:
        # Table not populated (or not populated enough) yet -- identify the
        # umpire for visibility, but don't act on an untrusted/empty table.
        return UmpireIntelFactor(
            umpire_name=name,
            league_mean_adjustment=0.0,
            factor_text=f"HP umpire: {name} (no zone-tendency data loaded)",
        )

    adj = table.get(name)
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


def get_umpire_zone_size(game_pk: Optional[int]) -> tuple[Optional[str], Optional[str]]:
    """
    NRFI-tier convenience wrapper: returns (umpire_zone_size, umpire_volatility)
    in the exact "wide"/"narrow"/"average" vocabulary
    models.nrfi_handicapper.environment_tier_multiplier expects, derived
    from the same computed zone-tendency table get_umpire_intel() uses.

    umpire_volatility is NOT computed by umpire_zone_compute (that would
    require pitch-to-pitch call-consistency variance, a separate
    computation this module doesn't attempt yet) -- always returns None for
    it, which environment_tier_multiplier already treats as "no adjustment"
    for that component. Don't fabricate a volatility read to fill the gap.
    """
    if game_pk is None:
        return None, None
    name = get_home_plate_umpire(game_pk)
    if name is None:
        return None, None
    table = _get_zone_tendency_table()
    if len(table) < MIN_TABLE_ENTRIES_TO_TRUST or name not in table:
        return None, None

    adj = table[name]
    # Reuse the same sign convention as get_umpire_intel(): negative adj =
    # pitcher-friendly/wider zone, positive = hitter-friendly/narrower zone.
    # A dead-zone band around 0 avoids over-labeling a near-neutral umpire.
    if adj <= -0.15:
        return "wide", None
    if adj >= 0.15:
        return "narrow", None
    return "average", None
