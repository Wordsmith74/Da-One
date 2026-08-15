"""
data/statcast_lineup_platoon.py

Fills the "Top 4" lineup-quality gap in models.nrfi_handicapper's Tier 2:
platoon-split OBP/ISO/wRC+-proxy for a team's first four batters, specific
to the OPPOSING starter's handedness, per the reference doc's "Use OBP,
ISO, and wRC+ specifically vs. the starter's handedness (platoon splits,
not season totals)."

Two responsibilities, kept honest and separate:

1. get_confirmed_top4(home_abbr, away_abbr, game_date) -- the actual posted
   batting order (first 4 spots) from MLB Stats API's boxscore endpoint.
   Lineups typically post 2-4 hours before first pitch (doc, section 6) --
   before that, this returns an empty/unconfirmed result rather than
   guessing from a "probable" lineup, which is exactly the kind of stale
   input the doc's "Common Pitfalls" section warns against ("a late scratch
   can flip a game's true probability"). missing_key_bat is set when a
   posted lineup differs from the team's typical top-4 (see
   _TYPICAL_TOP4_CACHE) -- a coarse but real, non-fabricated signal.

2. get_top4_platoon_splits(batter_ids, pitcher_throws, season) -- real
   season-to-date split stats (OBP, SLG->ISO) via MLB Stats API's own
   statSplits endpoint (sitCodes vl/vr = vs LHP/RHP), which is documented,
   free, and doesn't require pybaseball. wRC+ itself isn't published by
   MLB Stats API, so a wRC+-PROXY is derived here from OBP+ISO relative to
   league average -- clearly labeled as a proxy, not real wRC+, per this
   codebase's rule against silently substituting an approximation for the
   real thing (module docstring, and see models/nrfi_handicapper.py's own
   warning against exactly this kind of unlabeled substitution).

Fail-safe: every function returns None/empty on any failure. Written
without live network access in this sandbox -- MLB Stats API's statSplits
sitCodes (vl/vr) and boxscore battingOrder conventions are documented
elsewhere but unverified live here; spot-check before trusting in
production, same caveat this repo's other intelligence modules carry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import requests

from core.mlb_team_ids import MLB_TEAM_IDS

logger = logging.getLogger("betting_bot")

STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
_TIMEOUT = 10

# League-average anchors for the wRC+ proxy and defaults, roughly current
# era (per the reference doc's own league_obp/league_iso defaults already
# used in models.nrfi_handicapper.lineup_top4_tier_multiplier -- kept
# consistent with those rather than re-deriving separately).
_LEAGUE_OBP = 0.320
_LEAGUE_ISO = 0.150
_LEAGUE_WRC_PLUS = 100.0

# Remembers each team's most-recently-seen top-4 (by player id) so a newly
# posted lineup can be diffed against it to flag missing_key_bat. Process-
# level only -- intentionally not persisted, since "typical" should reflect
# recent games, not a stale multi-season memory.
_TYPICAL_TOP4_CACHE: dict[str, list[int]] = {}


@dataclass
class Top4Lineup:
    batter_ids: list[int] = field(default_factory=list)
    confirmed: bool = False
    missing_key_bat: bool = False


@dataclass
class Top4PlatoonSplits:
    top4_obp_vs_hand: Optional[float] = None
    top4_iso_vs_hand: Optional[float] = None
    top4_wrc_plus_vs_hand: Optional[float] = None
    batters_with_data: int = 0


def _resolve_game_pk(home_abbr: str, away_abbr: str, game_date: date) -> Optional[int]:
    """
    NOTE: previously matched on team.abbreviation without hydrate=team --
    the schedule endpoint's default team sub-object is just {id, name,
    link} without that hydrate, so abbreviation was always "" and this
    always returned None. Matching on id (via MLB_TEAM_IDS, present
    either way) fixes that; see the identical bug/fix in
    data/mlb_probable_pitchers.get_probable_pitchers and
    core/intelligence/umpire_intel.resolve_game_pk.
    """
    home_id = MLB_TEAM_IDS.get(home_abbr.upper())
    away_id = MLB_TEAM_IDS.get(away_abbr.upper())
    if home_id is None or away_id is None:
        logger.debug(
            f"[statcast_lineup_platoon] unrecognized abbreviation(s) "
            f"home={home_abbr!r} away={away_abbr!r} -- not in MLB_TEAM_IDS."
        )
        return None

    try:
        r = requests.get(
            f"{STATSAPI_BASE}/schedule",
            params={"sportId": 1, "date": game_date.isoformat(), "hydrate": "team"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        matches = []
        for day in r.json().get("dates", []):
            for g in day.get("games", []):
                h = (g.get("teams", {}).get("home", {}).get("team", {}) or {})
                a = (g.get("teams", {}).get("away", {}).get("team", {}) or {})
                if h.get("id") == home_id and a.get("id") == away_id:
                    matches.append(g.get("gamePk"))
        return matches[0] if len(matches) == 1 else None
    except Exception as exc:
        logger.debug(f"[statcast_lineup_platoon] schedule lookup failed: {exc!r}")
        return None


def get_confirmed_top4(
    home_abbr: str, away_abbr: str, game_date: Optional[date] = None,
) -> dict[str, Top4Lineup]:
    """
    Returns {"home": Top4Lineup, "away": Top4Lineup} from the posted
    batting order, or empty/unconfirmed Top4Lineup objects if lineups
    aren't posted yet for this game. Never guesses a lineup from a
    "probable" list -- an unconfirmed top-4 is exactly what
    nrfi_handicapper's lineup_top4_tier_multiplier is designed to receive
    as "no data" (all-None kwargs -> neutral 1.0), which is the honest
    outcome here, not a bug to patch around.
    """
    result = {"home": Top4Lineup(), "away": Top4Lineup()}
    game_date = game_date or date.today()
    game_pk = _resolve_game_pk(home_abbr, away_abbr, game_date)
    if game_pk is None:
        return result

    try:
        r = requests.get(f"{STATSAPI_BASE}/game/{game_pk}/boxscore", timeout=_TIMEOUT)
        r.raise_for_status()
        teams = r.json().get("teams", {})
        for side, abbr in (("home", home_abbr), ("away", away_abbr)):
            team_block = teams.get(side, {})
            batting_order = team_block.get("battingOrder") or []
            if not batting_order:
                continue
            # battingOrder is the list of player IDs in the confirmed
            # starting lineup order (MLB Stats API convention) -- take the
            # first four.
            top4_ids = [int(pid) for pid in batting_order[:4]]

            cache_key = abbr.upper()
            typical = _TYPICAL_TOP4_CACHE.get(cache_key)
            missing = bool(typical) and len(set(typical) & set(top4_ids)) < len(typical) - 1
            _TYPICAL_TOP4_CACHE[cache_key] = top4_ids

            result[side] = Top4Lineup(
                batter_ids=top4_ids, confirmed=True, missing_key_bat=missing,
            )
    except Exception as exc:
        logger.debug(f"[statcast_lineup_platoon] boxscore lookup failed for gamePk={game_pk}: {exc!r}")

    return result


def _get_batter_split(batter_id: int, sit_code: str, season: int) -> Optional[dict]:
    try:
        r = requests.get(
            f"{STATSAPI_BASE}/people/{batter_id}/stats",
            params={
                "stats": "statSplits", "group": "hitting",
                "sitCodes": sit_code, "season": season,
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        for block in r.json().get("stats", []):
            for split in block.get("splits", []):
                stat = split.get("stat") or {}
                obp = stat.get("obp")
                slg = stat.get("slg")
                avg = stat.get("avg")
                pa = stat.get("plateAppearances")
                if obp is not None and slg is not None and avg is not None:
                    return {
                        "obp": float(obp), "iso": float(slg) - float(avg),
                        "pa": int(pa) if pa is not None else 0,
                    }
        return None
    except Exception as exc:
        logger.debug(f"[statcast_lineup_platoon] split lookup failed for batter={batter_id}: {exc!r}")
        return None


def get_top4_platoon_splits(
    batter_ids: list[int], pitcher_throws: Optional[str], season: Optional[int] = None,
    min_pa_per_batter: int = 20,
) -> Top4PlatoonSplits:
    """
    Real season-to-date OBP/ISO vs. the given pitcher handedness for each
    batter, averaged (simple mean) across however many of the 4 batters
    have a real split with at least min_pa_per_batter plate appearances --
    a thin-PA platoon split (e.g. 5 PAs vs. LHP) is exactly the kind of
    noisy small sample the reference doc's "Reliability filter" section
    warns against, so those are excluded rather than diluting the average.

    wRC+ proxy: MLB Stats API doesn't publish wRC+. A rough proxy is
    derived as league_wrc_plus * (0.5*obp_ratio + 0.5*iso_ratio) --
    intentionally simple and clearly a proxy, not real park/league-adjusted
    wRC+. If you have a real wRC+ source (FanGraphs membership, Stathead),
    prefer wiring that in directly and pass top4_wrc_plus_vs_hand yourself.

    pitcher_throws: "L" or "R" (from data.mlb_probable_pitchers). Returns
    an all-None result if unknown -- platoon splits are meaningless without
    knowing which hand they're split against.
    """
    if not pitcher_throws or pitcher_throws.upper() not in ("L", "R"):
        return Top4PlatoonSplits()

    sit_code = "vl" if pitcher_throws.upper() == "L" else "vr"
    season = season or date.today().year

    obps, isos = [], []
    for bid in batter_ids[:4]:
        split = _get_batter_split(bid, sit_code, season)
        if split and split["pa"] >= min_pa_per_batter:
            obps.append(split["obp"])
            isos.append(split["iso"])

    if not obps:
        return Top4PlatoonSplits()

    top4_obp = sum(obps) / len(obps)
    top4_iso = sum(isos) / len(isos)
    obp_ratio = top4_obp / _LEAGUE_OBP if _LEAGUE_OBP else 1.0
    iso_ratio = top4_iso / _LEAGUE_ISO if _LEAGUE_ISO else 1.0
    wrc_plus_proxy = _LEAGUE_WRC_PLUS * (0.5 * obp_ratio + 0.5 * iso_ratio)

    return Top4PlatoonSplits(
        top4_obp_vs_hand=round(top4_obp, 4),
        top4_iso_vs_hand=round(top4_iso, 4),
        top4_wrc_plus_vs_hand=round(wrc_plus_proxy, 1),
        batters_with_data=len(obps),
    )
