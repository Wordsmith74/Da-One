"""
data/mlb_probable_pitchers.py

Resolves the two starting pitchers for a given matchup/date via MLB Stats
API (same free, no-key endpoint family already used by umpire_intel.py,
pitcher_intel.py, weather_intel.py), and the career-starts count that
models.nrfi_handicapper.nrfi_reliability_gate needs to decide whether a
first-inning read is trustworthy.

This is the missing link between core/game_markets.py (which only has team
names/abbreviations from the odds API) and every per-pitcher NRFI tier
(data/statcast_first_inning.py, the reliability gate, data/statcast_lineup_platoon.py's
handedness input).

Fail-safe contract: every public function returns None (or a ProbablePitcher
with is_known=False) on any failure -- missing probable pitcher, network
error, ambiguous schedule match -- never guesses. Callers already have to
handle "reliability gate fails closed on unknown data" per
nrfi_reliability_gate's own docstring, so an unresolved pitcher naturally
routes into that same fail-closed path rather than needing special-case
handling here.

Written without live network access in this sandbox -- the MLB Stats API
schedule/hydrate shapes below follow the same documented conventions
umpire_intel.py and pitcher_intel.py already rely on elsewhere in this repo;
spot-check field names against a live response before trusting in
production, same caveat those modules already carry.
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
_TIMEOUT = 10

# Process-level cache -- a probable-pitcher lookup for today's slate is the
# same answer for every market (moneyline, NRFI, K props, ...) that needs
# it, so don't re-hit the schedule endpoint per market per game.
_PROBABLE_CACHE: dict[str, "MatchupPitchers"] = {}
_CAREER_STARTS_CACHE: dict[int, Optional[int]] = {}


@dataclass
class ProbablePitcher:
    pitcher_id: Optional[int] = None
    full_name: Optional[str] = None
    throws: Optional[str] = None          # "L" or "R"
    career_starts: Optional[int] = None
    is_known: bool = False
    # Reliability-gate flags -- see models.nrfi_handicapper.nrfi_reliability_gate.
    # Each is a REAL, independently-sourced signal (see the functions below
    # that populate them); none are guessed when the underlying data is
    # unavailable -- they default to False, meaning "not flagged," not
    # "confirmed normal." A False here is "no disqualifying signal found,"
    # not a positive confirmation the start is routine.
    is_debut: bool = False
    is_injury_return_start: bool = False
    is_opener_or_bullpen_game: bool = False


@dataclass
class MatchupPitchers:
    home: ProbablePitcher
    away: ProbablePitcher


def _empty_matchup() -> MatchupPitchers:
    return MatchupPitchers(home=ProbablePitcher(), away=ProbablePitcher())


def get_career_starts(pitcher_id: int) -> Optional[int]:
    """
    Career MLB starts (gamesStarted, career pitching totals) for the
    reliability gate's ~50-start threshold. None on any failure -- the gate
    treats None as "fail closed," which is the correct behavior for an
    unresolved pitcher (see nrfi_reliability_gate docstring), not a bug to
    work around here.
    """
    if pitcher_id in _CAREER_STARTS_CACHE:
        return _CAREER_STARTS_CACHE[pitcher_id]

    result: Optional[int] = None
    try:
        r = requests.get(
            f"{STATSAPI_BASE}/people/{pitcher_id}/stats",
            params={"stats": "career", "group": "pitching"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        stats = r.json().get("stats", [])
        for block in stats:
            for split in block.get("splits", []):
                gs = (split.get("stat") or {}).get("gamesStarted")
                if gs is not None:
                    result = int(gs)
                    break
            if result is not None:
                break
    except Exception as exc:
        logger.debug(f"[mlb_probable_pitchers] career-starts lookup failed for {pitcher_id}: {exc!r}")
        result = None

    _CAREER_STARTS_CACHE[pitcher_id] = result
    return result


def _get_throws(pitcher_id: int) -> Optional[str]:
    try:
        r = requests.get(f"{STATSAPI_BASE}/people/{pitcher_id}", timeout=_TIMEOUT)
        r.raise_for_status()
        people = r.json().get("people", [])
        if not people:
            return None
        code = ((people[0].get("pitchHand") or {}).get("code") or "").upper()
        return code or None
    except Exception as exc:
        logger.debug(f"[mlb_probable_pitchers] throws lookup failed for {pitcher_id}: {exc!r}")
        return None


def get_mlb_debut_date(pitcher_id: int) -> Optional[str]:
    """
    Real career-debut date (MLB Stats API's own `mlbDebutDate` field on the
    /people/{id} endpoint), ISO format ("YYYY-MM-DD") or None if unresolvable.
    This is the direct, unambiguous signal for is_debut -- a TRUE career
    debut, not a proxy. (The reference doc's "season debut" language also
    covers a pitcher's first start of a given season after a healthy
    offseason, which this field alone doesn't capture -- that case is
    already covered independently by the low-career-starts branch of
    nrfi_reliability_gate itself, so it's not double-modeled here.)
    """
    try:
        r = requests.get(f"{STATSAPI_BASE}/people/{pitcher_id}", timeout=_TIMEOUT)
        r.raise_for_status()
        people = r.json().get("people", [])
        if not people:
            return None
        return people[0].get("mlbDebutDate")
    except Exception as exc:
        logger.debug(f"[mlb_probable_pitchers] debut-date lookup failed for {pitcher_id}: {exc!r}")
        return None


def get_career_pitching_role_totals(pitcher_id: int) -> tuple[Optional[int], Optional[int]]:
    """
    Returns (career_games_started, career_games_pitched) from MLB Stats
    API's career pitching totals -- the two real numbers
    is_opener_or_bullpen_game is derived from. (None, None) on failure.
    """
    try:
        r = requests.get(
            f"{STATSAPI_BASE}/people/{pitcher_id}/stats",
            params={"stats": "career", "group": "pitching"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        for block in r.json().get("stats", []):
            for split in block.get("splits", []):
                stat = split.get("stat") or {}
                gs = stat.get("gamesStarted")
                gp = stat.get("gamesPitched")
                if gs is not None and gp is not None:
                    return int(gs), int(gp)
        return None, None
    except Exception as exc:
        logger.debug(f"[mlb_probable_pitchers] career role-totals lookup failed for {pitcher_id}: {exc!r}")
        return None, None


# Below this career-games-pitched sample, a low starts/appearances ratio is
# too noisy to call "primarily a reliever" (e.g. a rookie's first few MLB
# outings). Matches this repo's general small-sample-skepticism convention.
_MIN_CAREER_APPEARANCES_FOR_ROLE_READ = 10
# A starts/appearances ratio below this, for a pitcher named as TODAY's
# probable starter, is the real signal an opener/bullpen game is planned --
# a bona fide starter's career ratio runs close to 1.0; a reliever's runs
# close to 0.0.
_OPENER_ROLE_RATIO_THRESHOLD = 0.30


def _infer_opener_or_bullpen_game(pitcher_id: int) -> bool:
    """
    Real, data-backed proxy for the reference doc's "opener/bullpen game"
    exclusion: a pitcher whose CAREER games-started/games-pitched ratio is
    low despite being listed as today's probable starter is very likely
    part of a planned opener/bullpen-game usage, not a conventional start.
    This is a proxy (not a direct "today is an opener game" flag, which
    MLB Stats API doesn't publish) -- it can occasionally misfire on a
    long-time reliever making a rare, fully conventional spot start, which
    is an acceptably conservative failure mode (excluding a possibly-fine
    game) given the doc's own bias toward not trusting unstable situations.
    """
    starts, appearances = get_career_pitching_role_totals(pitcher_id)
    if starts is None or appearances is None or appearances < _MIN_CAREER_APPEARANCES_FOR_ROLE_READ:
        return False
    return (starts / appearances) < _OPENER_ROLE_RATIO_THRESHOLD


def _infer_injury_return_start(pitcher_name: Optional[str], team_abbr: str) -> bool:
    """
    Real, keyword-derived proxy for "coming off an injury": checks whether
    this pitcher currently appears by name in
    data.rotowire_injuries_mlb.get_recent_mlb_injuries(team_abbr) -- RotoWire's
    free injury-NEWS feed (headlines like IL moves, activations, rehab
    updates). A name match means there's active, recent injury news about
    this pitcher, which is a reasonable (if coarse) signal they may be
    making their first start back -- it is NOT a confirmed "first start off
    IL" flag (that would require IL-activation-transaction data this repo
    doesn't have wired in), and can both over- and under-flag: a pitcher
    with lingering minor news who never actually missed a start would be
    over-flagged; a pitcher who returned from the IL with no recent news
    item (e.g. the news cycle already moved on) would be under-flagged.
    Given the reliability gate's fail-closed design, over-flagging (missing
    a bet you could have made) is the safer direction of error here.
    Returns False (not fabricated True) on any fetch/parse failure or
    missing name -- same fail-safe posture as every other module here.
    """
    if not pitcher_name:
        return False
    try:
        from data.rotowire_injuries_mlb import get_recent_mlb_injuries
    except Exception as exc:
        logger.debug(f"[mlb_probable_pitchers] rotowire_injuries_mlb import failed: {exc!r}")
        return False

    try:
        data = get_recent_mlb_injuries(team_abbr)
        if not data:
            return False
        target = pitcher_name.strip().lower()
        for inj in data.get("injuries", []):
            listed_name = ((inj.get("athlete") or {}).get("shortName") or "").strip().lower()
            if listed_name and (listed_name == target or listed_name in target or target in listed_name):
                return True
        return False
    except Exception as exc:
        logger.debug(f"[mlb_probable_pitchers] injury-return check failed for {pitcher_name}: {exc!r}")
        return False


def get_probable_pitchers(
    home_abbr: str, away_abbr: str, game_date: Optional[date] = None,
) -> MatchupPitchers:
    """
    Public entry point: resolves both starters for a matchup/date via MLB
    Stats API's schedule hydrate=probablePitcher, then fills handedness and
    career starts for each. Cached per (home_abbr, away_abbr, date) so a
    slate with multiple markets per game only pays this cost once.

    Ambiguous schedule matches (doubleheaders) are skipped, matching
    umpire_intel.resolve_game_pk's own "don't guess" behavior.
    """
    game_date = game_date or date.today()
    cache_key = f"{home_abbr.upper()}:{away_abbr.upper()}:{game_date.isoformat()}"
    if cache_key in _PROBABLE_CACHE:
        return _PROBABLE_CACHE[cache_key]

    result = _empty_matchup()
    home_id = MLB_TEAM_IDS.get(home_abbr.upper())
    away_id = MLB_TEAM_IDS.get(away_abbr.upper())
    if home_id is None or away_id is None:
        logger.debug(
            f"[mlb_probable_pitchers] unrecognized abbreviation(s) "
            f"home={home_abbr!r} away={away_abbr!r} -- not in MLB_TEAM_IDS."
        )
        _PROBABLE_CACHE[cache_key] = result
        return result

    try:
        r = requests.get(
            f"{STATSAPI_BASE}/schedule",
            params={
                "sportId": 1, "date": game_date.isoformat(),
                # "team" must be in hydrate for the schedule response to
                # include team.abbreviation at all -- without it the team
                # sub-object is just {id, name, link}. We match on team.id
                # instead (via MLB_TEAM_IDS above) since that's present
                # either way and isn't a fragile string-format dependency,
                # but keep "team" hydrated too for any other consumer of
                # this response shape that might want the name/abbrev.
                "hydrate": "team,probablePitcher",
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        matches = []
        for day in r.json().get("dates", []):
            for g in day.get("games", []):
                teams = g.get("teams", {})
                h = (teams.get("home", {}).get("team", {}) or {})
                a = (teams.get("away", {}).get("team", {}) or {})
                if h.get("id") == home_id and a.get("id") == away_id:
                    matches.append((teams.get("home", {}), teams.get("away", {})))

        if len(matches) == 1:
            home_side, away_side = matches[0]
            for side, key, team_abbr in (
                (home_side, "home", home_abbr), (away_side, "away", away_abbr),
            ):
                pp = (side.get("probablePitcher") or {})
                pid = pp.get("id")
                if pid is None:
                    continue
                pid_int = int(pid)
                debut_date = get_mlb_debut_date(pid_int)
                pitcher = ProbablePitcher(
                    pitcher_id=pid_int,
                    full_name=pp.get("fullName"),
                    throws=_get_throws(pid_int),
                    career_starts=get_career_starts(pid_int),
                    is_known=True,
                    is_debut=(debut_date == game_date.isoformat()),
                    is_injury_return_start=_infer_injury_return_start(pp.get("fullName"), team_abbr),
                    is_opener_or_bullpen_game=_infer_opener_or_bullpen_game(pid_int),
                )
                setattr(result, key, pitcher)
        elif len(matches) > 1:
            logger.debug(
                f"[mlb_probable_pitchers] ambiguous schedule match for "
                f"{away_abbr}@{home_abbr} on {game_date} ({len(matches)} games) -- skipping."
            )
    except Exception as exc:
        logger.debug(
            f"[mlb_probable_pitchers] probable-pitcher lookup failed for "
            f"{away_abbr}@{home_abbr} {game_date}: {exc!r}"
        )
        result = _empty_matchup()

    _PROBABLE_CACHE[cache_key] = result
    return result
