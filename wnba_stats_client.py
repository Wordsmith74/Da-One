"""
wnba_stats_client.py

Free, keyless client for the OFFICIAL league stats API — stats.wnba.com.
This is the same backend engine that powers stats.nba.com, just scoped to
LeagueID="10" (WNBA). No API key, no subscription, no rate-limit tier walls
like balldontlie's free plan.

Why this exists
----------------
balldontlie's free WNBA tier only covers /players and /games — everything
else (player_stats, team_stats, standings, injuries) is paywalled behind
their $9.99/mo "All-Star" plan (see core/bdl_wnba.py). ESPN's public site
API is free but only exposes box scores / schedules in a fairly shallow
shape. stats.wnba.com gives full team + player game logs, season averages,
and box scores for free — it just requires browser-like headers, because
the league's CDN (Akamai) blocks requests that look like bare scripts.

Endpoints used
--------------
  leaguegamelog   — every team OR player game log for a season, in one call
  boxscoretraditionalv2 — full box score (incl. per-player minutes) for one game
  commonteamroster      — current roster for a team

IMPORTANT — header requirement
-------------------------------
stats.wnba.com (like stats.nba.com) will 403 a plain `requests`/`urllib`
call with default headers. You MUST send a realistic browser User-Agent
plus an Origin/Referer pointing at wnba.com or the CDN blocks it. The
headers below are believed-correct as of 2026-06 but COULD NOT BE LIVE
TESTED in this sandbox (no network egress here) — verify the first live
call and adjust _HEADERS if you get a 403.

Cache
-----
Process-level dict, same pattern as bdl_wnba.py / game_logs.py.

Fallback
--------
Every public function returns None / [] on any failure. Callers must
fall back to the next source in the waterfall (ESPN, then synthetic).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from typing import Any

logger = logging.getLogger("betting_bot")

_BASE = "https://stats.wnba.com/stats"

# stats.wnba.com / stats.nba.com CDN requires a browser-like header set.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.wnba.com",
    "Referer": "https://www.wnba.com/",
    "x-nba-stats-origin": "stats",        # mirrors stats.nba.com convention
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
}

_LEAGUE_ID  = "10"   # WNBA
_TIMEOUT    = 8       # seconds
_MIN_GAP    = 0.6     # be polite — this is a free public endpoint, not a paid API
_LAST_CALL  = 0.0

# ---------------------------------------------------------------------------
# Process-level caches
# ---------------------------------------------------------------------------

_TEAM_LOG_CACHE:   dict[int, list[dict]] = {}   # season -> rows
_PLAYER_LOG_CACHE: dict[int, list[dict]] = {}   # season -> rows
_ROSTER_CACHE:     dict[str, list[dict]] = {}   # team_abbr -> players


# ---------------------------------------------------------------------------
# Low-level GET
# ---------------------------------------------------------------------------

def _get(endpoint: str, params: dict[str, Any]) -> dict | None:
    global _LAST_CALL
    gap = time.monotonic() - _LAST_CALL
    if gap < _MIN_GAP:
        time.sleep(_MIN_GAP - gap)

    qs  = urllib.parse.urlencode(params)
    url = f"{_BASE}/{endpoint}?{qs}"
    req = urllib.request.Request(url, headers=_HEADERS)

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            _LAST_CALL = time.monotonic()
            raw = resp.read()
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        _LAST_CALL = time.monotonic()
        logger.warning(
            "[wnba_stats] HTTP %s on %s — if this is a 403, the CDN is "
            "blocking the header set; update _HEADERS in wnba_stats_client.py",
            exc.code, endpoint,
        )
        return None
    except Exception as exc:
        _LAST_CALL = time.monotonic()
        logger.debug("[wnba_stats] GET %s failed: %s", endpoint, exc)
        return None


def _rows_from_resultsets(data: dict, set_name: str | None = None) -> list[dict]:
    """
    stats.wnba.com responses use the classic NBA-stats shape:
      {"resultSets": [{"name": ..., "headers": [...], "rowSet": [[...], ...]}]}
    Converts the first matching resultSet into a list of header->value dicts.
    """
    if not data:
        return []
    result_sets = data.get("resultSets") or data.get("resultSet")
    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    if not result_sets:
        return []

    target = result_sets[0]
    if set_name:
        for rs in result_sets:
            if rs.get("name") == set_name:
                target = rs
                break

    headers = target.get("headers", [])
    rows    = target.get("rowSet", [])
    return [dict(zip(headers, row)) for row in rows]


# ---------------------------------------------------------------------------
# Team game logs  (replaces / supplements core.intelligence.game_logs ESPN)
# ---------------------------------------------------------------------------

def get_team_game_totals(team_abbr: str, season: int | None = None, n: int = 15) -> list[float] | None:
    """
    Return the last *n* completed game totals (PTS + opponent PTS) for
    *team_abbr*, sourced from stats.wnba.com's leaguegamelog (PlayerOrTeam=T).

    Mirrors core.intelligence.game_logs.get_team_game_totals()'s return shape
    so it can be dropped in as a primary source ahead of the ESPN fallback.
    """
    season = season or date.today().year
    rows = _team_game_log(season)
    if not rows:
        return None

    abbr = team_abbr.upper()
    team_games = [r for r in rows if (r.get("TEAM_ABBREVIATION") or "").upper() == abbr]
    if not team_games:
        logger.debug("[wnba_stats] no team game log rows for %s season=%s", abbr, season)
        return None

    # Each row only has the team's own score (PTS) and PLUS_MINUS, not the
    # opponent's score directly — but PTS - PLUS_MINUS recovers it cleanly
    # (PLUS_MINUS = team_pts - opp_pts for that game).
    totals: list[tuple[str, float]] = []
    for r in team_games:
        try:
            pts = float(r.get("PTS") or 0)
            pm  = float(r.get("PLUS_MINUS") or 0)
            opp_pts = pts - pm
            game_total = pts + opp_pts
            game_date = r.get("GAME_DATE") or ""
            totals.append((game_date, round(game_total, 1)))
        except (TypeError, ValueError):
            continue

    totals.sort(key=lambda t: t[0])  # chronological, most recent last
    values = [t[1] for t in totals][-n:]
    if not values:
        return None

    logger.debug(
        "[wnba_stats] %s: %d real game totals via stats.wnba.com, mean=%.1f",
        abbr, len(values), sum(values) / len(values),
    )
    return values


def _team_game_log(season: int) -> list[dict]:
    if season in _TEAM_LOG_CACHE:
        return _TEAM_LOG_CACHE[season]

    data = _get("leaguegamelog", {
        "LeagueID": _LEAGUE_ID,
        "Season": str(season),
        "SeasonType": "Regular Season",
        "PlayerOrTeam": "T",
        "Counter": "1000",
        "Direction": "DESC",
        "Sorter": "DATE",
    })
    rows = _rows_from_resultsets(data)
    _TEAM_LOG_CACHE[season] = rows
    return rows


# ---------------------------------------------------------------------------
# Player game logs  (replaces / supplements core.bdl_wnba for free-tier use)
# ---------------------------------------------------------------------------

def get_player_stats(player_name: str, season: int | None = None) -> dict[str, list[float]] | None:
    """
    Full pipeline: look up *player_name* in the league player game log and
    return {MIN, REB, AST, PTS} lists, most-recent-first — same return
    shape as core.bdl_wnba.get_player_stats() so callers can try this
    first (free, no tier wall) and fall back to BDL/ESPN after.
    """
    season = season or date.today().year
    rows = _player_game_log(season)
    if not rows:
        return None

    name_lower = player_name.strip().lower()
    player_rows = [
        r for r in rows
        if (r.get("PLAYER_NAME") or "").strip().lower() == name_lower
    ]
    if not player_rows:
        # loose match on last name as a fallback
        last = name_lower.split()[-1] if name_lower.split() else name_lower
        player_rows = [
            r for r in rows
            if last in (r.get("PLAYER_NAME") or "").strip().lower()
        ]
    if not player_rows:
        logger.debug("[wnba_stats] no player game log rows for '%s'", player_name)
        return None

    player_rows.sort(key=lambda r: r.get("GAME_DATE") or "", reverse=True)

    def _f(row: dict, key: str) -> float:
        try:
            return float(row.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    return {
        "MIN": [_f(r, "MIN") for r in player_rows],
        "REB": [_f(r, "REB") for r in player_rows],
        "AST": [_f(r, "AST") for r in player_rows],
        "PTS": [_f(r, "PTS") for r in player_rows],
    }


def _player_game_log(season: int) -> list[dict]:
    if season in _PLAYER_LOG_CACHE:
        return _PLAYER_LOG_CACHE[season]

    data = _get("leaguegamelog", {
        "LeagueID": _LEAGUE_ID,
        "Season": str(season),
        "SeasonType": "Regular Season",
        "PlayerOrTeam": "P",
        "Counter": "2000",
        "Direction": "DESC",
        "Sorter": "DATE",
    })
    rows = _rows_from_resultsets(data)
    _PLAYER_LOG_CACHE[season] = rows
    return rows


# ---------------------------------------------------------------------------
# Roster  (useful for resolving position / active status for lineup_intel)
# ---------------------------------------------------------------------------

_TEAM_ID_BY_ABBR: dict[str, str] = {
    # stats.wnba.com TeamID values (same numeric space WNBA uses internally;
    # verify against /stats/leaguegamelog TEAM_ID column on first live run —
    # these were not network-verified in this sandbox).
    "ATL": "1611661320", "CHI": "1611661313", "CON": "1611661323",
    "DAL": "1611661321", "IND": "1611661325", "LVA": "1611661319",
    "LAS": "1611661324", "MIN": "1611661324", "NYL": "1611661317",
    "PHX": "1611661322", "SEA": "1611661328", "WAS": "1611661329",
    "GSV": "1611661330", "POR": "1611661331", "TOR": "1611661332",
}


def get_team_roster(team_abbr: str, season: int | None = None) -> list[dict]:
    """
    Return current roster rows [{PLAYER, POSITION, ...}] for *team_abbr*.
    Free, no key. Used to enrich injury reports with position when the
    injury source (e.g. RotoWire scrape) doesn't include it.
    """
    abbr = team_abbr.upper()
    if abbr in _ROSTER_CACHE:
        return _ROSTER_CACHE[abbr]

    team_id = _TEAM_ID_BY_ABBR.get(abbr)
    if not team_id:
        return []

    season = season or date.today().year
    data = _get("commonteamroster", {
        "LeagueID": _LEAGUE_ID,
        "TeamID": team_id,
        "Season": str(season),
    })
    rows = _rows_from_resultsets(data, set_name="CommonTeamRoster")
    _ROSTER_CACHE[abbr] = rows
    return rows
