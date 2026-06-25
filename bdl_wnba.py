"""
BallDontLie WNBA API client.

Provides per-player game logs and live player props for the WNBA
per-minute projection model.

Key design:
  - Targeted per-player fetches (search → stats) instead of bulk game scans
  - Process-level caches for player IDs and game log rows
  - 0.22 s inter-call sleep = ~4.5 req/sec (safely under 5/sec free-tier limit)
  - Graceful fallback when BALLDONTLIE_API_KEY is not set

Tier notes (as of 2026-06):
  - Free tier: /players (search), /games — works without subscription
  - All-Star ($9.99/sport/mo) and above: /player_stats, /player_season_stats,
    /odds/player_props, /standings
  - is_stats_available() returns False immediately after the first 401 on a
    stats endpoint so the caller can fall back without retrying.

API docs: https://www.balldontlie.io/openapi/wnba.yml
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_BDL_BASE       = "https://api.balldontlie.io"
_BDL_KEY        = os.environ.get("BALLDONTLIE_API_KEY", "")
_CURRENT_SEASON = 2026

# ── Process-level caches & tier state ────────────────────────────────────────
_PLAYER_ID_CACHE: dict[str, int | None] = {}  # lowercase full name → bdl player_id
_GAME_LOG_CACHE:  dict[tuple, list[dict]] = {} # (player_id, season) → game rows

_LAST_CALL_TS: float = 0.0
_MIN_CALL_GAP  = 0.22   # 4.5 req/sec max

# Set to True after first successful stats call; False after first 401 on stats.
# None = untested.  is_stats_available() returns True while None or True.
_STATS_TIER: bool | None = None


# ── Availability ─────────────────────────────────────────────────────────────

def is_available() -> bool:
    """True when BALLDONTLIE_API_KEY is configured."""
    return bool(_BDL_KEY)


def is_stats_available() -> bool:
    """
    False once we have confirmed that the current API key cannot access stats
    endpoints (401 returned).  True while untested or confirmed working.

    Use this to skip BDL stats lookups without wasting API calls when the
    account is on the free tier.
    """
    return _STATS_TIER is not False


# ── Low-level HTTP helper ─────────────────────────────────────────────────────

def _bdl_get(path: str, params: dict) -> dict | None:
    """
    Rate-limited GET to the BDL API.  Returns parsed JSON or None on error.

    Array params must be passed as {key: [v1, v2]} — urllib encodes them
    as key=v1&key=v2 which BDL accepts.
    """
    global _LAST_CALL_TS
    if not _BDL_KEY:
        return None

    # Enforce inter-call gap
    gap = time.monotonic() - _LAST_CALL_TS
    if gap < _MIN_CALL_GAP:
        time.sleep(_MIN_CALL_GAP - gap)

    qs  = urllib.parse.urlencode(params, doseq=True)
    url = f"{_BDL_BASE}{path}?{qs}"
    req = urllib.request.Request(url, headers={"Authorization": _BDL_KEY})

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            _LAST_CALL_TS = time.monotonic()
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        _LAST_CALL_TS = time.monotonic()
        if exc.code == 401:
            # Check if this is a stats/odds endpoint (free tier doesn't include them)
            _paywalled_prefixes = (
                "/wnba/v1/player_stats",
                "/wnba/v1/player_season",
                "/wnba/v1/team_stats",
                "/wnba/v1/team_season",
                "/wnba/v1/odds",
                "/wnba/v1/standings",
                "/wnba/v1/player_injuries",
            )
            if any(path.startswith(p) for p in _paywalled_prefixes):
                global _STATS_TIER
                _STATS_TIER = False
                logger.warning(
                    "[bdl_wnba] 401 Unauthorized on %s — API key is free tier; "
                    "WNBA stats/odds require All-Star plan ($9.99/sport/mo). "
                    "Falling back to ESPN for all WNBA player data.",
                    path,
                )
            else:
                logger.debug("[bdl_wnba] 401 on %s (non-stats endpoint)", path)
        else:
            logger.debug("[bdl_wnba] GET %s → HTTP %s", path, exc.code)
        return None
    except Exception as exc:
        _LAST_CALL_TS = time.monotonic()
        logger.debug(f"[bdl_wnba] GET {path} params={params} → {exc}")
        return None


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_min(min_str: str | None) -> float:
    """
    Parse BDL minutes string ('25:30', '25', '') to float minutes.
    '25:30' → 25.5, '25' → 25.0, '' → 0.0
    """
    if not min_str or min_str in ("", "0", "0:00", "00:00"):
        return 0.0
    if ":" in min_str:
        parts = min_str.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(min_str)
    except (ValueError, TypeError):
        return 0.0


# ── Player search ─────────────────────────────────────────────────────────────

def search_player(player_name: str) -> int | None:
    """
    Resolve a WNBA player name to a BDL player_id.

    Strategy:
      1. Exact first+last match against last-name search results
      2. Single-result last-name match (unambiguous)
      3. If still not found, try first-name search
    Results are process-cached to avoid repeat API calls.
    """
    name_lower = player_name.strip().lower()
    if name_lower in _PLAYER_ID_CACHE:
        return _PLAYER_ID_CACHE[name_lower]

    parts = name_lower.split()
    target_first = parts[0]  if len(parts) >= 1 else ""
    target_last  = parts[-1] if len(parts) >= 1 else ""

    def _try_find(result: dict | None) -> int | None:
        if not result:
            return None
        players = result.get("data", [])
        # Exact first+last match
        for p in players:
            fn = (p.get("first_name") or "").strip().lower()
            ln = (p.get("last_name")  or "").strip().lower()
            if fn == target_first and ln == target_last:
                return int(p["id"])
        # Unambiguous last-name match
        if len(players) == 1:
            ln = (players[0].get("last_name") or "").strip().lower()
            if ln == target_last:
                return int(players[0]["id"])
        return None

    # Try last name search first
    pid = _try_find(_bdl_get("/wnba/v1/players", {"search": target_last, "per_page": 20}))

    # Try first name if last name didn't resolve
    if pid is None and target_first != target_last:
        pid = _try_find(_bdl_get("/wnba/v1/players", {"search": target_first, "per_page": 20}))

    _PLAYER_ID_CACHE[name_lower] = pid
    if pid:
        logger.debug(f"[bdl_wnba] '{player_name}' → player_id={pid}")
    else:
        logger.debug(f"[bdl_wnba] player not found: '{player_name}'")
    return pid


# ── Game-log fetch ────────────────────────────────────────────────────────────

def get_player_game_logs(
    player_id: int,
    season:    int = _CURRENT_SEASON,
    per_page:  int = 30,
) -> list[dict]:
    """
    Fetch recent per-game stats for a BDL player_id.

    Returns a list of rows sorted most-recent-first:
      {min: float, reb: float, ast: float, date: str}

    DNP rows (min == 0) are excluded.
    Results are process-cached.
    """
    cache_key = (player_id, season)
    if cache_key in _GAME_LOG_CACHE:
        return _GAME_LOG_CACHE[cache_key]

    result = _bdl_get(
        "/wnba/v1/player_stats",
        {
            "player_ids[]": player_id,
            "seasons[]":    season,
            "per_page":     per_page,
        },
    )

    rows: list[dict] = []
    if result:
        for row in result.get("data", []):
            min_val = _parse_min(row.get("min"))
            if min_val == 0.0:
                continue   # DNP or missing — skip
            rows.append({
                "min":  min_val,
                "reb":  float(row.get("reb") or 0),
                "ast":  float(row.get("ast") or 0),
                "date": (row.get("game") or {}).get("date", ""),
            })
        # BDL returns chronological order — reverse to most-recent-first
        rows.sort(key=lambda r: r["date"], reverse=True)

    # If current season has no data yet, try previous season as fallback
    # (skip if we already know stats tier is unavailable)
    if not rows and season == _CURRENT_SEASON and _STATS_TIER is not False:
        result2 = _bdl_get(
            "/wnba/v1/player_stats",
            {
                "player_ids[]": player_id,
                "seasons[]":    _CURRENT_SEASON - 1,
                "per_page":     per_page,
            },
        )
        if result2:
            for row in result2.get("data", []):
                min_val = _parse_min(row.get("min"))
                if min_val == 0.0:
                    continue
                rows.append({
                    "min":  min_val,
                    "reb":  float(row.get("reb") or 0),
                    "ast":  float(row.get("ast") or 0),
                    "date": (row.get("game") or {}).get("date", ""),
                })
            rows.sort(key=lambda r: r["date"], reverse=True)

    _GAME_LOG_CACHE[cache_key] = rows
    return rows


# ── High-level stats entry point (called from player_props.py) ────────────────

def get_player_stats(
    player_name: str,
    season:      int = _CURRENT_SEASON,
) -> dict[str, list[float]] | None:
    """
    Full pipeline: search → fetch game logs → return cache-ready dict.

    Returns {MIN: [...], REB: [...], AST: [...]} most-recent-first, or None.
    Returns None immediately when the API key does not have stats tier access.
    """
    if _STATS_TIER is False:
        return None

    player_id = search_player(player_name)
    if not player_id:
        return None

    logs = get_player_game_logs(player_id, season=season)
    if not logs:
        return None

    return {
        "MIN": [r["min"] for r in logs],
        "REB": [r["reb"] for r in logs],
        "AST": [r["ast"] for r in logs],
    }


# ── Today's games ─────────────────────────────────────────────────────────────

def get_todays_game_ids(date_str: str) -> list[dict]:
    """
    Fetch today's WNBA games.

    date_str: "YYYY-MM-DD"
    Returns [{bdl_id, home_abbr, away_abbr, status}]
    """
    result = _bdl_get("/wnba/v1/games", {"dates[]": date_str, "per_page": 20})
    if not result:
        return []

    games = []
    for g in result.get("data", []):
        games.append({
            "bdl_id":    g["id"],
            "home_abbr": (g.get("home_team")    or {}).get("abbreviation", ""),
            "away_abbr": (g.get("visitor_team") or {}).get("abbreviation", ""),
            "status":    g.get("status", ""),
        })
    return games


# ── Live player props ─────────────────────────────────────────────────────────

def get_player_props(
    bdl_game_id: int,
    prop_types:  list[str] | None = None,
) -> list[dict]:
    """
    Fetch live WNBA player props for a BDL game_id.

    prop_types: optional filter e.g. ["rebounds", "assists", "points"].
    Returns [{player_id, prop_type, line, vendor, over_odds, under_odds}].

    Note: BDL player props are live-only; no historical data.
    """
    result = _bdl_get("/wnba/v1/odds/player_props", {"game_id": bdl_game_id})
    if not result:
        return []

    props = []
    for row in result.get("data", []):
        ptype = (row.get("prop_type") or "").lower().replace(" ", "_")
        if prop_types and ptype not in prop_types:
            continue

        try:
            line = float(row.get("line_value") or 0)
        except (TypeError, ValueError):
            continue
        if line <= 0:
            continue

        market = row.get("market") or {}
        props.append({
            "player_id":  row.get("player_id"),
            "prop_type":  ptype,
            "line":       line,
            "vendor":     row.get("vendor", ""),
            "over_odds":  market.get("over_odds"),
            "under_odds": market.get("under_odds"),
        })

    return props
