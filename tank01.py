"""
tank01.py

Tank01 RapidAPI fallback client — used when ESPN data is unavailable.

Confirmed active endpoints (RAPIDAPI_KEY subscription):
  MLB host  (tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com):
    /getMLBScoresOnly      — game scores by date
    /getMLBBettingOdds     — moneylines / totals by date
    /getMLBBoxScore        — per-game player stats  (plan-dependent; 403 → None)

  WNBA / multi-sport host (tank01-fantasy-stats.p.rapidapi.com):
    /getWNBAGamesForDate   — game schedule + scores by date
    /getWNBABoxScore       — per-game player stats
    /getWNBAPlayerInfo     — player metadata

Response envelope:  {"statusCode": 200, "body": <payload>}
All public functions return None on failure (403, 429, timeout, malformed).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger("betting_bot")

_KEY      = os.environ.get("RAPIDAPI_KEY", "")
_MLB_HOST = "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
_FAN_HOST = "tank01-fantasy-stats.p.rapidapi.com"
_TIMEOUT  = (3, 8)


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _get(host: str, path: str) -> Any | None:
    if not _KEY:
        logger.debug("[tank01] RAPIDAPI_KEY not set — skipping")
        return None
    headers = {
        "x-rapidapi-key":  _KEY,
        "x-rapidapi-host": host,
        "Content-Type":    "application/json",
    }
    url = f"https://{host}{path}"
    try:
        r = requests.get(url, headers=headers, timeout=_TIMEOUT)
        if r.status_code == 403:
            logger.debug(f"[tank01] endpoint not in plan: {path}")
            return None
        if r.status_code == 429:
            logger.warning(f"[tank01] rate-limited: {path} — skipping")
            return None
        r.raise_for_status()
        envelope = r.json()
        body = envelope.get("body")
        if not body:
            logger.debug(f"[tank01] empty body: {path}")
            return None
        return body
    except requests.Timeout:
        logger.warning(f"[tank01] timeout: {path}")
        return None
    except Exception as exc:
        logger.warning(f"[tank01] request error ({path}): {exc}")
        return None


# ---------------------------------------------------------------------------
# MLB endpoints
# ---------------------------------------------------------------------------

def get_mlb_scores(game_date_ymd: str) -> dict[str, Any] | None:
    """
    Scores keyed by gameID for a date in YYYYMMDD format.
    Returns None if endpoint is unavailable or not in plan.
    """
    return _get(_MLB_HOST, f"/getMLBScoresOnly?gameDate={game_date_ymd}")


def get_mlb_odds(game_date_ymd: str) -> list[dict] | None:
    """
    Betting odds list for a date in YYYYMMDD format.
    Returns list of game dicts or None.
    """
    body = _get(_MLB_HOST, f"/getMLBBettingOdds?gameDate={game_date_ymd}&itemFormat=list")
    return body if isinstance(body, list) else None


def get_mlb_boxscore(game_id: str) -> dict[str, Any] | None:
    """
    Per-game player stats for gameID format YYYYMMDD_AWAY@HOME.
    Returns None if not in plan (403) or unavailable.
    """
    return _get(_MLB_HOST, f"/getMLBBoxScore?gameID={game_id}")


# ---------------------------------------------------------------------------
# WNBA endpoints
# ---------------------------------------------------------------------------

def get_wnba_games(game_date_ymd: str) -> dict[str, Any] | None:
    """
    Games keyed by gameID for a date in YYYYMMDD format.
    Each value contains: away, home, gameStatus, gameDate, scores, etc.
    Returns None if unavailable.
    """
    return _get(_FAN_HOST, f"/getWNBAGamesForDate?gameDate={game_date_ymd}")


def get_wnba_boxscore(game_id: str) -> dict[str, Any] | None:
    """
    Per-game player stats for gameID format YYYYMMDD_AWAY@HOME.
    Payload contains playerStats dict keyed by playerID.
    """
    return _get(_FAN_HOST, f"/getWNBABoxScore?gameID={game_id}")


def get_wnba_player_info(player_id: str) -> dict[str, Any] | None:
    """Player metadata by Tank01 playerID."""
    return _get(_FAN_HOST, f"/getWNBAPlayerInfo?playerID={player_id}")


# ---------------------------------------------------------------------------
# Team abbreviation normalisation (ESPN / Odds-API → Tank01)
# ---------------------------------------------------------------------------

# Abbreviations that differ between the Odds API / ESPN and Tank01.
_WNBA_ABBR_ALIASES: dict[str, str] = {
    "WAS": "WSH",   # Washington Mystics  (Odds API uses WAS, Tank01 uses WSH)
    "WSH": "WSH",
    "LAS": "LV",    # Las Vegas Aces      (Odds API may use LAS)
    "GOL": "GS",    # Golden State Valkyries
    "GS":  "GS",
    "LV":  "LV",
    "NY":  "NY",
    "MIN": "MIN",
    "SEA": "SEA",
    "CHI": "CHI",
    "CON": "CON",
    "DAL": "DAL",
    "ATL": "ATL",
    "IND": "IND",
    "PHX": "PHX",
}


def normalise_wnba_abbr(abbr: str) -> str:
    """Return the Tank01 abbreviation for a given team code."""
    return _WNBA_ABBR_ALIASES.get(abbr.upper(), abbr.upper())
