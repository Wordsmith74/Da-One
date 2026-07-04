"""
Official MLB stat retrieval via the public MLB Stats API
(https://statsapi.mlb.com/api/v1/...). No API key required.

Endpoints used:
- schedule (find gamePk for a team+date)
- boxscore (pitcher strikeouts, final score, F5 line score)

TODO: this pipeline had a known hang risk in get_mlb_team_k_rate_allowed
(sequential per-game API calls). The same risk applies here if grading
many picks per day sequentially -- consider batching gamePk lookups per
(date) once, then reusing across all picks for that date, and adding a
per-request timeout (see `_get`).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import requests

logger = logging.getLogger("historical_grader.mlb")

BASE_URL = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT_SECS = 10

# Same map used in core/intelligence/bullpen_intel.py -- kept in sync
# manually since that module isn't safely importable here (it pulls in
# the full intelligence stack). team_abbr -> MLB Stats API team id.
_MLB_TEAM_IDS: dict[str, int] = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KC":  118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD":  135, "SF":  137, "SEA": 136,
    "STL": 138, "TB":  139, "TEX": 140, "TOR": 141, "WSH": 120,
}


def _get(url: str, params: Optional[dict] = None) -> Optional[dict]:
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECS)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.warning("mlb_api_request_failed", extra={"url": url, "error": str(e)})
        return None


def find_game_pk(team: str, game_date: str) -> Optional[int]:
    """
    team: full/common team name ("New York Yankees", "Yankees") OR a
    3-letter abbreviation ("NYY"). Abbreviations are resolved to a team id
    and matched exactly; names fall back to a substring match against the
    schedule's team names. Abbreviations used to fail here silently --
    e.g. "CHC" is not a substring of "Chicago Cubs" -- which surfaced as
    spurious mlb_game_not_found rejects for any pick whose matchup came
    from the abbreviated format ("SD@CHC_MLB_2026-07-01").
    """
    data = _get(f"{BASE_URL}/schedule", params={"sportId": 1, "date": game_date})
    if not data:
        return None

    team_id = _MLB_TEAM_IDS.get(team.strip().upper())

    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            home_team = game.get("teams", {}).get("home", {}).get("team", {})
            away_team = game.get("teams", {}).get("away", {}).get("team", {})
            if team_id is not None:
                if home_team.get("id") == team_id or away_team.get("id") == team_id:
                    return game.get("gamePk")
                continue
            home_name = home_team.get("name", "")
            away_name = away_team.get("name", "")
            if team.lower() in home_name.lower() or team.lower() in away_name.lower():
                return game.get("gamePk")
    logger.warning("mlb_game_not_found", extra={"team": team, "game_date": game_date})
    return None


def get_boxscore(game_pk: int) -> Optional[dict]:
    return _get(f"{BASE_URL}/game/{game_pk}/boxscore")


def get_pitcher_strikeouts(game_pk: int, player_name: str) -> Optional[int]:
    box = get_boxscore(game_pk)
    if not box:
        return None
    for side in ("home", "away"):
        players = box.get("teams", {}).get(side, {}).get("players", {})
        for _, pdata in players.items():
            full_name = pdata.get("person", {}).get("fullName", "")
            if full_name.lower() == player_name.lower():
                pitching = pdata.get("stats", {}).get("pitching", {})
                if "strikeOuts" in pitching:
                    return int(pitching["strikeOuts"])
    logger.warning(
        "mlb_pitcher_not_found_in_boxscore",
        extra={"game_pk": game_pk, "player_name": player_name},
    )
    return None


def get_game_total_runs(game_pk: int) -> Optional[float]:
    box = get_boxscore(game_pk)
    if not box:
        return None
    try:
        home_runs = box["teams"]["home"]["teamStats"]["batting"]["runs"]
        away_runs = box["teams"]["away"]["teamStats"]["batting"]["runs"]
        return float(home_runs) + float(away_runs)
    except KeyError as e:
        logger.warning(
            "mlb_total_runs_missing_field", extra={"game_pk": game_pk, "error": str(e)}
        )
        return None


def get_f5_total_runs(game_pk: int) -> Optional[float]:
    """First-5-innings total, summed from the linescore."""
    data = _get(f"{BASE_URL}/game/{game_pk}/linescore")
    if not data:
        return None
    innings = data.get("innings", [])[:5]
    if len(innings) < 5:
        logger.warning("mlb_f5_incomplete_innings", extra={"game_pk": game_pk})
        return None
    try:
        home_total = sum(i.get("home", {}).get("runs", 0) for i in innings)
        away_total = sum(i.get("away", {}).get("runs", 0) for i in innings)
        return float(home_total + away_total)
    except (TypeError, KeyError) as e:
        logger.warning("mlb_f5_parse_error", extra={"game_pk": game_pk, "error": str(e)})
        return None


def get_moneyline_winner(game_pk: int) -> Optional[str]:
    box = get_boxscore(game_pk)
    if not box:
        return None
    try:
        home_runs = box["teams"]["home"]["teamStats"]["batting"]["runs"]
        away_runs = box["teams"]["away"]["teamStats"]["batting"]["runs"]
        home_name = box["teams"]["home"].get("team", {}).get("name", "home")
        away_name = box["teams"]["away"].get("team", {}).get("name", "away")
        if home_runs > away_runs:
            return home_name
        elif away_runs > home_runs:
            return away_name
        return None  # unresolved / tie (shouldn't happen in MLB w/o extras)
    except KeyError as e:
        logger.warning("mlb_moneyline_missing_field", extra={"game_pk": game_pk, "error": str(e)})
        return None
