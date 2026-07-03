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
    team: full or common team name (e.g. "New York Yankees" or "Yankees")
    game_date: "YYYY-MM-DD"
    """
    data = _get(f"{BASE_URL}/schedule", params={"sportId": 1, "date": game_date})
    if not data:
        return None
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            home = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
            away = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "")
            if team.lower() in home.lower() or team.lower() in away.lower():
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
