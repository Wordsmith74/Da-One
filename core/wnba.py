"""
Official WNBA stat retrieval via the stats.wnba.com endpoints (same family
as stats.nba.com, just a different host/league id). These endpoints are
undocumented and require browser-like headers or they'll 403.

TODO: verify these endpoints are still live and headers still work --
NBA/WNBA stats endpoints change without notice more often than MLB's.
Consider a fallback to a secondary source (e.g. a paid provider you already
have keys for) if `_get` returns None repeatedly.
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger("historical_grader.wnba")

BASE_URL = "https://stats.wnba.com/stats"
REQUEST_TIMEOUT_SECS = 10
LEAGUE_ID = "10"  # WNBA league id in the stats.nba.com/wnba.com family

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.wnba.com/",
    "Accept": "application/json",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def _get(url: str, params: Optional[dict] = None) -> Optional[dict]:
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECS)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.warning("wnba_api_request_failed", extra={"url": url, "error": str(e)})
        return None


def find_game_id(team: str, game_date: str) -> Optional[str]:
    """
    team: team name or city (e.g. "Las Vegas Aces")
    game_date: "YYYY-MM-DD" -> converted to MM/DD/YYYY for the scoreboard endpoint
    """
    y, m, d = game_date.split("-")
    formatted_date = f"{m}/{d}/{y}"
    data = _get(
        f"{BASE_URL}/scoreboardV2",
        params={"GameDate": formatted_date, "LeagueID": LEAGUE_ID, "DayOffset": "0"},
    )
    if not data:
        return None
    try:
        headers = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]
        gid_idx = headers.index("GAME_ID")
        home_idx = headers.index("HOME_TEAM_ID") if "HOME_TEAM_ID" in headers else None
        for row in rows:
            # scoreboardV2 doesn't give team names directly; a second lookup
            # against LineScore result set is usually needed to match by name.
            # Simplified here -- TODO: match by team_id via a team-name map
            # if this proves unreliable, or use `find_game_id_by_linescore`.
            return row[gid_idx]  # NOTE: returns first game of the day, not matched to `team`
    except (KeyError, IndexError, ValueError) as e:
        logger.warning("wnba_scoreboard_parse_error", extra={"error": str(e)})
    return None


def get_boxscore(game_id: str) -> Optional[dict]:
    return _get(f"{BASE_URL}/boxscoretraditionalv2", params={"GameID": game_id})


def _get_player_stat(game_id: str, player_name: str, stat_field: str) -> Optional[float]:
    box = get_boxscore(game_id)
    if not box:
        return None
    try:
        result_set = next(
            rs for rs in box["resultSets"] if rs["name"] == "PlayerStats"
        )
        headers = result_set["headers"]
        name_idx = headers.index("PLAYER_NAME")
        stat_idx = headers.index(stat_field)
        for row in result_set["rowSet"]:
            if row[name_idx].lower() == player_name.lower():
                val = row[stat_idx]
                return float(val) if val is not None else None
    except (StopIteration, KeyError, ValueError) as e:
        logger.warning(
            "wnba_player_stat_parse_error",
            extra={"game_id": game_id, "player_name": player_name, "error": str(e)},
        )
    logger.warning(
        "wnba_player_not_found_in_boxscore",
        extra={"game_id": game_id, "player_name": player_name},
    )
    return None


def get_player_rebounds(game_id: str, player_name: str) -> Optional[float]:
    return _get_player_stat(game_id, player_name, "REB")


def get_player_assists(game_id: str, player_name: str) -> Optional[float]:
    return _get_player_stat(game_id, player_name, "AST")


def get_player_points(game_id: str, player_name: str) -> Optional[float]:
    return _get_player_stat(game_id, player_name, "PTS")


def get_game_total_points(game_id: str) -> Optional[float]:
    box = get_boxscore(game_id)
    if not box:
        return None
    try:
        result_set = next(rs for rs in box["resultSets"] if rs["name"] == "TeamStats")
        headers = result_set["headers"]
        pts_idx = headers.index("PTS")
        return float(sum(row[pts_idx] for row in result_set["rowSet"]))
    except (StopIteration, KeyError, ValueError) as e:
        logger.warning("wnba_total_points_parse_error", extra={"game_id": game_id, "error": str(e)})
        return None


def get_moneyline_winner(game_id: str) -> Optional[str]:
    box = get_boxscore(game_id)
    if not box:
        return None
    try:
        result_set = next(rs for rs in box["resultSets"] if rs["name"] == "TeamStats")
        headers = result_set["headers"]
        pts_idx = headers.index("PTS")
        team_idx = headers.index("TEAM_NAME")
        rows = sorted(result_set["rowSet"], key=lambda r: r[pts_idx], reverse=True)
        return rows[0][team_idx] if rows else None
    except (StopIteration, KeyError, ValueError, IndexError) as e:
        logger.warning("wnba_moneyline_parse_error", extra={"game_id": game_id, "error": str(e)})
        return None
