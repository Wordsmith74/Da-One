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
    team: team name, city, or 3-letter abbreviation (e.g. "Las Vegas Aces",
    "Aces", or "LVA"). Matched against the scoreboard's own LineScore data
    for that date (TEAM_ABBREVIATION / TEAM_CITY_NAME / TEAM_NAME) -- no
    separate abbreviation map needs to be kept in sync with the league.

    game_date: "YYYY-MM-DD" -> converted to MM/DD/YYYY for the scoreboard endpoint.

    Previously this ignored `team` entirely and returned whichever game
    happened to appear first in that day's scoreboard rowSet. That's
    harmless on a single-game day, but on any day with multiple WNBA games
    it could silently grade a pick against the wrong game's result with no
    error raised at all -- worse than a reject, since it wouldn't show up
    in grading_rejects.jsonl.
    """
    y, m, d = game_date.split("-")
    formatted_date = f"{m}/{d}/{y}"
    data = _get(
        f"{BASE_URL}/scoreboardV2",
        params={"GameDate": formatted_date, "LeagueID": LEAGUE_ID, "DayOffset": "0"},
    )
    if not data:
        return None

    team_query = team.strip().lower()

    try:
        result_sets = {rs["name"]: rs for rs in data["resultSets"]}
        line_score = result_sets["LineScore"]
        headers = line_score["headers"]
        gid_idx = headers.index("GAME_ID")
        abbr_idx = headers.index("TEAM_ABBREVIATION")
        city_idx = headers.index("TEAM_CITY_NAME") if "TEAM_CITY_NAME" in headers else None
        name_idx = headers.index("TEAM_NAME") if "TEAM_NAME" in headers else None

        matched_game_ids: set[str] = set()
        for row in line_score["rowSet"]:
            abbr = str(row[abbr_idx] or "").strip().lower()
            city = str(row[city_idx] or "").strip().lower() if city_idx is not None else ""
            name = str(row[name_idx] or "").strip().lower() if name_idx is not None else ""
            full_name = f"{city} {name}".strip()
            if team_query == abbr or (full_name and team_query in full_name):
                matched_game_ids.add(row[gid_idx])

        if not matched_game_ids:
            logger.warning("wnba_game_not_found", extra={"team": team, "game_date": game_date})
            return None
        if len(matched_game_ids) > 1:
            # A team plays at most once per day, so this means the match was
            # too loose (e.g. an ambiguous partial name) -- don't guess which
            # game is right, surface it instead of silently grading the wrong one.
            logger.warning(
                "wnba_game_id_ambiguous",
                extra={"team": team, "game_date": game_date, "candidates": list(matched_game_ids)},
            )
            return None
        return matched_game_ids.pop()
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
