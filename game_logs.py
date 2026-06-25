"""
game_logs.py

Fetches real game-total history (last N completed games) for WNBA and NBA
teams from the ESPN public API.  Replaces synthetic history in the Bayesian
simulation, giving the model actual variance from real scoring patterns
instead of a centred Gaussian.

ESPN endpoint used (no auth required, public):
  https://site.api.espn.com/apis/site/v2/sports/{path}/teams/{teamId}/schedule
    ?season={year}

Each completed game contributes one observation: home_score + away_score.
Returns observations in chronological order (most recent last), matching
the format expected by simulation_engine.estimate_player_metric().

Cache
-----
Process-level dict (_HISTORY_CACHE) stores results keyed by
(sport, team_abbr) so each team is fetched at most once per engine run.

Fallback
--------
Returns None on any error; caller must fall back to _synthetic_history().
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any
from urllib import request as urllib_req

logger = logging.getLogger("betting_bot")

# ---------------------------------------------------------------------------
# ESPN team ID maps  (abbreviation → ESPN internal numeric ID)
# ---------------------------------------------------------------------------

_WNBA_TEAM_ID: dict[str, int] = {
    # Original franchises — IDs verified from ESPN /apis/site/v2/sports/basketball/wnba/teams
    "ATL":  20,   # Atlanta Dream
    "CHI":  19,   # Chicago Sky
    "CON":  18,   # Connecticut Sun
    "DAL":   3,   # Dallas Wings
    "IND":   5,   # Indiana Fever
    "LVA":  17,   # Las Vegas Aces        (ESPN abbrev "LV")
    "LAS":   6,   # Los Angeles Sparks    (ESPN abbrev "LA")
    "MIN":   8,   # Minnesota Lynx
    "NYL":   9,   # New York Liberty      (ESPN abbrev "NY")
    "PHX":  11,   # Phoenix Mercury
    "SEA":  14,   # Seattle Storm
    "WAS":  16,   # Washington Mystics    (ESPN abbrev "WSH")
    # 2026 expansion franchises
    "GSV": 129689,  # Golden State Valkyries (ESPN abbrev "GS")
    "POR": 132052,  # Portland Fire
    "TOR": 131935,  # Toronto Tempo
}

_NBA_TEAM_ID: dict[str, int] = {
    "ATL":  1,  "BOS":  2,  "BKN": 17,  "CHA": 30,
    "CHI":  4,  "CLE":  5,  "DAL":  6,  "DEN":  7,
    "DET":  8,  "GSW":  9,  "HOU": 10,  "IND": 11,
    "LAC": 12,  "LAL": 13,  "MEM": 29,  "MIA": 14,
    "MIL": 15,  "MIN": 16,  "NOP":  3,  "NYK": 18,
    "OKC": 25,  "ORL": 19,  "PHI": 20,  "PHX": 21,
    "POR": 22,  "SAC": 23,  "SAS": 24,  "TOR": 28,
    "UTA": 26,  "WAS": 27,
}

_ESPN_SPORT_PATH: dict[str, str] = {
    "WNBA": "basketball/wnba",
    "NBA":  "basketball/nba",
}

_TEAM_ID_MAP: dict[str, dict[str, int]] = {
    "WNBA": _WNBA_TEAM_ID,
    "NBA":  _NBA_TEAM_ID,
}

# ---------------------------------------------------------------------------
# Process-level cache
# ---------------------------------------------------------------------------

_HISTORY_CACHE: dict[tuple[str, str], list[float]] = {}


# ---------------------------------------------------------------------------
# Internal fetch
# ---------------------------------------------------------------------------

def _espn_schedule(sport_path: str, team_id: int, season: int) -> list[dict] | None:
    """
    Fetch the ESPN team schedule for *team_id* and return the list of
    event dicts.  Returns None on any network / parse error.
    """
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/"
        f"{sport_path}/teams/{team_id}/schedule?season={season}"
    )
    try:
        with urllib_req.urlopen(url, timeout=8) as resp:
            data: dict[str, Any] = json.loads(resp.read().decode())
        events = data.get("events")
        if not isinstance(events, list):
            return None
        return events
    except Exception as exc:
        logger.debug(f"[game_logs] ESPN schedule fetch failed: {exc}")
        return None


def _extract_totals(events: list[dict]) -> list[float]:
    """
    Walk event dicts and return completed game totals (home + away score)
    in chronological order.
    """
    totals: list[float] = []
    for ev in events:
        for comp in ev.get("competitions", []):
            status = comp.get("status", {})
            completed = (
                status.get("type", {}).get("completed")
                or status.get("type", {}).get("state") == "post"
            )
            if not completed:
                continue
            competitors = comp.get("competitors", [])
            scores: list[float] = []
            for c in competitors:
                raw = c.get("score")
                if raw is None:
                    break
                # ESPN returns score as {"value": 78.0, "displayValue": "78"}
                # or as a plain numeric string
                if isinstance(raw, dict):
                    raw = raw.get("value") or raw.get("displayValue")
                try:
                    scores.append(float(raw))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    break
            if len(scores) == 2:
                totals.append(round(sum(scores), 1))
    return totals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_team_game_totals(
    sport: str,
    team_abbr: str,
    n: int = 15,
) -> list[float] | None:
    """
    Return the last *n* completed game totals (both teams combined) for
    *team_abbr* in *sport* (WNBA or NBA).

    Parameters
    ----------
    sport       : 'WNBA' | 'NBA'
    team_abbr   : Three-letter abbreviation, e.g. 'SEA', 'NYL', 'GSW'.
    n           : Number of most-recent games to return (default 15).

    Returns
    -------
    list[float]
        Game totals in chronological order, most recent last.
        e.g. [154.0, 169.0, 147.0, ...]
    None
        On any failure — caller must fall back to synthetic history.
    """
    sport_up = sport.upper()
    cache_key = (sport_up, team_abbr.upper())
    if cache_key in _HISTORY_CACHE:
        return _HISTORY_CACHE[cache_key]

    # WNBA: try the free stats.wnba.com client first (richer free data
    # than balldontlie's free tier; see core/wnba_stats_client.py).
    if sport_up == "WNBA":
        try:
            from core.wnba_stats_client import get_team_game_totals as _wnba_stats_totals
            values = _wnba_stats_totals(team_abbr, n=n)
            if values:
                _HISTORY_CACHE[cache_key] = values
                logger.debug(
                    f"[game_logs] {sport_up}/{team_abbr}: "
                    f"{len(values)} real game totals via stats.wnba.com"
                )
                return values
        except Exception as exc:
            logger.debug(f"[game_logs] stats.wnba.com failed for {team_abbr}: {exc}")
        logger.debug(f"[game_logs] stats.wnba.com had no data for {team_abbr}; trying ESPN…")

    sport_path = _ESPN_SPORT_PATH.get(sport_up)
    id_map     = _TEAM_ID_MAP.get(sport_up, {})
    team_id    = id_map.get(team_abbr.upper())

    if not sport_path or not team_id:
        logger.debug(
            f"[game_logs] No ESPN config for {sport_up}/{team_abbr} — "
            "falling back to synthetic history."
        )
        return None

    season = date.today().year
    events = _espn_schedule(sport_path, team_id, season)
    if events is None:
        return None

    all_totals = _extract_totals(events)
    if not all_totals:
        logger.debug(
            f"[game_logs] No completed games for {sport_up}/{team_abbr} "
            f"(season {season}) — falling back to synthetic history."
        )
        return None

    recent = all_totals[-n:]
    _HISTORY_CACHE[cache_key] = recent

    mean_val = round(sum(recent) / len(recent), 2)
    logger.debug(
        f"[game_logs] {sport_up}/{team_abbr}: "
        f"{len(recent)} real game totals, mean={mean_val}"
    )
    return recent
