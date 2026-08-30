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


def _extract_totals(events: list[dict]) -> list[tuple[str, float, int | None]]:
    """
    Walk event dicts and return (date, total, opponent_team_id) tuples for
    completed games (home + away score) in chronological order.

    The date string is carried alongside each total so callers can filter
    to games played on-or-before a given `as_of_date` — without it, a
    replay run has no way to exclude games that haven't happened yet as
    of the date being replayed. opponent_team_id is carried so
    get_head_to_head_totals() can filter to a specific matchup without a
    second network call per team.
    """
    totals: list[tuple[str, float, int | None]] = []
    for ev in events:
        ev_date = ev.get("date", "")  # ISO 8601, e.g. "2026-06-14T23:00Z"
        ev_date_str = ev_date[:10] if ev_date else ""
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
            team_ids: list[int] = []
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
                try:
                    team_ids.append(int(c.get("team", {}).get("id")))
                except (TypeError, ValueError):
                    pass
            if len(scores) == 2:
                total = round(sum(scores), 1)
                # SANITY FILTER (2026-08-12): found via replay -- ESPN
                # sometimes marks a postponed/rescheduled game as
                # "completed" with no score recorded, which _extract_totals
                # previously took as a literal 0.0 combined total. A real
                # WNBA/NBA game can never end 0-0, but a single 0.0 sitting
                # in a team's history barely moves the mean (invisible
                # there) while catastrophically inflating any variance
                # calculation built on this data -- one bad zero produced a
                # ~175-point residual, which alone explained every std
                # outlier (30-50) in a replay run, while every unaffected
                # team landed in a believable ~15-27 range. Floor chosen
                # conservatively: real WNBA/NBA combined totals are
                # essentially never below 100 even in a defensive slog, so
                # anything at or under 50 is corrupted data, not a real
                # final score.
                if total <= 50.0:
                    logger.debug(
                        f"[game_logs] Dropping implausible game total "
                        f"{total} on {ev_date_str} (likely a postponed/"
                        f"rescheduled game ESPN marked completed with no "
                        f"score) -- not a real final score."
                    )
                    continue
                totals.append((
                    ev_date_str,
                    total,
                    tuple(team_ids) if team_ids else None,
                ))
    return totals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_team_game_totals(
    sport: str,
    team_abbr: str,
    n: int = 15,
    as_of_date: str | None = None,
) -> list[float] | None:
    """
    Return the last *n* completed game totals (both teams combined) for
    *team_abbr* in *sport* (WNBA or NBA).

    Parameters
    ----------
    sport       : 'WNBA' | 'NBA'
    team_abbr   : Three-letter abbreviation, e.g. 'SEA', 'NYL', 'GSW'.
    n           : Number of most-recent games to return (default 15).
    as_of_date  : ISO 'YYYY-MM-DD' string. When set (replay mode), only
                  games played on-or-before this date are eligible, and
                  results are cached separately per date — this is what
                  makes the function safe to call from a replay loop over
                  historical dates instead of always describing "today".

    Returns
    -------
    list[float]
        Game totals in chronological order, most recent last.
        e.g. [154.0, 169.0, 147.0, ...]
    None
        On any failure — caller must fall back to synthetic history.
    """
    sport_up = sport.upper()
    cache_key = (sport_up, team_abbr.upper(), as_of_date)
    if cache_key in _HISTORY_CACHE:
        return _HISTORY_CACHE[cache_key]

    # NOTE (2026-08-29): this used to try `core.wnba_stats_client` first
    # for WNBA ("richer free data than balldontlie's free tier") -- that
    # module doesn't exist anywhere in this codebase (confirmed; see the
    # same dead-import problem already found and fixed in data/fetch.py
    # and core/player_props.py). The import always raised ImportError,
    # silently caught, and fell through to ESPN below every single time --
    # so behavior is unchanged by removing it, this just stops wasting a
    # guaranteed-failing import attempt on every call and stops the
    # docstring/comments implying a data source that was never real.
    # ESPN (below) is and has always been the actual data source in use.

    sport_path = _ESPN_SPORT_PATH.get(sport_up)
    id_map     = _TEAM_ID_MAP.get(sport_up, {})
    team_id    = id_map.get(team_abbr.upper())

    if not sport_path or not team_id:
        logger.debug(
            f"[game_logs] No ESPN config for {sport_up}/{team_abbr} — "
            "falling back to synthetic history."
        )
        return None

    season = date.fromisoformat(as_of_date).year if as_of_date else date.today().year
    events = _espn_schedule(sport_path, team_id, season)
    if events is None:
        return None

    all_totals = _extract_totals(events)
    if as_of_date is not None:
        all_totals = [(d, total, opp) for d, total, opp in all_totals if d and d <= as_of_date]
    if not all_totals:
        logger.debug(
            f"[game_logs] No completed games for {sport_up}/{team_abbr} "
            f"(season {season}, as_of={as_of_date}) — falling back to synthetic history."
        )
        return None

    recent = [total for _, total, _ in all_totals[-n:]]
    _HISTORY_CACHE[cache_key] = recent

    mean_val = round(sum(recent) / len(recent), 2)
    logger.debug(
        f"[game_logs] {sport_up}/{team_abbr}: "
        f"{len(recent)} real game totals, mean={mean_val}"
    )
    return recent


# ---------------------------------------------------------------------------
# Head-to-head (this season, this specific matchup)
# ---------------------------------------------------------------------------

_H2H_CACHE: dict[tuple[str, str, str, str | None], list[float]] = {}


def get_head_to_head_totals(
    sport: str,
    team_a_abbr: str,
    team_b_abbr: str,
    as_of_date: str | None = None,
) -> list[float] | None:
    """
    Return this season's completed game totals (both teams combined) for
    meetings specifically between *team_a_abbr* and *team_b_abbr* --
    e.g. if two teams have played twice already this season, this returns
    both of those two games' totals, chronological order.

    This is a genuinely different signal from get_team_game_totals(): a
    team's overall recent scoring average blends in how it plays against
    everyone, but two specific teams can produce very different totals
    against each other than either does against the league generally
    (pace mismatches, a defense that struggles specifically with one
    team's style, etc.) -- e.g. Indiana/New York produced a 158-point
    game in one meeting and 196 in another meeting this same season, a
    gap neither team's general season average would have flagged.

    Fetches team_a's full-season schedule once (one network call, same
    endpoint/cache pattern as get_team_game_totals) and filters to events
    where team_b's ESPN team id appears among the competitors -- no
    second network call needed.

    Parameters mirror get_team_game_totals(); see that function's
    docstring for as_of_date's replay-mode behavior.

    Returns
    -------
    list[float]
        Game totals from head-to-head meetings this season, chronological
        order (most recent last). Empty/None if the teams haven't played
        yet this season or on any lookup failure -- caller should treat
        this as "no h2h signal available" and fall back to
        get_team_game_totals()-only, not as an error.
    """
    sport_up = sport.upper()
    a_key = team_a_abbr.upper()
    b_key = team_b_abbr.upper()
    # Order-independent cache key -- "team_a vs team_b" and "team_b vs
    # team_a" are the same set of games.
    cache_key = (sport_up, *sorted((a_key, b_key)), as_of_date)
    if cache_key in _H2H_CACHE:
        return _H2H_CACHE[cache_key]

    sport_path = _ESPN_SPORT_PATH.get(sport_up)
    id_map     = _TEAM_ID_MAP.get(sport_up, {})
    team_a_id  = id_map.get(a_key)
    team_b_id  = id_map.get(b_key)

    if not sport_path or not team_a_id or not team_b_id:
        logger.debug(
            f"[game_logs] No ESPN config for {sport_up}/{a_key}-{b_key} h2h — "
            "no h2h signal available."
        )
        return None

    season = date.fromisoformat(as_of_date).year if as_of_date else date.today().year
    # Only need one side's schedule -- team_a's games already include
    # every meeting against team_b.
    events = _espn_schedule(sport_path, team_a_id, season)
    if events is None:
        return None

    all_totals = _extract_totals(events)
    h2h = [
        (d, total) for d, total, opp_ids in all_totals
        if opp_ids and team_b_id in opp_ids
    ]
    if as_of_date is not None:
        h2h = [(d, total) for d, total in h2h if d and d <= as_of_date]

    if not h2h:
        logger.debug(
            f"[game_logs] No h2h meetings yet this season for "
            f"{sport_up}/{a_key}-{b_key} (as_of={as_of_date})."
        )
        _H2H_CACHE[cache_key] = []
        return None

    values = [total for _, total in h2h]
    _H2H_CACHE[cache_key] = values
    logger.debug(
        f"[game_logs] {sport_up}/{a_key}-{b_key} h2h: "
        f"{len(values)} meeting(s) this season, totals={values}"
    )
    return values
