"""
Official WNBA stat retrieval via ESPN's public site API
(site.api.espn.com) -- NOT stats.wnba.com.

Why the switch: stats.wnba.com/stats.nba.com-family endpoints are
undocumented, require browser-like headers, and are widely known to block
requests from cloud/datacenter IP ranges outright, regardless of headers --
including GitHub Actions runners. Confirmed live: every single WNBA pick
failed grading in the CI workflow with wnba_api_request_failed, uniformly,
which is the signature of an IP-level block rather than a per-request
fluke. ESPN's site API is the same public endpoint already used elsewhere
in this repo (data/fetch.py:get_espn_wnba_scoreboard, and backtest.py's
legacy WNBA grader, both already relying on it) and is not known to block
CI IPs.

Data source and its real limitation
------------------------------------
Game-level data (final score, winner) comes straight from the scoreboard
endpoint's own per-competitor score/winner fields -- reliable, and always
present once ESPN marks the event as completed.

Player-prop data (rebounds/assists/points) is resolved in two steps:
  1. Try the scoreboard payload's per-game "leaders" blocks first (cheap --
     already fetched, no extra request). ESPN only populates a leaders
     block with that game's statistical LEADER(S) in each category, so
     this only ever resolves the top rebounder/assister in that game.
  2. Fall back to a full per-game box score via ESPN's
     `/summary?event=<id>` endpoint (get_player_boxscore_stat /
     _fetch_boxscore below). This carries every player who appeared in
     the game, keyed by the standard ESPN "boxscore.players[].statistics[]
     .athletes[]" shape (one block per team, a shared `labels` array e.g.
     ["MIN","FG","3PT","FT","OREB","DREB","REB","AST","STL","BLK","TO",
     "PF","+/-","PTS"], and each athlete's `stats` list positionally
     aligned to those labels).

This two-step approach was previously leaders-only, which meant any pick
on a player who wasn't that game's category leader was silently stuck
ungraded forever -- not an error, just never resolved (identical to the
existing "player DNP -> leave for a future pass" contract already used by
core/rebounds.py / core/assists.py -- see grade(), which already treats
actual_stat=None as "not gradeable this run", not a loss). The leaders
check is kept as step 1 (it's free once the scoreboard is already
fetched for find_game_id/game-level grading), with the box score as the
real source of truth for anyone not caught by it.

Function signatures are kept identical to the old stats.wnba.com module,
so core/historical_grader.py needs zero changes:
    find_game_id(team, game_date)              -> game_id | None
    get_player_rebounds(game_id, player_name)   -> float | None
    get_player_assists(game_id, player_name)    -> float | None
    get_player_points(game_id, player_name)     -> float | None
    get_game_total_points(game_id)              -> float | None
    get_moneyline_winner(game_id)                -> str | None  (winning team's display name)
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger("historical_grader.wnba")

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary"
REQUEST_TIMEOUT_SECS = 15

# game_id -> raw ESPN event dict, populated by find_game_id() so the
# game-level getters below (called right after, per pick) don't need a
# second network round trip for data they already have.
_EVENT_CACHE: dict[str, dict] = {}
# YYYYMMDD -> list of raw ESPN events, so a slate with many picks on the
# same date only hits the scoreboard endpoint once.
_SCOREBOARD_CACHE: dict[str, list[dict]] = {}
# game_id -> raw ESPN summary/boxscore payload. Populated lazily (only
# when a leaders-block lookup misses) and cached per game_id so multiple
# props on the same game (e.g. two players' rebounds) only cost one
# extra request, not one per pick.
_BOXSCORE_CACHE: dict[str, Optional[dict]] = {}

# ESPN's team-abbreviation slugs drift from the 3-letter codes used
# elsewhere in this engine (matchup strings, odds feeds). Same table as
# data/fetch.py's _ESPN_TEAM_ABBR_ALIAS (kept as a local copy rather than
# a shared import, matching that module's own precedent -- see its
# docstring). This was the actual cause of every WNBA grading pass
# rejecting with wnba_game_not_found: parse_matchup() derives `team` from
# the 3-letter code embedded in `matchup` (e.g. "LVA", "LAS", "NYL"), but
# find_game_id() only ever compared that against ESPN's own fields
# (displayName/shortDisplayName/location/name/abbreviation) with no
# aliasing, and ESPN's abbreviation for these four franchises is 2
# letters, not 3 -- so neither the exact-match nor the (>3-char-only)
# substring-match branch below could ever hit for them.
_ESPN_TEAM_ABBR_ALIAS = {
    "was": "wsh",  # WNBA Mystics
    "nyl": "ny",   # WNBA Liberty
    "las": "la",   # WNBA Sparks
    "lva": "lv",   # WNBA Aces
    "gsv": "gs",   # WNBA Valkyries
}


def _fetch_scoreboard(espn_date: str) -> list[dict]:
    if espn_date in _SCOREBOARD_CACHE:
        return _SCOREBOARD_CACHE[espn_date]
    events: list[dict] = []
    try:
        resp = requests.get(SCOREBOARD_URL, params={"dates": espn_date}, timeout=REQUEST_TIMEOUT_SECS)
        resp.raise_for_status()
        events = resp.json().get("events", [])
    except requests.RequestException as e:
        logger.warning(
            "wnba_api_request_failed",
            extra={"url": SCOREBOARD_URL, "date": espn_date, "error": str(e)},
        )
    _SCOREBOARD_CACHE[espn_date] = events
    return events


def find_game_id(team: str, game_date: str) -> Optional[str]:
    """
    team: team name, city, or abbreviation (e.g. "Las Vegas Aces", "Aces", "LVA").
    game_date: "YYYY-MM-DD" -> converted to ESPN's "YYYYMMDD".

    Matches against each event's own competitor team fields (displayName /
    shortDisplayName / location / name / abbreviation). A team plays at
    most once per day, so more than one match means the query string was
    too loose -- refuse to guess rather than silently grade the wrong game
    (same behavior the old stats.wnba.com module had).
    """
    espn_date = game_date.replace("-", "")
    events = _fetch_scoreboard(espn_date)
    if not events:
        return None

    team_query = team.strip().lower()
    # Also try ESPN's aliased form of the query (e.g. "lva" -> "lv") so
    # betting-side 3-letter codes that don't match ESPN's own abbreviation
    # still resolve. Keep both: some candidates (location, displayName)
    # are matched fine by the raw query already, and short-circuiting to
    # only the alias would break teams that aren't in the alias table.
    team_queries = {team_query, _ESPN_TEAM_ABBR_ALIAS.get(team_query, team_query)}
    matched: dict[str, dict] = {}

    for event in events:
        for comp in event.get("competitions", []):
            for competitor in comp.get("competitors", []):
                t = competitor.get("team", {}) or {}
                candidates = [
                    str(t.get("displayName") or ""),
                    str(t.get("shortDisplayName") or ""),
                    str(t.get("location") or ""),
                    str(t.get("name") or ""),
                    str(t.get("abbreviation") or ""),
                ]
                candidates = [c.strip().lower() for c in candidates if c.strip()]
                # Exact match on any field (raw or aliased query), OR a
                # substring match -- but only allow substring matching once
                # the query is long enough (>3 chars) to avoid a 2-3 letter
                # abbreviation spuriously matching inside an unrelated
                # longer string.
                if any(q == c for q in team_queries for c in candidates) or any(
                    len(q) > 3 and any(q in c for c in candidates) for q in team_queries
                ):
                    event_id = event.get("id")
                    if event_id:
                        matched[event_id] = event

    if not matched:
        logger.warning("wnba_game_not_found", extra={"team": team, "game_date": game_date})
        return None
    if len(matched) > 1:
        logger.warning(
            "wnba_game_id_ambiguous",
            extra={"team": team, "game_date": game_date, "candidates": list(matched.keys())},
        )
        return None

    game_id, event = next(iter(matched.items()))
    _EVENT_CACHE[game_id] = event
    return game_id


def _get_event(game_id: str) -> Optional[dict]:
    event = _EVENT_CACHE.get(game_id)
    if event is None:
        # Cache miss -- e.g. called out of order, or in a fresh process
        # that never called find_game_id for this game_id. Refetching
        # isn't possible without the original date, so this is a hard
        # miss rather than a silent guess.
        logger.warning("wnba_event_not_cached", extra={"game_id": game_id})
    return event


def _is_final(event: dict) -> bool:
    return bool((event.get("status", {}) or {}).get("type", {}).get("completed"))


def _competitors(event: dict) -> list[dict]:
    comps = event.get("competitions", [])
    return comps[0].get("competitors", []) if comps else []


def get_game_total_points(game_id: str) -> Optional[float]:
    event = _get_event(game_id)
    if not event or not _is_final(event):
        return None  # not final yet -- not gradeable this run, not an error
    try:
        return float(sum(int(c["score"]) for c in _competitors(event)))
    except (KeyError, ValueError, TypeError) as e:
        logger.warning("wnba_total_points_parse_error", extra={"game_id": game_id, "error": str(e)})
        return None


def get_moneyline_winner(game_id: str) -> Optional[str]:
    event = _get_event(game_id)
    if not event or not _is_final(event):
        return None
    competitors = _competitors(event)
    try:
        winner = next((c for c in competitors if c.get("winner") is True), None)
        if winner is None:
            # No explicit winner flag (shouldn't happen once completed,
            # but don't guess on an edge case) -- fall back to comparing
            # scores directly.
            ranked = sorted(competitors, key=lambda c: int(c["score"]), reverse=True)
            if len(ranked) < 2 or ranked[0]["score"] == ranked[1]["score"]:
                return None
            winner = ranked[0]
        return (winner.get("team", {}) or {}).get("displayName")
    except (KeyError, ValueError, TypeError) as e:
        logger.warning("wnba_moneyline_parse_error", extra={"game_id": game_id, "error": str(e)})
        return None


def _get_leader_stat(game_id: str, player_name: str, stat_category: str) -> Optional[float]:
    event = _get_event(game_id)
    if not event:
        return None
    player_query = player_name.strip().lower()
    for comp in event.get("competitions", []):
        for leader_group in comp.get("leaders", []):
            if leader_group.get("name") != stat_category:
                continue
            for leader in leader_group.get("leaders", []):
                athlete = str((leader.get("athlete", {}) or {}).get("displayName") or "").strip().lower()
                if athlete and athlete == player_query:
                    try:
                        return float(leader.get("value"))
                    except (TypeError, ValueError):
                        return None
    # Not an error condition -- the player just isn't this game's category
    # leader (see module docstring), or the game isn't final yet. Left
    # ungraded this run rather than guessed.
    logger.warning(
        "wnba_player_not_in_leaders",
        extra={"game_id": game_id, "player_name": player_name, "stat": stat_category},
    )
    return None


def _fetch_boxscore(game_id: str) -> Optional[dict]:
    """
    Fetch (and cache) the full per-game box score via ESPN's summary
    endpoint. Returns the raw JSON payload, or None if the request failed
    or the game has no boxscore section yet (e.g. not final). Cached by
    game_id -- including the None/failure case, so a bad game_id doesn't
    get re-requested on every subsequent pick in the same run.
    """
    if game_id in _BOXSCORE_CACHE:
        return _BOXSCORE_CACHE[game_id]
    payload: Optional[dict] = None
    try:
        resp = requests.get(SUMMARY_URL, params={"event": game_id}, timeout=REQUEST_TIMEOUT_SECS)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as e:
        logger.warning("wnba_boxscore_request_failed", extra={"game_id": game_id, "error": str(e)})
        payload = None
    _BOXSCORE_CACHE[game_id] = payload
    return payload


# ESPN's basketball boxscore `labels` array uses these short codes.
# Map our stat_category names (shared with the leaders-block lookup,
# e.g. "rebounds"/"assists"/"points") to the label(s) that hold the
# right column. REB is the combined total (not OREB+DREB) when present.
_BOXSCORE_STAT_LABELS: dict[str, tuple[str, ...]] = {
    "rebounds": ("REB",),
    "assists": ("AST",),
    "points": ("PTS",),
}


def get_player_boxscore_stat(game_id: str, player_name: str, stat_category: str) -> Optional[float]:
    """
    Look up a single stat for a player from the full per-game box score
    (as opposed to _get_leader_stat, which only ever finds that game's
    category leader). Covers every player who appeared in the game.

    Returns None (not gradeable this run, not an error) if: the summary
    request failed, the game has no boxscore yet (not final / postponed),
    the player didn't appear in it (DNP), or the stat value itself can't
    be parsed -- all logged, none raised, matching the rest of this
    module's "degrade gracefully" contract.
    """
    payload = _fetch_boxscore(game_id)
    if not payload:
        return None

    labels_to_try = _BOXSCORE_STAT_LABELS.get(stat_category)
    if not labels_to_try:
        logger.warning("wnba_boxscore_unknown_stat", extra={"stat": stat_category})
        return None

    player_query = player_name.strip().lower()
    try:
        team_blocks = (payload.get("boxscore") or {}).get("players") or []
        for team_block in team_blocks:
            for stat_group in team_block.get("statistics", []):
                labels = stat_group.get("labels") or []
                # Find the column index for whichever of our candidate
                # labels is present in this stat group.
                col_idx = next((labels.index(l) for l in labels_to_try if l in labels), None)
                if col_idx is None:
                    continue
                for athlete_entry in stat_group.get("athletes", []):
                    athlete = str((athlete_entry.get("athlete", {}) or {}).get("displayName") or "").strip().lower()
                    if athlete != player_query:
                        continue
                    stats = athlete_entry.get("stats") or []
                    if col_idx >= len(stats):
                        continue
                    raw_value = stats[col_idx]
                    # DNP rows typically render as "DNP" or "--" in this
                    # slot rather than a number -- treat as not gradeable
                    # rather than raising.
                    try:
                        return float(raw_value)
                    except (TypeError, ValueError):
                        logger.info(
                            "wnba_boxscore_player_dnp_or_unparsed",
                            extra={"game_id": game_id, "player_name": player_name, "raw_value": raw_value},
                        )
                        return None
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(
            "wnba_boxscore_parse_error",
            extra={"game_id": game_id, "player_name": player_name, "error": str(e)},
        )
        return None

    # Player never appeared in either team's statistics block at all
    # (e.g. inactive/not on the roster that night) -- not an error.
    logger.info("wnba_boxscore_player_not_found", extra={"game_id": game_id, "player_name": player_name})
    return None


def get_player_rebounds(game_id: str, player_name: str) -> Optional[float]:
    stat = _get_leader_stat(game_id, player_name, "rebounds")
    if stat is not None:
        return stat
    return get_player_boxscore_stat(game_id, player_name, "rebounds")


def get_player_assists(game_id: str, player_name: str) -> Optional[float]:
    stat = _get_leader_stat(game_id, player_name, "assists")
    if stat is not None:
        return stat
    return get_player_boxscore_stat(game_id, player_name, "assists")


def get_player_points(game_id: str, player_name: str) -> Optional[float]:
    return _get_leader_stat(game_id, player_name, "points")
