"""
core/player_props.py — Player prop candidate generator for the Bayesian engine.

Fetches today's player prop lines from two sources:
  1. The Odds API  (per-event endpoint) — primary source
  2. PropLine API  (sport-level endpoint) — supplementary bookmakers

Both bookmaker lists are merged before analysis so the engine sees the widest
possible book coverage (including Novig, Pinnacle, Smarkets).

Historical data
---------------
  NBA  : ESPN athlete season-average API, then synthetic history around it.
  MLB  : MLB Stats API season game-log, then synthetic history around it.
  Combo mkts : Sum of component ESPN stats (Pts+Reb = PTS + REB, etc.).
  Fallback   : Synthetic history anchored at a sport/market league mean when the
               external stat API is unavailable.

Supported markets
-----------------
  WNBA : (no player props — game totals only via game_markets.py)
  NBA  : player_points, player_rebounds, player_assists, player_blocks,
         player_steals, player_threes, player_turnovers,
         player_points_rebounds, player_points_assists, player_rebounds_assists,
         player_points_rebounds_assists
  MLB  : pitcher_strikeouts, pitcher_earned_runs
         (hits_allowed retired — removed from broadcast)
"""

from __future__ import annotations

import json
import logging
import os
import random
import statistics as _stats
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from core.market_intelligence import (
    compute_mis,
    compute_data_reliability,
    detect_sharp_action,
    detect_steam_move,
    detect_reverse_line_movement,
)

logger = logging.getLogger(__name__)

_API_KEY   = os.environ.get("THE_ODDS_API_KEY", "")
_ODDS_BASE = "https://api.the-odds-api.com/v4"

# ── Sport / market config ────────────────────────────────────────────────────

_SPORT_KEY: dict[str, str] = {
    "WNBA": "basketball_wnba",
    "NBA":  "basketball_nba",
    "MLB":  "baseball_mlb",
}

_ESPN_SPORT_PATH: dict[str, str] = {
    "WNBA": "basketball/wnba",
    "NBA":  "basketball/nba",
}

# Odds API market keys requested per sport (comma-separated for the endpoint)
# pitcher_hits_allowed retired from broadcast (poor model fit, removed 2026-06-06).
# Scope limited to the System Scope Definition (core/market_gate.py):
#   MLB  — pitcher_strikeouts only (pitcher_earned_runs removed — not in scope)
#   WNBA — player_rebounds + player_assists only (unchanged)
#   NBA  — no prop markets in scope (removed entirely)
_PROP_MARKETS: dict[str, str] = {
    "WNBA": "player_rebounds,player_assists",
    "MLB":  "pitcher_strikeouts",
}

# ── Performance-based market suspension ──────────────────────────────────────
# Markets removed from broadcast pending model recalibration.
# Evidence (43 graded picks):
#   batter_hits           → 43.8% WR, −31% ROI (16 picks) — no opposing pitcher adjustment
#   batter_total_bases    → removed per audit (same root cause as hits: no pitcher context)
#   player_points         → 28.6% WR, −39% ROI (7 picks)  — prior not role/matchup aware
#   pitcher_hits_allowed  → retired 2026-06-06 — model fit insufficient; market retired entirely
# Re-enable by removing from this set once the prior is recalibrated.
_SUSPENDED_PROP_MARKETS: frozenset[str] = frozenset({
    "batter_hits",
    "batter_total_bases",
    "player_points",
    "pitcher_hits_allowed",
})

# PropLine API market keys per sport — supplementary source for additional books.
# Includes markets PropLine carries that The Odds API may not serve per-event.
# pitcher_hits_allowed retired.
_PROPLINE_MARKETS: dict[str, list[str]] = {
    "WNBA": ["player_rebounds", "player_assists"],
    "MLB":  ["pitcher_strikeouts"],
    # NBA: no markets in scope
}

# Human-readable display name stored as `market` in each candidate
_MARKET_DISPLAY: dict[str, str] = {
    # Basketball
    "player_points":                 "Points",
    "player_rebounds":               "Rebounds",
    "player_assists":                "Assists",
    "player_blocks":                 "Blocks",
    "player_steals":                 "Steals",
    "player_threes":                 "3-Pointers",
    "player_turnovers":              "Turnovers",
    "player_points_rebounds":        "Pts+Reb",
    "player_points_assists":         "Pts+Ast",
    "player_rebounds_assists":       "Reb+Ast",
    "player_points_rebounds_assists":"Pts+Reb+Ast",
    # Baseball — pitching
    "pitcher_strikeouts":            "Strikeouts",
    "pitcher_earned_runs":           "Earned Runs",
    "pitcher_hits_allowed":          "Hits Allowed",
    # Baseball — batting
    "batter_hits":                   "Hits",
    "batter_total_bases":            "Total Bases",
}

# League-level priors (Bayesian mean + std) used when player stat lookup fails.
# Also gates which market keys are processed — keys absent here are skipped.
_PROP_PRIOR: dict[str, dict[str, float]] = {
    # Basketball — individual
    "player_points":                 {"mean": 15.0, "std": 5.0},
    "player_rebounds":               {"mean":  5.0, "std": 3.0},
    "player_assists":                {"mean":  3.5, "std": 2.0},
    "player_blocks":                 {"mean":  0.8, "std": 0.8},
    "player_steals":                 {"mean":  0.9, "std": 0.7},
    "player_threes":                 {"mean":  1.5, "std": 1.2},
    "player_turnovers":              {"mean":  2.5, "std": 1.5},
    # Basketball — combo
    "player_points_rebounds":        {"mean": 20.5, "std": 6.0},
    "player_points_assists":         {"mean": 18.5, "std": 5.5},
    "player_rebounds_assists":       {"mean":  9.0, "std": 3.5},
    "player_points_rebounds_assists":{"mean": 27.0, "std": 7.5},
    # Baseball — pitching
    "pitcher_strikeouts":            {"mean":  5.5, "std": 2.0},
    "pitcher_earned_runs":           {"mean":  2.5, "std": 1.5},
    "pitcher_hits_allowed":          {"mean":  5.0, "std": 2.0},
    # Baseball — batting
    "batter_hits":                   {"mean":  1.0, "std": 0.8},
    "batter_total_bases":            {"mean":  1.2, "std": 0.8},
}

# ESPN stat abbreviation(s) for each prop market.
# Combo markets use a tuple — the player avg is the SUM of the component stats.
_ESPN_STAT_COL: dict[str, str | tuple[str, ...]] = {
    # Individual stats
    "player_points":                 "PTS",
    "player_rebounds":               "REB",
    "player_assists":                "AST",
    "player_blocks":                 "BLK",
    "player_steals":                 "STL",
    "player_threes":                 "3PM",
    "player_turnovers":              "TO",
    # Combo markets (sum of components)
    "player_points_rebounds":        ("PTS", "REB"),
    "player_points_assists":         ("PTS", "AST"),
    "player_rebounds_assists":       ("REB", "AST"),
    "player_points_rebounds_assists":("PTS", "REB", "AST"),
}

# Max prop candidates returned per sport (keeps engine runtime manageable)
_MAX_CANDIDATES_PER_SPORT = 25

# Max events to process per sport per run (avoids flooding the Odds API)
_MAX_EVENTS_PER_SPORT = 6

# Rule 2: maximum allowed deviation between the best-price line and the
# consensus (median across all bookmakers) before a pick is rejected.
_LINE_CONSENSUS_MAX_DRIFT: float = 0.5

# Process-level registry: bet_id → metadata needed for pre-publish re-verification.
# Populated by get_player_prop_candidates(); consumed by core/line_validator.py.
_PROP_META: dict[str, dict] = {}

# MLB Stats API group + stat column for player stat history lookup.
_MLB_STAT_SPEC: dict[str, tuple[str, str]] = {
    # Pitching
    "pitcher_strikeouts":  ("pitching", "strikeOuts"),
    "pitcher_earned_runs": ("pitching", "earnedRuns"),
    "pitcher_hits_allowed":("pitching", "hits"),
    # Batting
    "batter_hits":         ("hitting",  "hits"),
    "batter_total_bases":  ("hitting",  "totalBases"),
}


# ── HTTP helper ──────────────────────────────────────────────────────────────

def _get_json(url: str, timeout: int = 6) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ── Synthetic history ────────────────────────────────────────────────────────

def _synthetic_history(
    mean: float, std: float, n: int = 15, seed: float | None = None
) -> list[float]:
    rng = random.Random(seed)
    return [max(0.0, round(rng.gauss(mean, std), 1)) for _ in range(n)]


# ── ESPN season-average lookup (WNBA / NBA) ───────────────────────────────────

def _espn_player_avg(player_name: str, sport: str, market: str) -> float | None:
    """
    Return a player's season average for the given prop market.
    Uses ESPN's athlete search + athlete stats endpoint.

    For combo markets (player_points_rebounds etc.) stat_col is a tuple —
    the returned value is the SUM of each component's season average.

    Returns None when the athlete is not found or the API is unavailable.
    """
    sport_path = _ESPN_SPORT_PATH.get(sport.upper())
    stat_col   = _ESPN_STAT_COL.get(market)
    if not sport_path or not stat_col:
        return None

    # Normalise to tuple so the rest of the function is uniform
    stat_cols: tuple[str, ...] = (stat_col,) if isinstance(stat_col, str) else stat_col
    target_cols = {c.upper() for c in stat_cols}

    try:
        encoded = urllib.parse.quote(player_name)
        search_url = (
            f"https://site.api.espn.com/apis/site/v2/sports/"
            f"{sport_path}/athletes?search={encoded}&limit=3"
        )
        sdata    = _get_json(search_url)
        athletes = sdata.get("items") or sdata.get("athletes", [])
        if not athletes:
            return None
        first      = athletes[0]
        athlete_id = str(first.get("id", ""))
        if not athlete_id:
            ref = first.get("$ref", "")
            athlete_id = ref.split("/")[-1].split("?")[0]
        if not athlete_id:
            return None

        stats_url = (
            f"https://site.api.espn.com/apis/site/v2/sports/"
            f"{sport_path}/athletes/{athlete_id}/stats"
        )
        data = _get_json(stats_url)

        # Collect all matching stat values from all split categories
        found: dict[str, float] = {}
        for cat in (
            data.get("splits", {}).get("categories", [])
            + data.get("statistics", {}).get("splits", {}).get("categories", [])
        ):
            for stat in cat.get("stats", []):
                abbr = stat.get("abbreviation", "").upper()
                if abbr in target_cols and abbr not in found:
                    try:
                        found[abbr] = float(stat["value"])
                    except (TypeError, ValueError) as _stat_exc:
                        logger.debug(f"[player_props] stat parse skip: {_stat_exc}")

        if not found:
            return None

        # Sum all requested components (works for both single and combo markets)
        return round(sum(found.get(c.upper(), 0.0) for c in stat_cols), 2)

    except Exception as _exc:
        logger.debug(f"[player_props] ESPN stat lookup failed: {_exc}")
    return None


# ── MLB Stats API — season avg + last-5 / last-10 game averages ─────────────

def _mlb_player_stats(
    player_name: str, market: str
) -> tuple[float | None, float | None, float | None]:
    """
    Return (season_avg, l5_avg, l10_avg) from the MLB Stats API game log.

    Uses a single player search → gamelog call so we don't double the API
    quota versus the old season-only approach.  Returns (None, None, None)
    when the player or market is not found.
    """
    spec = _MLB_STAT_SPEC.get(market)
    if not spec:
        return None, None, None
    group_key, stat_col = spec
    try:
        encoded    = urllib.parse.quote(player_name)
        search_url = (
            f"https://statsapi.mlb.com/api/v1/people/search"
            f"?names={encoded}&sportId=1"
        )
        sdata  = _get_json(search_url)
        people = sdata.get("people", [])
        if not people:
            return None, None, None
        pid = people[0]["id"]

        # Game log gives us per-game splits — from which we compute all three.
        gl_url = (
            f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
            f"?stats=gameLog&group={group_key}&season=2025&gameType=R"
        )
        gdata  = _get_json(gl_url)
        splits = []
        for block in gdata.get("stats", []):
            splits = block.get("splits", [])
            if splits:
                break
        if not splits:
            return None, None, None

        def _stat(s: dict) -> float:
            raw = s["stat"]
            return float(raw.get(stat_col) or 0)

        all_vals = [_stat(s) for s in splits]
        n        = len(all_vals)
        if n == 0:
            return None, None, None

        season_avg = round(sum(all_vals) / n, 2)
        l5_avg     = round(sum(all_vals[-5:]) / min(5, n), 2) if n >= 1 else None
        l10_avg    = round(sum(all_vals[-10:]) / min(10, n), 2) if n >= 5 else None

        return season_avg, l5_avg, l10_avg
    except Exception:
        return None, None, None


# ── ESPN per-game gamelog helper ──────────────────────────────────────────────

# ESPN stat label aliases — the gamelog endpoint uses the same abbreviations
# as the season stats endpoint for most markets, but some leagues use variants.
# Values are lists of accepted label strings (checked case-insensitively).
_ESPN_LABEL_ALIASES: dict[str, list[str]] = {
    "PTS": ["PTS", "POINTS"],
    "REB": ["REB", "TRB", "REBOUNDS"],
    "AST": ["AST", "ASSISTS"],
    "BLK": ["BLK", "BS",  "BLOCKS"],
    "STL": ["STL", "ST",  "STEALS"],
    "3PM": ["3PM", "3FGM", "TPM", "3FG"],
    "TO":  ["TO",  "TOV", "TURNOVERS"],
}


def _espn_athlete_id(player_name: str, sport_path: str) -> str | None:
    """Search ESPN for an athlete and return their numeric ID string, or None."""
    try:
        encoded = urllib.parse.quote(player_name)
        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/"
            f"{sport_path}/athletes?search={encoded}&limit=3"
        )
        data     = _get_json(url)
        athletes = data.get("items") or data.get("athletes", [])
        if not athletes:
            return None
        first = athletes[0]
        aid   = str(first.get("id", ""))
        if not aid:
            ref = first.get("$ref", "")
            aid = ref.split("/")[-1].split("?")[0]
        return aid or None
    except Exception as _exc:
        logger.debug(f"[player_props] ESPN athlete search failed ({player_name}): {_exc}")
        return None


def _espn_gamelog_per_game(
    athlete_id: str,
    sport_path: str,
    stat_cols: tuple[str, ...],
) -> list[float]:
    """
    Fetch an athlete's regular-season game log from ESPN and return a list of
    per-game stat totals (one float per game, ordered oldest → newest).

    For combo markets, each entry is the SUM of the component stats (e.g. PTS+REB).

    Returns an empty list when the endpoint is unavailable or the stat columns
    are not found in the gamelog labels.

    ESPN gamelog structure (regular-season type, id="2"):
      seasonTypes[i].categories[j].labels  — stat abbreviation list
      seasonTypes[i].categories[j].events[k].stats  — per-game values (same order)
    """
    # Build a lookup of target labels → canonical stat_col so we can match
    # ESPN's abbreviations case-insensitively and handle variants.
    target_upper = {c.upper() for c in stat_cols}
    alias_map: dict[str, str] = {}   # espn_label_upper → stat_col
    for col in stat_cols:
        col_up = col.upper()
        for alias in _ESPN_LABEL_ALIASES.get(col_up, [col_up]):
            alias_map[alias.upper()] = col_up

    try:
        url  = (
            f"https://site.api.espn.com/apis/site/v2/sports/"
            f"{sport_path}/athletes/{athlete_id}/gamelog"
        )
        data = _get_json(url, timeout=8)

        season_types = data.get("seasonTypes", [])
        if not season_types:
            return []

        # Prefer the Regular Season type (id="2"); fall back to first entry.
        reg = next(
            (st for st in season_types if str(st.get("id", "")) == "2"),
            season_types[0],
        )

        per_game: list[float] = []

        for cat in reg.get("categories", []):
            labels = [lbl.upper() for lbl in cat.get("labels", [])]

            # Build a col → [index, ...] map for this category
            col_indices: dict[str, list[int]] = {c: [] for c in target_upper}
            for idx, lbl in enumerate(labels):
                canon = alias_map.get(lbl)
                if canon and canon in col_indices:
                    col_indices[canon].append(idx)

            # Only proceed if ALL target columns have at least one matching index
            if not all(col_indices[c] for c in target_upper):
                continue

            for event in cat.get("events", []):
                raw_stats = event.get("stats", [])
                try:
                    total = 0.0
                    for col in target_upper:
                        for idx in col_indices[col]:
                            total += float(raw_stats[idx])
                    per_game.append(total)
                except (IndexError, TypeError, ValueError):
                    continue

            # First category that satisfies all columns is authoritative
            if per_game:
                break

        return per_game

    except Exception as _exc:
        logger.debug(
            f"[player_props] ESPN gamelog failed "
            f"(athlete={athlete_id}, cols={stat_cols}): {_exc}"
        )
        return []


# ── WNBA player stats cache (scoreboard + boxscore approach) ──────────────────
# ESPN's WNBA athlete search and gamelog endpoints both return 404.
# Solution: scan recent WNBA scoreboard dates, fetch game summaries, extract
# per-game REB/AST from boxscores.  Built once per process on first WNBA call.
# vals are stored most-recent-first (scoreboard scanned newest → oldest).

_WNBA_STATS_CACHE: dict[str, dict[str, list[float]]] = {}
_WNBA_CACHE_BUILT: bool = False
_WNBA_BDL_ATTEMPTED: set[str] = set()   # players already tried via BDL (avoid retries)

# Boxscore column → market key (prop markets)
_WNBA_BOX_COL: dict[str, str] = {
    "player_rebounds": "REB",
    "player_assists":  "AST",
}

# Additional boxscore columns captured for per-minute modeling (not prop markets)
_WNBA_CACHE_COLS: set[str] = {"REB", "AST", "MIN"}

# ── Minutes stability thresholds (L5 range = max - min) ──────────────────────
_WNBA_MIN_STABLE_RANGE   = 4.0   # ≤4 min range → "elite"
_WNBA_MIN_MODERATE_RANGE = 8.0   # 4–8 min range → "moderate"; >8 → "volatile"

# ── Role-based multipliers for per-minute projection ─────────────────────────
_WNBA_AST_ROLE_MULT: dict[str, float] = {
    "ballhandler": 1.15,   # primary creator — high AST/min
    "secondary":   1.05,   # facilitating scorer
    "off_ball":    0.80,   # limited creation
}
_WNBA_REB_ROLE_MULT: dict[str, float] = {
    "frontcourt": 1.10,    # primary rebounder
    "wing":       1.00,    # neutral
    "guard":      0.85,    # limited rebounding
}

# Per-minute rate thresholds for role classification
_WNBA_AST_HIGH  = 0.12   # ≥ → ballhandler
_WNBA_AST_LOW   = 0.07   # < → off_ball
_WNBA_REB_HIGH  = 0.20   # ≥ → frontcourt
_WNBA_REB_LOW   = 0.10   # < → guard


def _build_wnba_stats_cache(days_back: int = 21) -> None:
    """
    Populate _WNBA_STATS_CACHE by scanning recent WNBA scoreboard dates and
    extracting per-game stats from ESPN game summary boxscores.

    ESPN's athlete-level search and gamelog APIs return 404 for WNBA; this
    scoreboard approach is the only reliable ESPN data path for WNBA players.
    Called once per process on first WNBA prop candidate request.
    """
    global _WNBA_STATS_CACHE, _WNBA_CACHE_BUILT
    if _WNBA_CACHE_BUILT:
        return

    from datetime import date, timedelta
    today = date.today()
    target_cols = set(_WNBA_BOX_COL.values())   # {"REB", "AST"}

    # Phase 1: collect completed game IDs from scoreboard (one call per day)
    game_ids: list[str] = []
    for days_ago in range(1, days_back + 1):
        ds = (today - timedelta(days=days_ago)).strftime("%Y%m%d")
        try:
            data = _get_json(
                f"https://site.api.espn.com/apis/site/v2/sports/"
                f"basketball/wnba/scoreboard?dates={ds}",
                timeout=6,
            )
            for ev in data.get("events", []):
                comp = (ev.get("competitions") or [{}])[0]
                if comp.get("status", {}).get("type", {}).get("completed", False):
                    gid = comp.get("id")
                    if gid:
                        game_ids.append(str(gid))
        except Exception as _exc:
            logger.debug(f"[wnba_cache] scoreboard {ds}: {_exc}")

    logger.info(f"[wnba_cache] building from {len(game_ids)} completed WNBA games")

    # Columns that gate which stat groups we enter (must contain REB or AST)
    _stat_cols = set(_WNBA_BOX_COL.values())   # {"REB", "AST"}

    # Phase 2: fetch each game summary and extract player stats
    for gid in game_ids:
        try:
            summ = _get_json(
                f"https://site.api.espn.com/apis/site/v2/sports/"
                f"basketball/wnba/summary?event={gid}",
                timeout=8,
            )
        except Exception:
            continue

        for team_data in summ.get("boxscore", {}).get("players", []):
            for stat_group in team_data.get("statistics", []):
                names: list[str] = stat_group.get("names", [])
                # Only enter the stat group that contains REB or AST
                if not _stat_cols.intersection(names):
                    continue

                # Build index map for ALL desired columns (REB, AST + MIN)
                col_idx: dict[str, int] = {}
                for col in _WNBA_CACHE_COLS:
                    if col in names:
                        col_idx[col] = names.index(col)

                for ath_entry in stat_group.get("athletes", []):
                    ath   = ath_entry.get("athlete", {})
                    pname = (ath.get("displayName") or "").strip().lower()
                    if not pname:
                        continue
                    stats = ath_entry.get("stats", [])
                    if not stats:
                        continue  # DNP — skip, don't zero-pad

                    if pname not in _WNBA_STATS_CACHE:
                        _WNBA_STATS_CACHE[pname] = {c: [] for c in _WNBA_CACHE_COLS}

                    for col, idx in col_idx.items():
                        try:
                            raw = stats[idx]
                            if isinstance(raw, str) and "-" in raw:
                                raw = raw.split("-")[0]
                            _WNBA_STATS_CACHE[pname][col].append(float(raw))
                        except (IndexError, TypeError, ValueError):
                            pass

    _WNBA_CACHE_BUILT = True
    logger.info(  # noqa: E501 — keep the leading logger call
        f"[wnba_cache] ready: {len(_WNBA_STATS_CACHE)} WNBA players "
        f"from {len(game_ids)} games"
    )


def _ensure_wnba_cached(player_name: str) -> None:
    """
    Ensure the WNBA stats cache has data for *player_name*.

    Priority order:
      1. stats.wnba.com (FREE, no key — core.wnba_stats_client)
      2. BDL per-player fetch (fast, targeted — requires paid All-Star tier)
      3. ESPN bulk boxscore scan (once per process — covers all players at once)

    BDL is skipped when BALLDONTLIE_API_KEY is not configured or when the
    player was already attempted and not found (tracked by _WNBA_BDL_ATTEMPTED).
    """
    name_lower = player_name.strip().lower()

    # 1. Already in cache → nothing to do
    if name_lower in _WNBA_STATS_CACHE:
        return

    # 2. Try the free stats.wnba.com client first — no key, no tier wall.
    try:
        from core.wnba_stats_client import get_player_stats as _free_wnba_stats
        data = _free_wnba_stats(player_name)
        if data and data.get("MIN"):
            _WNBA_STATS_CACHE[name_lower] = data
            logger.info(
                f"[wnba_cache] stats.wnba.com ✓  {player_name} — "
                f"{len(data['MIN'])} games "
                f"(avg {sum(data['MIN'])/len(data['MIN']):.1f} min)"
            )
            return
        logger.debug(f"[wnba_cache] stats.wnba.com: no data for '{player_name}'")
    except Exception as _free_exc:
        logger.debug(f"[wnba_cache] stats.wnba.com error for '{player_name}': {_free_exc}")

    # 3. Try BDL (targeted, paid-tier for stats)
    if name_lower not in _WNBA_BDL_ATTEMPTED:
        _WNBA_BDL_ATTEMPTED.add(name_lower)
        try:
            from core import bdl_wnba
            if bdl_wnba.is_available() and bdl_wnba.is_stats_available():
                data = bdl_wnba.get_player_stats(player_name)
                if data and data.get("MIN"):
                    _WNBA_STATS_CACHE[name_lower] = data
                    logger.info(
                        f"[wnba_cache] BDL ✓  {player_name} — "
                        f"{len(data['MIN'])} games "
                        f"(avg {sum(data['MIN'])/len(data['MIN']):.1f} min)"
                    )
                    return
                else:
                    logger.debug(f"[wnba_cache] BDL: no data for '{player_name}'")
        except Exception as _bdl_exc:
            logger.debug(f"[wnba_cache] BDL error for '{player_name}': {_bdl_exc}")

    # 4. Fall back to ESPN bulk scan (idempotent, runs at most once per process)
    _build_wnba_stats_cache()


def _wnba_stats_from_cache(
    player_name: str, market: str
) -> tuple[float | None, float | None, float | None]:
    """
    Return (season_avg, l5_avg, l10_avg) for a WNBA player from the boxscore
    cache.  Ensures the cache is populated (BDL → ESPN fallback) on first call.

    Matching: exact display-name (case-insensitive), then last-name fallback.
    Returns (None, None, None) when the player or market is not found.
    """
    _ensure_wnba_cached(player_name)

    col = _WNBA_BOX_COL.get(market)
    if not col:
        return None, None, None

    name_lower = player_name.strip().lower()

    # Exact match first
    entry = _WNBA_STATS_CACHE.get(name_lower)

    # Last-name fallback (handles minor name format mismatches)
    if entry is None:
        parts = name_lower.split()
        last  = parts[-1] if parts else ""
        if last:
            for cached_name, data in _WNBA_STATS_CACHE.items():
                cached_parts = cached_name.split()
                if cached_parts and cached_parts[-1] == last:
                    entry = data
                    logger.debug(
                        f"[wnba_cache] last-name match: '{player_name}' → '{cached_name}'"
                    )
                    break

    if entry is None:
        logger.debug(f"[wnba_cache] no cached stats for '{player_name}'")
        return None, None, None

    vals = entry.get(col, [])   # most-recent-first order
    n    = len(vals)
    if not vals:
        return None, None, None

    season_avg = round(sum(vals) / n, 2)
    l5_avg  = round(sum(vals[:5])  / min(5,  n), 2) if n >= 3  else None
    l10_avg = round(sum(vals[:10]) / min(10, n), 2) if n >= 5  else None

    return season_avg, l5_avg, l10_avg


# ── WNBA per-minute × projected-minutes modeling ─────────────────────────────

def _wnba_minutes_classification(
    min_vals: list[float],
) -> tuple[float, float, str]:
    """
    From L5 minutes (most-recent-first) compute:
      (projected_minutes, minutes_range, stability_label)

    Projected minutes use a recency-weighted blend: most-recent game gets
    highest weight.  Stability is classified by L5 max−min range.
    """
    recent = [m for m in min_vals[:5] if m > 0]   # skip DNP
    n = len(recent)
    if n == 0:
        return 25.0, 0.0, "moderate"
    if n == 1:
        return round(recent[0], 1), 0.0, "elite"

    _w = [0.30, 0.25, 0.20, 0.15, 0.10][:n]
    _wsum = sum(_w)
    projected = sum(v * w for v, w in zip(recent, _w)) / _wsum
    minutes_range = max(recent) - min(recent)

    if minutes_range <= _WNBA_MIN_STABLE_RANGE:
        stability = "elite"
    elif minutes_range <= _WNBA_MIN_MODERATE_RANGE:
        stability = "moderate"
    else:
        stability = "volatile"

    return round(projected, 1), round(minutes_range, 1), stability


def _wnba_classify_role(player_name: str, market: str) -> str:
    """
    Infer a player's role from their season per-minute rates.

    player_assists  → "ballhandler" | "secondary" | "off_ball"
    player_rebounds → "frontcourt"  | "wing"      | "guard"
    """
    entry = _WNBA_STATS_CACHE.get(player_name.strip().lower())
    if entry is None:
        return "secondary" if market == "player_assists" else "wing"

    min_vals  = entry.get("MIN", [])
    valid_min = [m for m in min_vals if m > 0]
    if not valid_min:
        return "secondary" if market == "player_assists" else "wing"

    avg_min = sum(valid_min) / len(valid_min)
    if avg_min <= 0:
        return "secondary" if market == "player_assists" else "wing"

    if market == "player_assists":
        ast_vals = entry.get("AST", [])
        if not ast_vals:
            return "secondary"
        ast_per_min = (sum(ast_vals) / len(ast_vals)) / avg_min
        if ast_per_min >= _WNBA_AST_HIGH:
            return "ballhandler"
        if ast_per_min < _WNBA_AST_LOW:
            return "off_ball"
        return "secondary"

    if market == "player_rebounds":
        reb_vals = entry.get("REB", [])
        if not reb_vals:
            return "wing"
        reb_per_min = (sum(reb_vals) / len(reb_vals)) / avg_min
        if reb_per_min >= _WNBA_REB_HIGH:
            return "frontcourt"
        if reb_per_min < _WNBA_REB_LOW:
            return "guard"
        return "wing"

    return "secondary"


def _wnba_prop_projection(
    player_name: str,
    market: str,
    matchup_context: str = "",
    home_team: str = "",
    away_team: str = "",
    spread: float | None = None,
) -> tuple[float, float, float | None, float | None, float, float, str, str]:
    """
    Full WNBA per-minute × projected-minutes prop projection.

    Steps (per spec):
      1. Project minutes (L5-weighted, stability classified)
      2. Per-minute production rate (L5-weighted blend)
      3. Role adjustment (ballhandler / frontcourt / guard etc.)
      4. Opponent / game-environment adjustment (hook, neutral now)
      5. Final = rate × minutes × role_adj blended with legacy projection

    Returns:
      (weighted_proj, season_avg, l5_avg, l10_avg,
       projected_minutes, minutes_range, minutes_stability, role_label,
       blowout_level)
    """
    # Pre-compute blowout level from spread so early-exit paths can return it
    def _bl_from_spread(sp: float | None) -> str:
        if sp is None:
            return "none"
        a = abs(sp)
        return "heavy" if a >= 17.0 else "moderate" if a >= 10.0 else "none"

    _early_bl = _bl_from_spread(spread)

    _ensure_wnba_cached(player_name)

    col = _WNBA_BOX_COL.get(market)
    if not col:
        return 0.0, 0.0, None, None, 25.0, 0.0, "moderate", "unknown", _early_bl

    name_lower = player_name.strip().lower()
    entry = _WNBA_STATS_CACHE.get(name_lower)

    # Last-name fallback
    if entry is None:
        parts = name_lower.split()
        last  = parts[-1] if parts else ""
        if last:
            for cached_name, data in _WNBA_STATS_CACHE.items():
                if cached_name.split() and cached_name.split()[-1] == last:
                    entry = data
                    logger.debug(
                        f"[wnba_model] last-name match: '{player_name}' → '{cached_name}'"
                    )
                    break

    if entry is None:
        return 0.0, 0.0, None, None, 25.0, 0.0, "moderate", "unknown", _early_bl

    stat_vals = entry.get(col, [])    # most-recent-first
    min_vals  = entry.get("MIN", [])  # most-recent-first

    if not stat_vals:
        return 0.0, 0.0, None, None, 25.0, 0.0, "moderate", "unknown", _early_bl

    n_stat = len(stat_vals)
    n_min  = len(min_vals)

    # ── Raw stat averages (used by gatekeeper L5 brake + factor display) ──────
    season_avg = round(sum(stat_vals) / n_stat, 2)
    l5_avg  = round(sum(stat_vals[:5])  / min(5,  n_stat), 2) if n_stat >= 3 else None
    l10_avg = round(sum(stat_vals[:10]) / min(10, n_stat), 2) if n_stat >= 5 else None

    # ── Step 1: minutes projection & stability ────────────────────────────────
    if n_min >= 1:
        projected_minutes, minutes_range, minutes_stability = (
            _wnba_minutes_classification(min_vals)
        )
    else:
        projected_minutes, minutes_range, minutes_stability = 25.0, 0.0, "moderate"

    # ── Step 2: per-minute production rate (recency-weighted) ─────────────────
    # Pair stat with its corresponding minutes; skip games with 0 min played
    per_min_vals: list[float] = []
    for i, stat in enumerate(stat_vals):
        if i < n_min and min_vals[i] > 0:
            per_min_vals.append(stat / min_vals[i])

    if not per_min_vals:
        # No per-minute data — fall back to legacy weighted projection
        weighted_proj = _weighted_projection(season_avg, l5_avg, l10_avg)
        return (weighted_proj, season_avg, l5_avg, l10_avg,
                projected_minutes, minutes_range, minutes_stability, "unknown", _early_bl)

    # L5-prioritised rate blend
    _pw = [0.40, 0.25, 0.20, 0.10, 0.05][:len(per_min_vals)]
    _pwsum = sum(_pw)
    recent_rate  = sum(r * w for r, w in zip(per_min_vals[:len(_pw)], _pw)) / _pwsum
    season_rate  = sum(per_min_vals) / len(per_min_vals)
    blended_rate = recent_rate * 0.60 + season_rate * 0.40

    # ── Step 3: role adjustment ───────────────────────────────────────────────
    role_label = _wnba_classify_role(player_name, market)
    role_adj = (
        _WNBA_AST_ROLE_MULT.get(role_label, 1.0)
        if market == "player_assists"
        else _WNBA_REB_ROLE_MULT.get(role_label, 1.0)
    )

    # ── Step 4: opponent / game-environment adjustment ────────────────────────
    # Three layers from wnba_opp_intel:
    #   Layer 1 — shooting environment  (assists only): high pts-allowed → more assists
    #   Layer 2 — rebounding environment (rebounds only): high opp reb → fewer boards
    #   Layer 3 — blowout risk (all markets): large spread → fewer starter minutes
    opp_adj       = 1.0
    _blowout_mult = 1.0
    _blowout_level = _early_bl   # default: derived from spread; overwritten by opp_intel
    try:
        from core.intelligence.wnba_opp_intel import get_wnba_opp_intel
        _intel = get_wnba_opp_intel(home_team, away_team, market, spread)
        opp_adj        = (
            _intel.shooting_mult
            if market == "player_assists"
            else _intel.rebound_mult
        )
        _blowout_mult  = _intel.blowout_mult
        _blowout_level = _intel.blowout_level
        if _intel.diag:
            logger.debug(
                f"[wnba_model] {player_name} {market} opp_intel: {_intel.diag}"
            )
    except Exception as _intel_exc:
        logger.warning(
            f"[wnba_model] opp_intel failed for {player_name} {market} — "
            f"using spread-derived blowout_level='{_blowout_level}': {_intel_exc}"
        )

    # Apply blowout dampener to projected minutes before the rate calculation
    _effective_minutes = projected_minutes * _blowout_mult

    # ── Step 5: final projection ──────────────────────────────────────────────
    raw_proj    = blended_rate * _effective_minutes * role_adj * opp_adj
    legacy_proj = _weighted_projection(season_avg, l5_avg, l10_avg)
    # Blend per-minute model (70%) with legacy season-average model (30%)
    # for stability when the per-minute sample is small.
    final_proj  = round(raw_proj * 0.70 + legacy_proj * 0.30, 2)

    logger.debug(
        f"[wnba_model] {player_name} {market}: "
        f"rate={blended_rate:.3f}/min × {_effective_minutes:.1f}min "
        f"× role({role_label})×{role_adj:.2f} × opp_adj×{opp_adj:.3f} "
        f"→ raw={raw_proj:.2f} | legacy={legacy_proj:.2f} | final={final_proj:.2f} | "
        f"stability={minutes_stability} (range={minutes_range:.1f}min)"
    )

    return (final_proj, season_avg, l5_avg, l10_avg,
            projected_minutes, minutes_range, minutes_stability, role_label,
            _blowout_level)


# ── ESPN season-average + recent-form wrapper ─────────────────────────────────

def _espn_player_stats(
    player_name: str, sport: str, market: str
) -> tuple[float | None, float | None, float | None]:
    """
    Return (season_avg, l5_avg, l10_avg) for NBA/WNBA athletes via ESPN.

    WNBA short-circuit: ESPN's WNBA athlete search and gamelog APIs both return
    404, so WNBA stats are resolved via the scoreboard + boxscore cache instead.

    For NBA: tries the per-game gamelog endpoint first so the weighted projection
    can use real L5/L10 recent-form data instead of 100% season average.
    Falls back to the season-average-only stats endpoint when the gamelog
    is unavailable or returns fewer than 3 games.

    Returns (None, None, None) when the athlete or market is not found.
    """
    # WNBA: ESPN athlete search + gamelog both 404 — use boxscore cache
    if sport.upper() == "WNBA":
        return _wnba_stats_from_cache(player_name, market)

    sport_path = _ESPN_SPORT_PATH.get(sport.upper())
    stat_col   = _ESPN_STAT_COL.get(market)
    if not sport_path or not stat_col:
        return None, None, None

    stat_cols: tuple[str, ...] = (
        (stat_col,) if isinstance(stat_col, str) else stat_col
    )

    # Step 1 — resolve athlete ID (shared by both code paths below)
    athlete_id = _espn_athlete_id(player_name, sport_path)
    if not athlete_id:
        return None, None, None

    # Step 2 — attempt per-game gamelog
    per_game = _espn_gamelog_per_game(athlete_id, sport_path, stat_cols)

    if len(per_game) >= 3:
        n          = len(per_game)
        season_avg = round(sum(per_game) / n, 2)
        l5_avg     = round(sum(per_game[-5:])  / min(5,  n), 2)
        l10_avg    = round(sum(per_game[-10:]) / min(10, n), 2) if n >= 5 else None
        logger.debug(
            f"[player_props] ESPN gamelog OK — {player_name} {market} "
            f"n={n} season={season_avg} L5={l5_avg} L10={l10_avg}"
        )
        return season_avg, l5_avg, l10_avg

    # Step 3 — fall back to season-average-only stats endpoint
    logger.debug(
        f"[player_props] ESPN gamelog insufficient (n={len(per_game)}) "
        f"for {player_name} — falling back to season avg"
    )
    avg = _espn_player_avg(player_name, sport, market)
    return avg, None, None


# ── Weighted projection (40 % L5 · 30 % L10 · 20 % season · 10 % matchup) ──

def _weighted_projection(
    season_avg: float,
    l5_avg: float | None,
    l10_avg: float | None,
    matchup_adj: float = 0.0,
) -> float:
    """
    Compute a blended projection using recent-form weighting.

    When recency data is unavailable the weight is redistributed to season avg
    so the weights always sum to 1.0.
    """
    w_l5, w_l10, w_season, w_matchup = 0.40, 0.30, 0.20, 0.10

    # Redistribute missing weights to season_avg
    if l5_avg is None:
        w_season += w_l5
        w_l5 = 0.0
    if l10_avg is None:
        w_season += w_l10
        w_l10 = 0.0

    proj = (
        (l5_avg or 0.0) * w_l5
        + (l10_avg or 0.0) * w_l10
        + season_avg * w_season
        + (season_avg + matchup_adj) * w_matchup
    )
    return round(proj, 2)


# ── Odds API helpers ─────────────────────────────────────────────────────────

def _fetch_events(sport_key: str) -> list[dict]:
    url = f"{_ODDS_BASE}/sports/{sport_key}/events/?apiKey={_API_KEY}&dateFormat=iso"
    try:
        result = _get_json(url)
        return result if isinstance(result, list) else []
    except Exception as exc:
        logger.debug(f"[player_props] event list failed for {sport_key}: {exc}")
        return []


def _fetch_event_props(sport_key: str, event_id: str, markets: str) -> dict | None:
    url = (
        f"{_ODDS_BASE}/sports/{sport_key}/events/{event_id}/odds"
        f"?apiKey={_API_KEY}&regions=us&markets={markets}&oddsFormat=american"
    )
    try:
        return _get_json(url)
    except urllib.error.HTTPError as exc:
        logger.debug(f"[player_props] {event_id} props HTTP {exc.code}: {exc}")
        return None
    except Exception as exc:
        logger.debug(f"[player_props] {event_id} props error: {exc}")
        return None


# ── PropLine bookmaker merge ─────────────────────────────────────────────────

def _merge_propline_books(
    prop_map:       dict[tuple[str, str], dict[str, dict]],
    propline_books: list[dict],
    valid_markets:  set[str],
) -> None:
    """
    Merge bookmaker lines from PropLine into an existing prop_map in-place.

    propline_books is a list of bookmaker dicts already normalised by
    propline_client._normalize_outcomes() — all outcomes have name="over"/"under",
    a stripped player description, and a float point.

    Players already present in prop_map get additional book coverage (higher
    book_count → better MIS, better consensus, sharper sharp-action detection).
    Players only in PropLine are created fresh.
    """
    for bk in propline_books:
        bk_title = bk.get("title") or bk.get("key", "Unknown")
        for mkt in bk.get("markets", []):
            mkt_key = mkt.get("key", "")
            if mkt_key not in valid_markets:
                continue
            for outcome in mkt.get("outcomes", []):
                player_name = (outcome.get("description") or "").strip()
                direction   = (outcome.get("name") or "").lower()
                point       = outcome.get("point")
                price       = outcome.get("price")
                if (
                    not player_name
                    or direction not in ("over", "under")
                    or point is None
                    or price is None
                ):
                    continue
                key = (player_name, mkt_key)
                if key not in prop_map:
                    prop_map[key] = {}
                existing = prop_map[key].get(direction)
                if existing is None:
                    prop_map[key][direction] = {
                        "best_line":  float(point),
                        "best_odds":  int(price),
                        "best_book":  bk_title,
                        "book_count": 1,
                        "all_lines":  [float(point)],
                        "book_lines": [{"book": bk_title, "line": float(point)}],
                    }
                else:
                    # Avoid double-counting the same book if Odds API already has it
                    already = any(
                        bl["book"] == bk_title
                        for bl in existing.get("book_lines", [])
                    )
                    if already:
                        continue
                    new_pt = float(point)
                    # Skip alt-line variants (e.g. Pinnacle's 0.5 alongside the
                    # canonical 1.5 for batter_total_bases).  If the incoming line
                    # deviates more than the consensus-drift threshold from the
                    # current pool median it belongs to a different bet class.
                    if abs(new_pt - _stats.median(existing["all_lines"])) > _LINE_CONSENSUS_MAX_DRIFT:
                        continue
                    existing["all_lines"].append(new_pt)
                    existing["book_count"] += 1
                    existing.setdefault("book_lines", []).append(
                        {"book": bk_title, "line": new_pt}
                    )
                    if int(price) > existing["best_odds"]:
                        existing["best_line"] = new_pt
                        existing["best_odds"] = int(price)
                        existing["best_book"] = bk_title


# ── Player-prop helper: build team-matchup key for prop_grader ───────────────

def _prop_matchup_key(away_team: str, home_team: str, sport: str) -> str:
    """
    Build wager_details['team'] for player-prop bets so prop_grader can locate
    the right ESPN event without relying on the brute-force fallback.

    Format: "{AWAY_ABBR}v{HOME_ABBR}"  e.g. "PORvLA"
    prop_grader splits on "v" → target set {"POR", "LA"} → ESPN event match.

    WNBA: uses the verified ESPN abbreviation map from wnba_opp_intel.
    MLB/NBA: uses the first token of each full team name (e.g. "San" from
    "San Francisco Giants"), which isn't perfect but triggers the game-level
    fallback scan — far better than storing the player name.
    """
    try:
        if sport.upper() == "WNBA":
            from core.intelligence.wnba_opp_intel import _to_abbr
            a = _to_abbr(away_team) or away_team.split()[0].upper()[:4]
            h = _to_abbr(home_team) or home_team.split()[0].upper()[:4]
            return f"{a}v{h}"
        # MLB / NBA: full team names → pick the last word (city teams end in
        # a nickname, e.g. "Chicago Cubs" → "CUBS"; "San Francisco Giants" → "GIANTS")
        # but the prop_grader's fallback handles imprecise matches, so just use
        # a token that narrows the search more than the player name would.
        a_tok = away_team.split()[-1].upper()[:6]
        h_tok = home_team.split()[-1].upper()[:6]
        return f"{a_tok}v{h_tok}"
    except Exception:
        return "UNKvUNK"


# ── Public entry point ───────────────────────────────────────────────────────

def get_player_prop_candidates(
    sport: str,
    raw_mode: bool = False,
) -> list[dict[str, Any]]:
    """
    Return player prop candidates for today's games in *sport*.

    Each candidate dict is structurally identical to a game-total candidate and
    can be processed by _validate_candidate_for_simulation() → engine.analyze()
    → run_gatekeeper() without any modifications to the caller's pipeline.

    The `player` field is set to the player's name (non-null) so the existing
    thin-data check and the MiniApp's props-tab filter both work correctly.

    raw_mode=True (revalidation only):
        Skips Rule 2 (cross-bookmaker consensus gate) and uses the consensus
        median as the sportsbook_line so the revalidation engine can detect
        significant line movement even when books are currently spread apart.
    """
    from core.time_utils import now_est, convert_to_est
    from core.api_connector import normalize_api_timestamp

    sport_up    = sport.upper()
    api_sport   = _SPORT_KEY.get(sport_up)
    markets_str = _PROP_MARKETS.get(sport_up)

    # Strip performance-suspended markets from the API request
    if markets_str and _SUSPENDED_PROP_MARKETS:
        _active = [m for m in markets_str.split(",") if m.strip() not in _SUSPENDED_PROP_MARKETS]
        if len(_active) < len(markets_str.split(",")):
            logger.info(
                f"[player_props] {sport_up} suspended markets filtered: "
                + ", ".join(_SUSPENDED_PROP_MARKETS & set(markets_str.split(",")))
            )
        markets_str = ",".join(_active) if _active else None
    if not api_sport or not markets_str:
        return []

    today_et = now_est().date()
    events   = _fetch_events(api_sport)
    if not events:
        logger.debug(f"[player_props] no events found for {sport_up}")
        return []

    # ── Fetch PropLine supplementary data (one call covers all events) ────────
    # Done once here, outside the event loop, to avoid redundant API hits.
    # Apply the same suspension filter as the Odds API path — suspended markets
    # must not re-enter through PropLine.
    _propline_mkts = [
        m for m in _PROPLINE_MARKETS.get(sport_up, [])
        if m not in _SUSPENDED_PROP_MARKETS
    ]
    if _propline_mkts != _PROPLINE_MARKETS.get(sport_up, []):
        _filtered = set(_PROPLINE_MARKETS.get(sport_up, [])) & _SUSPENDED_PROP_MARKETS
        logger.info(
            f"[player_props] {sport_up} PropLine suspended markets filtered: "
            + ", ".join(sorted(_filtered))
        )
    _valid_markets   = set(_PROP_PRIOR.keys())   # same gate used in the loop
    _propline_all: dict[tuple[str, str], list[dict]] = {}
    if _propline_mkts:
        try:
            from core.propline_client import fetch_propline_books, _match_event
            _propline_all = fetch_propline_books(api_sport, _propline_mkts)
            logger.debug(
                f"[player_props] PropLine {sport_up}: "
                f"{len(_propline_all)} event(s) with data."
            )
        except Exception as _pl_exc:
            logger.warning(f"[player_props] PropLine fetch skipped: {_pl_exc}")

    candidates: list[dict[str, Any]] = []
    events_today = 0

    for event in events[:_MAX_EVENTS_PER_SPORT]:
        # Filter to today's games only
        commence = event.get("commence_time", "")
        try:
            game_time_utc = normalize_api_timestamp(commence)
            game_et       = convert_to_est(game_time_utc)
        except Exception:
            continue
        if game_et.date() != today_et:
            continue
        # Skip games that have already started (same rule as game totals)
        if game_et <= now_est():
            continue

        event_id  = event["id"]
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        # ── Fetch game spread for blowout-risk layer (WNBA only) ──────────
        # One Odds API call per event; result passed to _wnba_prop_projection.
        # Spread is the home-team line (negative = home is favoured).
        _game_spread: float | None = None
        if sport_up == "WNBA":
            try:
                _sp_data = _fetch_event_props(api_sport, event_id, "spreads")
                if _sp_data:
                    for _bk in _sp_data.get("bookmakers", []):
                        for _mkt in _bk.get("markets", []):
                            if _mkt.get("key") == "spreads":
                                for _oc in _mkt.get("outcomes", []):
                                    if _oc.get("name", "").lower() == home_team.lower():
                                        try:
                                            _game_spread = float(_oc["point"])
                                        except (KeyError, TypeError, ValueError):
                                            pass
                                        break
                            if _game_spread is not None:
                                break
                        if _game_spread is not None:
                            break
                logger.debug(
                    f"[player_props] WNBA spread for {away_team} @ {home_team}: "
                    f"{_game_spread:+.1f}" if _game_spread is not None
                    else f"[player_props] WNBA spread unavailable for {event_id}"
                )
            except Exception as _sp_exc:
                logger.debug(f"[player_props] spread fetch error: {_sp_exc}")

        props_data = _fetch_event_props(api_sport, event_id, markets_str)
        # Skip only when BOTH Odds API and PropLine have no data for this event.
        # WNBA player props are not on the Odds API (422) but ARE on PropLine —
        # don't discard the event; PropLine will populate the prop_map below.
        if not props_data and not _propline_all:
            continue

        events_today += 1

        # ── Rule 1 + 2: Collect ALL bookmaker lines per (player, market, direction).
        # Track every individual book's line for consensus (median) validation.
        # Structure: prop_map[(player, mkt_key)][direction] = {
        #   best_line, best_odds, best_book, book_count, all_lines: list[float]
        # }
        prop_map: dict[tuple[str, str], dict[str, dict]] = {}

        for bk in props_data.get("bookmakers", []):
            bk_title = bk.get("title") or bk.get("key", "Unknown")
            for mkt in bk.get("markets", []):
                mkt_key = mkt.get("key", "")
                if mkt_key not in _PROP_PRIOR:
                    continue
                for outcome in mkt.get("outcomes", []):
                    player_name = outcome.get("description", "").strip()
                    direction   = outcome.get("name", "").lower()
                    point       = outcome.get("point")
                    price       = outcome.get("price")
                    if (
                        not player_name
                        or direction not in ("over", "under")
                        or point is None
                        or price is None
                    ):
                        continue
                    key = (player_name, mkt_key)
                    if key not in prop_map:
                        prop_map[key] = {}
                    existing = prop_map[key].get(direction)
                    if existing is None:
                        prop_map[key][direction] = {
                            "best_line":  float(point),
                            "best_odds":  int(price),
                            "best_book":  bk_title,
                            "book_count": 1,
                            "all_lines":  [float(point)],
                            "book_lines": [{"book": bk_title, "line": float(point)}],
                        }
                    else:
                        new_pt = float(point)
                        # Skip alt-line variants: if this line deviates more than
                        # the consensus-drift threshold from the current pool median
                        # it is a different bet class (e.g. "2+ HRs" vs "1+ HR").
                        if abs(new_pt - _stats.median(existing["all_lines"])) > _LINE_CONSENSUS_MAX_DRIFT:
                            continue
                        existing["all_lines"].append(new_pt)
                        existing["book_count"] += 1
                        existing.setdefault("book_lines", []).append(
                            {"book": bk_title, "line": new_pt}
                        )
                        if int(price) > existing["best_odds"]:
                            existing["best_line"] = new_pt
                            existing["best_odds"] = int(price)
                            existing["best_book"] = bk_title

        # ── Merge PropLine supplementary bookmakers ───────────────────────────
        # Look up this game in the PropLine sport-level response by team names.
        # _propline_all is already fetched once before this loop.
        if _propline_all:
            try:
                from core.propline_client import _match_event
                _pl_books = _match_event(home_team, away_team, _propline_all)
                if _pl_books:
                    _merge_propline_books(prop_map, _pl_books, _valid_markets)
                    logger.debug(
                        f"[player_props] PropLine merged {len(_pl_books)} "
                        f"book(s) for {away_team} @ {home_team}"
                    )
            except Exception as _pl_exc:
                logger.debug(f"[player_props] PropLine merge error: {_pl_exc}")

        # ── Timestamp for this batch of lines ─────────────────────────────────
        _now_et     = datetime.now(timezone.utc)
        _verified_at = _now_et.strftime("%-I:%M %p ET")   # e.g. "9:02 AM ET"

        # ── Build one candidate per (player, market, direction) ───────────────
        for (player_name, mkt_key), sides in prop_map.items():
            prior       = _PROP_PRIOR[mkt_key]
            league_std  = prior["std"]

            # ── Resolve player stats (real data or NO PLAY) ───────────────────
            # WNBA: use the per-minute × projected-minutes model (full pipeline).
            # NBA/MLB: use the existing stat lookup + weighted projection.
            _proj_minutes:    float = 25.0
            _min_range:       float = 0.0
            _min_stability:   str   = "moderate"
            _role_label:      str   = "unknown"

            _blowout_level = "none"
            if sport_up == "WNBA":
                (weighted_proj, season_avg, l5_avg, l10_avg,
                 _proj_minutes, _min_range, _min_stability, _role_label,
                 _blowout_level) = (
                    _wnba_prop_projection(
                        player_name, mkt_key,
                        matchup_context=f"{away_team} @ {home_team}",
                        home_team=home_team,
                        away_team=away_team,
                        spread=_game_spread,
                    )
                )
            elif sport_up == "NBA":
                season_avg, l5_avg, l10_avg = _espn_player_stats(
                    player_name, sport_up, mkt_key
                )
                weighted_proj = _weighted_projection(
                    season_avg=season_avg or 0.0,
                    l5_avg=l5_avg,
                    l10_avg=l10_avg,
                )
            else:
                season_avg, l5_avg, l10_avg = _mlb_player_stats(
                    player_name, mkt_key
                )
                weighted_proj = _weighted_projection(
                    season_avg=season_avg or 0.0,
                    l5_avg=l5_avg,
                    l10_avg=l10_avg,
                )

            data_available = bool(season_avg and season_avg > 0)

            if not data_available:
                # Phase 1: hard block — no real stats = NO PLAY for all directions
                logger.info(
                    f"[player_props] NO PLAY: {player_name} {mkt_key} — "
                    "real stats unavailable, candidate blocked (no fallback allowed)"
                )
                continue

            # ── Workload scaling for pitcher strikeouts (MLB only) ────────────
            # Scale the historical projection by expected_ip / LEAGUE_AVG_IP so
            # a starter projected to throw 4.5 IP yields ~18% fewer K chances
            # than one projected to throw 6.0 IP.  Confidence-weighted blend
            # ensures a low-confidence workload blends back towards 1.0.
            _workload_confidence_tier = None   # read by gatekeeper Step 1b4 (OVER hook-risk penalty)
            if sport_up == "MLB" and mkt_key == "pitcher_strikeouts":
                try:
                    from core.pitcher_workload import (
                        get_pitcher_workload,
                        get_k_workload_scale,
                    )
                    from core.strikeout_matchup import get_k_matchup_scale
                    _wl = get_pitcher_workload(
                        pitcher_name = player_name,
                        event_id     = event_id,
                    )
                    _workload_confidence_tier = _wl.confidence_tier
                    # ── Layer 1: workload scale (innings-adjusted) ────────────
                    _k_scale = get_k_workload_scale(_wl)
                    if _k_scale != 1.0:
                        _orig = weighted_proj
                        weighted_proj = round(weighted_proj * _k_scale, 2)
                        logger.debug(
                            f"[player_props] K workload scale {player_name}: "
                            f"{_orig:.2f} → {weighted_proj:.2f} "
                            f"(×{_k_scale:.3f}, IP={_wl.expected_ip:.1f}, "
                            f"conf={_wl.confidence:.0f}%)"
                        )
                    # ── Layers 2–7: matchup scale (splits, lineup, SwStr%, CSW%, velo)
                    _matchup_scale = get_k_matchup_scale(
                        pitcher_name = player_name,
                        opp_abbr     = _wl.opp_abbr,
                        pitcher_id   = _wl.pitcher_id,
                        game_date    = _wl.game_date,
                    )
                    if _matchup_scale != 1.0:
                        _orig_m = weighted_proj
                        weighted_proj = round(weighted_proj * _matchup_scale, 2)
                        logger.debug(
                            f"[player_props] K matchup scale {player_name}: "
                            f"{_orig_m:.2f} → {weighted_proj:.2f} "
                            f"(×{_matchup_scale:.4f})"
                        )
                except Exception as _wl_exc:
                    logger.debug(
                        f"[player_props] workload/matchup K scale failed for "
                        f"{player_name}: {_wl_exc}"
                    )

            hist_mean = weighted_proj
            seed_val  = float(sum(ord(c) for c in (player_name + mkt_key)[:16]))

            # For WNBA: use real per-game logs from the ESPN cache when available
            # instead of synthetic data drawn from league_std. The synthetic fallback
            # over-states variance for consistent players (e.g. Kelsey Plum's real
            # σ=1.26 vs league_std=2.0), causing unnecessary stability rejections and
            # collapsing confidence to 58–63% even on high-edge candidates.
            # With real logs, NUTS infers the player's actual game-to-game sigma.
            if sport_up == "WNBA":
                _box_col   = _WNBA_BOX_COL.get(mkt_key)
                _real_logs = (
                    _WNBA_STATS_CACHE.get(player_name.lower(), {}).get(_box_col, [])
                    if _box_col else []
                )
                if len(_real_logs) >= 5:
                    hist = [float(v) for v in _real_logs]
                    # Use player's actual game-to-game SD as the sigma prior so NUTS
                    # starts from a realistic scale (still refined from the data).
                    league_std = max(0.3, round(_stats.stdev(_real_logs), 3))
                else:
                    hist = _synthetic_history(hist_mean, league_std, n=15, seed=seed_val)
            else:
                hist = _synthetic_history(hist_mean, league_std, n=15, seed=seed_val)

            display_market = _MARKET_DISPLAY.get(mkt_key, mkt_key.replace("_", " ").title())

            for direction, info in sides.items():
                line      = info["best_line"]
                odds      = info["best_odds"]
                book      = info["best_book"]
                book_cnt  = info["book_count"]
                all_lines = info["all_lines"]

                # ── Rule 2: Cross-bookmaker consensus validation ───────────────
                consensus_line = _stats.median(all_lines)
                drift = abs(line - consensus_line)
                if drift > _LINE_CONSENSUS_MAX_DRIFT:
                    if raw_mode:
                        line = consensus_line
                        logger.debug(
                            f"[player_props] raw_mode: Rule 2 bypassed for "
                            f"{player_name} {mkt_key} {direction} — using "
                            f"consensus {consensus_line:.1f}"
                        )
                    else:
                        logger.warning(
                            f"[player_props] LINE VALIDATION FAILED (Rule 2) — "
                            f"{player_name} {mkt_key} {direction}: best line {line} "
                            f"deviates {drift:.2f} from consensus {consensus_line:.1f} "
                            f"(threshold: {_LINE_CONSENSUS_MAX_DRIFT}). Skipping."
                        )
                        continue

                # ── Market Influence Score ────────────────────────────────────
                mis_score, mis_lbl = compute_mis(
                    all_lines=all_lines,
                    book_count=book_cnt,
                    best_line=line,
                    consensus_line=consensus_line,
                    sport=sport_up,
                )

                # ── Data Reliability Score ────────────────────────────────────
                drs = compute_data_reliability(
                    has_real_stats=data_available,
                    book_count=book_cnt,
                    has_l5=(l5_avg is not None),
                    has_l10=(l10_avg is not None),
                )

                # ── Phase 2: Sharp action, steam, RLM detection ───────────────
                _book_lines_dir = info.get("book_lines", [])
                _sharp_action   = detect_sharp_action(_book_lines_dir, direction)
                _steam_detected = detect_steam_move(all_lines, book_cnt, sport_up)
                _rlm_detected   = detect_reverse_line_movement(
                    _sharp_action.get("sharp_consensus_line"),
                    _sharp_action.get("rec_consensus_line"),
                    direction,
                )
                _sharp_signal   = _sharp_action["signal_type"]

                d_label     = "O" if direction == "over" else "U"
                safe_player = player_name.replace(" ", "_").replace("'", "").replace(".", "")
                bet_id      = f"prop_{safe_player}_{mkt_key}_{direction}_{event_id[:8]}"
                game_label  = (
                    f"{away_team.replace(' ','_')}@{home_team.replace(' ','_')}"
                    f"_{sport_up}"
                )

                # Build factor text with projection provenance
                proj_note = f"Season avg: {season_avg:.1f}"
                if l5_avg is not None:
                    proj_note += f" | L5: {l5_avg:.1f}"
                if l10_avg is not None:
                    proj_note += f" | L10: {l10_avg:.1f}"
                proj_note += f" | Proj: {weighted_proj:.1f}"
                if sport_up == "WNBA" and _min_stability != "unknown":
                    proj_note += (
                        f" | Min: {_proj_minutes:.0f} ({_min_stability}"
                        f", range {_min_range:.0f}min, {_role_label})"
                    )

                factor = (
                    f"{player_name} — {display_market} {d_label}{line} "
                    f"({book_cnt} book{'s' if book_cnt != 1 else ''}, best: {book}, "
                    f"consensus: {consensus_line:.1f}). "
                    f"{proj_note}. MIS: {mis_score}/100 ({mis_lbl}). "
                    f"Verified: {_verified_at}."
                )

                # ── Store metadata for pre-publish re-verification ────────────
                _PROP_META[bet_id] = {
                    "event_id":      event_id,
                    "api_sport":     api_sport,
                    "player":        player_name,
                    "market_key":    mkt_key,
                    "direction":     direction,
                    "opening_line":  line,
                    "consensus_line": consensus_line,
                    "verified_at":   _verified_at,
                }

                candidates.append({
                    "bet_id":                  bet_id,
                    "game_id":                 game_label,
                    "away_team":               away_team,
                    "home_team":               home_team,
                    "full_team_name":          player_name,
                    "team":                    _prop_matchup_key(
                                                   away_team, home_team, sport_up
                                               ),
                    "market":                  display_market,
                    "player":                  player_name,
                    "direction":               direction,
                    "sportsbook_line":         line,
                    "opening_line":            line,
                    "consensus_line":          consensus_line,
                    "verified_at":             _verified_at,
                    "american_odds":           odds,
                    "bookmaker_source":        book,
                    "book_count":              book_cnt,
                    "historical_data":         hist,
                    "league_mean":             hist_mean,
                    "league_std":              league_std,
                    "context":                 "regular",
                    "volatility_index":        None,
                    "recent_n":                5,
                    "factor":                  factor,
                    "game_time_utc":           game_time_utc,
                    # Phase 1 + MIS fields
                    "data_available":          data_available,
                    "data_reliability_score":  drs,
                    "mis_score":               mis_score,
                    "weighted_projection":     weighted_proj,
                    "season_avg":              season_avg,
                    "l5_avg":                  l5_avg,
                    "l10_avg":                 l10_avg,
                    "fallback_used":           False,
                    # WNBA per-minute model fields (read by gatekeeper Step 1b2 + 1b3)
                    "minutes_stability":       _min_stability,
                    "projected_minutes":       _proj_minutes,
                    "minutes_range":           _min_range,
                    "role_label":              _role_label,
                    "blowout_level":           _blowout_level,
                    # Workload risk (read by gatekeeper Step 1b4 — K OVER hook-risk penalty)
                    "workload_confidence_tier": _workload_confidence_tier,
                    # Phase 2 — market intelligence signals
                    "sharp_signal":            _sharp_signal,
                    "sharp_label":             _sharp_action["signal_label"],
                    "sharp_book_count":        _sharp_action["sharp_book_count"],
                    "steam_detected":          _steam_detected,
                    "rlm_detected":            _rlm_detected,
                })

    # Cap to the most-liquid props (highest book count) to keep engine runtime
    # manageable.  Both Over and Under for each player+market are included in
    # the sort so the engine can pick the better-valued direction.
    candidates.sort(key=lambda c: c["book_count"], reverse=True)
    capped = candidates[:_MAX_CANDIDATES_PER_SPORT]

    logger.info(
        f"[player_props] {sport_up}: {events_today} event(s) → "
        f"{len(candidates)} raw candidates → {len(capped)} after cap."
    )
    return capped


# ---------------------------------------------------------------------------
# Pre-publish re-verification helpers (used by core/line_validator.py)
# ---------------------------------------------------------------------------

def get_prop_meta(bet_id: str) -> dict | None:
    """Return stored line metadata for a prop pick, or None if not found."""
    return _PROP_META.get(bet_id)


def refresh_prop_line(
    api_sport: str,
    event_id: str,
    player: str,
    market_key: str,
    direction: str,
) -> float | None:
    """
    Re-fetch the current sportsbook line for a specific player prop.

    Returns the median line across all bookmakers that carry the market,
    or None if the line cannot be retrieved.
    """
    props_data = _fetch_event_props(api_sport, event_id, market_key)
    if not props_data:
        return None

    lines: list[float] = []
    player_lower    = player.lower()
    direction_lower = direction.lower()

    for bk in props_data.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt.get("key", "") != market_key:
                continue
            for outcome in mkt.get("outcomes", []):
                if (
                    outcome.get("description", "").strip().lower() == player_lower
                    and outcome.get("name", "").lower() == direction_lower
                ):
                    point = outcome.get("point")
                    if point is not None:
                        lines.append(float(point))

    if not lines:
        return None
    return _stats.median(lines)
