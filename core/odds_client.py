"""
odds_client.py

Fetches today's game slate and betting lines from The Odds API.

Key rotation
------------
Reads THE_ODDS_API_KEY, THE_ODDS_API_KEY_2, THE_ODDS_API_KEY_3 in order.
On HTTP 401/429 (exhausted / invalid) automatically rotates to the next key.

Returns candidates in the format expected by _get_game_candidates() in main.py.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib import request as urllib_req
from urllib.error import HTTPError
from zoneinfo import ZoneInfo

from core.market_intelligence import (
    SHARP_BOOKS,
    compute_mis,
    detect_sharp_action,
    detect_steam_move,
)

_ET = ZoneInfo("America/New_York")

BASE_URL = "https://api.the-odds-api.com/v4"

# ---------------------------------------------------------------------------
# Sport key mapping
# ---------------------------------------------------------------------------

_SPORT_KEY: dict[str, str] = {
    "WNBA": "basketball_wnba",
    "NBA":  "basketball_nba",
    "MLB":  "baseball_mlb",
}

# ---------------------------------------------------------------------------
# Full team name → short abbreviation
# ---------------------------------------------------------------------------

_ABBREV: dict[str, str] = {
    # WNBA (original franchises)
    "Atlanta Dream":              "ATL",
    "Chicago Sky":                "CHI",
    "Connecticut Sun":            "CON",
    "Dallas Wings":               "DAL",
    "Indiana Fever":              "IND",
    "Las Vegas Aces":             "LVA",
    "Los Angeles Sparks":         "LAS",
    "Minnesota Lynx":             "MIN",
    "New York Liberty":           "NYL",
    "Phoenix Mercury":            "PHX",
    "Seattle Storm":              "SEA",
    "Washington Mystics":         "WAS",
    # WNBA 2026 expansion franchises
    "Golden State Valkyries":     "GSV",
    "Portland Fire":              "POR",
    "Toronto Tempo":              "TOR",
    # NBA
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
    # MLB
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

# ---------------------------------------------------------------------------
# Bayesian priors — keyed by sport, then by market type.
#
# "totals"     = full-game total (both teams combined).  Used when the Odds
#                API is queried with markets=totals.
# "team_total" = one team's individual score.  Used if/when the team_totals
#                market is added to the fetch.
#
# These are the league-average values the synthetic history is centred on.
# Keeping them market-aware prevents a game-total prior (165 pts) from being
# applied to a team-total line (~82 pts), which would produce a massive
# z-score and confidence ≈ 99 % in the wrong direction.
# ---------------------------------------------------------------------------

_GAME_TOTAL_PRIOR: dict[str, dict[str, dict[str, float]]] = {
    "WNBA": {
        "totals":     {"mean": 165.0, "std": 8.0},
        "team_total": {"mean": 80.0,  "std": 6.0},
    },
    "NBA": {
        "totals":     {"mean": 222.0, "std": 10.0},
        "team_total": {"mean": 111.0, "std": 7.0},
    },
    "MLB": {
        "totals":     {"mean": 8.5,  "std": 2.0},
        "team_total": {"mean": 4.25, "std": 1.5},
    },
}

# ---------------------------------------------------------------------------
# MLB Stats API — team name → MLB Stats API teamId
# Used by get_mlb_game_totals_history() to fetch real scoring history.
# ---------------------------------------------------------------------------

_MLB_TEAM_ID: dict[str, int] = {
    "Arizona Diamondbacks": 109,
    "Atlanta Braves":       144,
    "Baltimore Orioles":    110,
    "Boston Red Sox":       111,
    "Chicago Cubs":         112,
    "Chicago White Sox":    145,
    "Cincinnati Reds":      113,
    "Cleveland Guardians":  114,
    "Colorado Rockies":     115,
    "Detroit Tigers":       116,
    "Houston Astros":       117,
    "Kansas City Royals":   118,
    "Los Angeles Angels":   108,
    "Los Angeles Dodgers":  119,
    "Miami Marlins":        146,
    "Milwaukee Brewers":    158,
    "Minnesota Twins":      142,
    "New York Mets":        121,
    "New York Yankees":     147,
    "Athletics":            133,
    "Oakland Athletics":    133,
    "Philadelphia Phillies":143,
    "Pittsburgh Pirates":   134,
    "San Diego Padres":     135,
    "San Francisco Giants": 137,
    "Seattle Mariners":     136,
    "St. Louis Cardinals":  138,
    "Tampa Bay Rays":       139,
    "Texas Rangers":        140,
    "Toronto Blue Jays":    141,
    "Washington Nationals": 120,
}

# Process-level cache: (teamId, as_of_date) → list of recent game totals.
# Populated on first fetch; reused for all subsequent candidates in the
# same run without additional API calls. Keyed on as_of_date (None in live
# mode) so a replay loop over multiple historical dates can't leak the
# first date's window into every other date.
_MLB_HISTORY_CACHE: dict[tuple[int, str | None], list[float]] = {}

# ---------------------------------------------------------------------------
# Key rotation
# ---------------------------------------------------------------------------

def _load_api_keys() -> list[str]:
    """Return all configured Odds API keys in rotation order."""
    keys: list[str] = []
    for name in ("THE_ODDS_API_KEY", "THE_ODDS_API_KEY_2", "THE_ODDS_API_KEY_3"):
        k = os.getenv(name, "").strip()
        if k:
            keys.append(k)
    return keys


def _fetch(path: str, params: str) -> Any:
    """
    GET request with automatic key rotation.

    Tries keys in order. On HTTP 401 or 429 (key exhausted / invalid)
    rotates to the next key. Raises RuntimeError if all keys fail.
    """
    keys = _load_api_keys()
    if not keys:
        raise RuntimeError(
            "No Odds API key configured. Set THE_ODDS_API_KEY in Secrets."
        )

    last_err: Exception | None = None
    for key in keys:
        url = f"{BASE_URL}/{path}?apiKey={key}&{params}"
        try:
            with urllib_req.urlopen(url, timeout=12) as resp:
                body = json.loads(resp.read().decode())
                remaining = resp.headers.get("x-requests-remaining", "?")
                print(
                    f"[odds_client] {path}  quota_remaining={remaining}  "
                    f"key=…{key[-6:]}",
                    flush=True,
                )
                return body
        except HTTPError as exc:
            if exc.code in (401, 429):
                print(
                    f"[odds_client] Key …{key[-6:]} returned HTTP {exc.code} "
                    "(exhausted/invalid) — rotating to next key.",
                    flush=True,
                )
                last_err = exc
                continue
            raise
    raise RuntimeError(
        f"All Odds API keys failed. Last error: {last_err}"
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _abbrev(full_name: str) -> str:
    return _ABBREV.get(full_name, full_name[:3].upper())


def _implied_prob(american_odds: int) -> float:
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    return 100 / (american_odds + 100)


def _seasonal_std_multiplier(sport: str) -> float:
    """
    Prior-uncertainty decay: as the season progresses we have more real data,
    so synthetic priors should be *tighter* (lower std multiplier).

    Calibration timeline (same for all sports):
        Pre-season (Oct/Nov):   mult = 1.40   ← high uncertainty, wide priors
        Early season (Dec/Jan): mult = 1.10
        Mid-season  (Jun):      mult = 0.70   ← 50 % reduction vs pre-season
        Late season (Aug/Sep):  mult = 0.50   ← high certainty, tight priors

    A 0.7 mid-season multiplier means each synthetic observation contributes
    half as much variance as pre-season, reflecting the accumulated in-season
    evidence that tightens our league-average estimates.
    """
    from datetime import date
    month = date.today().month
    # Map calendar month → multiplier (Northern-hemisphere US sports season)
    schedule = {
        10: 1.40, 11: 1.40,           # pre-season / early season
        12: 1.10,  1: 1.10,
        2:  0.90,  3: 0.90,
        4:  0.80,  5: 0.80,
        6:  0.70,  7: 0.70,           # mid-season (current)
        8:  0.60,  9: 0.50,           # late season / playoffs
    }
    return schedule.get(month, 0.70)


def _synthetic_history(
    seed: float,
    mean: float,
    std: float,
    n: int = 50,
    *,
    sport: str = "",
) -> list[float]:
    """
    Deterministic synthetic game-log history exactly centred at *mean*.

    Key guarantees:
    1. The sample mean equals *mean* exactly (centred sample) so the
       Bayesian posterior mean starts at league_mean, not shifted by
       random noise.  This prevents the prior from accidentally sitting
       at the market line and killing edge in both directions.
    2. The seed is game-specific (from the game_id hash) so different
       games produce different-looking histories even for the same sport.
    3. The spread uses a seasonal multiplier — tighter mid-season when
       we have more real data, wider pre-season under high uncertainty.
    4. Strict independence from sportsbook lines: *mean* is always the
       sport's league average, never the current market line.  The line
       enters only at the edge-derivation step.

    n=50 (default): posterior_std ≈ data_std/√50 = 5.6/7.07 ≈ 0.79 for WNBA,
    giving z≈1.90 for a 1.5-point edge → conf≈97.5 (Nuke-tier).
    Keeps NUTS sampling tractable (4 s → ~5 s per candidate).
    """
    mult = _seasonal_std_multiplier(sport)
    random.seed(int(seed) % (2 ** 31))
    raw = [random.gauss(0, std * mult) for _ in range(n)]
    # Force exact centering: subtract the empirical mean so sample_mean == 0,
    # then shift to league_mean.  The spread is preserved.
    sample_mean = sum(raw) / n
    return [round(mean + (r - sample_mean), 1) for r in raw]


def _mlb_dynamic_game_window(as_of_date: str | None = None) -> int:
    """
    Return how many completed games to pull for MLB history, scaled to how
    deep into the season we are.

    Why dynamic instead of a fixed 20:
      The Bayesian shrinkage prior (prior_weight_f5_games=10 in sport_config)
      carries 10 pseudo-game-count worth of weight. With only 20 real games
      the prior controls 10/(20+10) = 33% of the posterior -- fine in April
      when 20 games is most of the season, but by late June a team has ~80
      completed games and we were deliberately discarding 60 of them, keeping
      the posterior anchored near the 8.5-run league average even when a team
      is consistently over or under it.

    Window schedule (MLB plays ~27 games/month, ~1 per 1.37 days):
      Weeks  1-3  (<=20 games in):  all available (min 10)     -- early season
      Weeks  4-8  (21-54 games in): 40 games                   -- April/May
      Weeks  9-18 (55-108 games):   60 games                   -- June/August
      Weeks 19+   (>108 games):     80 games (~half season)    -- September

    *as_of_date* : when given (replay mode), "how deep into the season" is
    computed relative to that date instead of the real current date.
    """
    from datetime import date as _date
    from models.sport_config import MLB as _MLB_CFG

    start_m, start_d = _MLB_CFG["regular_season_start_month_day"]
    today = _date.fromisoformat(as_of_date) if as_of_date else _date.today()
    season_start = _date(today.year, start_m, start_d)
    days_in = max(0, (today - season_start).days)
    games_played_approx = int(days_in / 1.37)

    if games_played_approx <= 20:
        return max(10, games_played_approx)
    elif games_played_approx <= 54:
        return 40
    elif games_played_approx <= 108:
        return 60
    else:
        return 80


def get_mlb_game_totals_history(
    team_name: str,
    games: int | None = None,
    as_of_date: str | None = None,
) -> list[float] | None:
    """
    Fetch the last *games* completed regular-season game totals (home + away
    runs) for *team_name* from the MLB Stats API (free, no key required).

    *games* defaults to a dynamic window scaled to how deep into the MLB
    season we are (see _mlb_dynamic_game_window). Pass an explicit integer
    to override (useful for backtesting a fixed window).

    *as_of_date* : ISO 'YYYY-MM-DD' string. When set, only games played
    on-or-before this date are included, and results are cached separately
    per date — this is what makes the function safe to call from a replay
    loop over historical dates instead of always describing "today".

    Results are cached at process level so each team (and, in replay mode,
    each team+date) is only fetched once per engine run regardless of how
    many game candidates involve that team.

    Returns
    -------
    list[float]
        Actual game-total observations (e.g. [7.0, 9.0, 5.0, ...]) in
        chronological order, most recent last.
    None
        On any failure — caller should fall back to ``_synthetic_history()``.
    """
    from datetime import date as _date

    if games is None:
        games = _mlb_dynamic_game_window(as_of_date=as_of_date)

    team_id = _MLB_TEAM_ID.get(team_name)
    if team_id is None:
        print(
            f"[odds_client] MLB team ID not found for '{team_name}' "
            "— falling back to synthetic history.",
            flush=True,
        )
        return None

    cache_key = (team_id, as_of_date)
    if cache_key in _MLB_HISTORY_CACHE:
        return _MLB_HISTORY_CACHE[cache_key]

    try:
        season = _date.fromisoformat(as_of_date).year if as_of_date else _date.today().year
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&teamId={team_id}&season={season}&gameType=R"
        )
        with urllib_req.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        totals: list[float] = []
        for day in data.get("dates", []):
            day_date = day.get("date")
            if as_of_date is not None and (not day_date or day_date > as_of_date):
                continue
            for g in day.get("games", []):
                if g.get("status", {}).get("statusCode") != "F":
                    continue
                home_score = g["teams"]["home"].get("score")
                away_score = g["teams"]["away"].get("score")
                if home_score is not None and away_score is not None:
                    totals.append(float(home_score + away_score))

        if not totals:
            print(
                f"[odds_client] No completed MLB games for teamId={team_id} "
                f"('{team_name}', as_of={as_of_date}) — falling back to synthetic history.",
                flush=True,
            )
            return None

        recent = totals[-games:]
        _MLB_HISTORY_CACHE[cache_key] = recent
        mean_val = sum(recent) / len(recent)
        print(
            f"[odds_client] MLB real history: {team_name} (id={team_id}) "
            f"— {len(recent)} game totals (window={games}), mean={mean_val:.2f}",
            flush=True,
        )
        return recent

    except Exception as exc:
        print(
            f"[odds_client] MLB Stats API error for '{team_name}': {exc} "
            "— falling back to synthetic history.",
            flush=True,
        )
        return None


# ---------------------------------------------------------------------------
# Directive 1: Module-level helpers (hoisted out of the game loop)
# ---------------------------------------------------------------------------

def _validate_game(game: dict[str, Any]) -> tuple[bool, str]:
    """
    Fail-fast gate applied before any per-game processing.

    Returns (True, "") when the game record is usable, or (False, reason)
    to abort immediately.  Checking here — before timestamp parsing,
    bookmaker iteration, and history generation — eliminates wasted work
    on malformed records from the API response.
    """
    for field in ("home_team", "away_team", "id", "commence_time"):
        if not game.get(field):
            return False, f"missing field '{field}'"
    bk = game.get("bookmakers")
    if not isinstance(bk, list) or not bk:
        return False, "no bookmakers"
    return True, ""


def _best_side(
    lines: list[tuple[float, int, str]],
) -> tuple[float, int, str, int] | None:
    """
    Deduplicate bookmaker lines, keeping the best (highest) odds per book.

    Returns (line, odds, book_name, n_distinct_books) or None when fewer
    than 2 distinct books confirm the side (insufficient market consensus).

    Sharp-book preference: when a sharp book (Pinnacle, LowVig, BetAnySports)
    offers odds within 15 cents of the best available, it is preferred as the
    bookmaker_source so the card reflects the most efficient line in the market.

    Hoisted to module level from the game loop — defining functions inside
    a tight loop re-allocates the function object on every iteration.
    """
    if len(lines) < 2:
        return None
    per_book: dict[str, tuple[float, int]] = {}
    for ln, od, bk in lines:
        if bk not in per_book or od > per_book[bk][1]:
            per_book[bk] = (ln, od)
    if len(per_book) < 2:
        return None  # same book repeated — not true multi-book consensus
    by_odds = sorted(per_book.items(), key=lambda x: x[1][1], reverse=True)
    top_book, (top_line, top_od) = by_odds[0][0], by_odds[0][1]

    # Prefer sharp books when their odds are within 15 cents of the best.
    # Pinnacle and LowVig set the most efficient lines in the market — showing
    # their number gives the card more credibility as a sharp reference.
    _SHARP = {"Pinnacle", "pinnacle", "LowVig", "lowvig",
              "BetAnySports", "betanysports"}
    # Matchbook removed — EU-only book, consistently stale vs US market
    for book, (line, od) in by_odds:
        if book in _SHARP and od >= top_od - 15:
            top_book, top_line, top_od = book, line, od
            break

    return top_line, top_od, top_book, len(per_book)


def _et_day_bounds_utc(today_et: "date") -> tuple[str, str]:  # type: ignore[name-defined]
    """
    Directive 4: Compute commenceTimeFrom / commenceTimeTo for the Odds API.

    Returns ISO-8601 UTC strings bracketing the ET calendar day *today_et*,
    e.g. for 2026-06-01 ET (EDT = UTC-4):
        from  2026-06-01T04:00:00Z
        to    2026-06-02T04:00:00Z

    Filtering at the API level instead of locally reduces the response
    payload by 80-90 % on multi-sport days — no wasted bandwidth on
    yesterday's completed games or future-week matchups.
    """
    from datetime import date as _date, time as _time
    midnight_et  = datetime.combine(today_et, _time.min).replace(tzinfo=_ET)
    midnight_end = midnight_et + timedelta(days=1)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (
        midnight_et.astimezone(timezone.utc).strftime(fmt),
        midnight_end.astimezone(timezone.utc).strftime(fmt),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _fetch_historical(path: str, params: str, snapshot_iso: str) -> Any:
    """
    Fetch a historical snapshot from The Odds API's /v4/historical/ tree.

    Contract (per Odds API docs): the response body is wrapped as
    {"timestamp": ..., "previous_timestamp": ..., "next_timestamp": ...,
    "data": <same shape the live endpoint would return>}. This helper
    unwraps "data" so callers see the same shape as the live _fetch().

    NOTE: historical endpoints cost real API credits (10 credits per
    market/region/event for event-odds pulls) — only call when as_of_date
    is actually set.
    """
    hist_params = f"{params}&date={snapshot_iso}"
    raw = _fetch(f"historical/{path}", hist_params)
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    return raw


def fetch_todays_candidates(
    sport: str,
    as_of_date: str | None = None,
    snapshot_time: str = "10:00:00",
) -> list[dict[str, Any]]:
    """
    Return today's bet candidates for *sport*, using a two-tier strategy:

    Tier 1 — Slate cache (producer-consumer):
        Read from data/slate_cache/{SPORT}_{date}.json when the file exists
        and is ≤ 30 minutes old.  This decouples the analysis engine from
        the ingestion layer — repeated runs within a session share one API
        credit and the engine never blocks on network I/O.

    Tier 2 — Live Odds API fetch:
        On cache miss or stale entry, fetch from The Odds API with:
        • commenceTimeFrom / commenceTimeTo constrained to the ET calendar
          day (Directive 4 — API-level timezone-safe batching).
        • Per-game fail-fast validation before any processing (Directive 2).
        • Module-level helpers used throughout — no nested function defs
          inside the game loop (Directive 1 — native filtering).
        After a successful fetch, the validated candidates are written to
        the slate cache for the remainder of the session.

    Candidate structure
    -------------------
    Each dict contains: bet_id, game_id, away_team, home_team, team, market,
    direction, sportsbook_line, american_odds, bookmaker_source, book_count,
    historical_data, league_mean, league_std, context, volatility_index,
    recent_n, factor, game_time_utc.
    """
    from datetime import date as _date
    from core.api_connector import normalize_api_timestamp
    from core.time_utils import convert_to_est, now_est
    from core.slate_cache import read_slate, write_slate

    sport_up  = sport.upper()
    api_sport = _SPORT_KEY.get(sport_up)
    if not api_sport:
        return []

    prior = _GAME_TOTAL_PRIOR.get(sport_up, {}).get("totals", {"mean": 165.0, "std": 8.0})
    if as_of_date:
        today_et = _date.fromisoformat(as_of_date)
        date_str = as_of_date
        snapshot_iso = f"{as_of_date}T{snapshot_time}Z"
    else:
        today_et = now_est().date()
        date_str = today_et.isoformat()
        snapshot_iso = None

    # ── Tier 1: Slate cache ───────────────────────────────────────────────
    # In replay mode this reads/writes an isolated cache dir
    # (data/slate_cache_replay) instead of the live data/slate_cache — so a
    # backtest run shares one API credit across pipeline stages for the
    # same replay date without ever touching, or being served by, the live
    # cache.
    cache_dir = os.path.join("data", "slate_cache_replay") if as_of_date else None
    cached = read_slate(sport_up, date_str, cache_dir=cache_dir)
    if cached is not None:
        print(
            f"[odds_client] {sport_up} slate served from cache "
            f"({len(cached)} candidates — no API credit consumed).",
            flush=True,
        )
        return cached

    # ── Tier 2: Live API fetch with ET-bounded time filter, or the
    #    historical-snapshot endpoint when replaying a past date ──────────
    commence_from, commence_to = _et_day_bounds_utc(today_et)
    # regions=us,eu,us2 — us2 added so BetOnline (Odds API's "us2" region) is
    # actually returned. Previously only us (FanDuel/DK/BetMGM/Caesars-type
    # books) + eu (Pinnacle/etc.) were requested, so BetOnline never appeared
    # in bookmakers regardless of downstream weighting logic.
    params = (
        "regions=us,eu,us2&markets=totals&oddsFormat=american&dateFormat=iso"
        f"&commenceTimeFrom={commence_from}&commenceTimeTo={commence_to}"
    )

    try:
        if as_of_date is not None:
            raw_games = _fetch_historical(f"sports/{api_sport}/odds/", params, snapshot_iso)
        else:
            raw_games = _fetch(f"sports/{api_sport}/odds/", params)
    except Exception as exc:
        print(f"[odds_client] {sport} fetch failed: {exc}", flush=True)
        return []

    if not isinstance(raw_games, list):
        return []

    candidates: list[dict[str, Any]] = []

    for game in raw_games:
        # ── Directive 2: Fail-fast — abort this record immediately ───────
        ok, reason = _validate_game(game)
        if not ok:
            print(f"[odds_client] Skipping game (fail-fast): {reason}", flush=True)
            continue

        home_team = game["home_team"]
        away_team = game["away_team"]
        game_id   = game["id"]
        commence  = game["commence_time"]

        try:
            game_time_utc = normalize_api_timestamp(commence)
        except Exception as exc:
            print(f"[odds_client] Skipping game: bad commence_time '{commence}' — {exc}", flush=True)
            continue

        # Safety net: the API-level time filter should already restrict to
        # today, but keep this guard in case of DST edge cases.
        game_et = convert_to_est(game_time_utc)
        if game_et.date() != today_et:
            continue

        # Pre-game cutoff: skip any game that has already started so that
        # late manual runs never pick up in-play lines. In replay mode this
        # is anchored to the replay snapshot instant, not the live clock —
        # otherwise every game on a past date would look "already started".
        _cutoff_now = now_est() if as_of_date is None else convert_to_est(
            datetime.fromisoformat(snapshot_iso.replace("Z", "+00:00"))
        )
        if game_et <= _cutoff_now:
            print(
                f"[odds_client] Skipping {away_team} @ {home_team} — "
                f"game already started ({game_et.strftime('%H:%M ET')}).",
                flush=True,
            )
            continue

        # Collect over AND under lines per bookmaker
        over_lines:  list[tuple[float, int, str]] = []
        under_lines: list[tuple[float, int, str]] = []
        # Per-book lists for MIS computation and sharp-action detection
        book_lines_over:  list[dict] = []
        book_lines_under: list[dict] = []

        for bk in game.get("bookmakers", []):
            bk_title = bk.get("title") or bk.get("key", "Unknown")
            if bk_title == "Matchbook":  # EU-only, consistently stale vs US market
                continue
            for mkt in bk.get("markets", []):
                if mkt.get("key") != "totals":
                    continue
                for outcome in mkt.get("outcomes", []):
                    name  = outcome.get("name", "").lower()
                    pt    = outcome.get("point")
                    price = outcome.get("price")
                    # Directive 2: skip outcomes with missing required fields
                    if pt is None or price is None:
                        continue
                    if name == "over":
                        over_lines.append((float(pt), int(price), bk_title))
                        # "odds" added (previously line-only) — needed to devig
                        # a single book's own two-sided price later.
                        book_lines_over.append({"book": bk_title, "line": float(pt), "odds": int(price)})
                    elif name == "under":
                        under_lines.append((float(pt), int(price), bk_title))
                        book_lines_under.append({"book": bk_title, "line": float(pt), "odds": int(price)})

        if len(over_lines) < 2 and len(under_lines) < 2:
            continue

        home_abbr  = _abbrev(home_team)
        away_abbr  = _abbrev(away_team)
        game_label = f"{away_abbr}@{home_abbr}_{sport_up}_{date_str}"

        # Historical observations fed into the Bayesian engine.
        # MLB  — real game-total history from the MLB Stats API.
        # WNBA/NBA — real game-total history from ESPN (via game_logs module).
        # Fall back to synthetic history when any API is unavailable.
        game_seed = float(sum(ord(ch) for ch in game_id[:8]) % 100_000)

        # WNBA regime defaults — overridden below when sport is WNBA
        _wnba_league_std: float = prior["std"]
        _regime: str   = "neutral"
        _cet: float    = prior["mean"]

        # MLB pitching diagnostics — overridden below when sport is MLB
        _sp_factor = None
        _bp_factor = None

        def _combine_hists(h1: list[float], h2: list[float]) -> list[float]:
            """Interleave two teams' histories; append remainder of longer."""
            combined: list[float] = []
            for pair in zip(h1, h2):
                combined.extend(pair)
            longer = h1 if len(h1) >= len(h2) else h2
            shorter_len = min(len(h1), len(h2))
            combined.extend(longer[shorter_len:])
            return combined

        if sport_up == "MLB":
            home_hist = get_mlb_game_totals_history(home_team, as_of_date=as_of_date)
            away_hist = get_mlb_game_totals_history(away_team, as_of_date=as_of_date)
            real_hists = [h for h in (home_hist, away_hist) if h]
            if real_hists:
                shared_history = (
                    _combine_hists(real_hists[0], real_hists[1])
                    if len(real_hists) == 2
                    else real_hists[0]
                )
                effective_league_mean = round(
                    sum(shared_history) / len(shared_history), 2
                )
                print(
                    f"[odds_client] MLB real history: "
                    f"{away_abbr}@{home_abbr}  n={len(shared_history)}  "
                    f"mean={effective_league_mean}  (prior was {prior['mean']})",
                    flush=True,
                )
            else:
                shared_history = _synthetic_history(
                    game_seed, prior["mean"], prior["std"], sport=sport_up,
                )
                effective_league_mean = prior["mean"]

            # ── Starter + bullpen quality prior shift ──────────────────────
            # Full-game MLB totals were previously driven by team scoring
            # history alone (above). Fold in today's probable-starter FIP
            # (pitcher_intel — already built, was only used in revalidation)
            # and bullpen quality/fatigue (bullpen_intel — new) so pitching
            # actually moves the number, not just past runs scored.
            _pitching_adj = 0.0
            try:
                from core.intelligence.pitcher_intel import get_pitcher_intel
                _sp_factor = get_pitcher_intel(home_abbr, away_abbr, game_et.date())
                _pitching_adj += _sp_factor.league_mean_adjustment
            except Exception as exc:
                print(f"[odds_client] pitcher_intel unavailable: {exc}", flush=True)
            try:
                from core.intelligence.bullpen_intel import get_bullpen_intel
                _bp_factor = get_bullpen_intel(home_abbr, away_abbr, game_et.date())
                _pitching_adj += _bp_factor.league_mean_adjustment
            except Exception as exc:
                print(f"[odds_client] bullpen_intel unavailable: {exc}", flush=True)

            if _pitching_adj:
                effective_league_mean = round(effective_league_mean + _pitching_adj, 2)
                print(
                    f"[odds_client] MLB pitching adj: "
                    f"{away_abbr}@{home_abbr}  "
                    f"sp={getattr(_sp_factor, 'league_mean_adjustment', 0.0):+.2f}  "
                    f"bp={getattr(_bp_factor, 'league_mean_adjustment', 0.0):+.2f}  "
                    f"→ mean={effective_league_mean}",
                    flush=True,
                )
        elif sport_up in ("WNBA", "NBA"):
            try:
                from data.game_logs import get_team_game_totals
                home_hist_rg = get_team_game_totals(sport_up, home_abbr, as_of_date=as_of_date)
                away_hist_rg = get_team_game_totals(sport_up, away_abbr, as_of_date=as_of_date)
                real_hists_rg = [h for h in (home_hist_rg, away_hist_rg) if h]
                if real_hists_rg:
                    shared_history = (
                        _combine_hists(real_hists_rg[0], real_hists_rg[1])
                        if len(real_hists_rg) == 2
                        else real_hists_rg[0]
                    )
                    if sport_up == "WNBA":
                        # Regime Adjustment Protocol: use Contextual Expected
                        # Total rather than static season average as baseline.
                        from core.wnba_regime import compute_wnba_cet
                        _h_rg = home_hist_rg or []
                        _a_rg = away_hist_rg or []
                        _cet, _regime, _vol_mult = compute_wnba_cet(_h_rg, _a_rg)
                        effective_league_mean = _cet
                        _wnba_league_std = round(prior["std"] * _vol_mult, 2)
                        print(
                            f"[odds_client] WNBA CET: "
                            f"{away_abbr}@{home_abbr}  "
                            f"cet={_cet}  regime={_regime}  "
                            f"vol_mult={_vol_mult}  std={_wnba_league_std}  "
                            f"(prior mean was {prior['mean']})",
                            flush=True,
                        )
                    else:
                        effective_league_mean = round(
                            sum(shared_history) / len(shared_history), 2
                        )
                        _wnba_league_std = prior["std"]
                        _regime = "neutral"
                        _cet = effective_league_mean
                        print(
                            f"[odds_client] {sport_up} real history: "
                            f"{away_abbr}@{home_abbr}  n={len(shared_history)}  "
                            f"mean={effective_league_mean}  "
                            f"(prior was {prior['mean']})",
                            flush=True,
                        )
                else:
                    shared_history = _synthetic_history(
                        game_seed, prior["mean"], prior["std"], sport=sport_up,
                    )
                    if sport_up == "WNBA":
                        from core.wnba_regime import compute_wnba_cet
                        _cet, _regime, _vol_mult = compute_wnba_cet([], [])
                        effective_league_mean = _cet
                        _wnba_league_std = round(prior["std"] * _vol_mult, 2)
                    else:
                        effective_league_mean = prior["mean"]
                        _wnba_league_std = prior["std"]
                        _regime = "neutral"
                        _cet = prior["mean"]
            except Exception as _gl_exc:
                print(
                    f"[odds_client] game_logs fallback for {sport_up}: {_gl_exc}",
                    flush=True,
                )
                shared_history = _synthetic_history(
                    game_seed, prior["mean"], prior["std"], sport=sport_up,
                )
                effective_league_mean = prior["mean"]
                _wnba_league_std = prior["std"]
                _regime = "neutral"
                _cet = prior["mean"]
        else:
            shared_history = _synthetic_history(
                game_seed, prior["mean"], prior["std"], sport=sport_up,
            )
            effective_league_mean = prior["mean"]
            _wnba_league_std = prior["std"]
            _regime = "neutral"
            _cet = prior["mean"]

        # Cross-book consensus line (median of all books' over lines).
        # Dispersion = how far the best available line deviates from consensus.
        # A high dispersion means one book is significantly off-market — value.
        _consensus_line: float = 0.0
        _line_dispersion: float = 0.0
        if over_lines:
            _all_pts = sorted(pt for pt, _, _ in over_lines)
            _consensus_line = _all_pts[len(_all_pts) // 2]
            _line_dispersion = round(
                max(abs(pt - _consensus_line) for pt, _, _ in over_lines), 2
            )

        # Stale line drift thresholds — reject any direction where the best
        # available line deviates from multi-book consensus beyond this threshold.
        _STALE_DRIFT_THRESHOLD: dict[str, float] = {
            "MLB": 0.5, "NBA": 1.0, "WNBA": 0.5,
        }
        _drift_limit = _STALE_DRIFT_THRESHOLD.get(sport_up, 0.75)

        for direction, lines in (("over", over_lines), ("under", under_lines)):
            result = _best_side(lines)   # module-level — not redefined per game
            if result is None:
                continue
            best_line, best_odds, best_book, book_count = result

            # Stale line guard: reject when best line drifts too far from consensus.
            # Before giving up, retry among only the books whose lines are within
            # the drift limit — the highest-*odds* book (picked by _best_side above)
            # may be the off-market one while other books still have a normal,
            # in-consensus line worth using. Mirrors the fallback in game_markets.py.
            if _consensus_line and abs(best_line - _consensus_line) > _drift_limit:
                _within = [
                    (ln, od, bk) for ln, od, bk in lines
                    if abs(ln - _consensus_line) <= _drift_limit
                ]
                _fb = _best_side(_within) if _within else None
                if _fb:
                    best_line, best_odds, best_book, book_count = _fb
                else:
                    print(
                        f"[odds_client] STALE LINE rejected: {away_team}@{home_team} "
                        f"{direction} best={best_line} consensus={_consensus_line} "
                        f"drift={abs(best_line - _consensus_line):.2f} > {_drift_limit}",
                        flush=True,
                    )
                    continue
            d_label = "O" if direction == "over" else "U"
            bet_id  = f"{home_abbr}_total_{direction}_{game_id[:8]}"

            # Per-book lines for this direction
            _dir_book_lines = book_lines_over if direction == "over" else book_lines_under
            _all_line_pts   = [e["line"] for e in _dir_book_lines]
            # Opposing side's per-book lines (with odds) — needed downstream by
            # core/devig.py to pair each book's own-side/opposing-side prices
            # and devig them together. Not consumed yet in this file.
            _opp_book_lines = book_lines_under if direction == "over" else book_lines_over

            # ── MIS for game totals — fixes MIS=0 bug ──────────────────────
            _mis_score, _mis_lbl = compute_mis(
                all_lines=_all_line_pts,
                book_count=book_count,
                best_line=best_line,
                consensus_line=_consensus_line,
                sport=sport_up,
            )

            # ── Sharp action + steam detection ──────────────────────────────
            _sharp_action   = detect_sharp_action(_dir_book_lines, direction)
            _steam_detected = detect_steam_move(_all_line_pts, book_count, sport_up)

            # Cross-book note: flag when our line is off consensus by ≥ 0.5 pt
            _consensus_note = ""
            if _line_dispersion >= 0.5:
                _off_dir = "below" if best_line < _consensus_line else "above"
                _consensus_note = (
                    f" | Book consensus: {_consensus_line} "
                    f"({_off_dir} by {_line_dispersion:.1f}pt — value line detected)"
                )

            _pitching_note = ""
            if sport_up == "MLB":
                if _sp_factor is not None and _sp_factor.factor_text:
                    _pitching_note += f" | {_sp_factor.factor_text}"
                if _bp_factor is not None and _bp_factor.factor_text:
                    _pitching_note += f" | {_bp_factor.factor_text}"

            factor  = (
                f"{away_team} @ {home_team} — game total {best_line} "
                f"{d_label} ({book_count} books, best line: {best_book})."
                f"{_consensus_note}{_pitching_note}"
            )
            candidates.append({
                "bet_id":           bet_id,
                "game_id":          game_label,
                "away_team":        away_team,
                "home_team":        home_team,
                "full_team_name":   home_team,
                "team":             home_abbr,
                "market":           "Totals",
                "player":           None,
                "direction":        direction,
                "sportsbook_line":  best_line,
                "opening_line":     best_line,
                "american_odds":    best_odds,
                "bookmaker_source": best_book,
                "book_count":       book_count,
                "consensus_line":   _consensus_line,
                "line_dispersion":  _line_dispersion,
                "historical_data":  shared_history,
                "league_mean":      effective_league_mean,
                "league_std":       _wnba_league_std,
                "cet":              _cet,
                "scoring_regime":   _regime,
                "context":          "regular",
                "volatility_index": None,
                "recent_n":         5,
                "factor":           factor,
                # ── MLB starter + bullpen diagnostics (integrity_filters.py) ──
                "starting_pitcher":  (
                    f"{_sp_factor.away_pitcher}/{_sp_factor.home_pitcher}"
                    if sport_up == "MLB" and _sp_factor is not None and _sp_factor.combined_fip is not None
                    else None
                ),
                "sp_fip":           _sp_factor.combined_fip if (sport_up == "MLB" and _sp_factor) else None,
                "home_sp_era":      None,
                "bullpen_score":    _bp_factor.bullpen_score if (sport_up == "MLB" and _bp_factor) else None,
                "bullpen_fatigue":  _bp_factor.bullpen_fatigue if (sport_up == "MLB" and _bp_factor) else None,
                "bullpen_era":      _bp_factor.bullpen_era if (sport_up == "MLB" and _bp_factor) else None,
                "game_time_utc":    game_et,
                # Phase 2 — market intelligence
                "mis_score":        _mis_score,
                "book_lines":       _dir_book_lines,
                "opposing_book_lines": _opp_book_lines,
                "sharp_signal":     _sharp_action["signal_type"],
                "sharp_label":      _sharp_action["signal_label"],
                "sharp_book_count": _sharp_action["sharp_book_count"],
                "steam_detected":   _steam_detected,
                "rlm_detected":     False,
            })

    # ── Write to slate cache for remainder of session ─────────────────────
    # Args were previously swapped here (write_slate(key, date_str, candidates)
    # is the real signature) — every cache write was corrupted and silently
    # ignored on read, so the cache never actually saved API credits within
    # a run. In replay mode this writes to the isolated replay cache dir
    # (see cache_dir above), never the live cache.
    write_slate(sport_up, date_str, candidates, cache_dir=cache_dir)

    return candidates
