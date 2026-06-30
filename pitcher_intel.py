"""
pitcher_intel.py

Fetches MLB probable starting pitcher data and computes a quality-adjusted
effective FIP for use as the Bayesian prior shift in team-total bets.

Three-tier pitcher modeling
----------------------------

Tier 1 — Established starters (current-season IP >= 20)
    Rolling 10-game FIP with exponential decay (α = 0.2) so recent starts
    carry more weight.  FIP is computed from per-game components (HR, BB,
    HBP, K, IP) since the API does not pre-compute FIP.

Tier 2 — Limited-sample pitchers (IP < 20 this season)
    Discard current-season raw ERA (too noisy over < 20 IP).  Instead blend:
        effective = career_3yr_fip × w + LEAGUE_FIP × (1 − w)
    where w = min(career_ip / CAREER_IP_NORM, CAREER_MAX_WEIGHT).
    Career data is the aggregate of the 3 most recent prior seasons.

Tier 3 — No data (rookie / TBD)
    Fall back to LEAGUE_FIP, log at DEBUG, zero adjustment.

Prior shift formula
-------------------
    extra_runs = (effective_fip − LEAGUE_FIP) × ERA_SCALE   per starter
    adj        = Σ extra_runs across both starters, clipped to [ADJ_MIN, ADJ_MAX]

    ADJ_MAX = +5.0 prevents extreme single-game distortions.
    ADJ_MIN = −3.0 reflects that elite pitching matchups only compress
              totals so far before other run-scoring factors dominate.

Example — KC @ CIN, combined ERA 18.56
    Lyon Richardson rolling FIP:  ~12.x  →  extra ≈ +4.9
    Luinder Avila    rolling FIP:  ~5.x   →  extra ≈ +0.5
    Combined adj clipped to +5.0 → league_mean 8.5 → 13.5
    Line 9.5 is now 4 runs below the prior → strong OVER signal ✓

API calls
---------
  1. /api/v1/schedule?hydrate=probablePitcher          → pitcher IDs
  2. /api/v1/people/{id}/stats?stats=season             → current IP
  3a. (IP >= 20) /api/v1/people/{id}/stats?stats=gameLog → rolling FIP
  3b. (IP < 20)  /api/v1/people/{id}/stats?stats=career  → 3-yr career FIP
               + /api/v1/people/{id}/stats?stats=season&season=Y-1, Y-2

All calls use a 6-second timeout.  Any failure silently returns the
fallback (zero adjustment) so the engine is never blocked.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import date
from urllib import request as urllib_req
from urllib.error import URLError

logger = logging.getLogger("betting_bot")

# ---------------------------------------------------------------------------
# Process-level caches  (live for the duration of one picks run)
# ---------------------------------------------------------------------------
# Key: "AWAY@HOME_YYYY-MM-DD"  →  PitcherIntelFactor
# Prevents re-fetching the same game's pitchers for the OVER and UNDER
# candidate (two candidates share identical pitcher matchup data).
_RESULT_CACHE: dict[str, "PitcherIntelFactor"] = {}

# Key: "YYYY-MM-DD"  →  list[dict]  (schedule games with pitcher IDs)
_SCHEDULE_CACHE: dict[str, list[dict]] = {}

# Key: player_id  →  (effective_fip, method_tag)
# Prevents duplicate season/game-log/career calls across multiple games
# that share a pitcher (e.g. doubleheader, or just deduplication).
_FIP_CACHE: dict[int, tuple[float, str]] = {}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL        = "https://statsapi.mlb.com/api/v1"
_TIMEOUT         = 6          # seconds per HTTP request

# FIP
_FIP_CONSTANT    = 3.10       # standard FIP constant (varies ±0.15 by year)
_LEAGUE_FIP      = 4.20       # MLB average FIP ≈ ERA; used as baseline

# Adjustment geometry
_IP_PER_START    = 5.5        # typical starter innings faced per outing
_ERA_SCALE       = _IP_PER_START / 9.0   # ≈ 0.611 — ERA → per-game runs
_ADJ_MIN         = -3.0       # floor: elite pitching matchup (both aces)
_ADJ_MAX         = +5.0       # ceiling: historically bad pitching matchup

# Small-sample gatekeeping
_IP_THRESHOLD    = 20.0       # below this → discard current-season ERA
_CAREER_IP_NORM  = 150.0      # normaliser for career weight; 150 IP → max weight
_CAREER_MAX_WT   = 0.75       # maximum career data weight in the blend
_PRIOR_SEASONS   = 3          # how many past seasons to aggregate for career FIP

# Rolling window
_ROLLING_WINDOW  = 10         # number of recent game-log starts to use
_DECAY_ALPHA     = 0.2        # exponential decay factor (higher → more recent emphasis)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PitcherIntelFactor:
    home_pitcher:            str          = ""
    home_fip:                float | None = None   # effective FIP used for adj
    home_ip:                 float        = 0.0    # current-season IP
    away_pitcher:            str          = ""
    away_fip:                float | None = None
    away_ip:                 float        = 0.0
    combined_fip:            float | None = None
    league_mean_adjustment:  float        = 0.0
    factor_text:             str          = ""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _fetch(url: str) -> dict | list | None:
    """GET → parsed JSON, or None on any failure."""
    try:
        with urllib_req.urlopen(url, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except (URLError, OSError, json.JSONDecodeError) as exc:
        logger.debug(f"[pitcher_intel] HTTP error: {exc!r}")
        return None


def _parse_ip(ip_str: str | None) -> float:
    """
    Convert an MLB Stats API inningsPitched string to decimal innings.

    "5.2" → 5 + 2/3 = 5.667   (MLB uses .1 = 1 out, .2 = 2 outs)
    "0.0" → 0.0
    Returns 0.0 on any parse failure.
    """
    if not ip_str:
        return 0.0
    try:
        parts = str(ip_str).split(".")
        full  = int(parts[0])
        outs  = int(parts[1]) if len(parts) > 1 else 0
        return round(full + outs / 3.0, 4)
    except (ValueError, IndexError):
        return 0.0


def _fip(hr: int, bb: int, hbp: int, k: int, ip: float) -> float | None:
    """
    Compute FIP from counting stats.

    FIP = (13×HR + 3×(BB+HBP) − 2×K) / IP + FIP_CONSTANT

    Returns None when IP == 0 (undefined).
    """
    if ip <= 0:
        return None
    return round((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + _FIP_CONSTANT, 2)


def _exp_decay_avg(values: list[float], alpha: float = _DECAY_ALPHA) -> float:
    """
    Exponential-decay weighted average over a list of values.

    Weights: w_i = (1 − α)^(n − 1 − i)   where i=0 is the oldest entry.
    This gives the most recent value the highest weight.

    Edge case: empty list → returns LEAGUE_FIP as fallback.
    """
    if not values:
        return _LEAGUE_FIP
    n       = len(values)
    weights = [(1.0 - alpha) ** (n - 1 - i) for i in range(n)]
    total   = sum(weights)
    return round(sum(v * w for v, w in zip(values, weights)) / total, 2)


# ---------------------------------------------------------------------------
# Season / game-log fetching
# ---------------------------------------------------------------------------

def _season_stat(player_id: int, season: int) -> dict | None:
    """
    Return the pitching stat dict for *player_id* in *season*, or None.
    Keys include: inningsPitched, homeRuns, baseOnBalls, intentionalWalks,
    strikeOuts, hitByPitch, era.
    """
    url = (
        f"{_BASE_URL}/people/{player_id}/stats"
        f"?stats=season&group=pitching&season={season}"
    )
    data = _fetch(url)
    if not data or not isinstance(data, dict):
        return None
    for block in data.get("stats", []):
        splits = block.get("splits", [])
        if splits:
            return splits[0].get("stat")
    return None


def _career_stat(player_id: int) -> dict | None:
    """
    Return the career pitching stat dict, or None.
    """
    url = (
        f"{_BASE_URL}/people/{player_id}/stats"
        f"?stats=career&group=pitching"
    )
    data = _fetch(url)
    if not data or not isinstance(data, dict):
        return None
    for block in data.get("stats", []):
        splits = block.get("splits", [])
        if splits:
            return splits[0].get("stat")
    return None


def _game_log_splits(player_id: int, season: int) -> list[dict]:
    """
    Return per-game pitching splits for *player_id* in *season*.
    Each entry is the raw 'stat' dict from the API (has HR, BB, K, IP, etc.).
    Returns [] on any failure or when no starts are recorded.
    """
    url = (
        f"{_BASE_URL}/people/{player_id}/stats"
        f"?stats=gameLog&group=pitching&season={season}"
    )
    data = _fetch(url)
    if not data or not isinstance(data, dict):
        return []
    splits: list[dict] = []
    for block in data.get("stats", []):
        for split in block.get("splits", []):
            stat = split.get("stat")
            if stat:
                splits.append(stat)
    return splits


# ---------------------------------------------------------------------------
# Effective-FIP computation (Tier 1 and Tier 2)
# ---------------------------------------------------------------------------

def _rolling_fip(player_id: int, season: int) -> float | None:
    """
    Tier 1 — IP >= 20:  exponential-decay rolling FIP over the last
    ROLLING_WINDOW game-log starts.

    Starts with 0 IP are skipped (position player pitching, etc.).
    Returns None if fewer than 2 valid starts are found.
    """
    splits = _game_log_splits(player_id, season)
    if not splits:
        return None

    # Walk splits in chronological order (API returns newest first → reverse)
    splits = list(reversed(splits))

    fip_series: list[float] = []
    for stat in splits:
        ip  = _parse_ip(stat.get("inningsPitched"))
        if ip <= 0:
            continue
        hr  = int(stat.get("homeRuns",        0) or 0)
        bb  = int(stat.get("baseOnBalls",     0) or 0)
        hbp = int(stat.get("hitByPitch",      0) or 0)
        k   = int(stat.get("strikeOuts",      0) or 0)
        f   = _fip(hr, bb, hbp, k, ip)
        if f is not None:
            fip_series.append(f)

    # Keep only the last ROLLING_WINDOW starts
    fip_series = fip_series[-_ROLLING_WINDOW:]

    if len(fip_series) < 2:
        return None

    return _exp_decay_avg(fip_series)


def _career_blended_fip(player_id: int, current_season: int) -> float | None:
    """
    Tier 2 — IP < 20:  3-year career FIP blended with league baseline.

        effective = career_fip × w + LEAGUE_FIP × (1 − w)
        w = min(career_ip / CAREER_IP_NORM, CAREER_MAX_WT)

    Aggregates stats from the 3 prior seasons to form a sample-size-
    adjusted career FIP.  Falls back to career endpoint when prior-season
    data is sparse.
    """
    # ── Aggregate prior seasons ───────────────────────────────────────────
    agg_hr = agg_bb = agg_hbp = agg_k = 0
    agg_ip = 0.0
    seasons_found = 0

    for yr in range(current_season - 1, current_season - 1 - _PRIOR_SEASONS, -1):
        stat = _season_stat(player_id, yr)
        if not stat:
            continue
        ip = _parse_ip(stat.get("inningsPitched"))
        if ip <= 0:
            continue
        agg_hr  += int(stat.get("homeRuns",    0) or 0)
        agg_bb  += int(stat.get("baseOnBalls", 0) or 0)
        agg_hbp += int(stat.get("hitByPitch",  0) or 0)
        agg_k   += int(stat.get("strikeOuts",  0) or 0)
        agg_ip  += ip
        seasons_found += 1
        if seasons_found >= _PRIOR_SEASONS:
            break

    # If prior-season data is thin, supplement with the career endpoint
    if agg_ip < 30:
        career = _career_stat(player_id)
        if career:
            c_ip = _parse_ip(career.get("inningsPitched"))
            if c_ip > agg_ip:
                agg_hr  = int(career.get("homeRuns",    0) or 0)
                agg_bb  = int(career.get("baseOnBalls", 0) or 0)
                agg_hbp = int(career.get("hitByPitch",  0) or 0)
                agg_k   = int(career.get("strikeOuts",  0) or 0)
                agg_ip  = c_ip

    career_fip = _fip(agg_hr, agg_bb, agg_hbp, agg_k, agg_ip)
    if career_fip is None:
        return None   # no career data at all → caller uses league baseline

    # Blend weight proportional to career sample size (caps at CAREER_MAX_WT)
    w = min(agg_ip / _CAREER_IP_NORM, _CAREER_MAX_WT)
    blended = round(career_fip * w + _LEAGUE_FIP * (1.0 - w), 2)
    return blended


def _effective_fip(player_id: int, current_ip: float, season: int) -> tuple[float, str]:
    """
    Return (effective_fip, method_tag) for a pitcher.

    method_tag is a short label for the factor_text:
        'rolling'  — Tier 1: 10-game exponential-decay FIP
        'career'   — Tier 2: 3-yr career blend
        'league'   — Tier 3: no data, league baseline

    Guarantees a valid float is always returned (never None).
    Results are cached in _FIP_CACHE so OVER and UNDER candidates for the
    same game don't trigger duplicate API calls.
    """
    if player_id in _FIP_CACHE:
        return _FIP_CACHE[player_id]

    if current_ip >= _IP_THRESHOLD:
        fip = _rolling_fip(player_id, season)
        if fip is not None:
            result = (fip, "rolling")
            _FIP_CACHE[player_id] = result
            return result
        # Rolling failed — fall through to Tier 2
        logger.debug(
            f"[pitcher_intel] Rolling FIP unavailable for id={player_id}; "
            f"falling back to career blend."
        )

    # Tier 2 — small sample or rolling failed
    fip = _career_blended_fip(player_id, season)
    if fip is not None:
        result = (fip, "career")
        _FIP_CACHE[player_id] = result
        return result

    # Tier 3 — no data
    logger.debug(
        f"[pitcher_intel] No career data for id={player_id}; using league baseline."
    )
    result = (_LEAGUE_FIP, "league")
    _FIP_CACHE[player_id] = result
    return result


# ---------------------------------------------------------------------------
# Schedule / team matching
# ---------------------------------------------------------------------------

def _fetch_probable_starters(game_date: date) -> list[dict]:
    """
    Call the MLB schedule endpoint and return a list of game dicts:
      {home_team, away_team, home_id, home_name, away_id, away_name}
    where *_id is the MLB Stats API numeric player ID.

    Result cached in _SCHEDULE_CACHE for the run so all 12 candidates
    (6 games × OVER+UNDER) share one schedule fetch instead of 12.
    """
    date_str = game_date.strftime("%Y-%m-%d")
    if date_str in _SCHEDULE_CACHE:
        return _SCHEDULE_CACHE[date_str]

    url = (
        f"{_BASE_URL}/schedule"
        f"?sportId=1"
        f"&date={game_date.strftime('%Y-%m-%d')}"
        f"&hydrate=probablePitcher"
        f"&fields=dates,games,teams,home,away,team,name,probablePitcher,id,fullName"
    )
    data = _fetch(url)
    if not data or not isinstance(data, dict):
        return []

    games: list[dict] = []
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            teams     = game.get("teams", {})
            home_side = teams.get("home", {})
            away_side = teams.get("away", {})
            home_p    = home_side.get("probablePitcher", {})
            away_p    = away_side.get("probablePitcher", {})
            games.append({
                "home_team": home_side.get("team", {}).get("name", ""),
                "away_team": away_side.get("team", {}).get("name", ""),
                "home_id":   home_p.get("id"),
                "home_name": home_p.get("fullName", "TBD"),
                "away_id":   away_p.get("id"),
                "away_name": away_p.get("fullName", "TBD"),
            })
    _SCHEDULE_CACHE[date_str] = games
    return games


_TEAM_KEYWORDS: dict[str, str] = {
    "ARI": "Arizona",     "ATL": "Atlanta",      "BAL": "Baltimore",
    "BOS": "Boston",      "CHC": "Cubs",          "CWS": "White Sox",
    "CIN": "Cincinnati",  "CLE": "Cleveland",     "COL": "Colorado",
    "DET": "Detroit",     "HOU": "Houston",       "KC":  "Kansas City",
    "LAA": "Angels",      "LAD": "Dodgers",       "MIA": "Marlins",
    "MIL": "Milwaukee",   "MIN": "Minnesota",     "NYM": "Mets",
    "NYY": "Yankees",     "OAK": "Athletics",     "PHI": "Phillies",
    "PIT": "Pittsburgh",  "SD":  "San Diego",     "SF":  "San Francisco",
    "SEA": "Seattle",     "STL": "St. Louis",     "TB":  "Tampa Bay",
    "TEX": "Texas",       "TOR": "Toronto",       "WSH": "Washington",
}


def _team_matches(api_name: str, abbr: str) -> bool:
    keyword = _TEAM_KEYWORDS.get(abbr.upper(), "")
    return bool(keyword and keyword.lower() in api_name.lower())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_pitcher_intel(
    home_abbr: str,
    away_abbr: str,
    game_date: date | None = None,
) -> PitcherIntelFactor:
    """
    Return FIP-based league_mean adjustment for a given MLB game matchup.

    Parameters
    ----------
    home_abbr   : Home team abbreviation (e.g. 'CIN', 'TB').
    away_abbr   : Away team abbreviation (e.g. 'KC', 'BOS').
    game_date   : Date of the game; defaults to today (ET).

    Returns
    -------
    PitcherIntelFactor with league_mean_adjustment ready to be added to
    c["league_mean"] in _enrich_candidate().  Zero adjustment on any error.
    """
    try:
        return _compute(home_abbr, away_abbr, game_date)
    except Exception as exc:
        logger.debug(
            f"[pitcher_intel] Unexpected error for {away_abbr}@{home_abbr}: {exc}"
        )
        return PitcherIntelFactor()


def _compute(
    home_abbr: str,
    away_abbr: str,
    game_date: date | None,
) -> PitcherIntelFactor:
    if game_date is None:
        from core.time_utils import now_est
        game_date = now_est().date()

    # Top-level result cache: OVER and UNDER candidates for the same game
    # share identical pitcher data — only compute once per matchup per run.
    cache_key = f"{away_abbr.upper()}@{home_abbr.upper()}_{game_date}"
    if cache_key in _RESULT_CACHE:
        logger.debug(f"[pitcher_intel] Cache hit for {cache_key}")
        return _RESULT_CACHE[cache_key]

    season = game_date.year

    # ── Step 1: get probable pitcher IDs from schedule ────────────────────
    games = _fetch_probable_starters(game_date)
    if not games:
        logger.debug(
            f"[pitcher_intel] No schedule data from MLB Stats API for {game_date}"
        )
        return PitcherIntelFactor()

    match: dict | None = None
    for g in games:
        if _team_matches(g["home_team"], home_abbr) and \
           _team_matches(g["away_team"], away_abbr):
            match = g
            break

    if match is None:
        logger.debug(
            f"[pitcher_intel] No game found for {away_abbr}@{home_abbr} on {game_date}. "
            f"Available: {[(g['away_team'][:10], g['home_team'][:10]) for g in games]}"
        )
        return PitcherIntelFactor()

    home_name = match["home_name"]
    away_name = match["away_name"]
    home_id   = match["home_id"]
    away_id   = match["away_id"]

    # ── Step 2: fetch current-season IP for gatekeeping ───────────────────
    home_season_stat = _season_stat(home_id, season) if home_id else None
    away_season_stat = _season_stat(away_id, season) if away_id else None

    home_ip = _parse_ip(home_season_stat.get("inningsPitched")) if home_season_stat else 0.0
    away_ip = _parse_ip(away_season_stat.get("inningsPitched")) if away_season_stat else 0.0

    # ── Step 3: compute effective FIP per pitcher ─────────────────────────
    adj   = 0.0
    parts: list[str] = []

    home_fip: float | None = None
    away_fip: float | None = None

    if home_id:
        h_fip, h_method = _effective_fip(home_id, home_ip, season)
        home_fip = h_fip
        excess   = (h_fip - _LEAGUE_FIP) * _ERA_SCALE
        adj     += excess
        tag      = {"rolling": "10G", "career": "3yr", "league": "lg"}.get(h_method, h_method)
        parts.append(f"{home_name} FIP {h_fip:.2f} ({tag})")
    else:
        parts.append(f"{home_name} FIP N/A")

    if away_id:
        a_fip, a_method = _effective_fip(away_id, away_ip, season)
        away_fip = a_fip
        excess   = (a_fip - _LEAGUE_FIP) * _ERA_SCALE
        adj     += excess
        tag      = {"rolling": "10G", "career": "3yr", "league": "lg"}.get(a_method, a_method)
        parts.append(f"{away_name} FIP {a_fip:.2f} ({tag})")
    else:
        parts.append(f"{away_name} FIP N/A")

    adj = round(max(_ADJ_MIN, min(_ADJ_MAX, adj)), 2)

    combined_fip: float | None = None
    if home_fip is not None and away_fip is not None:
        combined_fip = round(home_fip + away_fip, 2)

    factor_text = f"SP FIP: {' | '.join(parts)}"
    if combined_fip is not None:
        factor_text += f" (Σ{combined_fip:.2f})"
    if adj != 0.0:
        arrow        = "↑" if adj > 0 else "↓"
        factor_text += f" → prior {arrow}{abs(adj):.1f}"

    logger.debug(
        f"[pitcher_intel] {away_abbr}@{home_abbr}: "
        f"home_fip={home_fip} (IP={home_ip:.1f}) "
        f"away_fip={away_fip} (IP={away_ip:.1f}) "
        f"adj={adj:+.2f}"
    )

    factor = PitcherIntelFactor(
        home_pitcher           = home_name,
        home_fip               = home_fip,
        home_ip                = home_ip,
        away_pitcher           = away_name,
        away_fip               = away_fip,
        away_ip                = away_ip,
        combined_fip           = combined_fip,
        league_mean_adjustment = adj,
        factor_text            = factor_text,
    )
    _RESULT_CACHE[cache_key] = factor
    return factor
