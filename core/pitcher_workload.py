"""
core/pitcher_workload.py

Pitcher Workload Model — MLB foundational forecasting layer.

Five-component weighted model
------------------------------
  Component 1  Sportsbook Signal     (40%) pitcher outs market → implied IP
  Component 2  Pitch Count Trend     (25%) recent pitch counts per start → IP
  Component 3  Opponent Difficulty   (15%) opposing offense OPS / K% → IP delta
  Component 4  Bullpen Fatigue       (10%) team BP usage (last 3 days) → IP delta
  Component 5  Manager Hook Profile  (10%) starter's own avg IP → hook tendency

Outputs per pitcher
-------------------
  expected_ip       projected starter innings (foundational MLB variable)
  expected_pitches  projected pitch count
  expected_outs     expected outs recorded (3 × expected_ip; internal)
  expected_ra       expected runs allowed (regression-adjusted; internal)
  expected_k        expected strikeouts
  confidence        0–100 composite score
  confidence_tier   "High" / "Medium" / "Low"
  discrepancy_flag  True when model vs sportsbook IP diverge ≥ 1.0 inning

Downstream consumers
--------------------
  player_props.py    scale pitcher_strikeouts projection by get_k_workload_scale()
  game_markets.py    adjust F5 eff_mean via get_f5_workload_adjustment()
  main.py            store workload in MLB game candidate for gatekeeper access

Never generate a betting pick from pitcher outs / expected_outs directly.
Workload data is internal signal only.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from urllib import request as urllib_req
from urllib.error import URLError

from models.bayesian import shrink_mlb_k_pct
from models.sport_config import MLB as _MLB_CFG

logger = logging.getLogger("betting_bot")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_MLB_BASE    = "https://statsapi.mlb.com/api/v1"
_ODDS_BASE   = "https://api.the-odds-api.com/v4"
_TIMEOUT     = 6   # seconds per HTTP request

# League baseline values  (2025–2026 MLB)
LEAGUE_AVG_IP      = 5.5     # average SP innings per start
LEAGUE_AVG_K9      = 9.2     # average K/9 for starting pitchers
LEAGUE_AVG_RA9     = 4.50    # average runs allowed / 9 innings
LEAGUE_AVG_PITCHES = 93      # average pitch count per start
_PITCHES_PER_INN   = 16.8    # league avg pitches per inning pitched

# F5 baseline: league-average runs expected in first 5 innings (both teams combined).
# Empirically lower than (LEAGUE_AVG_RA9 × 2 × 5/9 = 5.0) because SP ERA < bullpen ERA
# and starters typically face the lineup twice in those first five frames.
_LEAGUE_F5_RA = 4.68

# Confidence tier thresholds
_CONF_HIGH   = 70.0
_CONF_MEDIUM = 45.0

# Discrepancy threshold (innings)
_DISCREPANCY_THRESH = 1.0

# Component weights  (must sum to 1.0)
_WEIGHTS = {
    "sportsbook":    0.40,
    "pitch_trend":   0.25,
    "manager_hook":  0.10,
}

# Delta components — applied on top of the blended IP estimate
_DELTA_COMPS = ("opp_difficulty", "bullpen_fatigue")

# IP sanity bounds
_IP_MIN = 1.5
_IP_MAX = 9.0

# Rolling windows
_L3_WINDOW           = 3
_L5_WINDOW           = 5
_BP_FATIGUE_DAYS     = 3
_MANAGER_HOOK_WINDOW = 15   # recent starts for hook profile
_K_RA_WINDOW         = 10   # recent starts for K/9 and RA/9

# ─────────────────────────────────────────────────────────────────────────────
# Process-level caches  (live for the duration of one run)
# ─────────────────────────────────────────────────────────────────────────────

_WORKLOAD_CACHE:    dict[str, "WorkloadProjection"] = {}  # "{name}_{date}"
_GAME_LOG_CACHE:    dict[tuple[int, int], list[dict]] = {}  # (player_id, season)
_TEAM_STATS_CACHE:  dict[tuple[str, int], dict]  = {}  # (opp_abbr, season)
_BP_USAGE_CACHE:    dict[tuple[str, str], float | None] = {}  # (team_abbr, date_str)
_SB_OUTS_CACHE:     dict[str, float | None] = {}  # "event_{last12_of_name}"

# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkloadProjection:
    pitcher_name:     str
    pitcher_id:       int | None
    team_abbr:        str
    opp_abbr:         str
    game_date:        str          # "YYYY-MM-DD"

    # Primary forecast outputs
    expected_ip:      float = LEAGUE_AVG_IP
    expected_pitches: int   = LEAGUE_AVG_PITCHES
    expected_outs:    float = 0.0  # = expected_ip × 3
    expected_ra:      float = 0.0  # expected runs allowed
    expected_k:       float = 0.0  # expected strikeouts

    # Confidence
    confidence:       float = 50.0
    confidence_tier:  str   = "Medium"

    # Sportsbook component (for discrepancy detection)
    sb_implied_ip:    float | None = None

    # Discrepancy
    discrepancy_flag:   bool = False
    discrepancy_detail: str  = ""

    # Audit
    components_used: int = 0
    method:          str = "league_default"

    def __post_init__(self) -> None:
        self.expected_outs = round(self.expected_ip * 3, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(url: str) -> dict | list | None:
    try:
        with urllib_req.urlopen(url, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except (URLError, OSError, json.JSONDecodeError) as exc:
        logger.debug(f"[pitcher_workload] HTTP error: {exc!r}")
        return None


def _parse_ip(ip_str: str | None) -> float:
    """Convert MLB Stats API inningsPitched string to decimal IP.
    "5.2" → 5 + 2/3 = 5.667   (.1 = 1 out, .2 = 2 outs)
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


_TEAM_KEYWORDS: dict[str, str] = {
    "ARI": "Arizona",    "ATL": "Atlanta",     "BAL": "Baltimore",
    "BOS": "Boston",     "CHC": "Cubs",         "CWS": "White Sox",
    "CIN": "Cincinnati", "CLE": "Cleveland",    "COL": "Colorado",
    "DET": "Detroit",    "HOU": "Houston",      "KC":  "Kansas City",
    "LAA": "Angels",     "LAD": "Dodgers",      "MIA": "Marlins",
    "MIL": "Milwaukee",  "MIN": "Minnesota",    "NYM": "Mets",
    "NYY": "Yankees",    "OAK": "Athletics",    "PHI": "Phillies",
    "PIT": "Pittsburgh", "SD":  "San Diego",    "SF":  "San Francisco",
    "SEA": "Seattle",    "STL": "St. Louis",    "TB":  "Tampa Bay",
    "TEX": "Texas",      "TOR": "Toronto",      "WSH": "Washington",
}

_MLB_TEAM_IDS: dict[str, int] = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KC":  118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD":  135, "SF":  137, "SEA": 136,
    "STL": 138, "TB":  139, "TEX": 140, "TOR": 141, "WSH": 120,
}


def _team_matches(api_name: str, abbr: str) -> bool:
    keyword = _TEAM_KEYWORDS.get(abbr.upper(), "")
    return bool(keyword and keyword.lower() in api_name.lower())


def _get_game_log(player_id: int, season: int) -> list[dict]:
    """Fetch per-start pitching game log; result cached for the session.

    Returns most-recent-start-first, guaranteed by an explicit sort on
    _game_date -- do NOT rely on the MLB Stats API's raw split order for
    this. The raw order is chronological ascending (oldest start first),
    confirmed by data/fetch.py's get_mlb_player_game_logs() against the
    same endpoint, which explicitly reverses before slicing ("# Most
    recent games first"). This function previously returned splits
    un-reversed while callers (get_recent_k_bf_splits, _component_pitch_trend)
    assumed most-recent-first -- silently feeding them each pitcher's
    EARLIEST starts of the season (shorter, ramp-up-limited outings) as
    "recent" all season. Sorting explicitly here removes the ordering
    assumption entirely instead of relying on a second reversal to cancel
    out the first (2026-07-31 fix).
    """
    key = (player_id, season)
    if key in _GAME_LOG_CACHE:
        return _GAME_LOG_CACHE[key]
    url = (
        f"{_MLB_BASE}/people/{player_id}/stats"
        f"?stats=gameLog&group=pitching&season={season}"
    )
    data   = _fetch(url)
    splits: list[dict] = []
    if data and isinstance(data, dict):
        for block in data.get("stats", []):
            for split in block.get("splits", []):
                stat = split.get("stat")
                if stat:
                    stat["_game_date"] = split.get("date", "")
                    splits.append(stat)
    splits.sort(key=lambda s: s.get("_game_date", ""), reverse=True)
    _GAME_LOG_CACHE[key] = splits
    return splits


def get_recent_k_bf_splits(pitcher_id: int | None, season: int, n: int = 8) -> dict | None:
    """
    Real per-start (strikeouts, batters_faced) data for a pitcher's last `n`
    starts, feeding an empirical Binomial K% model (K ~ Binomial(BF, k_pct))
    -- used by core/player_props.py to price pitcher_strikeouts against
    models/monte_carlo.k_prop_edge_with_uncertainty instead of treating raw
    strikeout totals as Gaussian (Binomial is the correct shape for count
    data bounded by batters faced; see models/monte_carlo.py's docstring).

    Reuses _get_game_log()'s existing cache -- if get_pitcher_workload() /
    _component_pitch_trend() already pulled this pitcher's game log this
    run (they usually have, since this is meant to be called alongside
    get_pitcher_workload()), this costs zero extra API calls.

    Returns None when fewer than 6 real starts have usable battersFaced /
    strikeOuts data (e.g. a pitcher very early in the season) -- no
    synthetic backfill. An empirical std from a couple of starts is worse
    than admitting the data isn't there yet.

    AUDIT (2026-08-05): raised from 4 -> 6. The caller (core/player_props.py)
    used to treat None here as "fall back to the Gaussian/synthetic-history
    path" -- that fallback has since been removed for pitcher_strikeouts
    entirely (see player_props.py NO PLAY comment near _k_bf_skip_reason),
    so a None here now means "no pick," not "pick anyway with worse data."
    That raises the cost of a too-low floor: 4 starts was letting some very
    thin, high-variance samples (e.g. a pitcher who opened the season with
    two 2-inning openers) through as if they were trustworthy. 6 is still
    low by handicapping standards but keeps enough in-season pitchers
    eligible to be useful; revisit against core/calibration.py output once
    enough graded picks accumulate under the new gate.

    Also returns None when the recent-sample batters-faced total falls
    below sport_config.MLB["min_batters_faced_for_k_prop"] -- 4 very
    short/opener-style outings can clear the _MIN_STARTS floor on start
    count alone while still being too thin a real sample to trust (this
    config key existed but was never wired to anything before).

    k_pct_mean is Bayesian-shrunk (models.bayesian.shrink_mlb_k_pct) toward
    a blend of this pitcher's own season-to-date K% and the league average,
    weighted by sample size -- a raw small-sample recent K% (e.g. 9/24 in
    one hot start) is exactly the noisy "hot streak" number that function
    exists to correct before it ever reaches the Binomial simulation.
    k_pct_mean_raw carries the pre-shrinkage number through for logging.

    Returns dict with:
        n_starts       -- real starts used
        k_pct_mean     -- shrunk mean K% (strikeouts / batters_faced)
        k_pct_mean_raw -- unshrunk pooled recent K%, for debugging/logging
        k_pct_std      -- sample std of per-start K% (floored at 0.01)
        bf_mean        -- mean batters faced per start
        bf_std         -- sample std of batters faced per start (floored at 1.0)
    """
    _MIN_STARTS = 6
    if not pitcher_id:
        return None

    splits = _get_game_log(pitcher_id, season)
    if not splits:
        return None

    # _get_game_log returns most-recent-start-first (see _component_pitch_trend,
    # which reverses it to get chronological order) -- so a plain head-slice
    # here is already "last n starts".
    recent = splits[:n]

    bfs: list[float] = []
    ks: list[float] = []
    k_pcts: list[float] = []
    for stat in recent:
        bf = stat.get("battersFaced")
        k  = stat.get("strikeOuts")
        if not bf or bf <= 0 or k is None:
            continue
        bf = float(bf)
        bfs.append(bf)
        ks.append(float(k))
        k_pcts.append(float(k) / bf)

    if len(bfs) < _MIN_STARTS:
        return None

    total_bf = sum(bfs)
    total_ks = sum(ks)

    # ── Sample-size floor (sport_config.MLB["min_batters_faced_for_k_prop"]) ──
    _min_bf = _MLB_CFG.get("min_batters_faced_for_k_prop", 15)
    if total_bf < _min_bf:
        return None

    bf_mean = statistics.mean(bfs)
    # stdev needs >=2 points; with exactly _MIN_STARTS>=4 this is guaranteed,
    # but guard anyway in case a future _MIN_STARTS tweak drops it to 1.
    bf_std = statistics.stdev(bfs) if len(bfs) > 1 else max(1.0, bf_mean * 0.15)
    k_std  = statistics.stdev(k_pcts) if len(k_pcts) > 1 else max(0.02, statistics.mean(k_pcts) * 0.15)

    # ── Own-season K% prior (all season-to-date starts, not just the recent
    # window used above) -- feeds shrink_mlb_k_pct's own-season blend below.
    # Falls back to None (league-average-only prior) when season totals
    # aren't usable, same degrade-to-no-op discipline as the rest of this
    # function.
    season_bf = sum(
        float(s["battersFaced"]) for s in splits
        if s.get("battersFaced") and s["battersFaced"] > 0 and s.get("strikeOuts") is not None
    )
    season_ks = sum(
        float(s["strikeOuts"]) for s in splits
        if s.get("battersFaced") and s["battersFaced"] > 0 and s.get("strikeOuts") is not None
    )
    own_season_k_pct = (season_ks / season_bf) if season_bf > 0 else None

    shrunk_k_pct = shrink_mlb_k_pct(
        recent_ks=total_ks,
        recent_batters_faced=total_bf,
        own_season_k_pct=own_season_k_pct,
    )

    return {
        "n_starts":       len(bfs),
        "k_pct_mean":     round(shrunk_k_pct, 4),
        "k_pct_mean_raw": round(total_ks / total_bf, 4),
        "k_pct_std":      round(max(k_std, 0.01), 4),
        "bf_mean":        round(bf_mean, 2),
        "bf_std":         round(max(bf_std, 1.0), 2),
    }


def _get_probable_starters(date_str: str) -> list[dict]:
    """
    Fetch probable starters from MLB Stats API for a given date string.
    Reuses pitcher_intel's _SCHEDULE_CACHE when available to avoid double-fetching.
    Returns [{home_team, away_team, home_team_id, away_team_id,
              home_id, home_name, away_id, away_name}].
    """
    try:
        from core.intelligence.pitcher_intel import _SCHEDULE_CACHE
        if date_str in _SCHEDULE_CACHE:
            return _SCHEDULE_CACHE[date_str]
    except ImportError:
        pass

    url = (
        f"{_MLB_BASE}/schedule"
        f"?sportId=1&date={date_str}&hydrate=probablePitcher"
        f"&fields=dates,games,teams,home,away,team,id,name,"
        f"probablePitcher,fullName"
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
            home_team = home_side.get("team", {})
            away_team = away_side.get("team", {})
            games.append({
                "home_team":    home_team.get("name", ""),
                "away_team":    away_team.get("name", ""),
                "home_team_id": home_team.get("id"),
                "away_team_id": away_team.get("id"),
                "home_id":      home_p.get("id"),
                "home_name":    home_p.get("fullName", "TBD"),
                "away_id":      away_p.get("id"),
                "away_name":    away_p.get("fullName", "TBD"),
            })

    try:
        from core.intelligence.pitcher_intel import _SCHEDULE_CACHE
        _SCHEDULE_CACHE[date_str] = games
    except ImportError:
        pass

    return games


# ─────────────────────────────────────────────────────────────────────────────
# Component 1 — Sportsbook Signal (40%)
# ─────────────────────────────────────────────────────────────────────────────

def _component_sportsbook(
    pitcher_name: str,
    event_id: str,
) -> float | None:
    """
    Fetch pitcher outs market from The Odds API and convert to IP.
    Returns median implied IP or None when no market data is available.
    Note: outs data is consumed as internal signal only; never surfaced as a bet.
    """
    cache_key = f"{event_id}_{pitcher_name[-10:]}"
    if cache_key in _SB_OUTS_CACHE:
        return _SB_OUTS_CACHE[cache_key]

    api_key = os.environ.get("THE_ODDS_API_KEY", "")
    if not api_key:
        _SB_OUTS_CACHE[cache_key] = None
        return None

    url = (
        f"{_ODDS_BASE}/sports/baseball_mlb/events/{event_id}/odds"
        f"?apiKey={api_key}"
        f"&regions=us&markets=batter_pitcher_outs&oddsFormat=american"
    )
    data = _fetch(url)
    if not data or not isinstance(data, dict):
        _SB_OUTS_CACHE[cache_key] = None
        return None

    pitcher_lower = pitcher_name.lower()
    last_name     = pitcher_lower.split()[-1] if pitcher_lower else ""
    outs_lines: list[float] = []

    for bk in data.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt.get("key") != "batter_pitcher_outs":
                continue
            for out in mkt.get("outcomes", []):
                desc = (out.get("description") or "").lower()
                name = (out.get("name") or "").lower()
                pt   = out.get("point")
                if pt is None or name != "over":
                    continue
                if last_name and last_name in desc:
                    outs_lines.append(float(pt))

    if not outs_lines:
        _SB_OUTS_CACHE[cache_key] = None
        return None

    median_outs = statistics.median(outs_lines)
    implied_ip  = round(median_outs / 3.0, 2)

    logger.debug(
        f"[pitcher_workload] SB outs → {pitcher_name}: "
        f"{median_outs:.1f} outs → {implied_ip:.2f} IP "
        f"({len(outs_lines)} book line(s))"
    )
    _SB_OUTS_CACHE[cache_key] = implied_ip
    return implied_ip


# ─────────────────────────────────────────────────────────────────────────────
# Component 2 — Pitch Count Trend (25%)
# ─────────────────────────────────────────────────────────────────────────────

def _component_pitch_trend(player_id: int, season: int) -> float | None:
    """
    Derive expected IP from recent pitch count trend.
    Weighted blend: 40% L3 + 35% L5 + 25% season avg → projected pitches → IP.
    Returns None when fewer than 3 starts are available.
    """
    splits = _get_game_log(player_id, season)
    if not splits:
        return None

    splits_chrono = list(reversed(splits))
    pitch_counts: list[float] = []
    ip_per_start: list[float] = []

    for stat in splits_chrono:
        pitches = stat.get("numberOfPitches")
        ip      = _parse_ip(stat.get("inningsPitched"))
        if pitches is None or ip <= 0:
            continue
        pitch_counts.append(float(pitches))
        ip_per_start.append(ip)

    if len(pitch_counts) < 3:
        return None

    l3_avg     = sum(pitch_counts[-_L3_WINDOW:]) / _L3_WINDOW
    l5_slice   = pitch_counts[-_L5_WINDOW:] if len(pitch_counts) >= _L5_WINDOW else pitch_counts
    l5_avg     = sum(l5_slice) / len(l5_slice)
    season_avg = sum(pitch_counts) / len(pitch_counts)

    n = len(pitch_counts)
    if n >= _L5_WINDOW:
        proj_pitches = l3_avg * 0.40 + l5_avg * 0.35 + season_avg * 0.25
    else:
        proj_pitches = l3_avg * 0.55 + season_avg * 0.45

    # Derive this pitcher's actual IP-per-pitch ratio from history
    total_ip      = sum(ip_per_start)
    total_pitches = sum(pitch_counts[:len(ip_per_start)])
    ppi = (total_ip / total_pitches) if total_pitches > 0 else (1.0 / _PITCHES_PER_INN)

    proj_ip = round(proj_pitches * ppi, 2)
    logger.debug(
        f"[pitcher_workload] Pitch trend (id={player_id}): "
        f"L3={l3_avg:.0f}p L5={l5_avg:.0f}p → "
        f"{proj_pitches:.0f}p → {proj_ip:.2f} IP"
    )
    return proj_ip


# ─────────────────────────────────────────────────────────────────────────────
# Component 3 — Opponent Difficulty (15%) — delta IP
# ─────────────────────────────────────────────────────────────────────────────

def _component_opp_difficulty(opp_abbr: str, season: int) -> float | None:
    """
    Adjust expected IP for opposing lineup quality.
    High OPS → starter pulled sooner (negative delta).
    High K% → pitcher-friendly, deeper outing (positive delta).
    Returns IP delta in [-0.5, +0.5] or None when team data unavailable.
    """
    cache_key = (opp_abbr.upper(), season)
    if cache_key in _TEAM_STATS_CACHE:
        hitting = _TEAM_STATS_CACHE[cache_key]
    else:
        team_id = _MLB_TEAM_IDS.get(opp_abbr.upper())
        if not team_id:
            return None
        url = (
            f"{_MLB_BASE}/teams/{team_id}/stats"
            f"?stats=season&group=hitting&season={season}"
        )
        data    = _fetch(url)
        hitting = {}
        if data and isinstance(data, dict):
            for block in data.get("stats", []):
                splits = block.get("splits", [])
                if splits:
                    hitting = splits[0].get("stat", {})
                    break
        _TEAM_STATS_CACHE[cache_key] = hitting

    if not hitting:
        return None

    try:
        ops = float(hitting.get("ops") or 0)
        pa  = float(hitting.get("plateAppearances") or 0)
        ks  = float(hitting.get("strikeOuts") or 0)
        k_pct = (ks / pa) if pa > 0 else 0.23
    except (TypeError, ValueError, ZeroDivisionError):
        return None

    if ops <= 0:
        return None

    # OPS: every 0.030 above league avg (.720) → −0.06 IP
    LEAGUE_OPS   = 0.720
    ops_adj      = (LEAGUE_OPS - ops) * 2.0   # sign: strong offense → negative

    # K%: every 1 ppt above league avg (23%) → +0.04 IP (pitcher-friendly)
    LEAGUE_K_PCT = 0.23
    k_adj        = (k_pct - LEAGUE_K_PCT) * 4.0

    total = round(ops_adj + k_adj, 2)
    total = max(-0.50, min(0.50, total))
    logger.debug(
        f"[pitcher_workload] Opp difficulty {opp_abbr}: "
        f"OPS={ops:.3f} K%={k_pct:.1%} → adj={total:+.2f} IP"
    )
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Component 4 — Bullpen Fatigue (10%) — delta IP
# ─────────────────────────────────────────────────────────────────────────────

def _component_bullpen_fatigue(
    team_abbr: str,
    game_date: date,
    season: int,
) -> float | None:
    """
    DISABLED (2026-08-29) -- see below. Always returns None (component
    skipped) until this is rebuilt on a per-pitcher endpoint.

    What this was supposed to measure: how heavily the team's bullpen has
    been used in the last 3 days (heavy BP usage -> manager lets tonight's
    starter go longer; fresh BP -> quicker hook).

    Why it's disabled: `/teams/{id}/stats?stats=gameLog&group=pitching`
    returns one TEAM-AGGREGATE row per game (~9 total IP for a regulation
    game), not a per-pitcher breakdown -- there is no field in this
    response that separates the starter's innings from the bullpen's. The
    old code guessed the starter's IP as `min(total_ip, LEAGUE_AVG_IP)`
    and called the remainder "bullpen IP". For a normal 9-inning game that
    arithmetic reduces to a near-constant
    `9 - LEAGUE_AVG_IP = 9 - 5.5 = 3.5` regardless of whether the actual
    starter that day threw 3 innings (heavy real bullpen usage, true value
    ~6) or 8 innings (almost no bullpen usage, true value ~1) -- the
    formula can't tell those two very different nights apart, because
    total_ip is ~9 either way and the cap always pins sp_est to 5.5. So in
    the common case this component wasn't measuring bullpen fatigue at
    all, just producing a roughly-constant pseudo-signal (see the removed
    `sp_est = min(total_ip, LEAGUE_AVG_IP)` line in version control for
    the exact prior formula) that happened to mostly cancel out against
    LEAGUE_BP_3D by construction -- confident-looking but not real
    information, which is worse than admitting we don't have it.

    Real fix requires per-pitcher game logs for the team's actual relief
    corps (e.g. iterate roster relievers' individual
    `/people/{id}/stats?stats=gameLog` and sum IP for appearances with
    gamesStarted=0 in the trailing window) rather than trying to back out
    a starter/reliever split from a team aggregate. Not implemented here
    -- flagging as a real, if minor (10% weight, ±0.25/+0.30 IP delta
    range), TODO rather than shipping the misleading version.
    """
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Component 5 — Manager Hook Profile (10%)
# ─────────────────────────────────────────────────────────────────────────────

def _component_manager_hook(player_id: int, season: int) -> float | None:
    """
    Infer manager's hook tendency from this starter's actual avg IP per start.
    Returns absolute IP estimate (not a delta) based on recent history.
    Returns None when fewer than 3 starts are available.
    """
    splits = _get_game_log(player_id, season)
    if not splits:
        return None

    splits_chrono = list(reversed(splits))[-_MANAGER_HOOK_WINDOW:]
    ip_values = [
        _parse_ip(s.get("inningsPitched"))
        for s in splits_chrono
        if _parse_ip(s.get("inningsPitched")) > 0
    ]
    if len(ip_values) < 3:
        return None

    avg_ip = round(sum(ip_values) / len(ip_values), 2)
    logger.debug(
        f"[pitcher_workload] Manager hook (id={player_id}): "
        f"avg IP={avg_ip:.2f} over {len(ip_values)} starts"
    )
    return avg_ip


# ─────────────────────────────────────────────────────────────────────────────
# Secondary stats — K/9 and RA/9 from game log
# ─────────────────────────────────────────────────────────────────────────────

def _derive_k9(player_id: int, season: int) -> float:
    splits        = _get_game_log(player_id, season)
    splits_chrono = list(reversed(splits))[-_K_RA_WINDOW:]
    total_k = 0
    total_ip = 0.0
    for stat in splits_chrono:
        ip = _parse_ip(stat.get("inningsPitched"))
        if ip <= 0:
            continue
        total_k  += int(stat.get("strikeOuts", 0) or 0)
        total_ip += ip
    if total_ip <= 0:
        return LEAGUE_AVG_K9
    return round(total_k / total_ip * 9.0, 2)


def _derive_ra9(player_id: int, season: int) -> float:
    """RA/9 regressed 30% towards league mean to dampen small-sample noise."""
    splits        = _get_game_log(player_id, season)
    splits_chrono = list(reversed(splits))[-_K_RA_WINDOW:]
    total_er = 0
    total_ip = 0.0
    for stat in splits_chrono:
        ip = _parse_ip(stat.get("inningsPitched"))
        if ip <= 0:
            continue
        total_er += int(stat.get("earnedRuns", 0) or 0)
        total_ip += ip
    if total_ip <= 0:
        return LEAGUE_AVG_RA9
    raw = total_er / total_ip * 9.0
    return round(raw * 0.70 + LEAGUE_AVG_RA9 * 0.30, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Core aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _build_projection(
    pitcher_name: str,
    pitcher_id:   int | None,
    team_abbr:    str,
    opp_abbr:     str,
    game_date:    date,
    event_id:     str | None,
) -> WorkloadProjection:
    season = game_date.year

    # ── Collect absolute-IP estimates from components 1, 2, 5 ───────────────
    abs_ips:    dict[str, float] = {}
    confidence  = 50.0
    n_comps     = 0
    method_parts: list[str] = []

    # Component 1 — Sportsbook
    sb_ip: float | None = None
    if pitcher_id and event_id:
        try:
            sb_ip = _component_sportsbook(pitcher_name, event_id)
        except Exception as exc:
            logger.debug(f"[pitcher_workload] C1 sportsbook failed: {exc}")
    if sb_ip is not None:
        abs_ips["sportsbook"] = sb_ip
        confidence += 15.0
        n_comps    += 1
        method_parts.append(f"sb={sb_ip:.1f}")

    # Component 2 — Pitch count trend
    trend_ip: float | None = None
    if pitcher_id:
        try:
            trend_ip = _component_pitch_trend(pitcher_id, season)
        except Exception as exc:
            logger.debug(f"[pitcher_workload] C2 pitch_trend failed: {exc}")
    if trend_ip is not None:
        abs_ips["pitch_trend"] = trend_ip
        confidence += 10.0
        n_comps    += 1
        method_parts.append(f"trend={trend_ip:.1f}")

    # Component 5 — Manager hook (also sets a data-driven base IP)
    hook_ip: float | None = None
    if pitcher_id:
        try:
            hook_ip = _component_manager_hook(pitcher_id, season)
        except Exception as exc:
            logger.debug(f"[pitcher_workload] C5 manager_hook failed: {exc}")
    if hook_ip is not None:
        abs_ips["manager_hook"] = hook_ip
        n_comps    += 1
        method_parts.append(f"hook={hook_ip:.1f}")

    # ── Weighted blend of available absolute-IP components ───────────────────
    if abs_ips:
        total_w    = sum(_WEIGHTS[k] for k in abs_ips if k in _WEIGHTS)
        total_w    = total_w or 1.0   # fallback: equal weight
        blended_ip = sum(
            _WEIGHTS.get(k, 0.10) * v for k, v in abs_ips.items()
        ) / total_w
    else:
        blended_ip = LEAGUE_AVG_IP   # full league default

    # ── Component 3 — Opponent difficulty (delta) ────────────────────────────
    opp_adj = 0.0
    if opp_abbr:
        try:
            adj = _component_opp_difficulty(opp_abbr, season)
            if adj is not None:
                opp_adj  = adj
                n_comps += 1
                method_parts.append(f"opp={adj:+.2f}")
        except Exception as exc:
            logger.debug(f"[pitcher_workload] C3 opp_difficulty failed: {exc}")

    # ── Component 4 — Bullpen fatigue (delta) ────────────────────────────────
    bp_adj = 0.0
    if team_abbr:
        try:
            adj = _component_bullpen_fatigue(team_abbr, game_date, season)
            if adj is not None:
                bp_adj   = adj
                n_comps += 1
                method_parts.append(f"bp={adj:+.2f}")
        except Exception as exc:
            logger.debug(f"[pitcher_workload] C4 bullpen_fatigue failed: {exc}")

    # ── Final expected IP ─────────────────────────────────────────────────────
    raw_ip      = blended_ip + opp_adj + bp_adj
    expected_ip = round(max(_IP_MIN, min(_IP_MAX, raw_ip)), 2)

    # ── Secondary stats ───────────────────────────────────────────────────────
    k9  = LEAGUE_AVG_K9
    ra9 = LEAGUE_AVG_RA9
    if pitcher_id:
        try:
            k9 = _derive_k9(pitcher_id, season)
        except Exception:
            pass
        try:
            ra9 = _derive_ra9(pitcher_id, season)
        except Exception:
            pass

    expected_k  = round(k9  * expected_ip / 9.0, 2)
    expected_ra = round(ra9 * expected_ip / 9.0, 2)
    exp_pitches = int(round(expected_ip * _PITCHES_PER_INN))

    # ── Confidence scoring ────────────────────────────────────────────────────
    # Boost when model and sportsbook closely agree; penalise divergence
    if sb_ip is not None:
        gap = abs(expected_ip - sb_ip)
        if gap < 0.50:
            confidence += 10.0
        elif gap >= _DISCREPANCY_THRESH:
            confidence -= 5.0

    confidence = round(min(95.0, max(20.0, confidence)), 1)
    tier       = (
        "High"   if confidence >= _CONF_HIGH   else
        "Medium" if confidence >= _CONF_MEDIUM else
        "Low"
    )

    # ── Discrepancy detection ─────────────────────────────────────────────────
    disc_flag   = False
    disc_detail = ""
    if sb_ip is not None:
        gap = abs(expected_ip - sb_ip)
        if gap >= _DISCREPANCY_THRESH:
            disc_flag   = True
            direction   = "longer" if expected_ip > sb_ip else "shorter"
            disc_detail = (
                f"Model={expected_ip:.1f} IP is {direction} than "
                f"sportsbook {sb_ip:.1f} IP by {gap:.1f} innings"
            )
            logger.info(
                f"[pitcher_workload] DISCREPANCY {pitcher_name}: {disc_detail}"
            )

    method = ", ".join(method_parts) if method_parts else "league_default"

    return WorkloadProjection(
        pitcher_name     = pitcher_name,
        pitcher_id       = pitcher_id,
        team_abbr        = team_abbr,
        opp_abbr         = opp_abbr,
        game_date        = game_date.strftime("%Y-%m-%d"),
        expected_ip      = expected_ip,
        expected_pitches = exp_pitches,
        expected_outs    = round(expected_ip * 3, 1),
        expected_ra      = expected_ra,
        expected_k       = expected_k,
        confidence       = confidence,
        confidence_tier  = tier,
        sb_implied_ip    = sb_ip,
        discrepancy_flag = disc_flag,
        discrepancy_detail = disc_detail,
        components_used  = n_comps,
        method           = method,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_pitcher_workload(
    pitcher_name: str,
    team_abbr:    str  = "",
    opp_abbr:     str  = "",
    *,
    game_date:  date | None = None,
    event_id:   str | None  = None,
    pitcher_id: int | None  = None,
) -> WorkloadProjection:
    """
    Return a WorkloadProjection for a single MLB starting pitcher.

    Parameters
    ----------
    pitcher_name : Full name (e.g. 'Gerrit Cole')
    team_abbr    : Pitcher's team abbreviation (optional; used for BP fatigue)
    opp_abbr     : Opposing team abbreviation (optional; used for opp difficulty)
    game_date    : Date of start (defaults to today ET)
    event_id     : Odds API event ID — enables sportsbook component
    pitcher_id   : MLB Stats API player ID (skips schedule lookup when provided)
    """
    from core.time_utils import now_est

    if game_date is None:
        game_date = now_est().date()

    cache_key = f"{pitcher_name}_{game_date.isoformat()}"
    if cache_key in _WORKLOAD_CACHE:
        return _WORKLOAD_CACHE[cache_key]

    # ── Resolve pitcher ID via schedule when not provided ────────────────────
    if pitcher_id is None:
        try:
            games = _get_probable_starters(game_date.strftime("%Y-%m-%d"))
            pitcher_lower = pitcher_name.lower()
            for g in games:
                home_match = g.get("home_name", "").lower() == pitcher_lower
                away_match = g.get("away_name", "").lower() == pitcher_lower
                # Primary: name+team match
                if home_match and (
                    not team_abbr or _team_matches(g.get("home_team", ""), team_abbr)
                ):
                    pitcher_id = g["home_id"]
                    if not team_abbr:
                        team_abbr = next(
                            (k for k, v in _TEAM_KEYWORDS.items()
                             if v.lower() in g.get("home_team", "").lower()), ""
                        )
                    if not opp_abbr:
                        opp_abbr = next(
                            (k for k, v in _TEAM_KEYWORDS.items()
                             if v.lower() in g.get("away_team", "").lower()), ""
                        )
                    break
                if away_match and (
                    not team_abbr or _team_matches(g.get("away_team", ""), team_abbr)
                ):
                    pitcher_id = g["away_id"]
                    if not team_abbr:
                        team_abbr = next(
                            (k for k, v in _TEAM_KEYWORDS.items()
                             if v.lower() in g.get("away_team", "").lower()), ""
                        )
                    if not opp_abbr:
                        opp_abbr = next(
                            (k for k, v in _TEAM_KEYWORDS.items()
                             if v.lower() in g.get("home_team", "").lower()), ""
                        )
                    break
        except Exception as exc:
            logger.debug(f"[pitcher_workload] schedule lookup failed: {exc}")

    try:
        proj = _build_projection(
            pitcher_name = pitcher_name,
            pitcher_id   = pitcher_id,
            team_abbr    = team_abbr,
            opp_abbr     = opp_abbr,
            game_date    = game_date,
            event_id     = event_id,
        )
    except Exception as exc:
        logger.warning(f"[pitcher_workload] projection failed for {pitcher_name}: {exc}")
        proj = WorkloadProjection(
            pitcher_name = pitcher_name,
            pitcher_id   = pitcher_id,
            team_abbr    = team_abbr,
            opp_abbr     = opp_abbr,
            game_date    = game_date.strftime("%Y-%m-%d"),
        )
        proj.__post_init__()

    logger.info(
        f"[pitcher_workload] {pitcher_name} ({team_abbr or '?'} vs {opp_abbr or '?'}): "
        f"IP={proj.expected_ip:.2f} K={proj.expected_k:.1f} "
        f"RA={proj.expected_ra:.1f} conf={proj.confidence:.0f}% "
        f"({proj.confidence_tier}) [{proj.method}]"
    )

    _WORKLOAD_CACHE[cache_key] = proj
    return proj


def get_game_workload_pair(
    home_abbr: str,
    away_abbr: str,
    *,
    game_date:    date | None = None,
    home_event_id: str | None = None,
    away_event_id: str | None = None,
) -> tuple["WorkloadProjection | None", "WorkloadProjection | None"]:
    """
    Resolve and return (home_WorkloadProjection, away_WorkloadProjection).
    Uses the MLB schedule to look up probable pitcher names and IDs.
    Returns (None, None) when the game cannot be found on the schedule.
    """
    from core.time_utils import now_est

    if game_date is None:
        game_date = now_est().date()

    try:
        games = _get_probable_starters(game_date.strftime("%Y-%m-%d"))
    except Exception as exc:
        logger.debug(f"[pitcher_workload] schedule lookup failed: {exc}")
        return None, None

    match: dict | None = None
    for g in games:
        if (_team_matches(g.get("home_team", ""), home_abbr) and
                _team_matches(g.get("away_team", ""), away_abbr)):
            match = g
            break

    if match is None:
        logger.debug(
            f"[pitcher_workload] No game found for {away_abbr}@{home_abbr} "
            f"on {game_date}."
        )
        return None, None

    home_wl = get_pitcher_workload(
        pitcher_name = match.get("home_name", "TBD"),
        team_abbr    = home_abbr,
        opp_abbr     = away_abbr,
        game_date    = game_date,
        event_id     = home_event_id,
        pitcher_id   = match.get("home_id"),
    )
    away_wl = get_pitcher_workload(
        pitcher_name = match.get("away_name", "TBD"),
        team_abbr    = away_abbr,
        opp_abbr     = home_abbr,
        game_date    = game_date,
        event_id     = away_event_id,
        pitcher_id   = match.get("away_id"),
    )
    return home_wl, away_wl


# ─────────────────────────────────────────────────────────────────────────────
# Downstream utility functions used by consumers
# ─────────────────────────────────────────────────────────────────────────────

def get_ramp_signal(wl: "WorkloadProjection", status_history: list[dict] | None = None) -> dict:
    """
    Thin wrapper around models/ramp_detection.py for MLB starters.

    Pulls the pitcher's last-3-start IP (recent) vs. everything before that
    in the same season's game log (baseline) -- reusing _get_game_log()'s
    cache, so this costs no extra HTTP call when called after
    get_pitcher_workload() for the same pitcher/date -- and flags a
    workload ramp-up (e.g. a starter stretched out gradually after an IL
    stint) so callers can discount a projection that would otherwise trust
    a fresh-off-IL sample at face value.

    Returns {"ramp_flag": False} (safe no-op) whenever there's no
    pitcher_id or too little game-log history to compare recent vs.
    baseline -- this must never raise, same contract as the rest of this
    module's public functions.
    """
    from models.ramp_detection import auto_adjust_workload_input

    if not wl.pitcher_id:
        return {"ramp_flag": False}
    try:
        season = int(wl.game_date[:4]) if wl.game_date else date.today().year
        log = _get_game_log(wl.pitcher_id, season)
        ip_values = [_parse_ip(g.get("inningsPitched")) for g in log]
        ip_values = [v for v in ip_values if v > 0]
        if len(ip_values) < _L3_WINDOW + 1:
            return {"ramp_flag": False}
        recent   = ip_values[-_L3_WINDOW:]
        baseline = ip_values[:-_L3_WINDOW]
        return auto_adjust_workload_input(
            recent_values=recent, baseline_values=baseline,
            sport="mlb_pitcher", status_history=status_history,
        )
    except Exception as exc:
        logger.debug(f"[pitcher_workload] ramp signal failed for {wl.pitcher_name}: {exc}")
        return {"ramp_flag": False}


def get_k_workload_scale(wl: "WorkloadProjection") -> float:
    """
    Strikeout projection scale factor for player_props.py.

    Scale = expected_ip / LEAGUE_AVG_IP, confidence-weighted so a
    low-confidence workload blends towards the neutral 1.0.
    Clamped to [0.55, 1.45].

    Example:
        Pitcher projects 4.8 IP (conf=78%) → scale = 0.873 × 0.78 + 1.0 × 0.22 ≈ 0.90
        → model projects 10% fewer strikeouts vs raw historical average.
    """
    raw_scale  = wl.expected_ip / LEAGUE_AVG_IP
    conf_w     = wl.confidence / 100.0
    blended    = raw_scale * conf_w + 1.0 * (1.0 - conf_w)
    return round(max(0.55, min(1.45, blended)), 3)


def get_f5_workload_adjustment(
    home_wl: "WorkloadProjection",
    away_wl: "WorkloadProjection",
) -> float:
    """
    League-mean adjustment (runs) for F5 total candidates in game_markets.py.

    Logic:
      For each starter, estimate runs allowed in the first 5 innings.
      Compare combined F5 RA to the league baseline (_LEAGUE_F5_RA = 4.68 runs).
      Return the delta, dampened by 0.40.

    When a starter projects short (< 5 IP), bullpen ERA (assumed +20%) fills
    the remaining F5 innings, raising expected runs.  Deep starters suppress
    the F5 total.

    Clamped to [-1.2, +1.2] runs.
    """
    def _f5_ra(wl: "WorkloadProjection") -> float:
        ip      = max(0.1, wl.expected_ip)
        f5_sp   = min(ip, 5.0)
        sp_ra   = wl.expected_ra * (f5_sp / ip)
        bp_inn  = max(0.0, 5.0 - f5_sp)
        bp_ra   = (LEAGUE_AVG_RA9 * 1.20) * bp_inn / 9.0
        return sp_ra + bp_ra

    expected = _f5_ra(home_wl) + _f5_ra(away_wl)
    delta    = (expected - _LEAGUE_F5_RA) * 0.40
    return round(max(-1.2, min(1.2, delta)), 2)
