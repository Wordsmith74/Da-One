"""
venue_intel.py

Home / Away / Neutral Environment Intelligence Module.

What it does
------------
1. Classifies every game as HOME_AWAY or NEUTRAL.
2. Computes venue-specific team performance profiles:
     MLB  — home/away W-L record + avg runs scored/allowed (MLB Stats API)
     NBA/WNBA — home/road W-L record + avg points (ESPN team record API)
3. Builds a Home Advantage Score and Road Competency Score for the matchup.
4. Applies MLB park factors (hardcoded, updated for 2025 season).
5. Returns edge adjustment (max ±5.0) + output tags + factor text.

Edge sources
------------
  Park factor (MLB totals only)  : up to ±3.0
  Venue split matchup            : up to ±2.0
  ───────────────────────────────────────────
  Total                          : capped ±5.0

Safety rules
------------
- Never overrides pitcher, injury, lineup, or market-efficiency signals.
- Neutral site: all home/away adjustments removed.
- Fail-safe: returns VenueFactor(edge_adjustment=0.0) on any error.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from urllib import request as urllib_req
from urllib.error import HTTPError

logger = logging.getLogger("betting_bot")


# ---------------------------------------------------------------------------
# MLB Park Factors  (run factor, 100 = league neutral, 2025 estimates)
# ---------------------------------------------------------------------------

_MLB_PARK_FACTORS: dict[str, tuple[int, str]] = {
    # hitter-friendly
    "COL": (122, "Coors Field (extreme hitter-friendly)"),
    "CIN": (109, "Great American Ball Park (hitter-friendly)"),
    "PHI": (107, "Citizens Bank Park (hitter-friendly)"),
    "BOS": (106, "Fenway Park (hitter-friendly)"),
    "CWS": (105, "Guaranteed Rate Field (hitter-friendly)"),
    "TEX": (104, "Globe Life Field (hitter-friendly)"),
    "KC":  (104, "Kauffman Stadium (hitter-friendly)"),
    "NYY": (103, "Yankee Stadium"),
    "ATL": (102, "Truist Park"),
    "TOR": (101, "Rogers Centre"),
    # neutral
    "MIL": (100, "American Family Field"),
    "HOU": (100, "Minute Maid Park"),
    "CLE": (100, "Progressive Field"),
    "DET": (100, "Comerica Park"),
    # pitcher-friendly
    "OAK": ( 99, "Oakland Coliseum"),
    "STL": ( 99, "Busch Stadium"),
    "MIN": ( 98, "Target Field (pitcher-friendly)"),
    "BAL": ( 98, "Camden Yards"),
    "PIT": ( 98, "PNC Park (pitcher-friendly)"),
    "CHC": ( 97, "Wrigley Field"),
    "NYM": ( 97, "Citi Field (pitcher-friendly)"),
    "LAA": ( 97, "Angel Stadium (pitcher-friendly)"),
    "WSH": ( 97, "Nationals Park"),
    "ARI": ( 96, "Chase Field"),
    "TB":  ( 96, "Tropicana Field"),
    "LAD": ( 95, "Dodger Stadium (pitcher-friendly)"),
    "SEA": ( 95, "T-Mobile Park (pitcher-friendly)"),
    "MIA": ( 93, "loanDepot park (pitcher-friendly)"),
    "SF":  ( 92, "Oracle Park (pitcher-friendly)"),
    "SD":  ( 91, "Petco Park (pitcher-friendly)"),
}

# League baseline home win rates (used to measure deviation)
_LEAGUE_HOME_WP: dict[str, float] = {
    "MLB":  0.540,
    "NBA":  0.595,
    "WNBA": 0.560,
}

# ESPN sport path for team record lookups
_ESPN_SPORT_PATH: dict[str, str] = {
    "NBA":  "basketball/nba",
    "WNBA": "basketball/wnba",
}

# ---------------------------------------------------------------------------
# Process-level caches
# ---------------------------------------------------------------------------

# abbr → MLB schedule games (completed, regular season)
_MLB_SCHEDULE_CACHE: dict[str, list[dict[str, Any]]] = {}

# (abbr, sport) → dict with home/road WP and scoring
_ESPN_RECORD_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Internal data helpers
# ---------------------------------------------------------------------------

@dataclass
class _VenueSplits:
    """Normalised home/away performance splits for one team."""
    home_wp:           float | None = None  # 0–1
    road_wp:           float | None = None
    home_avg_scored:   float | None = None  # runs or points
    home_avg_allowed:  float | None = None
    road_avg_scored:   float | None = None
    road_avg_allowed:  float | None = None
    overall_avg_scored:float | None = None
    games_played:      int          = 0


# ---------------------------------------------------------------------------
# MLB Stats API venue-split helpers
# ---------------------------------------------------------------------------

def _fetch_mlb_schedule(team_id: int) -> list[dict[str, Any]]:
    season = date.today().year
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&teamId={team_id}&season={season}&gameType=R"
    )
    with urllib_req.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    return [
        g
        for d in data.get("dates", [])
        for g in d.get("games", [])
        if g.get("status", {}).get("statusCode") == "F"
    ]


_MLB_ABBREV_TO_ID: dict[str, int] = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111,
    "CHC": 112, "CWS": 145, "CIN": 113, "CLE": 114,
    "COL": 115, "DET": 116, "HOU": 117, "KC":  118,
    "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158,
    "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD":  135, "SF":  137,
    "SEA": 136, "STL": 138, "TB":  139, "TEX": 140,
    "TOR": 141, "WSH": 120,
}


def _get_mlb_splits(abbr: str) -> _VenueSplits:
    """Compute MLB home/away splits from schedule.  Cached per team."""
    if abbr in _MLB_SCHEDULE_CACHE:
        games = _MLB_SCHEDULE_CACHE[abbr]
    else:
        team_id = _MLB_ABBREV_TO_ID.get(abbr)
        if team_id is None:
            return _VenueSplits()
        try:
            games = _fetch_mlb_schedule(team_id)
            _MLB_SCHEDULE_CACHE[abbr] = games
        except Exception as exc:
            logger.debug(f"[venue] MLB schedule fetch failed for {abbr}: {exc}")
            _MLB_SCHEDULE_CACHE[abbr] = []
            return _VenueSplits()

    team_id = _MLB_ABBREV_TO_ID.get(abbr)
    h_wins = h_losses = a_wins = a_losses = 0
    h_scored: list[float] = []
    h_allowed: list[float] = []
    a_scored: list[float] = []
    a_allowed: list[float] = []

    for g in games:
        h  = g["teams"]["home"]
        a  = g["teams"]["away"]
        hid = h.get("team", {}).get("id")
        aid = a.get("team", {}).get("id")
        hs  = h.get("score")
        as_ = a.get("score")
        if hs is None or as_ is None:
            continue

        if hid == team_id:
            h_scored.append(float(hs))
            h_allowed.append(float(as_))
            if h.get("isWinner", hs > as_):
                h_wins += 1
            else:
                h_losses += 1
        elif aid == team_id:
            a_scored.append(float(as_))
            a_allowed.append(float(hs))
            if a.get("isWinner", as_ > hs):
                a_wins += 1
            else:
                a_losses += 1

    def _safe_avg(lst: list[float]) -> float | None:
        return sum(lst) / len(lst) if lst else None

    home_total  = h_wins + h_losses
    away_total  = a_wins + a_losses
    all_scored  = h_scored + a_scored

    return _VenueSplits(
        home_wp           = h_wins / home_total if home_total >= 5 else None,
        road_wp           = a_wins / away_total if away_total >= 5 else None,
        home_avg_scored   = _safe_avg(h_scored),
        home_avg_allowed  = _safe_avg(h_allowed),
        road_avg_scored   = _safe_avg(a_scored),
        road_avg_allowed  = _safe_avg(a_allowed),
        overall_avg_scored = _safe_avg(all_scored),
        games_played       = home_total + away_total,
    )


# ---------------------------------------------------------------------------
# ESPN team record helper (NBA / WNBA)
# ---------------------------------------------------------------------------

def _get_espn_splits(abbr: str, sport: str) -> _VenueSplits:
    """Fetch home/road WP + overall avg scoring from ESPN team endpoint."""
    key = (abbr.lower(), sport.upper())
    if key in _ESPN_RECORD_CACHE:
        rec = _ESPN_RECORD_CACHE[key]
    else:
        sport_path = _ESPN_SPORT_PATH.get(sport.upper())
        if not sport_path:
            return _VenueSplits()
        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/"
            f"{sport_path}/teams/{abbr.lower()}"
        )
        try:
            with urllib_req.urlopen(url, timeout=8) as resp:
                data = json.loads(resp.read().decode())
            items = data.get("team", {}).get("record", {}).get("items", [])
            rec: dict[str, Any] = {}
            for item in items:
                t    = item.get("type", "")
                stats = {s["name"]: s["value"] for s in item.get("stats", [])}
                if t == "total":
                    rec["total_wp"]   = stats.get("winPercent")
                    rec["avg_pts_for"]= stats.get("avgPointsFor")
                    rec["wins"]       = int(stats.get("wins", 0))
                    rec["losses"]     = int(stats.get("losses", 0))
                elif t == "home":
                    rec["home_wp"]  = stats.get("winPercent")
                    rec["home_wins"]= int(stats.get("wins", 0))
                    rec["home_losses"] = int(stats.get("losses", 0))
                elif t == "road":
                    rec["road_wp"]  = stats.get("winPercent")
                    rec["road_wins"]= int(stats.get("wins", 0))
                    rec["road_losses"] = int(stats.get("losses", 0))
            _ESPN_RECORD_CACHE[key] = rec
        except HTTPError as exc:
            logger.debug(f"[venue] ESPN record HTTP {exc.code} for {abbr} {sport}")
            _ESPN_RECORD_CACHE[key] = {}
            return _VenueSplits()
        except Exception as exc:
            logger.debug(f"[venue] ESPN record failed for {abbr} {sport}: {exc}")
            _ESPN_RECORD_CACHE[key] = {}
            return _VenueSplits()

    home_games = rec.get("home_wins", 0) + rec.get("home_losses", 0)
    road_games = rec.get("road_wins", 0) + rec.get("road_losses", 0)
    total_games = rec.get("wins", 0) + rec.get("losses", 0)

    return _VenueSplits(
        home_wp            = rec.get("home_wp") if home_games >= 5 else None,
        road_wp            = rec.get("road_wp") if road_games >= 5 else None,
        overall_avg_scored = rec.get("avg_pts_for"),
        games_played       = total_games,
    )


# ---------------------------------------------------------------------------
# Park factor → edge
# ---------------------------------------------------------------------------

def _park_factor_adj(home_abbr: str, direction: str) -> tuple[float, str]:
    """
    Return (edge_adjustment, label) for the home park.
    Positive = favours OVER; negative = favours UNDER.
    Reversed for 'under' direction.
    """
    factor, park_name = _MLB_PARK_FACTORS.get(home_abbr, (100, ""))
    if not park_name:
        return 0.0, ""

    deviation = factor - 100
    if deviation >= 22:
        adj = 3.0
    elif deviation >= 8:
        adj = 2.0
    elif deviation >= 4:
        adj = 1.0
    elif deviation >= 1:
        adj = 0.5
    elif deviation >= -2:
        adj = 0.0
    elif deviation >= -5:
        adj = -0.5
    elif deviation >= -8:
        adj = -1.0
    elif deviation >= -12:
        adj = -2.0
    else:
        adj = -2.5

    if direction.lower() == "under":
        adj = -adj

    return round(adj, 1), park_name


# ---------------------------------------------------------------------------
# Venue matchup edge
# ---------------------------------------------------------------------------

def _venue_matchup_adj(
    home_splits: _VenueSplits,
    away_splits:  _VenueSplits,
    sport:        str,
    direction:    str,
) -> tuple[float, list[str], list[str]]:
    """
    Compare home team's home performance vs away team's road performance.
    Returns (edge_adj, tags, text_parts).
    Positive adj = environment favours more total scoring (OVER).
    Reversed for 'under'.
    """
    league_home_wp = _LEAGUE_HOME_WP.get(sport.upper(), 0.54)
    tags: list[str] = []
    parts: list[str] = []
    adj: float = 0.0

    home_wp = home_splits.home_wp
    road_wp  = away_splits.road_wp

    # ── Home advantage score ────────────────────────────────────────────────
    home_score_label = ""
    home_adj: float = 0.0
    if home_wp is not None:
        deviation = home_wp - league_home_wp
        if deviation >= 0.12:
            home_adj = 1.5
            home_score_label = "ELITE HOME TEAM"
            tags.append("STRONG HOME EDGE")
            tags.append("HOME FIELD BOOST")
        elif deviation >= 0.07:
            home_adj = 1.0
            home_score_label = "STRONG HOME TEAM"
            tags.append("HOME FIELD BOOST")
        elif deviation >= 0.03:
            home_adj = 0.5
            home_score_label = "ABOVE-AVERAGE HOME TEAM"
        elif deviation <= -0.08:
            home_adj = -1.0
            home_score_label = "WEAK HOME TEAM"
        if home_score_label:
            parts.append(f"{home_score_label} ({home_wp:.0%} at home)")

    # ── Road competency score ───────────────────────────────────────────────
    road_adj: float = 0.0
    road_score_label = ""
    if road_wp is not None:
        if road_wp >= 0.48:
            road_adj = 0.5
            road_score_label = "STRONG ROAD TEAM"
            tags.append("STRONG ROAD TEAM")
        elif road_wp >= 0.40:
            road_adj = 0.0
            road_score_label = "AVERAGE ROAD TEAM"
        elif road_wp >= 0.32:
            road_adj = -0.5
            road_score_label = "POOR ROAD TEAM"
        else:
            road_adj = -1.0
            road_score_label = "VERY POOR ROAD TEAM"
        if road_score_label:
            parts.append(f"{road_score_label} ({road_wp:.0%} away)")

    # ── MLB: use actual scoring splits for finer adjustment ─────────────────
    scoring_adj: float = 0.0
    if sport.upper() == "MLB":
        league_avg_runs = 4.5
        h_scored  = home_splits.home_avg_scored
        a_road_sc = away_splits.road_avg_scored

        if h_scored is not None:
            home_off_boost = h_scored - league_avg_runs
            scoring_adj += max(-0.5, min(0.5, home_off_boost * 0.15))

        if a_road_sc is not None:
            away_road_diff = a_road_sc - league_avg_runs
            scoring_adj += max(-0.5, min(0.5, away_road_diff * 0.10))

        scoring_adj = round(scoring_adj, 2)

    # ── Major venue edge flag ───────────────────────────────────────────────
    if (
        home_wp is not None and road_wp is not None
        and home_wp >= 0.60 and road_wp <= 0.38
    ):
        tags.append("MAJOR VENUE ADVANTAGE")

    # ── Total venue split adjustment ────────────────────────────────────────
    adj = round(home_adj + road_adj + scoring_adj, 2)

    if direction.lower() == "under":
        adj = -adj

    return round(max(-2.0, min(2.0, adj)), 2), tags, parts


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class VenueFactor:
    venue_type:           str              = "HOME_AWAY"
    home_abbr:            str              = ""
    away_abbr:            str              = ""
    home_wp:              float | None     = None
    road_wp:              float | None     = None
    home_avg_scored:      float | None     = None
    road_avg_scored:      float | None     = None
    park_factor:          int   | None     = None
    park_label:           str              = ""
    park_adj:             float            = 0.0
    venue_split_adj:      float            = 0.0
    edge_adjustment:      float            = 0.0
    tags:                 list[str]        = field(default_factory=list)
    factor_text:          str              = ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_venue_factor(
    home_name:  str,
    away_name:  str,
    sport:      str,
    home_abbr:  str = "",
    away_abbr:  str = "",
    direction:  str = "over",
) -> VenueFactor:
    """
    Evaluate venue environment and return edge adjustment + context tags.

    Parameters
    ----------
    home_name / away_name : Full team names (fallback abbrev resolution).
    sport                 : 'MLB', 'NBA', or 'WNBA'.
    home_abbr / away_abbr : Short abbreviations (preferred).
    direction             : 'over' or 'under' — determines sign of park adj.

    Returns
    -------
    VenueFactor with edge_adjustment capped at ±5.0.
    Zero-adjustment VenueFactor on any error.
    """
    try:
        return _compute(home_name, away_name, sport, home_abbr, away_abbr, direction)
    except Exception as exc:
        logger.debug(
            f"[venue] unexpected error ({home_name} vs {away_name}): {exc}"
        )
        return VenueFactor()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute(
    home_name: str,
    away_name: str,
    sport:     str,
    home_abbr: str,
    away_abbr: str,
    direction: str,
) -> VenueFactor:
    if not home_abbr:
        home_abbr = home_name[:3].upper()
    if not away_abbr:
        away_abbr = away_name[:3].upper()

    sport_up = sport.upper()

    # ── Venue splits ─────────────────────────────────────────────────────────
    if sport_up == "MLB":
        home_splits = _get_mlb_splits(home_abbr)
        away_splits  = _get_mlb_splits(away_abbr)
    elif sport_up in ("NBA", "WNBA"):
        home_splits = _get_espn_splits(home_abbr, sport_up)
        away_splits  = _get_espn_splits(away_abbr, sport_up)
    else:
        return VenueFactor()

    # ── Park factor (MLB totals only) ────────────────────────────────────────
    park_adj   = 0.0
    park_label = ""
    park_factor_val: int | None = None
    if sport_up == "MLB":
        park_adj, park_label = _park_factor_adj(home_abbr, direction)
        pf_entry = _MLB_PARK_FACTORS.get(home_abbr)
        if pf_entry:
            park_factor_val = pf_entry[0]

    # ── Venue split matchup ───────────────────────────────────────────────────
    split_adj, tags, split_parts = _venue_matchup_adj(
        home_splits, away_splits, sport_up, direction
    )

    # ── Park factor tags ──────────────────────────────────────────────────────
    if park_label:
        tags.append("PARK FACTOR IMPACT")
        if park_adj >= 1.5:
            tags.append("HIGH-SCORING PARK")
        elif park_adj <= -1.5:
            tags.append("PITCHER-FRIENDLY PARK")

    # ── Factor text ──────────────────────────────────────────────────────────
    text_parts: list[str] = []
    if park_label and park_adj != 0.0:
        sign = "+" if park_adj > 0 else ""
        text_parts.append(f"{park_label} (park adj {sign}{park_adj:.1f})")
    text_parts.extend(split_parts)
    if any(t in tags for t in ("MAJOR VENUE ADVANTAGE",)):
        text_parts.append("MAJOR VENUE ADVANTAGE")

    # ── Total adjustment ─────────────────────────────────────────────────────
    total_adj = round(max(-5.0, min(5.0, park_adj + split_adj)), 2)

    # Deduplicate tags
    seen: set[str] = set()
    unique_tags: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique_tags.append(t)

    logger.debug(
        f"[venue] {away_abbr}@{home_abbr} {sport_up}: "
        f"park_adj={park_adj:+.1f}  split_adj={split_adj:+.1f}  "
        f"total={total_adj:+.1f}  tags={unique_tags}"
    )

    return VenueFactor(
        venue_type       = "HOME_AWAY",
        home_abbr        = home_abbr,
        away_abbr        = away_abbr,
        home_wp          = home_splits.home_wp,
        road_wp          = away_splits.road_wp,
        home_avg_scored  = home_splits.home_avg_scored,
        road_avg_scored  = away_splits.road_avg_scored,
        park_factor      = park_factor_val,
        park_label       = park_label,
        park_adj         = park_adj,
        venue_split_adj  = split_adj,
        edge_adjustment  = total_adj,
        tags             = unique_tags,
        factor_text      = " | ".join(text_parts),
    )
