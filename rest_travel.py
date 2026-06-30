"""
rest_travel.py

Pulls each team's recent schedule from the ESPN public API, calculates
rest days since the last completed game, and estimates travel miles via
haversine distance between city coordinates.

Edge adjustments applied
------------------------
  Back-to-back (0 rest days)      : -2.5
  Short rest   (1 rest day)       : -1.0
  Normal rest  (2 rest days)      : ±0.0
  Extended rest (3+ rest days)    : +0.5
  Long travel  (>1 500 mi) + ≤1 day rest : additional -1.0

All adjustments are capped at ±4.0.

Data resilience
---------------
Uses data_fetcher.fetch_espn() — strict 3-second timeout, ESPN primary →
ESPN fallback → RotoWire waterfall (Rule 1 + Rule 2).  Every source
failure is logged as a "Source Unavailable" event (Rule 4).
Response structure is validated before use (Rule 3).

Fail-safe: returns RestTravelFactor() with zero adjustment on any error.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("betting_bot")

from core.data_fetcher import fetch_espn          # Rule 1+2
from core.data_validator import validate_schedule  # Rule 3

_SPORT_PATHS: dict[str, str] = {
    "WNBA": "basketball/wnba",
    "NBA":  "basketball/nba",
    "MLB":  "baseball/mlb",
}


# ---------------------------------------------------------------------------
# City coordinates for haversine travel distance
# ---------------------------------------------------------------------------

# fmt: off
_CITY_COORDS: dict[str, tuple[float, float]] = {
    # WNBA
    "ATL": (33.749, -84.388),   "CHI": (41.883, -87.632),
    "CON": (41.430, -72.110),   "DAL": (32.776, -96.797),
    "IND": (39.768, -86.158),   "LAS": (36.169, -115.140),
    "LVA": (36.169, -115.140),  "MIN": (44.977, -93.265),
    "NYL": (40.714, -74.006),   "PHO": (33.448, -112.074),
    "SEA": (47.606, -122.332),  "WAS": (38.907, -77.037),
    # NBA (overlaps with WNBA where city is shared)
    "BOS": (42.361, -71.058),   "BKN": (40.682, -73.975),
    "CHA": (35.228, -80.843),   "CLE": (41.499, -81.695),
    "DEN": (39.740, -104.984),  "DET": (42.332, -83.047),
    "GSW": (37.768, -122.388),  "HOU": (29.763, -95.363),
    "LAC": (34.052, -118.243),  "LAL": (34.052, -118.243),
    "MEM": (35.149, -90.049),   "MIA": (25.774, -80.190),
    "MIL": (43.045, -87.907),   "NOP": (29.951, -90.072),
    "NYK": (40.714, -74.006),   "OKC": (35.467, -97.519),
    "ORL": (28.538, -81.379),   "PHI": (39.953, -75.165),
    "PHX": (33.448, -112.074),  "POR": (45.523, -122.676),
    "SAC": (38.582, -121.494),  "SAS": (29.424, -98.494),
    "TOR": (43.651, -79.347),   "UTA": (40.758, -111.891),
    # MLB
    "ARI": (33.448, -112.074),  "BAL": (39.283, -76.622),
    "CHC": (41.883, -87.632),   "CWS": (41.833, -87.634),
    "CIN": (39.103, -84.512),   "COL": (39.740, -104.984),
    "KC":  (39.100, -94.577),   "LAA": (33.800, -117.883),
    "LAD": (34.052, -118.243),  "NYM": (40.714, -74.006),
    "NYY": (40.714, -74.006),   "OAK": (37.751, -122.200),
    "PIT": (40.441, -79.996),   "SD":  (32.715, -117.157),
    "SF":  (37.768, -122.388),  "STL": (38.627, -90.198),
    "TB":  (27.771, -82.636),   "TEX": (32.750, -97.082),
    "WSH": (38.907, -77.037),
}
# fmt: on


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two (lat, lon) points."""
    R = 3_958.8  # Earth radius in miles
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _travel_miles(home_abbr: str, away_abbr: str) -> float | None:
    """Miles the away team traveled to reach the home venue. None if unknown."""
    home = _CITY_COORDS.get(home_abbr.upper())
    away = _CITY_COORDS.get(away_abbr.upper())
    if home is None or away is None:
        return None
    return _haversine_miles(away[0], away[1], home[0], home[1])


# ---------------------------------------------------------------------------
# ESPN schedule parsing
# ---------------------------------------------------------------------------

def _fetch_last_game_date(team_abbr: str, sport: str) -> datetime | None:
    """
    Return the UTC datetime of the team's most recently completed game,
    or None if the schedule cannot be fetched or validated.
    """
    sport_path = _SPORT_PATHS.get(sport.upper())
    if not sport_path:
        return None

    # Rule 1+2: fetch with strict timeout + waterfall failover
    result = fetch_espn(f"{sport_path}/teams/{team_abbr}/schedule")
    if not result.ok:
        logger.debug(
            f"[rest_travel] All sources failed for {team_abbr} schedule "
            f"(last error: {result.error})"
        )
        return None

    # Rule 3: validate structure before using data
    validation = validate_schedule(result.data)
    if not validation.valid:
        logger.debug(
            f"[rest_travel] Data integrity: {validation.reason} for {team_abbr}"
            + (f"  missing={validation.missing_fields}" if validation.missing_fields else "")
        )
        return None

    events = result.data.get("events", [])  # type: ignore[union-attr]
    now_utc = datetime.now(timezone.utc)
    completed_dates: list[datetime] = []

    for event in events:
        status = event.get("status", {}).get("type", {})
        if not status.get("completed", False):
            continue
        raw_date = event.get("date", "")
        if not raw_date:
            continue
        try:
            if raw_date.endswith("Z"):
                raw_date = raw_date[:-1] + "+00:00"
            dt = datetime.fromisoformat(raw_date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < now_utc:
                completed_dates.append(dt)
        except ValueError:
            continue

    return max(completed_dates) if completed_dates else None


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class RestTravelFactor:
    rest_days:       int | None   = None
    travel_miles:    float | None = None
    edge_adjustment: float        = 0.0
    factor_text:     str          = ""


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def get_rest_travel_factor(
    team_abbr: str,
    sport: str,
    game_time_utc: datetime | None = None,
) -> RestTravelFactor:
    """
    Compute rest-days and travel-miles for *team_abbr* and return an
    edge adjustment (positive = favourable, negative = penalise).

    Fail-safe: returns RestTravelFactor() with zero adjustment on any error.
    """
    try:
        return _compute(team_abbr, sport, game_time_utc)
    except Exception as exc:
        logger.debug(f"[rest_travel] Unexpected error for {team_abbr}: {exc}")
        return RestTravelFactor()


def _compute(
    team_abbr: str,
    sport: str,
    game_time_utc: datetime | None,
) -> RestTravelFactor:
    last_game = _fetch_last_game_date(team_abbr, sport)
    if last_game is None:
        return RestTravelFactor()

    reference = game_time_utc if game_time_utc else datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)

    rest_days = max(0, (reference.date() - last_game.date()).days)

    # Edge adjustment from rest
    if rest_days == 0:
        adj = -2.5
        rest_label = "back-to-back"
    elif rest_days == 1:
        adj = -1.0
        rest_label = "1 day rest"
    elif rest_days == 2:
        adj = 0.0
        rest_label = "2 days rest"
    else:
        adj = 0.5
        rest_label = f"{rest_days} days rest"

    # Travel adjustment (away-team heuristic: use team coords as proxy)
    miles = _travel_miles(team_abbr, team_abbr)  # self → 0 for home games
    travel_label = ""
    travel_adj   = 0.0

    # Better: if we know the matchup, travel is away→home.
    # Without opponent info, estimate 0 for now; caller can enrich later.

    # Cap total adjustment
    total_adj = round(max(-4.0, min(4.0, adj + travel_adj)), 2)

    parts = [rest_label]
    if travel_label:
        parts.append(travel_label)
    factor_text = f"{team_abbr}: {', '.join(parts)}"

    logger.debug(
        f"[rest_travel] {team_abbr} {sport}: rest={rest_days}d  adj={total_adj:+.1f}"
    )

    return RestTravelFactor(
        rest_days       = rest_days,
        travel_miles    = miles,
        edge_adjustment = total_adj,
        factor_text     = factor_text,
    )


def get_matchup_travel_factor(
    home_abbr: str,
    away_abbr: str,
    sport: str,
    game_time_utc: datetime | None = None,
) -> dict[str, RestTravelFactor]:
    """
    Compute rest+travel factors for both teams in a matchup.
    Returns dict with keys 'home' and 'away'.
    """
    miles = _travel_miles(home_abbr, away_abbr)

    home_factor = get_rest_travel_factor(home_abbr, sport, game_time_utc)
    away_factor = get_rest_travel_factor(away_abbr, sport, game_time_utc)

    # Apply travel penalty to away team
    if miles is not None:
        away_factor.travel_miles = miles
        if miles > 1_500:
            extra = -1.0 if (away_factor.rest_days or 99) <= 1 else -0.5
            away_factor.edge_adjustment = round(
                max(-4.0, away_factor.edge_adjustment + extra), 2
            )
            away_factor.factor_text += f", {int(miles):,}mi travel"

    return {"home": home_factor, "away": away_factor}
