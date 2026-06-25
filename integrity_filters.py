"""
integrity_filters.py — V3.0 Sport-Specific Integrity Filters

Each Diamond/Nuke candidate must pass mandatory element checks before
publication. Missing elements downgrade the pick one tier per element.
Multiple missing elements (2+) cause the pick to be discarded entirely.

MLB required elements (Diamond + Nuke):
  1. Starting pitcher projection (ERA/FIP/xFIP)
  2. Bullpen score
  3. Park factor score
  4. Weather score
  5. Market agreement score

NBA required elements (Diamond + Nuke):
  1. Injury projection
  2. Rotation projection
  3. Pace projection
  4. Market agreement
  5. Rest analysis

WNBA required elements (Diamond + Nuke):
  1. Injury projection
  2. Pace projection
  3. Rotation projection
  4. Market agreement
  5. Travel/rest analysis
"""

from __future__ import annotations
from typing import Any


def _check_mlb_integrity(candidate: dict[str, Any]) -> list[str]:
    """Return list of missing MLB game-market integrity elements."""
    missing: list[str] = []

    # 1. Starting pitcher projection
    has_sp = bool(
        candidate.get("starting_pitcher")
        or candidate.get("sp_era") is not None
        or candidate.get("sp_fip") is not None
        or candidate.get("home_sp_era") is not None
        or candidate.get("pitcher_intel")
    )
    if not has_sp:
        missing.append("starting_pitcher_projection")

    # 2. Bullpen score
    has_bullpen = (
        candidate.get("bullpen_score") is not None
        or candidate.get("bullpen_fatigue") is not None
        or candidate.get("bullpen_era") is not None
    )
    if not has_bullpen:
        missing.append("bullpen_score")

    # 3. Park factor
    has_park = (
        candidate.get("park_factor") is not None
        or candidate.get("venue_factor") is not None
        or candidate.get("park_factor_runs") is not None
        or candidate.get("venue_edge_adjustment") is not None
    )
    if not has_park:
        missing.append("park_factor_score")

    # 4. Weather score
    has_weather = (
        candidate.get("weather_score") is not None
        or candidate.get("weather") is not None
        or candidate.get("wind_factor") is not None
        or candidate.get("temperature") is not None
    )
    if not has_weather:
        missing.append("weather_score")

    # 5. Market agreement score
    if candidate.get("market_agreement_score") is None:
        missing.append("market_agreement_score")

    return missing


def _check_nba_integrity(candidate: dict[str, Any]) -> list[str]:
    """Return list of missing NBA game-market integrity elements."""
    missing: list[str] = []

    # 1. Injury projection
    has_injury = (
        candidate.get("injury_report") is not None
        or candidate.get("lineup_intel") is not None
        or candidate.get("injury_impact") is not None
        or candidate.get("injury_score") is not None
    )
    if not has_injury:
        missing.append("injury_projection")

    # 2. Rotation projection
    has_rotation = (
        candidate.get("rotation") is not None
        or candidate.get("rotation_score") is not None
        or candidate.get("lineup_intel") is not None
        or candidate.get("usage_redistribution") is not None
    )
    if not has_rotation:
        missing.append("rotation_projection")

    # 3. Pace projection
    has_pace = (
        candidate.get("pace") is not None
        or candidate.get("pace_projection") is not None
        or candidate.get("offensive_pace") is not None
        or candidate.get("team_pace") is not None
    )
    if not has_pace:
        missing.append("pace_projection")

    # 4. Market agreement
    if candidate.get("market_agreement_score") is None:
        missing.append("market_agreement_score")

    # 5. Rest analysis
    has_rest = (
        candidate.get("rest_days") is not None
        or candidate.get("rest_analysis") is not None
        or candidate.get("back_to_back") is not None
        or candidate.get("rest_factor") is not None
    )
    if not has_rest:
        missing.append("rest_analysis")

    return missing


def _check_wnba_integrity(candidate: dict[str, Any]) -> list[str]:
    """Return list of missing WNBA game-market integrity elements."""
    missing: list[str] = []

    # 1. Injury projection
    has_injury = (
        candidate.get("injury_report") is not None
        or candidate.get("lineup_intel") is not None
        or candidate.get("injury_impact") is not None
        or candidate.get("injury_score") is not None
    )
    if not has_injury:
        missing.append("injury_projection")

    # 2. Pace projection
    has_pace = (
        candidate.get("pace") is not None
        or candidate.get("pace_projection") is not None
        or candidate.get("offensive_pace") is not None
    )
    if not has_pace:
        missing.append("pace_projection")

    # 3. Rotation projection
    has_rotation = (
        candidate.get("rotation") is not None
        or candidate.get("rotation_score") is not None
        or candidate.get("lineup_intel") is not None
    )
    if not has_rotation:
        missing.append("rotation_projection")

    # 4. Market agreement
    if candidate.get("market_agreement_score") is None:
        missing.append("market_agreement_score")

    # 5. Travel/rest analysis
    has_travel_rest = (
        candidate.get("rest_days") is not None
        or candidate.get("travel_factor") is not None
        or candidate.get("back_to_back") is not None
        or candidate.get("rest_travel") is not None
    )
    if not has_travel_rest:
        missing.append("travel_rest_analysis")

    return missing


def run_integrity_filter(
    candidate: dict[str, Any],
    sport: str,
    is_game_market: bool = True,
) -> tuple[int, list[str]]:
    """
    Run the integrity filter for the given sport and candidate.

    Player props skip the filter (no downgrade applied).

    Args:
        candidate:      The enriched candidate dict from the pipeline.
        sport:          Sport key ("MLB", "NBA", "WNBA").
        is_game_market: False for player-prop candidates.

    Returns:
        (tier_downgrades, missing_elements)
            tier_downgrades == 0  → all present, no downgrade
            tier_downgrades == 1  → one missing, drop one tier
            tier_downgrades >= 2  → multiple missing, discard pick
    """
    if not is_game_market:
        return 0, []

    s = sport.upper()
    if s == "MLB":
        missing = _check_mlb_integrity(candidate)
    elif s == "NBA":
        missing = _check_nba_integrity(candidate)
    elif s == "WNBA":
        missing = _check_wnba_integrity(candidate)
    else:
        return 0, []

    return len(missing), missing
