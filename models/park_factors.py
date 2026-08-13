"""
models/park_factors.py

Static reference tables consumed by the NRFI environment tier
(models.nrfi_handicapper.environment_tier_multiplier) and by
models/weather_intel.py's new get_nrfi_environment_inputs().

Two tables, kept deliberately separate from behavior:

1. PARK_RUN_FACTOR -- full-game, multi-year park run factors (1.0 =
   league-neutral). These are well-established, slow-moving numbers (unlike
   umpire tendency or first-inning splits, park effects are stable across
   seasons) commonly published by ESPN/FanGraphs/Statcast park-factor
   pages. Values below are rounded, order-of-magnitude-correct approximations
   for the current park era -- REPLACE WITH THE LATEST PUBLISHED FIGURES
   (e.g. Statcast's park factor page, updated each offseason) rather than
   trusting these indefinitely; ballparks get humidors, fence changes, and
   multi-year factors drift. environment_tier_multiplier already dampens
   this via NRFI_PARK_SCALE (sport_config.MLB) since only a fraction of a
   full-game park effect plausibly applies to a single first inning.

2. PARK_CF_AZIMUTH_DEG -- approximate center-field orientation (compass
   bearing a ball hit to straightaway CF travels, 0=North/360, 90=East,
   180=South, 270=West) for each open-air park, used to resolve Open-Meteo's
   wind DIRECTION against "blowing out to CF" vs. "blowing in" rather than
   just using raw wind speed with no direction context (see
   weather_intel.get_nrfi_environment_inputs). Approximate -- sourced from
   published stadium-orientation references; good enough for a coarse
   "mostly out / mostly in / crosswind" bucket, not precise degree-level
   physics.
"""
from __future__ import annotations

from typing import Optional

# Full-game park run factor, 1.0 = neutral. Domes/retractables get a value
# close to neutral by design (climate-controlled). Rounded approximations --
# refresh from a current published source periodically.
PARK_RUN_FACTOR: dict[str, float] = {
    "ARI": 1.02, "ATL": 1.01, "BAL": 0.97, "BOS": 1.06, "CHC": 1.02,
    "CWS": 1.00, "CIN": 1.07, "CLE": 0.97, "COL": 1.28, "DET": 0.97,
    "HOU": 1.00, "KC": 1.01, "LAA": 0.98, "LAD": 0.97, "MIA": 0.94,
    "MIL": 1.01, "MIN": 1.00, "NYM": 0.96, "NYY": 1.03, "OAK": 0.92,
    "PHI": 1.04, "PIT": 0.96, "SD": 0.95, "SF": 0.92, "SEA": 0.95,
    "STL": 0.98, "TB": 0.94, "TEX": 1.03, "TOR": 1.02, "WSH": 0.99,
}

# Approximate CF bearing in degrees. Open-air parks only listed with real
# orientation research value; domes/retractables (mostly closed) are
# omitted since wind direction doesn't apply when the roof's shut anyway.
PARK_CF_AZIMUTH_DEG: dict[str, float] = {
    "ATL": 15, "BAL": 30, "BOS": 55, "CHC": 30, "CWS": 30,
    "CIN": 90, "CLE": 3, "COL": 65, "DET": 75, "KC": 45,
    "LAA": 30, "LAD": 30, "MIN": 15, "NYM": 33, "NYY": 75,
    "PHI": 5, "PIT": 30, "SD": 5, "SF": 95, "STL": 45,
    "WSH": 30,
}


def get_park_run_factor(team_abbr: str) -> float:
    """Neutral (1.0) fallback for unknown abbreviations -- never guesses low/high."""
    return PARK_RUN_FACTOR.get(team_abbr.upper(), 1.0)


def get_wind_out_component(
    team_abbr: str, wind_mph: Optional[float], wind_direction_deg: Optional[float],
) -> float:
    """
    Resolves a raw wind speed+direction reading into a signed "blowing out
    to CF" component in mph: positive = blowing out, negative = blowing in,
    magnitude scaled toward zero for pure crosswinds. Returns 0.0 (neutral)
    for domes, unknown parks, or missing direction data -- a false-neutral
    is safer than guessing a direction effect from speed alone (this is the
    same fail-safe posture weather_intel.py already documents for its own
    K-rate wind decision).
    """
    if wind_mph is None or wind_direction_deg is None:
        return 0.0
    cf_azimuth = PARK_CF_AZIMUTH_DEG.get(team_abbr.upper())
    if cf_azimuth is None:
        return 0.0

    import math
    # Open-Meteo reports wind_direction as the direction the wind is
    # COMING FROM. "Blowing out to CF" means wind is coming from behind
    # home plate toward CF, i.e. from the direction roughly opposite the
    # CF azimuth (CF azimuth + 180).
    blowing_from_deg = cf_azimuth + 180.0
    delta = abs(((wind_direction_deg - blowing_from_deg) + 180) % 360 - 180)
    # delta near 0 -> wind is coming from behind home plate -> blowing OUT.
    # delta near 180 -> wind coming from CF -> blowing IN.
    # Scale by cos(delta) so a pure crosswind (delta=90) contributes ~0.
    component = wind_mph * math.cos(math.radians(delta))
    return round(component, 2)
