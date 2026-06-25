"""
edge_calibrator.py — V3.0 Edge Recalibration System

Translates inflated simulation-native edge percentages into calibrated
values reflecting realistic sportsbook market inefficiency ranges.

The Bayesian simulation produces raw edges of 10–50 % because model
posteriors are tighter than real-world game variance warrants.  These
numbers are mathematically consistent but not interpretable as true
sportsbook edge — they conflate model confidence with market inefficiency.

V3.0 Edge Classification (calibrated scale):
  1–2 %  = Small Edge
  2–4 %  = Strong Edge
  4–6 %  = Elite Edge
  6–8 %  = Rare Edge
  8%+    = Exceptional Edge
  > 10%  = Requires auto-validation before publication

Calibration formulas
--------------------
Game markets (totals, team_total, spreads, moneyline):
    raw [floor, 50%] → calibrated [1%, cal_max] proportionally.
    raw < floor      → near-zero (sub-threshold, won't qualify).

Player props:
    raw [floor, 50%] → calibrated [2%, 15%].
    Slightly less compression — props carry more genuine model edge
    relative to thinly-priced team-level game markets.
"""

from __future__ import annotations

_GAME_MARKET_SUBSTRINGS: tuple[str, ...] = (
    "totals",
    "team_total",
    "spreads",
    "team_spread",
    "moneyline",
    "h2h",
    "alternate_",
    "run_line",
    "puck_line",
    "first_half",
    "first_quarter",
    "live_",
    "game_total",
    "full_game",
    "draw_no_bet",
    "double_chance",
)


def is_game_market(market: str) -> bool:
    m = market.strip().lower().replace(" ", "_")
    return any(m == s or m.startswith(s) for s in _GAME_MARKET_SUBSTRINGS)


# Raw-edge floor below which a game-market candidate is treated as sub-threshold.
# Below this the calibrated output is near-zero and won't reach any tier.
_GAME_RAW_FLOOR: dict[str, float] = {
    "MLB":  10.0,
    "NBA":   9.0,
    "WNBA":  8.0,
}
_GAME_RAW_FLOOR_DEFAULT = 10.0

# Calibrated maximum for game markets per sport
_GAME_CAL_MAX: dict[str, float] = {
    "MLB":  10.0,
    "NBA":  12.0,
    "WNBA": 10.0,
}
_GAME_CAL_MAX_DEFAULT = 10.0

# Raw-edge floor for player props
_PROP_RAW_FLOOR: dict[str, float] = {
    "MLB":  8.0,
    "NBA":  8.0,
    "WNBA": 8.0,
}
_PROP_RAW_FLOOR_DEFAULT = 8.0
_PROP_CAL_MAX = 15.0

# Validation threshold — any calibrated edge above this triggers auto-validation
EDGE_VALIDATION_THRESHOLD = 10.0


def calibrate_edge(raw_edge: float, sport: str, market: str) -> float:
    """
    Compress a raw simulation edge percentage into a calibrated value.

    Args:
        raw_edge: Edge % as output by the simulation (typically 0–50%).
        sport:    Sport key ("MLB", "NBA", "WNBA").
        market:   Market name (normalized or raw).

    Returns:
        Calibrated edge % in [0, 15]. Never negative.
    """
    raw = float(raw_edge)
    s   = sport.upper()

    if is_game_market(market):
        floor   = _GAME_RAW_FLOOR.get(s, _GAME_RAW_FLOOR_DEFAULT)
        cal_max = _GAME_CAL_MAX.get(s, _GAME_CAL_MAX_DEFAULT)

        if raw <= 0.0:
            return 0.0
        if raw < floor:
            # Sub-floor: scale to 0–1 % to preserve ordering
            return round(max(0.0, raw / floor * 1.0), 2)

        # Proportional map: [floor, 50%] → [1%, cal_max]
        cal = 1.0 + (raw - floor) / (50.0 - floor) * (cal_max - 1.0)
        return round(min(cal_max, cal), 2)

    else:
        # Player props — lighter compression
        floor = _PROP_RAW_FLOOR.get(s, _PROP_RAW_FLOOR_DEFAULT)

        if raw <= 0.0:
            return 0.0
        if raw < floor:
            return round(max(0.0, raw / floor * 2.0), 2)

        # Proportional map: [floor, 50%] → [2%, 15%]
        cal = 2.0 + (raw - floor) / (50.0 - floor) * (_PROP_CAL_MAX - 2.0)
        return round(min(_PROP_CAL_MAX, cal), 2)


def classify_edge(calibrated_edge: float) -> str:
    """Return the V3.0 edge classification label for a calibrated edge value."""
    if calibrated_edge >= 8.0:
        return "Exceptional Edge"
    if calibrated_edge >= 6.0:
        return "Rare Edge"
    if calibrated_edge >= 4.0:
        return "Elite Edge"
    if calibrated_edge >= 2.0:
        return "Strong Edge"
    if calibrated_edge >= 1.0:
        return "Small Edge"
    return "Marginal"


def requires_validation(calibrated_edge: float) -> bool:
    """
    True when calibrated edge exceeds 10 %.
    Per V3.0 spec: any projected edge above 10 % requires automatic
    validation before publication.
    """
    return calibrated_edge > EDGE_VALIDATION_THRESHOLD
