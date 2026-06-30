"""
confidence_caps.py — V3.0 Confidence Ceiling System

Confidence must be earned through validated data, market confirmation,
historical calibration, and predictive accuracy.

No market may exceed these ceilings regardless of projected edge.
A realistic 82-confidence play is superior to a misleading 99-confidence play.

V3.0 spec ceilings by tier:

  MLB Totals  — Nuke:85  Diamond:80  Gold Standard:75
  NBA Totals  — Nuke:88  Diamond:83  Gold Standard:78
  WNBA Totals — Nuke:86  Diamond:81  Gold Standard:76
  Props (all) — Nuke:92  Diamond:87  Gold Standard:82
"""

from __future__ import annotations

# CONFIDENCE_CAPS[sport][market_category][tier_name] → float ceiling
CONFIDENCE_CAPS: dict[str, dict[str, dict[str, float]]] = {
    "MLB": {
        "totals": {"Nuke": 85.0, "Diamond": 80.0, "Gold Standard": 75.0},
        "props":  {"Nuke": 92.0, "Diamond": 87.0, "Gold Standard": 82.0},
    },
    "NBA": {
        "totals": {"Nuke": 88.0, "Diamond": 83.0, "Gold Standard": 78.0},
        "props":  {"Nuke": 92.0, "Diamond": 87.0, "Gold Standard": 82.0},
    },
    "WNBA": {
        "totals": {"Nuke": 86.0, "Diamond": 81.0, "Gold Standard": 76.0},
        "props":  {"Nuke": 92.0, "Diamond": 87.0, "Gold Standard": 82.0},
    },
}

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


def _is_game_market(market: str) -> bool:
    m = market.strip().lower().replace(" ", "_")
    return any(m == s or m.startswith(s) for s in _GAME_MARKET_SUBSTRINGS)


def market_category(market: str) -> str:
    """Return 'totals' for game markets, 'props' for player props."""
    return "totals" if _is_game_market(market) else "props"


def get_confidence_cap(sport: str, market: str, tier_name: str) -> float:
    """
    Return the V3.0 maximum allowed confidence for this sport × market × tier.

    Falls back to 99.0 (effectively uncapped) when the combination is not
    in the spec table (e.g., unknown sport or tier = DISCARD).
    """
    sport_caps = CONFIDENCE_CAPS.get(sport.upper())
    if not sport_caps:
        return 99.0
    cat       = market_category(market)
    cat_caps  = sport_caps.get(cat, sport_caps.get("totals", {}))
    return float(cat_caps.get(tier_name, 99.0))


def apply_confidence_cap(
    confidence: float,
    sport: str,
    market: str,
    tier_name: str,
) -> tuple[float, bool]:
    """
    Clamp confidence to the V3.0 ceiling.

    Returns:
        (capped_confidence, was_capped)
            was_capped is True when the input exceeded the ceiling.
    """
    cap   = get_confidence_cap(sport, market, tier_name)
    capped = min(float(confidence), cap)
    return capped, capped < float(confidence)
