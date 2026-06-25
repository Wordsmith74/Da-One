"""
core/market_weights.py — Market-specific confidence modifiers for player props.

Pipeline A (game markets) : no modifier applied — existing grading model unchanged.
Pipeline B (player props) : confidence adjusted by market tier before grading.

Adjustment sequence (inside run_gatekeeper):
  1. Edge calculation         — upstream (player_props.py / simulation engine)
  2. Market confidence mod    ← this module  (Step 0 in run_gatekeeper)
  3. Tier evaluation          — evaluate_tier() in decision_gatekeeper.py

Modifier values
  Tier 1 markets  : +5  (highest trust — stable, modelable, low variance)
  Tier 2 markets  : +3  (moderate trust — multiple paths to success)
  Tier 3 markets  : +1  (slight trust advantage over neutral)
  Tier 4 / neutral:  0
  Restricted      : -5  (high variance — deprioritised for Nuke/Diamond)

Restricted market tier cap (applied after stamp_tier):
  Restricted markets are capped at Edge tier unless the raw edge % exceeds
  the sport-specific "extraordinary edge" floor, in which case Diamond is
  allowed. Nuke is never granted to a restricted market.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Game market detection — Pipeline A  (never modified)
# ---------------------------------------------------------------------------

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
    """Return True if market belongs to Pipeline A (game markets — not modified)."""
    m = market.strip().lower().replace(" ", "_")
    return any(m == s or m.startswith(s) for s in _GAME_MARKET_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Confidence modifiers — WNBA
# ---------------------------------------------------------------------------

_WNBA_MODIFIERS: dict[str, float] = {
    # Tier 1 — stable roles, consistent minutes, slower book adjustment
    "player_assists":  +5.0,
    "player_rebounds": +5.0,
    # Tier 2 — multiple statistical paths to cover
    "player_points_rebounds_assists": +3.0,
    "player_points_assists":          +3.0,
    "player_rebounds_assists":        +3.0,
    # Tier 3 — no trust penalty, small bonus
    "player_points": +1.0,
    "player_threes": +1.0,
    # Restricted — high variance, deprioritised
    "player_steals": -5.0,
    "player_blocks": -5.0,
    "first_basket":  -5.0,
    "double_double": -5.0,
    "triple_double": -5.0,
}

# ---------------------------------------------------------------------------
# Confidence modifiers — NBA
# ---------------------------------------------------------------------------

_NBA_MODIFIERS: dict[str, float] = {
    # Tier 1 — minutes/usage/rotation-driven, highly modelable
    "player_rebounds": +5.0,
    "player_assists":  +5.0,
    # Tier 2 — combo markets with multiple paths
    "player_points_rebounds_assists": +3.0,
    "player_points_assists":          +3.0,
    "player_rebounds_assists":        +3.0,
    # Tier 3
    "player_points":    +1.0,
    # Neutral
    "player_threes":    0.0,
    "player_turnovers": 0.0,
    # Restricted
    "player_blocks":  -5.0,
    "player_steals":  -5.0,
    "first_basket":   -5.0,
    "triple_double":  -5.0,
    "double_double":  -5.0,
}

# ---------------------------------------------------------------------------
# Confidence modifiers — MLB
# ---------------------------------------------------------------------------

_MLB_MODIFIERS: dict[str, float] = {
    # Tier 1 — most modelable prop in the system
    "pitcher_strikeouts": +5.0,
    # Tier 2 — moderate trust
    "batter_total_bases": +3.0,
    # Tier 3 — small bonus
    "batter_hits_runs_rbis":   +1.0,
    "batter_rbis":             +1.0,
    "batter_runs_scored":      +1.0,
    "pitcher_hits_allowed":    -99.0,   # retired market — penalise so it never qualifies
    # Tier 4 / neutral
    "batter_hits":          0.0,
    "pitcher_earned_runs":  0.0,
    # Restricted — heavy variance penalty
    "batter_home_runs":    -5.0,
    "batter_stolen_bases": -5.0,
    "anytime_home_run":    -5.0,
    "first_inning_over":   -5.0,
    "first_inning_under":  -5.0,
    "first_inning_hits":   -5.0,
    "first_inning_runs":   -5.0,
}

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_SPORT_MODIFIERS: dict[str, dict[str, float]] = {
    "WNBA": _WNBA_MODIFIERS,
    "NBA":  _NBA_MODIFIERS,
    "MLB":  _MLB_MODIFIERS,
}

_RESTRICTED_MARKETS: dict[str, frozenset[str]] = {
    "WNBA": frozenset({
        "player_steals", "player_blocks",
        "first_basket", "double_double", "triple_double",
    }),
    "NBA": frozenset({
        "player_blocks", "player_steals",
        "first_basket", "triple_double", "double_double",
    }),
    "MLB": frozenset({
        "batter_home_runs", "batter_stolen_bases",
        "anytime_home_run",
        "first_inning_over", "first_inning_under",
        "first_inning_hits", "first_inning_runs",
    }),
}

# Edge % needed for a restricted market to be promoted past EDGE tier (Diamond max)
_RESTRICTED_DIAMOND_FLOOR: dict[str, float] = {
    "WNBA": 20.0,
    "NBA":  22.0,
    "MLB":  25.0,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_market_confidence_modifier(market: str, sport: str) -> float:
    """
    Return the confidence adjustment for a player-prop market.

    Returns 0.0 for game markets (Pipeline A) and unknown markets.
    Cap behaviour: applied by the caller after evaluating the tier.
    """
    if is_game_market(market):
        return 0.0

    norm = market.strip().lower().replace(" ", "_")
    sport_map = _SPORT_MODIFIERS.get(sport.upper())
    if sport_map is None:
        return 0.0

    if norm in sport_map:
        return sport_map[norm]

    # Prefix match for families like "first_inning_*"
    for key, val in sport_map.items():
        if norm.startswith(key):
            return val

    return 0.0


def is_restricted_market(market: str, sport: str) -> bool:
    """Return True if market is in the restricted (high-variance) set."""
    if is_game_market(market):
        return False
    norm = market.strip().lower().replace(" ", "_")
    return norm in _RESTRICTED_MARKETS.get(sport.upper(), frozenset())


def restricted_diamond_floor(sport: str) -> float:
    """Edge % that allows a restricted market to reach Diamond (never Nuke)."""
    return _RESTRICTED_DIAMOND_FLOOR.get(sport.upper(), 25.0)


def market_weight_label(market: str, sport: str) -> str:
    """Human-readable label for the modifier applied to a market."""
    if is_game_market(market):
        return "game market (Pipeline A — no modifier)"
    mod = get_market_confidence_modifier(market, sport)
    if mod > 0:
        tier_name = {5.0: "Tier 1", 3.0: "Tier 2", 1.0: "Tier 3"}.get(mod, f"+{mod:.0f}")
        return f"{tier_name} prop (+{mod:.0f} conf)"
    if mod < 0:
        return f"restricted prop ({mod:.0f} conf)"
    return "neutral prop (no modifier)"
