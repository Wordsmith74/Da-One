"""
core/wnba_regime.py

WNBA Totals Regime Adjustment Protocol.

Computes a Contextual Expected Total (CET) for each WNBA game by combining
weighted recent-form averages from both teams' full-game scoring histories.

The CET replaces the static league-average prior (165 pts) as the baseline
against which sportsbook lines are measured, eliminating false edges created
by anchoring high-total games (172–180) against a fixed 165 mean.

Regime classification and volatility adjustment:
  Low      (CET ≤ 164) : tight confidence bands, vol_multiplier = 0.85
  Neutral  (165–171)   : standard confidence bands, vol_multiplier = 1.00
  High     (CET ≥ 172) : wide confidence bands,   vol_multiplier = 1.20

Usage (game_markets, odds_client):
    from core.wnba_regime import compute_wnba_cet
    cet, regime, vol_mult = compute_wnba_cet(home_hist, away_hist)

    # Full-game total context: use cet as effective_league_mean
    effective_league_mean = cet
    adjusted_std = prior["std"] * vol_mult

    # Team-total context: split CET in half per team
    eff_mean = cet / 2.0
    eff_std  = prior["std"] * 0.65 * vol_mult
"""

from __future__ import annotations

import logging

logger = logging.getLogger("betting_bot")

# ---------------------------------------------------------------------------
# Weighting profile for recent-form average
# ---------------------------------------------------------------------------

_L10_WEIGHT    = 0.70   # last 10 games — primary signal
_L20_WEIGHT    = 0.20   # games 11–20  — secondary
_SEASON_WEIGHT = 0.10   # older games   — stabilisation only

# ---------------------------------------------------------------------------
# Regime thresholds
# ---------------------------------------------------------------------------

_LOW_THRESHOLD  = 164.5   # ≤ this → low environment
_HIGH_THRESHOLD = 171.5   # ≥ this → high environment

_FALLBACK_CET = 165.0     # used when both histories are empty

_VOL_MULTIPLIER: dict[str, float] = {
    "low":     0.85,
    "neutral": 1.00,
    "high":    1.20,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weighted_mean(history: list[float]) -> float:
    """
    Compute a recency-weighted mean from a chronological game-total history
    (most recent entry is LAST).

    Weighting:
        last 10 games : 70 %
        games 11–20   : 20 %
        older games   : 10 %
    """
    if not history:
        return _FALLBACK_CET

    n = len(history)
    # Reverse so index 0 = most recent
    rev = list(reversed(history))

    if n <= 3:
        # Too few games — plain mean, no weighting
        return sum(rev) / n

    l10    = rev[:10]
    l11_20 = rev[10:20]
    older  = rev[20:]

    # Build weighted average from whatever buckets exist
    parts: list[tuple[list[float], float]] = []
    if l10:
        parts.append((l10, _L10_WEIGHT))
    if l11_20:
        parts.append((l11_20, _L20_WEIGHT if older else (1.0 - _L10_WEIGHT)))
    if older:
        parts.append((older, _SEASON_WEIGHT))

    total_w = sum(w for _, w in parts)
    if total_w <= 0:
        return sum(rev) / n

    weighted_sum = sum(
        (sum(g) / len(g)) * w
        for g, w in parts
        if g
    )
    return weighted_sum / total_w


def classify_regime(cet: float) -> str:
    """Return 'low' | 'neutral' | 'high' based on the CET."""
    if cet <= _LOW_THRESHOLD:
        return "low"
    if cet >= _HIGH_THRESHOLD:
        return "high"
    return "neutral"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_wnba_cet(
    home_hist: list[float],
    away_hist: list[float],
) -> tuple[float, str, float]:
    """
    Compute the Contextual Expected Total (CET) for a WNBA game.

    Parameters
    ----------
    home_hist : list[float]
        Combined game totals (both teams' scores) for the home team's recent
        games, in chronological order (most recent LAST).
        Returned by core.intelligence.game_logs.get_team_game_totals().
    away_hist : list[float]
        Same structure for the away team.

    Returns
    -------
    cet : float
        Contextual Expected Total — the game-specific projection baseline.
    regime : str
        'low' | 'neutral' | 'high'
    vol_multiplier : float
        Multiplier to apply to league_std (0.85 / 1.00 / 1.20).
    """
    home_env = _weighted_mean(home_hist) if home_hist else _FALLBACK_CET
    away_env = _weighted_mean(away_hist) if away_hist else _FALLBACK_CET

    # CET is the mean of both teams' scoring environments
    cet = round((home_env + away_env) / 2.0, 2)
    regime = classify_regime(cet)
    vol_mult = _VOL_MULTIPLIER[regime]

    logger.debug(
        "[wnba_regime] CET=%.1f  regime=%s  vol_mult=%.2f  "
        "(home_env=%.1f, away_env=%.1f)",
        cet, regime, vol_mult, home_env, away_env,
    )
    return cet, regime, vol_mult
