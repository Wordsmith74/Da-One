"""
wnba_regime.py

Regime Adjustment Protocol for WNBA totals.

WNBA team scoring is far more volatile game-to-game than NBA (smaller
rosters, bigger star-dependency, more variable pace), so a flat season
average is a weak baseline for "expected total" on any given night.

compute_wnba_cet() computes a Contextual Expected Total (CET) from each
team's recent game-total history, plus a regime label and a volatility
multiplier used to widen/narrow the Bayesian prior's std before sampling.

Regimes
-------
"hot"      both teams trending above their own season norm recently
"cold"     both teams trending below their own season norm recently
"mixed"    one team hot, one cold (or insufficient signal either way)
"neutral"  no usable history for either team (synthetic-history fallback)

This is intentionally simple (recency-weighted average + a hot/cold
volatility bump) rather than a full opponent-adjusted model -- it exists to
give the Bayesian prior a *contextual* center instead of a season-long flat
average, not to be a complete scoring model on its own.
"""
from __future__ import annotations


def _recency_weighted_mean(values: list[float]) -> float:
    """Weight the most recent games more heavily (simple linear ramp)."""
    if not values:
        return 0.0
    n = len(values)
    weights = list(range(1, n + 1))  # oldest=1 ... newest=n
    total_w = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total_w


def compute_wnba_cet(
    home_hist: list[float],
    away_hist: list[float],
) -> tuple[float, str, float]:
    """
    Compute Contextual Expected Total, regime label, and volatility
    multiplier for a WNBA matchup.

    Args:
        home_hist: Home team's recent game-total history (points scored,
                   most-recent-last).
        away_hist: Away team's recent game-total history, same convention.

    Returns:
        (cet, regime, vol_mult)
            cet      : float -- contextual expected total for this game.
            regime   : "hot" | "cold" | "mixed" | "neutral"
            vol_mult : float -- multiplier applied to the prior's std.
                       >1.0 widens uncertainty (regime is noisy/mixed),
                       <1.0 narrows it (both teams trending consistently).
    """
    if not home_hist and not away_hist:
        return 0.0, "neutral", 1.0

    home_recent = _recency_weighted_mean(home_hist) if home_hist else None
    away_recent = _recency_weighted_mean(away_hist) if away_hist else None

    # Season-long flat average per side, for "trending above/below own norm".
    home_season = sum(home_hist) / len(home_hist) if home_hist else None
    away_season = sum(away_hist) / len(away_hist) if away_hist else None

    def _trend(recent: float | None, season: float | None) -> str | None:
        if recent is None or season is None or season == 0:
            return None
        delta_pct = (recent - season) / season
        if delta_pct >= 0.05:
            return "hot"
        if delta_pct <= -0.05:
            return "cold"
        return "flat"

    home_trend = _trend(home_recent, home_season)
    away_trend = _trend(away_recent, away_season)

    # CET = recency-weighted blend of both sides' recent scoring; falls
    # back to whichever side has data if only one team has history.
    recents = [r for r in (home_recent, away_recent) if r is not None]
    cet = round(sum(recents) / len(recents), 2) if recents else 0.0

    trends = [t for t in (home_trend, away_trend) if t is not None]
    if trends and all(t == "hot" for t in trends):
        regime, vol_mult = "hot", 0.85
    elif trends and all(t == "cold" for t in trends):
        regime, vol_mult = "cold", 0.85
    elif trends:
        regime, vol_mult = "mixed", 1.15
    else:
        regime, vol_mult = "neutral", 1.0

    return cet, regime, vol_mult
