"""
Season-phase detection (regular vs. postseason) and postseason adjustments.

MLB and WNBA postseasons behave differently and must not share an adjustment
curve:
  - MLB postseason: tighter bullpen usage, more pitcher-grade information
    (less rest, leverage-based usage) -- starters often see SHORTER outings,
    affecting K-prop projections.
  - WNBA postseason: roles tighten around high-usage starters, bench minutes
    shrink -- a role player's regular-season per-30 rate becomes less
    representative, while a starter's usage typically goes UP.

Both adjustments are real effects with academic/industry support broadly, but
the exact multipliers below are reasonable, labeled estimates, not fitted
coefficients -- recalibrate from real postseason data once a few years of
results exist (see models/backtest.py for the mechanism to do that).
"""
from datetime import datetime
from models.sport_config import MLB, WNBA

_CONFIGS = {"mlb": MLB, "wnba": WNBA}


def detect_phase(date_str, sport):
    """date_str: 'YYYY-MM-DD'. sport: 'mlb' or 'wnba'."""
    if sport not in _CONFIGS:
        raise ValueError(f"sport must be 'mlb' or 'wnba' -- got {sport!r}")
    cfg = _CONFIGS[sport]
    d = datetime.strptime(date_str, "%Y-%m-%d").date()

    start_m, start_d = cfg["regular_season_start_month_day"]
    end_m, end_d = cfg["regular_season_end_month_day"]
    season_start = d.replace(month=start_m, day=start_d)
    season_end = d.replace(month=end_m, day=end_d)

    if d < season_start:
        return "offseason"
    if d <= season_end:
        return "regular"
    return "postseason"


def adjust_for_postseason(value, entity_type, is_starter_or_high_usage=False, postseason_sample_size=0):
    """
    entity_type: 'mlb_pitcher' or 'wnba_player' -- selects the adjustment rule.
    Returns dict: {adjusted_value, note}

    The adjustment shrinks toward the regular-season value as postseason_sample_size
    grows (i.e. trust the new postseason-specific signal more once there's
    actually a postseason sample to trust, rather than applying a fixed haircut
    all October/September long).
    """
    if entity_type == "mlb_pitcher":
        # Shorter playoff outings -> discount workload-derived projections (K props)
        # by up to 12%, fading out as postseason sample size grows.
        max_adjustment = 0.12
        decay = min(1.0, postseason_sample_size / 5)  # fully faded out after 5 playoff starts
        adjustment = -max_adjustment * (1 - decay)
        adjusted = value * (1 + adjustment)
        note = f"MLB postseason workload discount applied: {adjustment*100:.1f}%"
    elif entity_type == "wnba_player":
        # High-usage starters: roles tighten, usage often rises in playoffs.
        # Bench/low-usage players: minutes compress -- this function is only
        # meant to be called for the player being projected, with the caller
        # deciding is_starter_or_high_usage from their regular-season role.
        max_adjustment = 0.08 if is_starter_or_high_usage else -0.10
        decay = min(1.0, postseason_sample_size / 4)
        adjustment = max_adjustment * (1 - decay)
        adjusted = value * (1 + adjustment)
        note = f"WNBA postseason {'usage bump' if is_starter_or_high_usage else 'bench compression'}: {adjustment*100:.1f}%"
    else:
        raise ValueError(f"entity_type must be 'mlb_pitcher' or 'wnba_player' -- got {entity_type!r}")

    return {"adjusted_value": adjusted, "note": note}
