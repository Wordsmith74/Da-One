"""
MLB-ONLY advanced metric blending.

There is no WNBA equivalent of CSW%/SIERA/park-factor in this file on purpose:
basketball doesn't have a "park," and the analogous basketball advanced stats
(true shooting %, usage rate, defensive matchup difficulty) are different
enough mechanically that they live in models/wnba_advanced.py instead of being
forced into this file's shape. Do not import this module for WNBA processing.
"""
from models.sport_config import MLB


def project_k_pct_advanced(csw_pct, swstr_pct, raw_k_pct):
    """
    Blends CSW% (called+swinging strike %) and SwStr% with the shrunk raw K%
    to get a more stable strikeout-rate projection. CSW%/SwStr% are *process*
    stats (how often a pitcher generates whiffs/called strikes per pitch) and
    are more predictive going forward than recent K% outcomes alone, which can
    be inflated/deflated by sequencing luck.

    Falls back gracefully to raw_k_pct alone if advanced columns are missing
    (e.g. pybaseball lookup failed or name match was empty) -- this MUST NOT
    raise, since run_pipeline.py depends on this never crashing a live run.
    """
    if csw_pct is None and swstr_pct is None:
        return raw_k_pct

    weights = []
    values = []
    if csw_pct is not None:
        # CSW% correlates strongly with K% league-wide; treat it as a strong signal
        values.append(csw_pct * 1.05)
        weights.append(0.45)
    if swstr_pct is not None:
        # SwStr% alone underestimates total K% (doesn't count called third strikes)
        values.append(swstr_pct * 1.85)
        weights.append(0.25)

    values.append(raw_k_pct)
    weights.append(1.0 - sum(weights))  # remaining weight on the shrunk raw rate

    blended = sum(v * w for v, w in zip(values, weights))
    return max(0.0, min(1.0, blended))


def pitcher_quality_factor(siera):
    """
    Converts a SIERA into a multiplicative run-environment factor centered at 1.0
    (league-average SIERA ~= 4.00). Lower SIERA -> tougher pitcher -> factor < 1
    (suppresses runs scored against them); higher SIERA -> factor > 1.
    Missing SIERA falls back to neutral 1.0 rather than crashing.
    """
    if siera is None:
        return 1.0
    league_avg_siera = 4.00
    # Cap the swing so one extreme SIERA value (small sample) can't blow up the sim
    factor = 1.0 + (siera - league_avg_siera) * 0.06
    return max(0.7, min(1.3, factor))


def f5_park_factor(full_game_park_factor):
    """
    Scales a full-game park factor down for First-5-Innings use. Starters
    typically face the order ~2x in 5 innings vs ~3x in 9, and bullpen/late-game
    park effects (twilight, wind shifts) shouldn't bleed into an F5 number.
    Scale factor pulled from sport_config.MLB so it's not a silent magic number.
    """
    scale = MLB["f5_park_scale"]
    # Move the factor toward 1.0 (neutral) by (1 - scale), i.e. partially apply it
    return 1.0 + (full_game_park_factor - 1.0) * scale
