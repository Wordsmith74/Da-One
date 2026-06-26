"""
Ramp-up / workload-limitation detection.

Purpose: catch the case where a recent sample looks great in isolation but is
actually a player on a workload limit (pitcher fresh off IL, getting stretched
out gradually; basketball player on a minutes restriction) -- betting their
*projected* workload as if it matches their *career-normal* workload is a
classic way models lose money even when the underlying rate stat is right.

This is shared infrastructure but is explicitly sport-aware: MLB tracks
innings pitched per start, WNBA tracks minutes per game, and the drop
threshold that counts as "a real ramp-up flag" is different for each
(see models/sport_config.py -- MLB["ramp_drop_threshold_pct"] vs
WNBA["ramp_drop_threshold_pct"]). Passing the wrong sport string here is a
silent miscalibration, so `sport` is required and validated.
"""
from models.sport_config import MLB, WNBA

_VALID_SPORTS = {"mlb_pitcher": MLB, "wnba_player": WNBA}


def auto_adjust_workload_input(recent_values, baseline_values, sport, status_history=None):
    """
    recent_values: most recent N values of the workload metric (IP for MLB
                    pitchers, minutes for WNBA players)
    baseline_values: a longer trailing window of the same metric, used as
                    the "normal" baseline to compare against
    sport: 'mlb_pitcher' or 'wnba_player' -- selects which config block's
                    drop-threshold and discount window apply
    status_history: optional list of recent injury/availability status dicts;
                    used only to decide whether a discount window should be
                    applied at all (e.g. fresh off an IL/inactive stint)

    Returns dict: {adjusted_value, ramp_flag, baseline_mean, recent_mean}
    """
    if sport not in _VALID_SPORTS:
        raise ValueError(
            f"sport must be one of {list(_VALID_SPORTS)} -- got {sport!r}. "
            f"Refusing to silently apply a generic threshold across sports."
        )
    cfg = _VALID_SPORTS[sport]

    recent_clean = [v for v in (recent_values or []) if v is not None]
    baseline_clean = [v for v in (baseline_values or []) if v is not None]

    if not recent_clean:
        return {"adjusted_value": 0.0, "ramp_flag": True, "baseline_mean": 0.0, "recent_mean": 0.0}

    recent_mean = sum(recent_clean) / len(recent_clean)
    baseline_mean = sum(baseline_clean) / len(baseline_clean) if baseline_clean else recent_mean

    drop_pct = 0.0
    if baseline_mean > 0:
        drop_pct = (baseline_mean - recent_mean) / baseline_mean * 100

    returning_from_layoff = _is_returning_from_layoff(status_history)
    ramp_flag = drop_pct >= cfg["ramp_drop_threshold_pct"] or returning_from_layoff

    if ramp_flag:
        # Trust the (lower) recent workload more than the stale baseline when
        # ramp-up is suspected -- weight recent 70/30 over baseline rather than
        # averaging evenly, since the baseline reflects a workload they may not
        # be cleared for yet.
        adjusted_value = recent_mean * 0.7 + baseline_mean * 0.3
    else:
        # No ramp concern -- blend recent and baseline evenly, recent slightly
        # favored since it's more current.
        adjusted_value = recent_mean * 0.6 + baseline_mean * 0.4

    return {
        "adjusted_value": adjusted_value,
        "ramp_flag": ramp_flag,
        "baseline_mean": baseline_mean,
        "recent_mean": recent_mean,
        "drop_pct": round(drop_pct, 1),
        "sport": sport,
    }


def _is_returning_from_layoff(status_history):
    if not status_history:
        return False
    statuses = [s.get("status", "").lower() for s in status_history if isinstance(s, dict)]
    return any(s in ("il", "inactive", "out", "day-to-day", "questionable") for s in statuses)
