"""
core/stability_filter.py

Model Stability Filter — relative uncertainty framework.

Per the Stability Filter Reform spec, the PRIMARY stability metric is
relative stability: posterior standard deviation expressed as a percentage
of the projected outcome (σ / |mean| × 100).

RELATIVE STABILITY TIERS
  Elite       < 8%    — highly concentrated; may qualify for premium confidence
  Acceptable  8–12%   — fully eligible; most profitable plays land here
  Caution     12–15%  — eligible with moderated confidence if edge/market confirms
  Reject      > 15%   — rejected; insufficient precision for recommendation

ABSOLUTE GUARDRAILS  (secondary — emergency only)
  Purpose: detect extreme volatility, data corruption, or simulation failures.
  They are NOT the primary acceptance criterion.
  MLB  : σ > 5.0
  NBA  : σ > 20.0
  WNBA : σ > 20.0

Exposed API (backward-compatible):
  check_stability(sport, posterior_std, posterior_mean=None) → (bool, str)
  stability_label(sport, posterior_std, posterior_mean=None) → str
  stability_tier(relative_pct)                               → str
"""
from __future__ import annotations

# ── Relative stability tiers (primary filter) ────────────────────────────────
_REL_ELITE:      float = 8.0    # %
_REL_ACCEPTABLE: float = 12.0   # %
_REL_CAUTION:    float = 15.0   # %
# > _REL_CAUTION  →  REJECT

# ── Absolute guardrails (emergency secondary filter only) ─────────────────────
_ABS_GUARDRAILS: dict[str, float] = {
    "MLB":  5.0,
    "NBA":  20.0,
    "WNBA": 20.0,
}
_ABS_DEFAULT = 20.0


def stability_tier(relative_pct: float) -> str:
    """Map a relative-stability percentage to its tier label."""
    if relative_pct < _REL_ELITE:
        return "ELITE"
    if relative_pct < _REL_ACCEPTABLE:
        return "ACCEPTABLE"
    if relative_pct < _REL_CAUTION:
        return "CAUTION"
    return "REJECT"


def check_stability(
    sport: str,
    posterior_std: float,
    posterior_mean: float | None = None,
) -> tuple[bool, str]:
    """
    Evaluate whether a simulation result is stable enough to publish.

    Primary  : relative stability  (σ / |mean|).
    Secondary: absolute guardrails (emergency).

    Returns:
        (True,  detail_str)  — stable (Elite / Acceptable / Caution); proceed
        (False, reason)      — unstable; reject as No Play
    """
    sport_up = sport.upper()

    # ── Primary: relative stability ───────────────────────────────────────────
    if posterior_mean is not None and abs(posterior_mean) > 1e-6:
        rel_pct = posterior_std / abs(posterior_mean) * 100.0
        tier    = stability_tier(rel_pct)

        if tier == "REJECT":
            return False, (
                f"relative σ={rel_pct:.1f}% > {_REL_CAUTION:.0f}% threshold "
                f"({sport_up}  σ={posterior_std:.3f}  proj={posterior_mean:.2f})"
            )

        tier_tag = f" [{tier}]" if tier != "ACCEPTABLE" else ""
        return True, f"rel={rel_pct:.1f}%{tier_tag}"

    # ── Fallback: no mean available — use absolute guardrail only ─────────────
    guardrail = _ABS_GUARDRAILS.get(sport_up, _ABS_DEFAULT)
    if posterior_std > guardrail:
        return False, (
            f"absolute guardrail breached: σ={posterior_std:.3f} > {guardrail:.1f} "
            f"({sport_up}, no projection mean available)"
        )

    return True, f"abs_guard=OK σ={posterior_std:.3f}"


def stability_label(
    sport: str,
    posterior_std: float,
    posterior_mean: float | None = None,
) -> str:
    """Short human-readable label for logging."""
    is_stable, reason = check_stability(sport, posterior_std, posterior_mean)
    if is_stable:
        return f"stable({reason})"
    return f"UNSTABLE({reason})"
