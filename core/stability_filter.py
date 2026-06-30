"""
core/stability_filter.py

Model Stability Filter — relative uncertainty framework.

Per the Stability Filter Reform spec, the PRIMARY stability metric is
relative stability: posterior standard deviation expressed as a percentage
of the projected outcome (σ / |mean| × 100).

RELATIVE STABILITY TIERS (per sport)
  Tiers are sport-specific, mirroring the absolute guardrails below, since
  different sports/markets carry inherently different variance profiles
  (e.g. WNBA rebounds/assists props are noisier than MLB strikeout props
  due to smaller sample sizes and higher game-to-game variance).

  MLB / NBA (default)
    Elite       < 8%
    Acceptable  8–12%
    Caution     12–15%
    Reject      > 15%

  WNBA
    Elite       < 15%
    Acceptable  16–20%
    Caution     21–28%
    Reject      > 28%

  NOTE: these WNBA cutoffs are a starting estimate based on the observed
  reject distribution (15.3-37.5%), not a backtested calibration. Re-run
  scripts/backtest_stability_tiers.py against historical settled props
  before trusting them in production.

ABSOLUTE GUARDRAILS  (secondary — emergency only)
  Purpose: detect extreme volatility, data corruption, or simulation failures.
  They are NOT the primary acceptance criterion.
  MLB  : σ > 5.0
  NBA  : σ > 20.0
  WNBA : σ > 20.0

LOW-PROJECTION ABSOLUTE FLOOR  (tertiary — overrides relative tier)
  Purpose: relative σ/|mean| is structurally unreliable when the
  projection itself is tiny (e.g. a bench player projected for ~2
  rebounds). A small absolute σ (≈0.6-1.1) produces a huge relative
  percentage there even though the underlying simulation is fine.
  Below the per-sport projection threshold, switch to an absolute σ
  ceiling instead of the relative-percentage tiers.
  MLB  : proj < 1.5  → σ ceiling 0.8
  NBA  : proj < 5.0  → σ ceiling 2.5
  WNBA : proj < 5.0  → σ ceiling 2.0

  NOTE: starting estimates, not backtested. TODO: replace with
  scripts/backtest_stability_tiers.py output once enough low-projection
  settled props exist.

Exposed API (backward-compatible):
  check_stability(sport, posterior_std, posterior_mean=None) → (bool, str)
  stability_label(sport, posterior_std, posterior_mean=None) → str
  stability_tier(relative_pct, sport="")                      → str
"""
from __future__ import annotations

# ── Relative stability tiers (primary filter) ────────────────────────────────
# Per-sport, mirroring the _ABS_GUARDRAILS pattern below. Each entry is
# (elite_ceiling, acceptable_ceiling, caution_ceiling) in percent.
# > caution_ceiling → REJECT.
_REL_TIERS: dict[str, tuple[float, float, float]] = {
    "MLB":  (8.0, 12.0, 15.0),
    "NBA":  (8.0, 12.0, 15.0),
    # WNBA props run noisier (smaller minutes samples, higher game-to-game
    # variance on rebounds/assists) — observed rejects clustered 15.3-37.5%,
    # so a 15% ceiling was zeroing the sport out entirely. Widened to match
    # that distribution instead of MLB/NBA's tighter ceiling. TODO: replace
    # with backtested values once enough settled WNBA samples exist.
    "WNBA": (15.0, 20.0, 28.0),
}
_REL_DEFAULT: tuple[float, float, float] = (8.0, 12.0, 15.0)

# ── Absolute guardrails (emergency secondary filter only) ─────────────────────
_ABS_GUARDRAILS: dict[str, float] = {
    "MLB":  5.0,
    "NBA":  20.0,
    "WNBA": 20.0,
}
_ABS_DEFAULT = 20.0

# ── Low-projection absolute floor (tertiary — overrides relative tier) ────────
# When |posterior_mean| falls below this threshold, relative σ/|mean| becomes
# a noisy, misleading metric (tiny denominator inflates the percentage even
# when absolute σ is unremarkable). Below threshold, fall back to a flat
# absolute σ ceiling instead of the relative tiers.
# Each entry is (proj_threshold, abs_sigma_ceiling).
_LOW_PROJ_FLOOR: dict[str, tuple[float, float]] = {
    "MLB":  (1.5, 0.8),
    "NBA":  (5.0, 2.5),
    "WNBA": (5.0, 2.0),
}
_LOW_PROJ_DEFAULT: tuple[float, float] = (5.0, 2.5)


def stability_tier(relative_pct: float, sport: str = "") -> str:
    """Map a relative-stability percentage to its tier label, per sport."""
    elite, acceptable, caution = _REL_TIERS.get(sport.upper(), _REL_DEFAULT)
    if relative_pct < elite:
        return "ELITE"
    if relative_pct < acceptable:
        return "ACCEPTABLE"
    if relative_pct < caution:
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

    # ── Tertiary: low-projection absolute floor ───────────────────────────────
    # Relative σ/|mean| is unreliable when the projection itself is tiny
    # (e.g. bench-level rebounds/assists props). Below the per-sport
    # threshold, bypass the relative tiers and use a flat σ ceiling instead.
    if posterior_mean is not None and abs(posterior_mean) > 1e-6:
        proj_threshold, sigma_ceiling = _LOW_PROJ_FLOOR.get(sport_up, _LOW_PROJ_DEFAULT)
        if abs(posterior_mean) < proj_threshold:
            if posterior_std > sigma_ceiling:
                rel_pct = posterior_std / abs(posterior_mean) * 100.0
                return False, (
                    f"low-proj absolute floor breached: σ={posterior_std:.3f} > "
                    f"{sigma_ceiling:.2f} ({sport_up}, proj={posterior_mean:.2f} < "
                    f"{proj_threshold:.1f}; rel={rel_pct:.1f}% ignored as unreliable "
                    f"at this scale)"
                )
            rel_pct = posterior_std / abs(posterior_mean) * 100.0
            return True, (
                f"low-proj abs_floor=OK σ={posterior_std:.3f} "
                f"(proj={posterior_mean:.2f}, rel={rel_pct:.1f}% ignored)"
            )

    # ── Primary: relative stability ───────────────────────────────────────────
    if posterior_mean is not None and abs(posterior_mean) > 1e-6:
        rel_pct = posterior_std / abs(posterior_mean) * 100.0
        tier    = stability_tier(rel_pct, sport_up)

        if tier == "REJECT":
            _, _, caution_ceiling = _REL_TIERS.get(sport_up, _REL_DEFAULT)
            return False, (
                f"relative σ={rel_pct:.1f}% > {caution_ceiling:.0f}% threshold "
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
