"""
core/composite_confidence_score.py — Pick Ranking Governance Protocol v1.0

Computes the Composite Confidence Score (CCS) that governs pick ranking
within the dominant-filter pool.  CCS replaces the legacy formula
(edge×0.6 + conf×0.4) everywhere pool-ranking decisions are made.

The best pick is not necessarily the largest edge.  The best pick is the
candidate with the strongest combination of reliability, signal agreement,
validated edge, robustness, and controlled volatility.

Five weighted factors (all normalized 0–100):

  Factor 1 — Projection Reliability  (35%)
      How trustworthy is the underlying projection?
      Sub: data reliability score (DRS), L5/L10 recency, line stability.

  Factor 2 — Signal Agreement         (25%)
      How many independent model components support the same conclusion?
      Sub: MIS (sharp/CLV/RLM/steam), Market Agreement Score, directional confirmation.

  Factor 3 — Edge Strength            (20%)
      How large and market-validated is the edge?
      Sub: calibrated edge, effective edge vs raw ratio.

  Factor 4 — Volatility Adjustment    (10%)
      How stable is the situation? (lower vol = higher score)
      Sub: line velocity stability, market liquidity, odds value.

  Factor 5 — Market Efficiency        (10%)
      How historically predictable is this market type?
      Signal-layer-depth priors; updated as graded data accumulates.

Sensitivity Multiplier (0.78–1.00)
      Applied on top of the five-factor raw CCS.
      Tests: "If projection shifted 5% against our position, would the
      play remain valid?"  Fragile edges are penalised so a large-but-fragile
      edge never outranks a smaller-but-robust one.

Public API
----------
  compute_ccs(bd, ld) → (float, str)
      Returns (ccs_score_0_to_100, robustness_label).

  sensitivity_label(posterior_mean, sportsbook_line, direction) → str
      Standalone helper for logging / display.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from output.telegram_formatter import BetDisplay

logger = logging.getLogger("betting_bot")

# ---------------------------------------------------------------------------
# Factor weights — must sum to 1.0
# ---------------------------------------------------------------------------
_W_RELIABILITY = 0.35
_W_AGREEMENT   = 0.25
_W_EDGE        = 0.20
_W_VOLATILITY  = 0.10
_W_EFFICIENCY  = 0.10

# ---------------------------------------------------------------------------
# Market efficiency priors (0–100 scale)
# Starting estimates based on signal-layer depth per market.
# ---------------------------------------------------------------------------
_MARKET_EFFICIENCY_PRIOR: dict[str, float] = {
    # MLB — pitching props carry the deepest signal stack
    # (workload model, 7-layer K matchup, Savant metrics, FIP/ERA priors)
    "pitcher_strikeouts":  62.0,
    "pitcher_earned_runs": 55.0,
    "totals":              55.0,
    "team_total":          57.0,
    # NBA / WNBA — game totals solid; player props limited by data availability
    "player_rebounds":                   58.0,
    "player_assists":                    56.0,
    "player_points":                     50.0,   # suspended market, but still scored
    "player_steals":                     52.0,
    "player_blocks":                     52.0,
    "player_threes":                     51.0,
    "player_turnovers":                  50.0,
    "player_points_rebounds":            55.0,
    "player_points_assists":             54.0,
    "player_rebounds_assists":           55.0,
    "player_points_rebounds_assists":    53.0,
}
_MARKET_EFFICIENCY_DEFAULT = 53.0

# ---------------------------------------------------------------------------
# Sensitivity test thresholds
# ---------------------------------------------------------------------------
# Gap retention = (gap_after_5pct_adverse_shift) / original_gap
_ROBUST_FLOOR   = 0.75   # ≥ 75% → robust
_MODERATE_FLOOR = 0.45   # ≥ 45% → moderate
_FRAGILE_FLOOR  = 0.20   # ≥ 20% → somewhat fragile
# < 0.20 → fragile

# Multipliers applied to raw CCS
_MULT_ROBUST         = 1.00
_MULT_MODERATE       = 0.92
_MULT_SOMEWHAT       = 0.85
_MULT_FRAGILE        = 0.78


# ---------------------------------------------------------------------------
# Sensitivity test
# ---------------------------------------------------------------------------

def sensitivity_label(
    posterior_mean: float,
    sportsbook_line: float,
    direction: str,
) -> str:
    """Return just the robustness label without the multiplier."""
    _, label = _sensitivity(posterior_mean, sportsbook_line, direction)
    return label


def _sensitivity(
    posterior_mean: float,
    sportsbook_line: float,
    direction: str,
) -> tuple[float, str]:
    """
    Compute the sensitivity robustness multiplier for a pick.

    Logic
    -----
    gap      = projection distance from the line in the bet direction.
    shift    = 5% of the projected value, applied against the position.
    retention = (gap − shift) / gap — how much headroom survives the shift.

    A dominant pitcher projected at 7.5K over a 5.5 line has a 2.0K gap;
    after a 5% adverse shift (0.375K) the retention is 1.625/2.0 = 81% → robust.

    A marginal play with projection 6.0K over a 5.5 line has a 0.5K gap;
    after the same shift the retention is 0.125/0.5 = 25% → somewhat fragile.
    """
    dir_lower = direction.strip().lower()
    gap = (
        (posterior_mean - sportsbook_line) if dir_lower == "over"
        else (sportsbook_line - posterior_mean)
    )

    if gap <= 0.0:
        # Projection already at or against us — fragile by definition
        return _MULT_FRAGILE, "fragile"

    shift     = abs(posterior_mean) * 0.05
    remaining = gap - shift
    retention = remaining / gap

    if retention >= _ROBUST_FLOOR:
        return _MULT_ROBUST, "robust"
    if retention >= _MODERATE_FLOOR:
        return _MULT_MODERATE, "moderate"
    if retention >= _FRAGILE_FLOOR:
        return _MULT_SOMEWHAT, "somewhat_fragile"
    return _MULT_FRAGILE, "fragile"


# ---------------------------------------------------------------------------
# Factor 1: Projection Reliability
# ---------------------------------------------------------------------------

def _f1_reliability(bd: "BetDisplay", wd: dict[str, Any]) -> float:
    """
    How trustworthy is the underlying projection?

    Sub-weights:
      45% — DRS (data completeness, sample size quality, 0–100)
      25% — data recency (L5/L10 availability from real game log)
      30% — line stability (market not moving; velocity-based, 0–10 → 0–100)
    """
    c = bd.bet.raw_result or {}

    drs = float(
        bd.bet.data_reliability_score
        or wd.get("data_reliability_score")
        or 70
    )

    l5  = c.get("l5_avg")
    l10 = c.get("l10_avg")
    if l5 is not None and l10 is not None:
        recency = 100.0
    elif l10 is not None:
        recency = 70.0
    elif l5 is not None:
        recency = 60.0
    else:
        recency = 40.0   # season average only

    line_stability = min(100.0, float(wd.get("stability_score", 5.0)) * 10.0)

    return 0.45 * drs + 0.25 * recency + 0.30 * line_stability


# ---------------------------------------------------------------------------
# Factor 2: Signal Agreement
# ---------------------------------------------------------------------------

def _f2_agreement(bd: "BetDisplay", wd: dict[str, Any]) -> float:
    """
    How many independent model components support the same conclusion?

    Sub-weights:
      50% — MIS (market intelligence score: CLV, RLM, steam, sharp action)
      35% — Market Agreement Score (multi-model signal convergence)
      15% — directional sharp confirmation (are sharps on our side?)
    """
    c   = bd.bet.raw_result or {}
    mis = float(bd.bet.mis_score or 0)
    mas = float(c.get("market_agreement_score", 50))

    direction     = bd.bet.direction.lower()
    sharp_sig     = str(c.get("sharp_signal") or "").lower()
    rlm_detected  = bool(c.get("rlm_detected", False))
    steam_detected = bool(c.get("steam_detected", False))

    confirms = (
        (direction == "over"  and "over"  in sharp_sig) or
        (direction == "under" and "under" in sharp_sig)
    )
    opposes = (
        (direction == "over"  and "under" in sharp_sig) or
        (direction == "under" and "over"  in sharp_sig)
    )

    if confirms:
        sharp_score = 80.0
        if rlm_detected:
            sharp_score += 10.0    # RLM on our side is a strong validation
        if steam_detected:
            sharp_score += 10.0    # steam confirmation is additional weight
    elif opposes:
        sharp_score = 20.0         # market pushing against us — penalty
    else:
        sharp_score = 50.0         # no sharp signal — neutral

    return 0.50 * mis + 0.35 * mas + 0.15 * sharp_score


# ---------------------------------------------------------------------------
# Factor 3: Edge Strength
# ---------------------------------------------------------------------------

def _f3_edge(bd: "BetDisplay", wd: dict[str, Any]) -> float:
    """
    How large and market-validated is the edge?

    Sub-weights:
      50% — calibrated edge (10% edge → 100; proportional)
      50% — effective edge vs raw ratio (market-adjustment agreement)
    """
    raw_edge = float(bd.bet.edge_percentage)
    eff_edge = float(wd.get("effective_edge") or raw_edge)

    edge_score = min(100.0, raw_edge / 10.0 * 100.0)

    eff_ratio = (
        min(1.0, eff_edge / raw_edge) * 100.0
        if raw_edge > 0.0
        else 50.0
    )

    return 0.50 * edge_score + 0.50 * eff_ratio


# ---------------------------------------------------------------------------
# Factor 4: Volatility Adjustment
# ---------------------------------------------------------------------------

def _f4_volatility(bd: "BetDisplay", wd: dict[str, Any]) -> float:
    """
    How stable / low-variance is the situation? (higher = less volatile)

    Sub-weights:
      45% — market intelligence score (book depth, line tightness, consensus; 0–100)
      35% — market liquidity (book count normalized; 0–10 → 0–100)
      20% — odds value (heavy juice or very long odds signal instability)

    Note: stability_score (line velocity) was removed here because it already
    appears in Factor 1 (_f1_reliability, 30% sub-weight).  MIS is an
    independent signal measuring market depth and consensus quality.
    """
    mis  = min(100.0, float(bd.bet.mis_score or 0))
    liq  = min(100.0, float(wd.get("liquidity_score",  5.0)) * 10.0)

    odds = float(bd.bet.american_odds or -110)
    if odds < 0:
        juice = abs(odds)
        if juice <= 110:
            odds_score = 100.0
        elif juice <= 130:
            odds_score = 80.0
        elif juice <= 150:
            odds_score = 65.0
        elif juice <= 200:
            odds_score = 45.0
        else:
            odds_score = 25.0
    else:
        # Plus-money — attractive but thinner/more volatile market
        odds_score = 70.0 if odds < 150 else 55.0

    return 0.45 * mis + 0.35 * liq + 0.20 * odds_score


# ---------------------------------------------------------------------------
# Factor 5: Market Efficiency
# ---------------------------------------------------------------------------

def _f5_efficiency(market: str) -> float:
    """
    How historically predictable is this market type?

    Uses signal-layer-depth priors calibrated to the engine's architecture.
    Falls back to the default for unknown markets.
    """
    mkt = market.strip().lower().replace(" ", "_")
    return _MARKET_EFFICIENCY_PRIOR.get(mkt, _MARKET_EFFICIENCY_DEFAULT)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_ccs(
    bd: "BetDisplay",
    ld: dict[str, Any],
) -> tuple[float, str]:
    """
    Compute the Composite Confidence Score for a single pick.

    Args:
        bd: BetDisplay wrapping the core Bet with candidate raw_result.
        ld: Log dict for this bet — must contain a "wager_details" key
            populated by build_pick_pipeline_for_sport() (has stability_score,
            liquidity_score, effective_edge, sharp/RLM/steam signals, etc.).

    Returns:
        (ccs_score, robustness_label)
          ccs_score        — float in [0, 100]; higher = stronger candidate.
          robustness_label — "robust" | "moderate" | "somewhat_fragile" | "fragile".

    Mutations
        None.  All scoring is read-only.  Callers store results in wager_details.
    """
    wd = ld.get("wager_details") or {}
    c  = bd.bet.raw_result or {}

    # ── Five factors ──────────────────────────────────────────────────────────
    f1 = _f1_reliability(bd, wd)
    f2 = _f2_agreement(bd, wd)
    f3 = _f3_edge(bd, wd)
    f4 = _f4_volatility(bd, wd)
    f5 = _f5_efficiency(bd.bet.market)

    raw_ccs = (
        _W_RELIABILITY * f1
        + _W_AGREEMENT * f2
        + _W_EDGE      * f3
        + _W_VOLATILITY * f4
        + _W_EFFICIENCY * f5
    )

    # ── Sensitivity test ──────────────────────────────────────────────────────
    proj  = float(
        c.get("weighted_projection")
        or c.get("posterior_mean")
        or bd.bet.sportsbook_line
    )
    line  = float(bd.bet.sportsbook_line)
    direc = bd.bet.direction.lower()

    sens_mult, robustness = _sensitivity(proj, line, direc)

    ccs = round(raw_ccs * sens_mult, 2)

    logger.debug(
        "[CCS] %-32s  R=%5.1f A=%5.1f E=%5.1f V=%5.1f M=%5.1f | "
        "raw=%5.1f × %.2f (%s) → CCS=%5.1f",
        bd.bet.bet_id[:32],
        f1, f2, f3, f4, f5,
        raw_ccs, sens_mult, robustness, ccs,
    )

    return ccs, robustness
