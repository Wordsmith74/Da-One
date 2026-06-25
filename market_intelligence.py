"""
core/market_intelligence.py — Market Intelligence Score (MIS) + Sharp Action Framework

Phase 2 upgrade: sport-calibrated liquidity, sharp-book detection, reverse line
movement (RLM), steam move detection, line velocity, and effective-edge calculation.

Priority hierarchy (immutable — governs ALL market evaluation):
  1. Data Reliability   — DRS gate: no real stats = NO PLAY
  2. Model Projection   — Bayesian posterior drives the recommendation
  3. Effective Edge     — Raw edge adjusted (±) by market signals
  4. Simulation Results — Monte Carlo win probability
  5. Sharp Signals      — Sharp-book confirmation / contradiction
  6. Consensus Strength — Multi-book agreement
  7. Liquidity Score    — Market depth (book count, sport-calibrated)
  8. Market Stability   — Line velocity and direction

Market signals ENHANCE confidence — they never independently CREATE,
REMOVE, or REVERSE a model recommendation.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Book classification
# ---------------------------------------------------------------------------

# Known sharp / market-making books (set lines early; sharp money tracks them)
# Includes no-vig betting exchanges (Novig, Matchbook, Smarkets) which provide
# the truest market-implied probabilities — no vig = most accurate true-odds signal.
SHARP_BOOKS: frozenset[str] = frozenset({
    "pinnacle",
    "betcris",
    "circa",
    "bookmaker",
    "draftkings",
    "williamhill_us",
    "betway",
    # No-vig / betting exchanges (PropLine API additions)
    "novig",
    "matchbook",
    "smarkets",
    "onexbet",
})

# Recreational books (mostly public action; less predictive of true line)
RECREATIONAL_BOOKS: frozenset[str] = frozenset({
    "fanduel",
    "betmgm",
    "caesars",
    "bovada",
    "mybookieag",
    "barstool",
    "unibet_us",
    "pointsbetus",
    "sugarhouse",
    "betrivers",
    "betus",
    "wynnbet",
    "betonlineag",
    "lowvig",
    "superbook",
})


# ---------------------------------------------------------------------------
# Sport-calibrated liquidity thresholds
# ---------------------------------------------------------------------------

# WNBA is a thin market — fewer books is normal; calibrate accordingly
SPORT_LIQUIDITY_THRESHOLDS: dict[str, dict[str, int]] = {
    "WNBA":    {"full": 4,  "moderate": 2},
    "NBA":     {"full": 7,  "moderate": 3},
    "MLB":     {"full": 7,  "moderate": 3},
    "default": {"full": 6,  "moderate": 3},
}


# ---------------------------------------------------------------------------
# Priority order (immutable)
# ---------------------------------------------------------------------------

PRIORITY_ORDER: tuple[str, ...] = (
    "data_reliability",
    "model_projection",
    "effective_edge",
    "simulation_results",
    "sharp_signals",
    "consensus_strength",
    "liquidity_score",
    "market_stability",
)


# ---------------------------------------------------------------------------
# Effective edge bounds
# ---------------------------------------------------------------------------

_MAX_EDGE_BOOST   = 5.0   # max positive shift from market signals
_MAX_EDGE_PENALTY = 7.0   # max negative shift from market signals


# ---------------------------------------------------------------------------
# Market Influence Score
# ---------------------------------------------------------------------------

def compute_mis(
    all_lines:      list[float],
    book_count:     int,
    best_line:      float,
    consensus_line: float,
    sport:          str = "",
) -> tuple[int, str]:
    """
    Compute a Market Influence Score (0-100) using sport-calibrated thresholds.

    Components
    ----------
    1. Book liquidity (0-30)  — sport-calibrated book-count score
    2. Line tightness (0-40)  — spread across all books
    3. Consensus quality(0-30)— best-line proximity to median

    Returns (score, label)  where label ∈ {"Strong signal", "Moderate signal", "Weak signal"}
    """
    thresholds = SPORT_LIQUIDITY_THRESHOLDS.get(
        sport.upper(), SPORT_LIQUIDITY_THRESHOLDS["default"]
    )
    full_books = thresholds["full"]

    # ── 1. Book liquidity (0-30) ─────────────────────────────────────────────
    liquidity = min(1.0, book_count / full_books) * 30.0

    # ── 2. Line tightness (0-40) ─────────────────────────────────────────────
    if len(all_lines) >= 2:
        spread    = max(all_lines) - min(all_lines)
        tightness = max(0.0, 1.0 - spread / 3.0) * 40.0
    else:
        tightness = 10.0

    # ── 3. Consensus quality (0-30) ──────────────────────────────────────────
    drift   = abs(best_line - consensus_line)
    quality = max(0.0, 1.0 - drift / 1.5) * 30.0

    score = int(round(liquidity + tightness + quality))
    score = max(0, min(100, score))

    if score >= 80:
        label = "Strong signal"
    elif score >= 60:
        label = "Moderate signal"
    else:
        label = "Weak signal"

    return score, label


# ---------------------------------------------------------------------------
# Sharp Action Detection
# ---------------------------------------------------------------------------

def detect_sharp_action(
    book_lines: list[dict[str, Any]],
    direction:  str = "over",
) -> dict[str, Any]:
    """
    Detect sharp-book vs recreational-book consensus disagreement.

    Parameters
    ----------
    book_lines : list of {"book": str, "line": float}
    direction  : "over" or "under"

    Returns
    -------
    dict with:
        sharp_present        : bool
        sharp_book_count     : int
        rec_book_count       : int
        sharp_consensus_line : float | None
        rec_consensus_line   : float | None
        signal_type          : "sharp_confirm" | "sharp_contrary" | "no_sharp" | "neutral"
        signal_label         : str
    """
    sharp_lines: list[float] = []
    rec_lines:   list[float] = []

    for entry in book_lines:
        raw_key  = (entry.get("book") or "").lower().replace(" ", "").replace("-", "")
        line_val = entry.get("line")
        if line_val is None:
            continue
        # Match against known sharp books (flexible string match)
        is_sharp = any(
            sb.replace("_", "").replace(" ", "") in raw_key
            for sb in SHARP_BOOKS
        )
        if is_sharp:
            sharp_lines.append(float(line_val))
        else:
            rec_lines.append(float(line_val))

    sharp_present        = len(sharp_lines) > 0
    sharp_consensus_line = _median(sharp_lines) if sharp_lines else None
    rec_consensus_line   = _median(rec_lines)   if rec_lines   else None

    if not sharp_present:
        signal_type  = "no_sharp"
        signal_label = "No sharp book coverage"
    elif sharp_consensus_line is None or rec_consensus_line is None:
        signal_type  = "neutral"
        signal_label = "Insufficient data for comparison"
    else:
        diff = sharp_consensus_line - rec_consensus_line
        # For OVER: sharp lower than rec → sharp has lower line → confirms OVER
        # For UNDER: sharp higher than rec → sharp has higher line → confirms UNDER
        threshold = 0.25
        if direction == "over":
            if diff < -threshold:
                signal_type  = "sharp_confirm"
                signal_label = (
                    f"Sharp confirm: books at {sharp_consensus_line:.1f} "
                    f"vs rec {rec_consensus_line:.1f}"
                )
            elif diff > threshold:
                signal_type  = "sharp_contrary"
                signal_label = (
                    f"Sharp contrary: books at {sharp_consensus_line:.1f} "
                    f"vs rec {rec_consensus_line:.1f}"
                )
            else:
                signal_type  = "neutral"
                signal_label = "Sharp/rec aligned"
        else:
            if diff > threshold:
                signal_type  = "sharp_confirm"
                signal_label = (
                    f"Sharp confirm: books at {sharp_consensus_line:.1f} "
                    f"vs rec {rec_consensus_line:.1f}"
                )
            elif diff < -threshold:
                signal_type  = "sharp_contrary"
                signal_label = (
                    f"Sharp contrary: books at {sharp_consensus_line:.1f} "
                    f"vs rec {rec_consensus_line:.1f}"
                )
            else:
                signal_type  = "neutral"
                signal_label = "Sharp/rec aligned"

    return {
        "sharp_present":        sharp_present,
        "sharp_book_count":     len(sharp_lines),
        "rec_book_count":       len(rec_lines),
        "sharp_consensus_line": sharp_consensus_line,
        "rec_consensus_line":   rec_consensus_line,
        "signal_type":          signal_type,
        "signal_label":         signal_label,
    }


# ---------------------------------------------------------------------------
# Reverse Line Movement (RLM) Detection
# ---------------------------------------------------------------------------

def detect_reverse_line_movement(
    sharp_line: float | None,
    rec_line:   float | None,
    direction:  str,
) -> bool:
    """
    Detect a Reverse Line Movement proxy.

    Public money usually bets the popular side (OVER on props, favorite on spreads).
    RLM = line moves against the popular side, indicating sharp money on the other side.

    Proxy (no bet-% data available): sharp books posting a meaningfully
    different line than recreational books signals RLM.

    For OVER picks  : sharp line >= rec line + 0.5 → sharp pushed it up (harder OVER)
    For UNDER picks : sharp line <= rec line - 0.5 → sharp pushed it down (harder UNDER)

    Returns True if RLM pattern is detected.
    """
    if sharp_line is None or rec_line is None:
        return False
    diff = sharp_line - rec_line
    if direction == "over":
        return diff >= 0.5
    else:
        return diff <= -0.5


# ---------------------------------------------------------------------------
# Steam Move Detection
# ---------------------------------------------------------------------------

def detect_steam_move(
    all_lines:  list[float],
    book_count: int,
    sport:      str = "",
) -> bool:
    """
    Detect a steam move proxy signal.

    A steam move = multiple sharp books move simultaneously → market reaches
    consensus very quickly. Proxy: tight line spread across many books indicates
    rapid consensus (steam settled and market followed).

    Returns True if steam pattern detected.
    """
    thresholds = SPORT_LIQUIDITY_THRESHOLDS.get(
        sport.upper(), SPORT_LIQUIDITY_THRESHOLDS["default"]
    )

    if book_count < thresholds["moderate"] or len(all_lines) < 2:
        return False

    spread = max(all_lines) - min(all_lines)
    return spread <= 0.5 and book_count >= thresholds["full"]


# ---------------------------------------------------------------------------
# Line Velocity Tracking
# ---------------------------------------------------------------------------

def compute_line_velocity(
    opening_line:  float,
    current_line:  float,
    hours_elapsed: float = 1.0,
) -> dict[str, Any]:
    """
    Compute line movement velocity since opening.

    Returns dict with:
        magnitude  : float — absolute change (points)
        direction  : "up" | "down" | "flat"
        velocity   : float — pts/hr
        rapid_move : bool  — velocity > 0.5 pts/hr signals sharp action
    """
    magnitude = round(abs(current_line - opening_line), 2)
    if magnitude < 0.05:
        mv_dir = "flat"
    elif current_line > opening_line:
        mv_dir = "up"
    else:
        mv_dir = "down"

    velocity   = round(magnitude / max(hours_elapsed, 0.1), 3)
    rapid_move = velocity > 0.5

    return {
        "magnitude":  magnitude,
        "direction":  mv_dir,
        "velocity":   velocity,
        "rapid_move": rapid_move,
    }


# ---------------------------------------------------------------------------
# Effective Edge
# ---------------------------------------------------------------------------

def _american_to_implied_pct(odds: int | float) -> float:
    """Convert American odds to implied probability (0–100 %)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100) * 100.0
    return 100.0 / (odds + 100) * 100.0


def detect_soft_line(
    model_prob:    float,
    american_odds: int | float,
    sport:         str = "default",
) -> bool:
    """
    Fix 5: Return True when the bookmaker's implied probability is
    meaningfully lower than the model's probability.

    A 'soft' line means the market hasn't sharpened to our model's level —
    could indicate early posting, liability shading, or model overconfidence.
    Either way, the excess gap is a bearish signal for pure edge plays.

    Thresholds (model_prob − implied_prob):
      WNBA  : > 10 pp  (thinner market, softer lines are more common)
      NBA   : > 12 pp
      MLB   : > 12 pp
    """
    implied    = _american_to_implied_pct(american_odds)
    thresholds = {"WNBA": 10.0, "NBA": 12.0, "MLB": 12.0}
    threshold  = thresholds.get(sport.upper(), 12.0)
    return (model_prob - implied) > threshold


def compute_effective_edge(
    raw_edge:           float,
    sharp_signal:       str  = "no_sharp",
    rlm_detected:       bool = False,
    steam_detected:     bool = False,
    mis_score:          int  = 0,
    soft_line_detected: bool = False,
) -> float:
    """
    Adjust raw model edge using market signals to produce effective edge.

    Rules (Priority order items 5-8):
    - sharp_confirm     : +2.0%
    - sharp_contrary    : -3.5%
    - steam_detected    : +1.0%
    - rlm_detected      : +1.5%
    - mis >= 80         : +0.5%
    - mis < 40          : -1.0%
    - soft_line_detected: -1.5%  (Fix 5)

    Clamp: total adjustment within [-_MAX_EDGE_PENALTY, +_MAX_EDGE_BOOST].
    The model recommendation (raw_edge direction) is NEVER reversed by this.
    """
    adj = 0.0

    if sharp_signal == "sharp_confirm":
        adj += 2.0
    elif sharp_signal == "sharp_contrary":
        adj -= 3.5

    if steam_detected:
        adj += 1.0

    if rlm_detected:
        adj += 1.5

    if soft_line_detected:
        adj -= 1.5   # Fix 5: soft book line is a bearish signal

    adj = max(-_MAX_EDGE_PENALTY, min(_MAX_EDGE_BOOST, adj))
    return round(raw_edge + adj, 2)


# ---------------------------------------------------------------------------
# Data Reliability Score
# ---------------------------------------------------------------------------

def compute_data_reliability(
    has_real_stats:  bool,
    book_count:      int,
    has_l5:          bool = False,
    has_l10:         bool = False,
    has_injury_data: bool = False,
) -> int:
    """
    Compute a Data Reliability Score (0-100).

    Thresholds:
      < 40  NO PLAY — insufficient data
      40-59 Value tier only
      60-74 S tier maximum
      75+   Full tier access (S+/Nuke)
    """
    score = 0
    if has_real_stats:
        score += 40
    if has_l5:
        score += 20
    if has_l10:
        score += 15
    if book_count >= 4:
        score += 15
    elif book_count >= 2:
        score += 8
    if has_injury_data:
        score += 10
    return min(100, score)


# ---------------------------------------------------------------------------
# Tier eligibility
# ---------------------------------------------------------------------------

def tier_eligibility(data_reliability_score: int) -> dict[str, bool]:
    """Return tier access flags based on DRS."""
    return {
        "allow_play":   data_reliability_score >= 40,
        "allow_s":      data_reliability_score >= 60,
        "allow_s_plus": data_reliability_score >= 75,
    }


def mis_label(score: int) -> str:
    """Human-readable label for a MIS score."""
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Moderate"
    return "Weak"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0
