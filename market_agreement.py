"""
market_agreement.py — V3.0 Market Agreement Score

Computes a 0–100 agreement score that reflects how strongly sharp books,
line movement, and market signals confirm the model's direction.

Score interpretation:
  80–100 : Strong Agreement  → Nuke eligible
  60–79  : Moderate Agreement→ Diamond max unless other signals override
  40–59  : Mixed Signals     → Edge tier max
  < 40   : Disagreement      → confidence + edge penalties applied

Required for Nuke classification: score ≥ 65
Required for Diamond classification: score ≥ 45
"""

from __future__ import annotations

# Minimum agreement score per tier
AGREEMENT_FLOOR: dict[str, int] = {
    "Nuke":    65,
    "Diamond": 45,
    "Edge":    20,
}

# Penalties applied when market disagrees with the model
SHARP_CONTRARY_CONF_PENALTY = 5.0
LOW_AGREEMENT_CONF_PENALTY  = 3.0   # agreement < 40

# Edge reduction when market strongly disagrees
LOW_AGREEMENT_EDGE_PENALTY = 0.5    # agreement < 30


def compute_market_agreement(
    sharp_signal:   str  = "no_sharp",
    rlm_detected:   bool = False,
    steam_detected: bool = False,
    mis_score:      int  = 0,
    line_move_dir:  str  = "",
) -> int:
    """
    Compute a 0–100 market agreement score from available signals.

    A score near 100 means sharp action, line movement, and market
    intelligence all confirm the model.  Near 0 means the market is
    actively moving against the model direction.

    Args:
        sharp_signal:   "sharp_confirm" | "sharp_contrary" | "no_sharp"
        rlm_detected:   Reverse line movement (smart money vs. public)
        steam_detected: Steam move (sharp syndicate coordinated action)
        mis_score:      Market Intelligence Score 0–100 from market_intelligence.py
        line_move_dir:  "confirming" | "opposing" | "" (from line_movement module)

    Returns:
        Integer score 0–100.
    """
    score = 50  # neutral baseline

    # Sharp signal: ±25 pts (highest weight per spec)
    if sharp_signal == "sharp_confirm":
        score += 25
    elif sharp_signal == "sharp_contrary":
        score -= 25

    # RLM: +15 pts (smart money moving with model)
    if rlm_detected:
        score += 15

    # Steam: +10 pts (coordinated sharp action)
    if steam_detected:
        score += 10

    # MIS contribution: ±10 pts
    if mis_score >= 70:
        score += 10
    elif mis_score >= 50:
        score += 5
    elif mis_score < 30:
        score -= 10
    elif mis_score < 40:
        score -= 5

    # Line movement direction: ±10 pts
    move = (line_move_dir or "").lower()
    if move == "confirming":
        score += 10
    elif move == "opposing":
        score -= 10

    return max(0, min(100, score))


def agreement_confidence_penalty(agreement_score: int, sharp_signal: str) -> float:
    """
    Confidence reduction (positive = penalty) due to market disagreement.
    Returns 0.0 if markets are neutral or confirming.
    """
    penalty = 0.0
    if sharp_signal == "sharp_contrary":
        penalty += SHARP_CONTRARY_CONF_PENALTY
    if agreement_score < 40:
        penalty += LOW_AGREEMENT_CONF_PENALTY
    return penalty


def agreement_edge_penalty(agreement_score: int) -> float:
    """
    Edge reduction (positive = penalty) when market strongly disagrees.
    Returns 0.0 when agreement is adequate.
    """
    return LOW_AGREEMENT_EDGE_PENALTY if agreement_score < 30 else 0.0


def tier_passes_agreement(tier_name: str, agreement_score: int) -> bool:
    """
    True when the agreement score meets the minimum floor for the tier.
    Picks that fail are downgraded one tier.
    """
    return agreement_score >= AGREEMENT_FLOOR.get(tier_name, 0)
