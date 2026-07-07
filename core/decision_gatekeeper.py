"""
decision_gatekeeper.py

Final filter before any data is sent to the notification system.
Evaluates edge/confidence thresholds, detects conflicting bet signals,
and applies same-game correlation (SGC) rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("betting_bot")

try:
    from core.reject_logger import log_rejected_bet_obj as _log_reject
    _REJECT_LOGGER_AVAILABLE = True
except ImportError:
    _REJECT_LOGGER_AVAILABLE = False

from core.market_weights import (
    get_market_confidence_modifier,
    is_restricted_market,
    restricted_diamond_floor,
    market_weight_label,
)

# ── V3.0 calibration modules ─────────────────────────────────────────────────
try:
    from core.confidence_caps import apply_confidence_cap, market_category as _mkt_cat
    from core.market_agreement import (
        tier_passes_agreement as _tier_passes_agreement,
        agreement_confidence_penalty as _agree_conf_penalty,
        agreement_edge_penalty as _agree_edge_penalty,
    )
    from core.integrity_filters import run_integrity_filter as _run_integrity_filter
    from core.edge_calibrator import is_game_market as _is_game_mkt_for_gk
    _V3_MODULES_AVAILABLE = True
except ImportError:
    _V3_MODULES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

class Tier(str, Enum):
    NUKE    = "Nuke"
    DIAMOND = "Diamond"
    GOLD    = "Gold Standard"   # replaces Edge — assigned by ranked selection
    DISCARD = "DISCARD"


# Sport-specific quality thresholds: (tier, min_edge_%, min_confidence)
# Both conditions must be satisfied for a bet to enter the eligible pool.
#
# Tier assignment above the minimum bar is handled by assign_ranked_tiers():
#   Rank 1 (highest composite score) → Nuke
#   Rank 2                           → Diamond
#   All remaining eligible picks     → Gold Standard
#
# V3.0 Calibrated Thresholds (edge on 0–12% calibrated scale).
# Confidence CAPS (from confidence_caps.py) applied in Step 1c:
#   MLB Totals  — Nuke≤85  Diamond≤80  Gold Standard≤75
#   NBA Totals  — Nuke≤88  Diamond≤83  Gold Standard≤78
#   WNBA Totals — Nuke≤86  Diamond≤81  Gold Standard≤76
#   Props (all) — Nuke≤92  Diamond≤87  Gold Standard≤82
SPORT_TIER_THRESHOLDS: dict[str, list[tuple[Tier, float, float]]] = {
    "MLB": [
        (Tier.NUKE,    3.5, 85.0),
        (Tier.DIAMOND, 2.0, 78.0),
        (Tier.GOLD,    1.0, 68.0),
    ],
    "NBA": [
        (Tier.NUKE,    4.0, 85.0),
        (Tier.DIAMOND, 2.5, 78.0),
        (Tier.GOLD,    1.5, 68.0),
    ],
    "WNBA": [
        (Tier.NUKE,    3.5, 85.0),
        (Tier.DIAMOND, 2.0, 78.0),
        (Tier.GOLD,    1.0, 68.0),
    ],
}
_DEFAULT_SPORT_KEY = "MLB"
# Backward-compat alias — external callers that imported TIER_THRESHOLDS get MLB defaults
TIER_THRESHOLDS = SPORT_TIER_THRESHOLDS[_DEFAULT_SPORT_KEY]

# ---------------------------------------------------------------------------
# Market-specific tier overrides (replaces SPORT_TIER_THRESHOLDS for one market)
# ---------------------------------------------------------------------------
# Data-driven replacement for the generic sport-wide Nuke/Diamond/Gold cuts, for
# markets that have graded enough of their own history to support a dedicated
# search (core/threshold_optimizer.py) rather than reusing the sport default.
#
# pitcher_strikeouts (MLB Ks) — output/threshold_recommendations.json,
# generated 2026-07-07, sport="MLB" market_class="prop", n_graded=99:
#   Nuke:    edge>=3.72%, conf>=82.9  -> n=20, 17W-3L,  win%=85.0%, roi=+51.5%
#   Diamond: edge>=3.72%, conf>=81.7  -> n=31, 25W-6L,  win%=80.7%, roi=+42.8%
#   Gold:    edge>=3.72%, conf>=77.9  -> n=41, 32W-9L,  win%=78.1%, roi=+14.7%
# (consensus_evaluated=False for this group -- only 8/99 picks had
# side_agreement_frac populated, so no consensus axis is folded in here.)
# Replaces MLB's generic SPORT_TIER_THRESHOLDS (3.5/85, 2.0/78, 1.0/68) for THIS
# market only; every other MLB market is untouched. Re-run
# core/threshold_optimizer.py periodically and update this table as more MLB Ks
# picks get graded -- do not hand-tune it without re-running the search.
_MARKET_TIER_THRESHOLDS: dict[str, list[tuple[Tier, float, float]]] = {
    "pitcher_strikeouts": [
        (Tier.NUKE,    3.72, 82.9),
        (Tier.DIAMOND, 3.72, 81.7),
        (Tier.GOLD,    3.72, 77.9),
    ],
}

# ---------------------------------------------------------------------------
# Market-specific entry floors (Upgrade 2 — v2.1 spec)
# ---------------------------------------------------------------------------
# Per-market minimum (edge_%, confidence) that a bet must clear BEFORE tier
# evaluation begins.  These replace the universal Gold Standard threshold as
# the entry gate for every in-scope market, producing a tighter signal set.
#
# Key:   normalized market name (market_normalized() output)
# Value: (min_edge_pct, min_confidence)   — 0.0 means "no floor on that axis"
_MARKET_ENTRY_FLOORS: dict[str, tuple[float, float]] = {
    "pitcher_strikeouts": (2.0, 68.0),   # MLB Ks — Gold bar (corrected from 78.0)
                                          # historical performance; Gold-tier Ks ran 5W-12L
                                          # (29.4%) so Gold is excluded from this market entirely.
                                          # (Formerly (4.5, 65.0), a floor below Gold's own
                                          # 68-confidence bar, which let the losing tier through.)
    "first_5_total":      (4.0, 60.0),   # MLB F5 total (Bayesian)
    "first_5_ml":         (4.0, 60.0),   # MLB F5 moneyline
    "first_5_rl":         (4.0, 60.0),   # MLB F5 run line
    "nrfi":               (4.0, 60.0),   # No Run First Inning
    "yrfi":               (4.0, 60.0),   # Yes Run First Inning
    "player_assists":     (1.0, 60.0),   # WNBA assists
    "player_rebounds":    (1.0, 60.0),   # WNBA rebounds
    "h2h":                (0.0, 68.0),   # WNBA ML — market_normalized("h2h") -> "h2h"
                                          # (game_markets._MARKET_BUNDLE["WNBA"] == "h2h" only)
    "team_total":         (0.0, 68.0),   # WNBA team total. NOTE: game_markets._MARKET_BUNDLE
                                          # no longer requests team_totals for WNBA (spreads and
                                          # team-totals were removed from scope) -- this floor is
                                          # currently unreachable and kept only for when/if that
                                          # market is re-enabled with its own graded history.
    # "moneyline" removed as a duplicate of "h2h": market_normalized() never emits
    # "moneyline" for a raw odds-API market key, and MLB moneyline is not in scope
    # (see game_markets._MARKET_BUNDLE["MLB"]), so no MLB entry existed for it either.
    # "run_line" moved OUT of this shared dict -- see _MARKET_ENTRY_FLOORS_BY_SPORT below.
}

# Full-game run line / spread / total floors, split per sport instead of one shared
# "run_line" entry. The old single entry -- (0.0, 68.0), commented "MLB + WNBA" -- meant
# MLB's full-game run line was gated by a floor that was really WNBA's default (0 edge,
# 68 conf), with no MLB-specific graded evidence behind it at all. That is no longer
# acceptable: per threshold_optimizer.py's own stated policy ("this tool will not emit a
# threshold it cannot support with evidence"), a market with no graded history of its own
# should not silently inherit another sport's number just because the value happens to be
# a shared default.
#
# Evidence check (output/threshold_recommendations.json, generated 2026-07-07):
#   MLB / market_class=game -> n_graded=1 -> status="insufficient_data" (need >= 15)
# MLB full-game run line therefore has NO calibrated floor of its own yet. Rather than
# reuse WNBA's number, it is aligned with MLB's OTHER precomputed game-market entries
# (first_5_ml / first_5_rl / first_5_total, all (4.0, 60.0)) as the closest same-sport,
# same-pipeline analogue, and is clearly labeled provisional pending its own graded sample.
# WNBA's run_line floor is kept separately (and is currently unreachable -- see
# _MARKET_BUNDLE["WNBA"] == "h2h" only) so the two can never silently re-merge into one
# shared constant again.
_MARKET_ENTRY_FLOORS_BY_SPORT: dict[str, dict[str, tuple[float, float]]] = {
    "MLB": {
        "run_line": (4.0, 60.0),   # full-game run line / spread -- provisional, MLB-only,
                                    # aligned to MLB's own first_5_rl/ml/total floors.
                                    # Re-derive from output/threshold_recommendations.json
                                    # once MLB game_class n_graded >= 15.
        # "total"/"totals": no entry -- full-game MLB total is not currently in scope
        # (game_markets._MARKET_BUNDLE["MLB"] only requests totals_first_5_innings, not a
        # full-game total), so there is nothing here to gate yet. Add once it's wired.
    },
    "WNBA": {
        "run_line": (0.0, 68.0),   # WNBA-only historical default. Currently unreachable --
                                    # game_markets._MARKET_BUNDLE["WNBA"] == "h2h" only, so
                                    # WNBA no longer requests spreads. Kept so that if spreads
                                    # are re-enabled for WNBA, they use this WNBA-labeled
                                    # floor rather than falling back onto MLB's.
    },
}

# WNBA blowout confidence penalties (Step 1b3)
_BLOWOUT_CONF_PENALTY: dict[str, float] = {
    "moderate": 6.0,    # spread 10–17 pts → −6 confidence
    "heavy":    10.0,   # spread ≥ 17 pts  → −10 confidence
}

# ---------------------------------------------------------------------------
# Nuke projection cushion multipliers (per market type)
# ---------------------------------------------------------------------------
# For OVER picks: model projection must exceed line × cushion to qualify Nuke.
# For UNDER picks: projection must be ≤ line × cushion_max_under.
_S_PLUS_CUSHION_OVER: dict[str, float] = {
    "hits":               2.2,
    "batter_hits":        2.2,
    "strikeouts":         1.6,
    "pitcher_strikeouts": 1.6,
    "points":             1.4,
    "player_points":      1.4,
    "rebounds":           1.5,
    "player_rebounds":    1.5,
    "assists":            1.5,
    "player_assists":     1.5,
    "totals":             1.08,
    "team_total":         1.08,
}
_S_PLUS_DEFAULT_CUSHION_OVER  = 1.4
_S_PLUS_CUSHION_UNDER_MAX: dict[str, float] = {
    "totals":     0.93,
    "team_total": 0.93,
}
_S_PLUS_DEFAULT_CUSHION_UNDER = 0.92

# ---------------------------------------------------------------------------
# Fix 4: Market concentration cap — max S+ picks per market type per run
# ---------------------------------------------------------------------------
_SPLUS_MARKET_CAP = 4


# ---------------------------------------------------------------------------
# Bet data structure
# ---------------------------------------------------------------------------

@dataclass
class Bet:
    """
    Represents a single evaluated betting signal.

    Attributes
    ----------
    bet_id          : Unique string identifier (e.g. "WNBA_SEA_CHI_player_pts").
    team            : Team abbreviation or name the bet is tied to.
    market          : What is being bet on (lowercase_underscore): 'team_total',
                      'player_points', 'player_assists', 'team_spread', etc.
    direction       : 'over' or 'under'.
    sportsbook_line : The published line.
    edge_percentage : Model edge vs. the 50% breakeven point (positive = favours direction).
    confidence_score: A 0–100 confidence score derived from posterior tightness,
                      sample size, or a composite model signal.
    player          : (optional) Player name/ID for player-prop bets.
    game_id         : Unique identifier for the game this bet belongs to
                      (e.g. "PHX@MIN_WNBA_2026-06-01"). Used for same-game
                      correlation detection. Empty string = unknown game.
    tier            : Populated by evaluate_tier(); None until evaluated.
    flagged         : True if any gatekeeper rule has flagged this bet.
    flag_reason     : Human-readable reason for any flag.
    raw_result      : Arbitrary dict carrying upstream engine output for reference.
    """

    bet_id:           str
    team:             str
    market:           str
    direction:        str
    sportsbook_line:  float
    edge_percentage:  float
    confidence_score: float
    player:           str | None       = None
    game_id:          str              = ""
    american_odds:          float            = 0.0
    data_reliability_score: int              = 100
    mis_score:              int              = 0
    tier:                   Tier | None      = None
    flagged:                bool             = False
    flag_reason:            str              = ""
    raw_result:             dict[str, Any]   = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Market normalization
# ---------------------------------------------------------------------------

def market_normalized(market: str) -> str:
    """
    Normalize a market string to lowercase_underscore format.
    Accepts both 'Player Points' and 'player_points' forms.

    Prop display-name aliases (e.g. 'Assists' stored on Bet.market when
    the pick originates from player_props.py) are mapped to their canonical
    internal keys so publication-whitelist lookups work correctly.
    """
    _DISPLAY_ALIASES: dict[str, str] = {
        "assists":          "player_assists",
        "rebounds":         "player_rebounds",
        "points":           "player_points",
        "strikeouts":       "pitcher_strikeouts",
        "player_strikeouts":"pitcher_strikeouts",
        "blocks":           "player_blocks",
        "steals":           "player_steals",
        "reb+ast":          "player_rebounds_assists",
        "pts+reb+ast":      "player_points_rebounds_assists",
        "pts+reb":          "player_points_rebounds",
        "pts+ast":          "player_points_assists",
        # Full-game total, as produced by core/odds_client.py's
        # fetch_todays_candidates() (candidate["market"] == "Totals").
        # Kept distinct from "first_5_total" (a different bet type) --
        # see core/market_gate.py.
        "totals":           "game_total",
        "total":             "game_total",
    }
    norm = market.strip().lower().replace(" ", "_")
    return _DISPLAY_ALIASES.get(norm, norm)


# ---------------------------------------------------------------------------
# Core gatekeeper functions
# ---------------------------------------------------------------------------

def evaluate_tier(
    edge_percentage: float,
    confidence_score: float,
    sport: str = _DEFAULT_SPORT_KEY,
    market: str | None = None,
) -> Tier | None:
    """
    Categorise a bet signal by edge and confidence using sport-specific thresholds,
    unless `market` has its own data-driven override in _MARKET_TIER_THRESHOLDS
    (e.g. MLB pitcher_strikeouts), in which case that replaces the sport default
    entirely for this call.

    Tier rules (both conditions must be met):
        Nuke    — edge ≥ 16/15/14 %  AND  confidence ≥ 85  (MLB/NBA/WNBA)
        Diamond — edge ≥ 13/12/11 %  AND  confidence ≥ 78
        Edge    — edge ≥ 10/ 9/ 8 %  AND  confidence ≥ 68
        None    — below Edge thresholds → suppressed
    """
    thresholds = None
    if market is not None:
        thresholds = _MARKET_TIER_THRESHOLDS.get(market_normalized(market))
    if thresholds is None:
        thresholds = SPORT_TIER_THRESHOLDS.get(sport.upper(), SPORT_TIER_THRESHOLDS[_DEFAULT_SPORT_KEY])
    for tier, min_edge, min_conf in thresholds:
        if edge_percentage >= min_edge and confidence_score >= min_conf:
            return tier
    return None


def stamp_tier(bet: Bet, sport: str = _DEFAULT_SPORT_KEY) -> Bet:
    """
    Convenience helper: call evaluate_tier() and write the result onto a Bet.
    """
    bet.tier = evaluate_tier(bet.edge_percentage, bet.confidence_score, sport, bet.market)
    return bet


# ---------------------------------------------------------------------------
# Same-Game Correlation (SGC) matrix
# ---------------------------------------------------------------------------
# Keys are (market_a, market_b, same_team) with markets sorted alphabetically.
# Values are Pearson correlation coefficients ρ ∈ [-1, +1].
#
# Positive ρ  → legs fire together more often than independence assumes →
#               true combined probability is HIGHER than P(A)×P(B).
# Negative ρ  → legs fight each other → true combined probability is LOWER.
# ---------------------------------------------------------------------------

SGC_CORRELATION_MATRIX: dict[tuple[str, str, bool], float] = {
    # ── Same-team positive correlations ──────────────────────────────────────
    # Team total ↔ player scoring volume
    ("player_assists",     "team_total",         True): +0.50,
    ("player_hits",        "team_total",         True): +0.55,
    ("player_points",      "team_total",         True): +0.65,
    ("player_rebounds",    "team_total",         True): +0.30,
    ("player_total_bases", "team_total",         True): +0.50,
    # Spread ↔ team total (winning big = covering)
    ("team_spread",        "team_total",         True): +0.70,
    # Player ↔ player (different players, same team)
    ("player_assists",     "player_points",      True): +0.40,
    ("player_hits",        "player_total_bases", True): +0.60,
    ("player_hits",        "player_hits",        True): +0.25,
    ("player_points",      "player_points",      True): +0.30,
    ("player_points",      "player_rebounds",    True): +0.25,
    ("player_rebounds",    "player_rebounds",    True): +0.20,

    # ── Cross-team negative correlations (same game) ──────────────────────
    # Two opposing spreads are mutually exclusive
    ("team_spread",        "team_spread",        False): -1.00,
    # A's big win → B can't cover spread
    ("team_spread",        "team_total",         False): -0.60,
    # Opponent's player vs our team total (defensive drag)
    ("player_points",      "team_total",         False): -0.15,
    ("player_hits",        "team_total",         False): -0.15,
    ("player_points",      "team_spread",        False): -0.15,
}

# Threshold below which a same-game pair is considered negatively correlated
# and blocked from appearing together in a parlay (Rule 4).
SGC_NEGATIVE_BLOCK_THRESHOLD: float = -0.30


def _sgc_key(mkt_a: str, mkt_b: str, same_team: bool) -> tuple[str, str, bool]:
    """Return a canonical sorted key for the SGC matrix."""
    a, b = (mkt_a, mkt_b) if mkt_a <= mkt_b else (mkt_b, mkt_a)
    return (a, b, same_team)


def classify_sgc_pairs(bets: list[Bet]) -> list[tuple[Bet, Bet, float]]:
    """
    Identify all same-game bet pairs and return their correlation coefficient ρ.

    Only pairs that share a non-empty game_id and appear in SGC_CORRELATION_MATRIX
    are returned. Pairs not in the matrix have ρ = 0.0 (treated as independent)
    and are excluded from the result list.

    Args:
        bets: Any list of Bet objects (need not be the full batch).

    Returns:
        List of (bet_a, bet_b, rho) — one entry per correlated same-game pair.
    """
    pairs: list[tuple[Bet, Bet, float]] = []
    n = len(bets)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = bets[i], bets[j]
            if not a.game_id or a.game_id != b.game_id:
                continue
            same_team = a.team.lower() == b.team.lower()
            key = _sgc_key(
                market_normalized(a.market),
                market_normalized(b.market),
                same_team,
            )
            rho = SGC_CORRELATION_MATRIX.get(key, 0.0)
            if rho != 0.0:
                pairs.append((a, b, rho))
    return pairs


# ---------------------------------------------------------------------------
# Conflict detection (Rules 1 & 2 — negative correlations, same team)
# ---------------------------------------------------------------------------

_SCORING_VOLUME_MARKETS = {
    "team_total",
    "player_points",
    "player_assists",
    "player_rebounds",
}

_SPREAD_CORRELATED_MARKETS = {
    "team_total",
    "team_spread",
}


def check_for_conflicts(list_of_bets: list[Bet], sport: str = _DEFAULT_SPORT_KEY) -> list[Bet]:
    """
    Identify negative correlations across a batch of bets and flag them.

    Conflict rules detected
    -----------------------
    1. **Volume conflict** — Same team, both markets are scoring-volume
       markets, but directions are opposite.

    2. **Spread/Total conflict** — Same team, one bet is on team_total and
       the other is on the team_spread, but their implied directions conflict.

    Resolution
    ----------
    - A conflict is only raised when BOTH bets have a tier assigned (i.e.,
      both would independently pass the gatekeeper).  If one side has no
      tier it would be discarded regardless — flagging the good pick because
      its worthless mirror exists is incorrect and suppresses real edge.
    - Both conflicting bets are flagged with `flagged = True` and a
      human-readable `flag_reason`.
    - The bet with the **lower edge** additionally has its `confidence_score`
      reduced by 10 points (floor 0) and its tier re-evaluated.
    """
    n = len(list_of_bets)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = list_of_bets[i], list_of_bets[j]

            if a.team.lower() != b.team.lower():
                continue

            conflict_type = _detect_conflict_type(a, b)
            if conflict_type is None:
                continue

            # A "Volume conflict" is only a real correlation when it involves
            # a team-level market (team_total) or the SAME player betting
            # both directions. Two different players' individual props
            # (e.g. Player A rebounds UNDER vs. Player B rebounds OVER on
            # the same team) are not actually correlated -- one player going
            # under their own line says nothing about another player's own
            # line -- so skip flagging those as conflicts.
            if conflict_type == "Volume conflict" and a.player and b.player:
                if a.player.lower() != b.player.lower():
                    continue

            # Only treat as a real conflict when both sides would be approved.
            # Generating OVER+UNDER per game is intentional — the worthless
            # side (tier=None, negative edge) should not block the good side.
            if a.tier is None or b.tier is None:
                continue

            reason = (
                f"{conflict_type}: '{a.market}' {a.direction.upper()} "
                f"conflicts with '{b.market}' {b.direction.upper()} "
                f"on team '{a.team}'"
            )

            a.flagged = True
            b.flagged = True
            a.flag_reason = _append_flag(a.flag_reason, reason)
            b.flag_reason = _append_flag(b.flag_reason, reason)

            lower = a if a.edge_percentage <= b.edge_percentage else b
            lower.confidence_score = max(0.0, lower.confidence_score - 10.0)
            lower.tier = evaluate_tier(lower.edge_percentage, lower.confidence_score, sport, lower.market)

    # ── Rule 6: Market Uniqueness Rule ───────────────────────────────────────
    # Within a single analysis run, only ONE recommendation direction may exist
    # per (game_id, market_norm) pair.  Opposing bets on the same game+market
    # (e.g. Over 216.5 AND Under 216.0 for the same full-game total) represent
    # a signal conflict.  The lower-edge bet is flagged; the stronger is kept.
    _gm_groups: dict[tuple[str, str], list[Bet]] = {}
    for _b in list_of_bets:
        if not _b.game_id or _b.tier is None or _b.flagged:
            continue
        _gm_groups.setdefault((_b.game_id, market_normalized(_b.market)), []).append(_b)

    for (_gid, _mn), _grp in _gm_groups.items():
        _overs  = [_b for _b in _grp if _b.direction.lower() == "over"]
        _unders = [_b for _b in _grp if _b.direction.lower() == "under"]
        if not (_overs and _unders):
            continue
        _all_sides = _overs + _unders
        _best_side = max(_all_sides, key=lambda _b: _b.edge_percentage)
        for _loser in _all_sides:
            if _loser is _best_side:
                continue
            _r = (
                f"Market uniqueness conflict ({_mn}): "
                f"{_loser.direction.upper()} opposes stronger "
                f"{_best_side.direction.upper()} "
                f"(edge {_best_side.edge_percentage:.2f}%) in game {_gid}"
            )
            _loser.flagged = True
            _loser.flag_reason = _append_flag(_loser.flag_reason, _r)
            _loser.confidence_score = max(0.0, _loser.confidence_score - 10.0)
            _loser.tier = evaluate_tier(_loser.edge_percentage, _loser.confidence_score, sport, _loser.market)

    return list_of_bets


# ---------------------------------------------------------------------------
# Rule 3 — Prop-consensus block
# ---------------------------------------------------------------------------

_PROP_MARKETS = {
    "player_points",
    "player_assists",
    "player_rebounds",
    "player_threes",
    "player_hits",
    "player_total_bases",
    "pitcher_strikeouts",
}

_PROP_CONSENSUS_THRESHOLD = 0.60   # ≥60 % of same-team props opposing → block
_PROP_CONSENSUS_MIN_PROPS  = 2     # need at least this many props to form a consensus


def _check_prop_consensus(bets: list[Bet], sport: str = _DEFAULT_SPORT_KEY) -> None:
    """
    Rule 3: Flag a team_total bet when ≥60 % of same-team, same-game player
    prop bets oppose its direction. Mutates bets in place.

    Requires game_id to be set on bets. Bets without a game_id are skipped.
    """
    # Group by game_id
    by_game: dict[str, list[Bet]] = {}
    for bet in bets:
        if bet.game_id:
            by_game.setdefault(bet.game_id, []).append(bet)

    for game_bets in by_game.values():
        team_totals = [
            b for b in game_bets
            if market_normalized(b.market) == "team_total" and not b.flagged
        ]

        for tt in team_totals:
            same_team_props = [
                b for b in game_bets
                if market_normalized(b.market) in _PROP_MARKETS
                and b.team.lower() == tt.team.lower()
            ]

            if len(same_team_props) < _PROP_CONSENSUS_MIN_PROPS:
                continue

            opposing = [
                p for p in same_team_props
                if p.direction.lower() != tt.direction.lower()
            ]
            ratio = len(opposing) / len(same_team_props)

            if ratio >= _PROP_CONSENSUS_THRESHOLD:
                reason = (
                    f"Prop-consensus conflict: {len(opposing)}/{len(same_team_props)} "
                    f"props for {tt.team} oppose "
                    f"{market_normalized(tt.market)} {tt.direction.upper()} "
                    f"({ratio:.0%} opposition — threshold {_PROP_CONSENSUS_THRESHOLD:.0%})"
                )
                tt.flagged = True
                tt.flag_reason = _append_flag(tt.flag_reason, reason)
                tt.confidence_score = max(0.0, tt.confidence_score - 15.0)
                tt.tier = evaluate_tier(tt.edge_percentage, tt.confidence_score, sport, tt.market)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_conflict_type(a: Bet, b: Bet) -> str | None:
    """Return a conflict-type label if a and b conflict, else None."""
    dir_a = a.direction.lower()
    dir_b = b.direction.lower()
    opposite = dir_a != dir_b

    if not opposite:
        return None

    mkt_a = market_normalized(a.market)
    mkt_b = market_normalized(b.market)

    if mkt_a in _SCORING_VOLUME_MARKETS and mkt_b in _SCORING_VOLUME_MARKETS:
        return "Volume conflict"

    if (
        mkt_a in _SPREAD_CORRELATED_MARKETS
        and mkt_b in _SPREAD_CORRELATED_MARKETS
        and mkt_a != mkt_b
    ):
        return "Spread/Total conflict"

    return None


def _append_flag(existing: str, new_reason: str) -> str:
    if existing:
        return f"{existing}; {new_reason}"
    return new_reason


# ---------------------------------------------------------------------------
# Gatekeeper pipeline — convenience entry point used by DecisionOrchestrator
# ---------------------------------------------------------------------------

def run_gatekeeper(bets: list[Bet], sport: str = _DEFAULT_SPORT_KEY) -> dict[str, list[Bet]]:
    """
    Full gatekeeper pipeline: stamp tiers, detect conflicts, partition results.

    Rules applied in order
    ----------------------
    1. Tier stamping — assign Nuke/Diamond/Edge/None using sport-specific thresholds.
    2. Volume & Spread/Total conflicts (same team, opposite direction).
    3. Prop-consensus block — team_total suppressed when ≥60 % of same-team
       props in the same game oppose its direction (requires game_id).

    Returns
    -------
    dict with keys:
        'approved'  — Tier Nuke/Diamond/Edge bets with no conflicts
        'flagged'   — Bets that triggered any rule (manual review)
        'discarded' — Bets below threshold (tier is None)
    """
    # ── Step 0: Market confidence modifier (Pipeline B — player props only) ──
    # Applied AFTER edge calculation, BEFORE tier evaluation.
    # Game markets (Pipeline A) are skipped — their grading model is unchanged.
    for bet in bets:
        modifier = get_market_confidence_modifier(bet.market, sport)
        if modifier != 0.0:
            raw_conf = bet.confidence_score
            bet.confidence_score = min(100.0, max(0.0, bet.confidence_score + modifier))
            label    = market_weight_label(bet.market, sport)
            sign     = f"+{modifier:.0f}" if modifier > 0 else f"{modifier:.0f}"
            bet.flag_reason = _append_flag(
                bet.flag_reason,
                f"Market weight [{label}]: conf {sign}pts "
                f"({raw_conf:.1f}→{bet.confidence_score:.1f})",
            )

    # ── Step 0.5: Market-specific entry floors ────────────────────────────────
    # Each in-scope market has (min_edge, min_confidence) floors that act as
    # the entry gate BEFORE tier evaluation.  A bet that doesn't clear its
    # market's floor is discarded immediately — no tier is ever assigned.
    for bet in bets:
        mkt_norm = market_normalized(bet.market)
        # Sport-specific floor takes priority (e.g. MLB run_line vs WNBA run_line are no
        # longer the same entry) -- fall back to the sport-agnostic table only for markets
        # that are inherently single-sport already (Ks, F5 markets, NRFI/YRFI, assists, etc.)
        floor = _MARKET_ENTRY_FLOORS_BY_SPORT.get(sport.upper(), {}).get(mkt_norm)
        if floor is None:
            floor = _MARKET_ENTRY_FLOORS.get(mkt_norm)
        if floor is None:
            continue   # no market-specific floor → universal threshold applies
        min_edge, min_conf = floor
        below_edge = min_edge > 0 and bet.edge_percentage < min_edge
        below_conf = min_conf > 0 and bet.confidence_score < min_conf
        if below_edge or below_conf:
            parts: list[str] = []
            if below_edge:
                parts.append(
                    f"edge {bet.edge_percentage:.2f}% < floor {min_edge:.1f}%"
                )
            if below_conf:
                parts.append(
                    f"conf {bet.confidence_score:.1f} < floor {min_conf:.0f}"
                )
            reason = f"Market entry floor [{mkt_norm}]: {'; '.join(parts)}"
            bet.tier        = None
            bet.flagged     = True
            bet.flag_reason = _append_flag(bet.flag_reason, reason)
            logger.debug(f"[gatekeeper] FLOOR    {bet.bet_id}: {reason}")
            import datetime as _dt_floor
            _log_reject(bet, sport, _dt_floor.date.today().isoformat(),
                        "market_entry_floor", reason_override=reason)

    # Step 1: stamp tiers (sport-specific thresholds)
    for bet in bets:
        if bet.tier is None and bet.flagged:
            continue   # already discarded in Step 0.5 — do not reassign tier
        stamp_tier(bet, sport)

    # ── Step 1b: Restricted market tier cap ───────────────────────────────────
    # Restricted markets (high-variance: steals, blocks, HR, etc.) are capped
    # at EDGE unless the raw edge is extraordinary (≥ sport floor → Diamond max).
    # Nuke is never granted to a restricted market regardless of edge.
    _dia_floor = restricted_diamond_floor(sport)
    for bet in bets:
        if bet.tier is None or bet.flagged:
            continue
        if not is_restricted_market(bet.market, sport):
            continue
        if bet.tier == Tier.NUKE:
            reason = (
                f"Restricted market cap ({bet.market}): Nuke not permitted — "
                f"capped at Diamond (edge={bet.edge_percentage:.1f}%)"
            )
            bet.tier        = Tier.DIAMOND
            bet.flag_reason = _append_flag(bet.flag_reason, reason)
        if bet.tier == Tier.DIAMOND and bet.edge_percentage < _dia_floor:
            reason = (
                f"Restricted market cap ({bet.market}): Diamond requires "
                f"edge ≥{_dia_floor:.0f}%, got {bet.edge_percentage:.1f}% — "
                f"capped at Gold Standard"
            )
            bet.tier        = Tier.GOLD
            bet.flag_reason = _append_flag(bet.flag_reason, reason)

    # ── Step 1b2: WNBA minutes stability tier cap ─────────────────────────────
    # Player props backed by volatile playing-time patterns carry additional
    # uncertainty that standard edge/confidence thresholds don't capture.
    #
    # Cap rules (apply only when minutes_stability is present in raw_result):
    #   "volatile" (L5 range > 8 min) → max Gold Standard (Nuke→Gold, Diamond→Gold)
    #   "moderate" (L5 range 4–8 min) → max Diamond (Nuke→Diamond)
    #   "elite"    (L5 range ≤ 4 min) → no additional cap applied
    #
    # Only enforced for WNBA player_assists and player_rebounds markets.
    _WNBA_STABILITY_MKTS = {"player_assists", "player_rebounds"}
    for bet in bets:
        if bet.tier is None or bet.flagged:
            continue
        mkt_norm = market_normalized(bet.market)
        if mkt_norm not in _WNBA_STABILITY_MKTS:
            continue
        stability = bet.raw_result.get("minutes_stability")
        if stability is None or stability == "unknown":
            continue

        _mrange_raw = bet.raw_result.get("minutes_range", None)
        _mrange_str = f"{_mrange_raw:.0f}min" if isinstance(_mrange_raw, (int, float)) else "?min"

        if stability == "volatile":
            if bet.tier in (Tier.NUKE, Tier.DIAMOND):
                reason = (
                    f"WNBA minutes stability cap (volatile, range={_mrange_str}): "
                    f"{bet.tier.value} → Gold Standard"
                )
                bet.tier        = Tier.GOLD
                bet.flag_reason = _append_flag(bet.flag_reason, reason)
                logger.debug(f"[gatekeeper] {bet.bet_id}: {reason}")

        elif stability == "moderate":
            if bet.tier == Tier.NUKE:
                reason = (
                    f"WNBA minutes stability cap (moderate, range={_mrange_str}): "
                    f"Nuke → Diamond"
                )
                bet.tier        = Tier.DIAMOND
                bet.flag_reason = _append_flag(bet.flag_reason, reason)
                logger.debug(f"[gatekeeper] {bet.bet_id}: {reason}")
        # "elite" → no cap, tier unchanged

    # ── Step 1b3: WNBA blowout confidence penalty ─────────────────────────────
    # High-spread WNBA games create garbage-time risk that compresses starter
    # minutes.  The projection layer (player_props.py) already dampens the
    # minute estimate via blowout_mult; this step adds a direct confidence
    # penalty to capture the remaining uncertainty before V3.0 caps are applied.
    #
    # "moderate" (spread 10–17 pts) → −6 confidence
    # "heavy"    (spread ≥ 17 pts)  → −10 confidence
    # Missing blowout_level on a WNBA prop is treated as "none" (no penalty)
    # because the projection layer already applied the conservative default mult.
    # Applies ONLY to WNBA player_assists and player_rebounds markets.
    _WNBA_BLOWOUT_MKTS = frozenset({"player_assists", "player_rebounds"})

    for bet in bets:
        if bet.tier is None or bet.flagged:
            continue
        if sport.upper() != "WNBA":
            continue
        mkt_norm = market_normalized(bet.market)
        if mkt_norm not in _WNBA_BLOWOUT_MKTS:
            continue

        blowout_level = bet.raw_result.get("blowout_level", "none")
        penalty       = _BLOWOUT_CONF_PENALTY.get(blowout_level, 0.0)
        if penalty <= 0:
            continue

        raw_conf = bet.confidence_score
        bet.confidence_score = max(0.0, bet.confidence_score - penalty)
        reason = (
            f"WNBA blowout [{blowout_level}]: "
            f"conf −{penalty:.0f}pts ({raw_conf:.1f}→{bet.confidence_score:.1f})"
        )
        bet.flag_reason = _append_flag(bet.flag_reason, reason)
        bet.tier        = evaluate_tier(bet.edge_percentage, bet.confidence_score, sport, bet.market)
        if bet.tier is None:
            bet.flagged = True
        logger.debug(f"[gatekeeper] BLOWOUT  {bet.bet_id}: {reason}")

    # ── Step 1c: V3.0 Confidence Caps ─────────────────────────────────────────
    # Prevent inflated Bayesian posteriors from generating misleadingly high
    # confidence values.  Caps are applied per-sport × per-market × per-tier.
    if _V3_MODULES_AVAILABLE:
        for bet in bets:
            if bet.tier is None:
                continue
            capped_conf, was_capped = apply_confidence_cap(
                bet.confidence_score, sport, bet.market, bet.tier.value
            )
            if was_capped:
                bet.flag_reason = _append_flag(
                    bet.flag_reason,
                    f"V3.0 confidence cap [{bet.tier.value}/"
                    f"{_mkt_cat(bet.market)}]: "
                    f"{bet.confidence_score:.1f}→{capped_conf:.1f}",
                )
                bet.confidence_score = capped_conf

    # ── Step 1.6: Market signal hard filters (early exit — before MAS penalties) ─
    # Three conditions warrant immediate discard regardless of raw edge/confidence.
    # Running BEFORE Step 1d avoids wasting conf/edge deductions on picks that
    # will be hard-blocked anyway (DC-2 fix: sharp_contrary no longer applies
    # both a MAS-based conf penalty AND a separate hard block).
    #
    #   1. effective_edge < 0   — market intelligence wiped the model edge.
    #   2. sharp_contrary       — confirmed sharp money is on the other side.
    #   3. edge_decay > 3.0     — line moved > 3 pts against pick since model time.
    _EFF_EDGE_FLOOR = 0.0
    _EDGE_DECAY_MAX = 3.0
    _SHARP_BLOCK    = "sharp_contrary"

    for bet in bets:
        if bet.tier is None or bet.flagged:
            continue
        rd = bet.raw_result

        eff_edge  = rd.get("effective_edge")
        sharp_sig = str(rd.get("sharp_signal") or "")
        decay     = rd.get("edge_decay")

        if eff_edge is not None and float(eff_edge) < _EFF_EDGE_FLOOR:
            reason = (
                f"Market signal filter: effective_edge={float(eff_edge):.2f}% < 0 — "
                f"market intelligence wiped the model edge"
            )
            bet.tier        = None
            bet.flagged     = True
            bet.flag_reason = _append_flag(bet.flag_reason, reason)
            logger.debug(f"[gatekeeper] REJECT mkt-signal  {bet.bet_id}: {reason}")
            continue

        if _SHARP_BLOCK in sharp_sig:
            reason = (
                f"Market signal filter: sharp_signal='{sharp_sig}' — "
                f"confirmed sharp money is against this pick"
            )
            bet.tier        = None
            bet.flagged     = True
            bet.flag_reason = _append_flag(bet.flag_reason, reason)
            logger.debug(f"[gatekeeper] REJECT mkt-signal  {bet.bet_id}: {reason}")
            continue

        if decay is not None and float(decay) > _EDGE_DECAY_MAX:
            reason = (
                f"Market signal filter: edge_decay={float(decay):.1f} > {_EDGE_DECAY_MAX} — "
                f"line moved materially against pick since model time"
            )
            bet.tier        = None
            bet.flagged     = True
            bet.flag_reason = _append_flag(bet.flag_reason, reason)
            logger.debug(f"[gatekeeper] REJECT mkt-signal  {bet.bet_id}: {reason}")

    # ── Step 1d: V3.0 Market Agreement + Integrity Filter ─────────────────────
    # Market Agreement: picks with agreement score below the tier floor are
    # downgraded one tier OR have confidence/edge penalties applied — not both
    # (DC-3 fix: tier-downgrade and conf/edge penalty are now mutually exclusive
    # so a single low-MAS reading doesn't cascade through two separate mechanisms).
    # Integrity Filter: missing required elements downgrade Nuke/Diamond picks
    # one tier per element; 2+ missing elements discard the pick entirely.
    if _V3_MODULES_AVAILABLE:
        _tier_order = [Tier.NUKE, Tier.DIAMOND, Tier.GOLD]

        for bet in bets:
            if bet.tier is None or bet.flagged:
                continue
            rd = bet.raw_result

            # Market agreement score (computed in main.py pipeline)
            mas = rd.get("market_agreement_score")
            if mas is not None:
                mas = int(mas)
                _sharp = str(rd.get("sharp_signal", "no_sharp"))

                _was_downgraded = not _tier_passes_agreement(bet.tier.value, mas)
                if _was_downgraded:
                    # Tier downgrade path — confidence/edge penalties are NOT
                    # additionally applied (one mechanism per MAS reading).
                    _idx = next(
                        (i for i, t in enumerate(_tier_order) if t == bet.tier), None
                    )
                    _ma_reason = (
                        f"V3.0 market agreement ({mas}/100) below floor "
                        f"for {bet.tier.value} → downgraded one tier"
                    )
                    if _idx is not None and _idx + 1 < len(_tier_order):
                        bet.tier = _tier_order[_idx + 1]
                    else:
                        bet.tier = None
                        bet.flagged = True
                    bet.flag_reason = _append_flag(bet.flag_reason, _ma_reason)
                else:
                    # Tier passed the agreement floor — apply soft conf/edge
                    # penalties when agreement is weak but not floor-failing.
                    _conf_pen = _agree_conf_penalty(mas, _sharp)
                    _edge_pen = _agree_edge_penalty(mas)
                    if _conf_pen > 0:
                        bet.confidence_score = max(0.0, bet.confidence_score - _conf_pen)
                        bet.tier = evaluate_tier(bet.edge_percentage, bet.confidence_score, sport, bet.market)
                        if bet.tier is None:
                            bet.flagged = True
                    if _edge_pen > 0:
                        bet.edge_percentage = round(
                            max(0.0, bet.edge_percentage - _edge_pen), 2
                        )

            # Integrity filter — only Diamond/Nuke game-market picks
            if bet.tier in (Tier.NUKE, Tier.DIAMOND) and not bet.flagged:
                _is_gm = _is_game_mkt_for_gk(bet.market)
                _n_miss, _missing = _run_integrity_filter(
                    rd, sport, is_game_market=_is_gm
                )
                if _n_miss >= 2:
                    _if_reason = (
                        f"V3.0 integrity filter: {_n_miss} required elements "
                        f"missing ({', '.join(_missing[:3])}) — discarded"
                    )
                    bet.tier    = None
                    bet.flagged = True
                    bet.flag_reason = _append_flag(bet.flag_reason, _if_reason)
                elif _n_miss == 1:
                    _if_reason = (
                        f"V3.0 integrity filter: 1 element missing "
                        f"({_missing[0]}) — downgraded one tier"
                    )
                    _idx = next(
                        (i for i, t in enumerate(_tier_order) if t == bet.tier), None
                    )
                    if _idx is not None and _idx + 1 < len(_tier_order):
                        bet.tier = _tier_order[_idx + 1]
                    else:
                        bet.tier = None
                        bet.flagged = True
                    bet.flag_reason = _append_flag(bet.flag_reason, _if_reason)

    # ── Step 1.4b: Strikeout prop tier rescue ───────────────────────────────
    # MLB pitcher strikeout props use a lower confidence floor (68) because
    # the workload + K-matchup model provides reliable K-rate priors that
    # partially substitute for posterior confidence.  Picks that were rejected
    # by evaluate_tier() (needs conf ≥ 68) but still meet conf ≥ 68 and
    # edge ≥ 1.0 are re-admitted at Gold Standard before the prop floor check.
    _STRIKEOUT_RESCUE_CONF = 68.0
    _STRIKEOUT_RESCUE_EDGE = 1.0

    for bet in bets:
        if bet.tier is not None:
            continue   # already approved — nothing to rescue
        if bet.flagged:
            continue   # rejected by market entry floor or prior rule — do not rescue
        mkt = market_normalized(bet.market)
        if "strikeouts" not in mkt:
            continue
        if (bet.confidence_score >= _STRIKEOUT_RESCUE_CONF
                and bet.edge_percentage >= _STRIKEOUT_RESCUE_EDGE):
            bet.tier    = Tier.GOLD
            bet.flagged = False
            bet.flag_reason = None
            logger.debug(
                f"[gatekeeper] STRIKEOUT RESCUE  {bet.bet_id}: "
                f"edge={bet.edge_percentage:.2f}%  conf={bet.confidence_score:.1f} "
                f"→ re-admitted at Gold Standard (strikeout conf floor=68)"
            )

    # ── Step 1.5: Prop confidence floor ─────────────────────────────────────
    # Player props (player_* / pitcher_* / batter_*) are high-variance markets.
    # Require a minimum confidence for any prop to be eligible for broadcast.
    # Strikeout props use a lower floor (68) — workload model compensates.
    # All other props use the standard floor (75).
    _PROP_CONF_FLOOR          = 75.0
    _STRIKEOUT_PROP_CONF_FLOOR = 68.0
    _PROP_PREFIXES             = ("player_", "pitcher_", "batter_")

    for bet in bets:
        if bet.tier is None or bet.flagged:
            continue
        mkt = market_normalized(bet.market)
        if not any(mkt.startswith(pfx) for pfx in _PROP_PREFIXES):
            continue   # game market — no prop floor applied
        floor = _STRIKEOUT_PROP_CONF_FLOOR if "strikeouts" in mkt else _PROP_CONF_FLOOR
        if bet.confidence_score < floor:
            reason = (
                f"Prop confidence floor: {mkt} requires conf ≥ {floor:.0f} "
                f"for broadcast eligibility, got {bet.confidence_score:.1f}"
            )
            bet.tier        = None
            bet.flagged     = True
            bet.flag_reason = _append_flag(bet.flag_reason, reason)
            logger.debug(
                f"[gatekeeper] REJECT prop  {bet.bet_id}: {reason}"
            )

    # Step 2: same-team direction conflicts
    check_for_conflicts(bets, sport)

    # Step 3: prop-consensus block (requires game_id on bets)
    _check_prop_consensus(bets, sport)

    # ── Step 4: Near-miss flagging ────────────────────────────────────────────
    # Bets that miss the Gold Standard minimum by < 2.5 % edge OR < 5 conf pts
    # are flagged for secondary analysis rather than being silently discarded.
    _sport_thresholds     = SPORT_TIER_THRESHOLDS.get(sport.upper(), SPORT_TIER_THRESHOLDS[_DEFAULT_SPORT_KEY])
    _EDGE_MIN_EDGE        = _sport_thresholds[-1][1]   # lowest publishable tier min_edge
    _EDGE_MIN_CONF        = _sport_thresholds[-1][2]   # lowest publishable tier min_conf
    _NEAR_MISS_EDGE_FLOOR = max(0.1, _EDGE_MIN_EDGE - 0.5)   # V3.0: tighter window on calibrated scale
    _NEAR_MISS_CONF_FLOOR = _EDGE_MIN_CONF - 5.0

    for bet in bets:
        if bet.tier is None and not bet.flagged:
            if (
                bet.edge_percentage  >= _NEAR_MISS_EDGE_FLOOR
                and bet.confidence_score >= _NEAR_MISS_CONF_FLOOR
            ):
                gap_edge = _EDGE_MIN_EDGE - bet.edge_percentage
                gap_conf = _EDGE_MIN_CONF - bet.confidence_score
                bet.flagged    = True
                bet.flag_reason = _append_flag(
                    bet.flag_reason,
                    f"Near-miss: edge={bet.edge_percentage:.1f}% "
                    f"(need ≥{_EDGE_MIN_EDGE}%, gap={gap_edge:+.1f}); "
                    f"conf={bet.confidence_score:.1f} "
                    f"(need ≥{_EDGE_MIN_CONF}, gap={gap_conf:+.1f}) — "
                    f"secondary analysis recommended "
                    f"(check injury reports & line movement)."
                )

    # ── Step 5: Plus-money confidence floor ─────────────────────────────────
    # Plus-money lines (odds > 0) have a lower implied probability (<50 %), so
    # even a modest model result easily clears the standard edge threshold.
    # We require +3 confidence points above the tier floor for any plus-money
    # bet to ensure genuine model conviction.
    #
    # Floors (tier base + 3):
    #   Nuke:         85 → 88  (plus-money)
    #   Diamond:      78 → 81  (plus-money)
    #   Gold Standard:68 → 71  (plus-money)
    _PLUS_MONEY_CONF_FLOORS: list[tuple[Tier, float]] = [
        (Tier.NUKE,    88.0),
        (Tier.DIAMOND, 81.0),
        (Tier.GOLD,    71.0),
    ]

    for bet in bets:
        if bet.tier is None or bet.flagged:
            continue
        if bet.american_odds <= 0:
            continue  # minus-money or unknown — standard thresholds apply

        for tier, min_conf in _PLUS_MONEY_CONF_FLOORS:
            if bet.tier == tier:
                if bet.confidence_score < min_conf:
                    reason = (
                        f"Plus-money confidence floor: odds={bet.american_odds:+.0f} "
                        f"requires conf ≥ {min_conf:.0f} for {tier.value} tier, "
                        f"got {bet.confidence_score:.1f}"
                    )
                    bet.flagged     = True
                    bet.flag_reason = _append_flag(bet.flag_reason, reason)
                    bet.tier        = None
                break

    # ── Step 6: L5 cold-streak brake (DC-1 fix) ──────────────────────────────
    # For OVER picks: if L5 average < line, recent form contradicts the pick.
    # For UNDER picks: if L5 average > line, recent hot streak is a headwind.
    #
    # DC-1 fix: L5 data already shaped the NUTS posterior (lower mean → lower
    # edge/confidence).  Applying a full 25-pt penalty on top double-counts
    # the same observation.  We now apply a reduced secondary brake:
    #   • l5_in_simulation=True  (default — L5 was in NUTS): max 10 pts, ×20
    #   • l5_in_simulation=False (season-avg-only posterior): max 25 pts, ×40
    # The secondary brake catches extreme cold streaks the posterior may
    # under-weight when sample size is small.
    for bet in bets:
        if bet.tier is None or bet.flagged:
            continue
        l5 = bet.raw_result.get("l5_avg")
        if l5 is None:
            continue
        line      = float(bet.sportsbook_line)
        if line <= 0:
            continue
        direction = bet.direction.lower()

        breached = (direction == "over" and l5 < line) or (
                    direction == "under" and l5 > line)
        if not breached:
            continue

        gap_ratio = abs(l5 - line) / max(line, 0.1)

        l5_was_in_sim = bet.raw_result.get("l5_in_simulation", True)
        if l5_was_in_sim:
            penalty    = round(min(10.0, gap_ratio * 20.0), 1)
            brake_note = "reduced — L5 in NUTS posterior"
        else:
            penalty    = round(min(25.0, gap_ratio * 40.0), 1)
            brake_note = "full — season-avg-only posterior"

        side   = "cold" if direction == "over" else "hot"
        reason = (
            f"L5 {side}-streak brake [{brake_note}]: L5={l5:.2f} "
            f"{'<' if direction == 'over' else '>'} line={line:.2f} — "
            f"recent form contradicts {direction.upper()}; "
            f"confidence −{penalty:.0f}pts"
        )
        bet.confidence_score = max(0.0, bet.confidence_score - penalty)
        bet.tier             = evaluate_tier(bet.edge_percentage, bet.confidence_score, sport, bet.market)
        bet.flag_reason      = _append_flag(bet.flag_reason, reason)
        if bet.tier is None:
            bet.flagged = True

    # ── Step 7: Nuke projection cushion gate ─────────────────────────────────
    # Model projection must clear the sportsbook line by a market-specific
    # multiplier to qualify as Nuke. Thin cushion = elevated bust risk.
    for bet in bets:
        if bet.tier != Tier.NUKE or bet.flagged:
            continue
        proj = bet.raw_result.get("weighted_projection")
        if proj is None:
            continue
        line = float(bet.sportsbook_line)
        if line <= 0:
            continue
        mkt       = market_normalized(bet.market)
        direction = bet.direction.lower()

        if direction == "over":
            cushion  = _S_PLUS_CUSHION_OVER.get(mkt, _S_PLUS_DEFAULT_CUSHION_OVER)
            required = line * cushion
            if proj < required:
                reason = (
                    f"Nuke cushion gate: proj={proj:.2f} < "
                    f"line×{cushion:.2f}={required:.2f} — "
                    f"insufficient projection margin; confidence −10pts"
                )
                bet.confidence_score = max(0.0, bet.confidence_score - 10.0)
                bet.tier             = evaluate_tier(bet.edge_percentage, bet.confidence_score, sport, bet.market)
                bet.flag_reason      = _append_flag(bet.flag_reason, reason)
                if bet.tier is None:
                    bet.flagged = True
        else:  # under
            cushion_max  = _S_PLUS_CUSHION_UNDER_MAX.get(mkt, _S_PLUS_DEFAULT_CUSHION_UNDER)
            required_max = line * cushion_max
            if proj > required_max:
                reason = (
                    f"Nuke cushion gate: proj={proj:.2f} > "
                    f"line×{cushion_max:.2f}={required_max:.2f} — "
                    f"projection too close to line for Nuke UNDER; confidence −10pts"
                )
                bet.confidence_score = max(0.0, bet.confidence_score - 10.0)
                bet.tier             = evaluate_tier(bet.edge_percentage, bet.confidence_score, sport, bet.market)
                bet.flag_reason      = _append_flag(bet.flag_reason, reason)
                if bet.tier is None:
                    bet.flagged = True

    # ── Step 8: Market concentration cap ─────────────────────────────────────
    # Limit Nuke picks to _SPLUS_MARKET_CAP per market type per run.
    # Sort by edge descending so the weakest picks are demoted first.
    _splus_market_count: dict[str, int] = {}
    _splus_candidates = sorted(
        [b for b in bets if b.tier == Tier.NUKE and not b.flagged],
        key=lambda b: b.edge_percentage,
        reverse=True,
    )
    for bet in _splus_candidates:
        mkt = market_normalized(bet.market)
        _splus_market_count[mkt] = _splus_market_count.get(mkt, 0) + 1
        if _splus_market_count[mkt] > _SPLUS_MARKET_CAP:
            reason = (
                f"Market concentration cap: {mkt} already has "
                f"{_SPLUS_MARKET_CAP} Nuke pick(s) — demoted to Diamond "
                f"(edge={bet.edge_percentage:.1f}%)"
            )
            bet.tier        = Tier.DIAMOND
            bet.flag_reason = _append_flag(bet.flag_reason, reason)

    approved  = [b for b in bets if b.tier is not None and not b.flagged]
    flagged   = [b for b in bets if b.flagged]
    discarded = [b for b in bets if b.tier is None and not b.flagged]

    # ── Rejection reason logging ──────────────────────────────────────────────
    # Log a descriptive reason for every bet that did not reach approved status.
    # This creates a clear audit trail in betting_bot.log for each engine run.
    for bet in flagged:
        logger.debug(
            f"[gatekeeper] FLAGGED  {bet.bet_id:40s} | "
            f"edge={bet.edge_percentage:.2f}%  conf={bet.confidence_score:.1f} | "
            f"{bet.flag_reason or 'no reason recorded'}"
        )
    for bet in discarded:
        sport_key = sport.upper()
        thresholds = SPORT_TIER_THRESHOLDS.get(sport_key, SPORT_TIER_THRESHOLDS[_DEFAULT_SPORT_KEY])
        min_edge   = thresholds[-1][1]
        min_conf   = thresholds[-1][2]
        logger.debug(
            f"[gatekeeper] DISCARD  {bet.bet_id:40s} | "
            f"edge={bet.edge_percentage:.2f}% (need ≥{min_edge})  "
            f"conf={bet.confidence_score:.1f} (need ≥{min_conf}) | "
            f"{bet.flag_reason or 'below Gold Standard threshold'}"
        )
    logger.info(
        f"[gatekeeper] [{sport}] approved={len(approved)}  "
        f"flagged={len(flagged)}  discarded={len(discarded)}"
    )

    # ── Shadow reject log ─────────────────────────────────────────────────────
    if _REJECT_LOGGER_AVAILABLE:
        import datetime as _dt
        _slate = _dt.date.today().isoformat()
        for bet in discarded:
            _log_reject(bet, sport, _slate, "gatekeeper")
        for bet in flagged:
            _log_reject(bet, sport, _slate, "gatekeeper_flagged")

    return {"approved": approved, "flagged": flagged, "discarded": discarded}
