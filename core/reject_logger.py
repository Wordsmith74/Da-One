"""
core/reject_logger.py

Unified shadow logger for every bet that enters the pipeline but never publishes.

Every rejection — regardless of which layer kills it — appends one JSON record
to  data/bet_rejects.jsonl.  The `stage` field tells you where it was killed:

    pre_sim        — failed validation before the NUTS sampler ran
    stability      — posterior σ/|mean| exceeded the threshold
    gatekeeper     — edge/confidence below floor, market signal hard-block,
                     prop conf floor, plus-money floor, near-miss, MAS penalty,
                     L5 cold-streak brake, Nuke cushion gate, or integrity filter
    governance_hold — passed gatekeeper but held by signal confirmation (< 3 cycles)
    conflict_hold  — passed gatekeeper + signal gate but blocked by conflict guardian

Schema
------
{
  "date":           "2026-06-15",          # slate date (ET)
  "timestamp":      "2026-06-15T13:00Z",   # wall-clock UTC ISO-8601
  "sport":          "MLB",
  "stage":          "gatekeeper",
  "reason":         "Prop confidence floor: ...",
  "bet_id":         "prop_Chase_Burns_...",
  "market":         "pitcher_strikeouts",
  "player":         "Chase Burns",
  "team":           "METSvREDS",
  "direction":      "under",
  "sportsbook_line": 7.5,
  "edge":           8.2,
  "confidence":     64.1,
  "projection":     6.1,
  "sigma":          0.48,
  "rel_sigma_pct":  7.9,
  "game_id":        "New_York_Mets@Cincinnati_Reds_MLB",
  "away_team":      "New York Mets",
  "home_team":      "Cincinnati Reds",
  "bookmaker":      "Novig",
  "flag_reason":      "Market entry floor [player_assists]: edge 2.10% < floor 4.0%",
  "minutes_stability": "volatile",
  "minutes_range":    9.0,
  "blowout_level":    "moderate"
}

New fields (added to isolate the four WNBA-specific gatekeeper layers
without string-matching `reason`):

    flag_reason       — full accumulated bet.flag_reason at log time. May be
                         a semicolon-joined chain if the bet survived several
                         penalty steps (e.g. blowout penalty AND stability
                         cap) before being discarded. `reason` stays the
                         specific trigger for *this* log call; `flag_reason`
                         is the whole trail.
    minutes_stability — bet.raw_result["minutes_stability"]: "elite" |
                         "moderate" | "volatile" | "unknown" | None.
                         Drives the Step 1b2 tier cap.
    minutes_range     — bet.raw_result["minutes_range"] (L5 minutes range,
                         float). Underlying number behind minutes_stability.
    blowout_level     — bet.raw_result["blowout_level"]: "none" | "moderate"
                         | "heavy" | None. Drives the Step 1b3 confidence
                         penalty.
    ramp_flag         — bet.raw_result["ramp_flag"]: bool | None. True when
                         the subject player is either on a minutes-drop ramp
                         pattern or carries a live questionable/day-to-day/
                         probable status tag (models/ramp_detection.py +
                         core/player_props.py). Drives the Step 0.5b hard
                         entry gate for player_assists/player_rebounds.

With these, you can filter bet_rejects.jsonl by stage="gatekeeper" and
sport="WNBA", then group by which of flag_reason's component steps
("Market entry floor", "WNBA minutes stability cap", "WNBA blowout",
"V3.0 confidence cap") actually appears, instead of guessing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.decision_gatekeeper import Bet

logger = logging.getLogger("betting_bot")

_REJECT_LOG: Path = Path(__file__).resolve().parent.parent / "data" / "bet_rejects.jsonl"


def log_rejected_bet(
    *,
    sport: str,
    slate_date: str,
    stage: str,
    reason: str,
    bet_id: str = "",
    market: str = "",
    player: str = "",
    team: str = "",
    direction: str = "",
    sportsbook_line: float | None = None,
    edge: float | None = None,
    confidence: float | None = None,
    projection: float | None = None,
    sigma: float | None = None,
    rel_sigma_pct: float | None = None,
    game_id: str = "",
    away_team: str = "",
    home_team: str = "",
    bookmaker: str = "",
    flag_reason: str = "",
    minutes_stability: str | None = None,
    minutes_range: float | None = None,
    blowout_level: str | None = None,
    ramp_flag: bool | None = None,
) -> None:
    """Append one rejection record to data/bet_rejects.jsonl."""
    record: dict[str, Any] = {
        "date":            slate_date,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "sport":           sport,
        "stage":           stage,
        "reason":          reason,
        "bet_id":          bet_id or None,
        "market":          market or None,
        "player":          player or None,
        "team":            team or None,
        "direction":       direction or None,
        "sportsbook_line": sportsbook_line,
        "edge":            round(edge, 4) if edge is not None else None,
        "confidence":      round(confidence, 2) if confidence is not None else None,
        "projection":      round(projection, 4) if projection is not None else None,
        "sigma":           round(sigma, 4) if sigma is not None else None,
        "rel_sigma_pct":   round(rel_sigma_pct, 2) if rel_sigma_pct is not None else None,
        "game_id":         game_id or None,
        "away_team":       away_team or None,
        "home_team":       home_team or None,
        "bookmaker":       bookmaker or None,
        "flag_reason":     flag_reason or None,
        "minutes_stability": minutes_stability,
        "minutes_range":   round(minutes_range, 1) if minutes_range is not None else None,
        "blowout_level":   blowout_level,
        "ramp_flag":       ramp_flag,
    }
    try:
        _REJECT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _REJECT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.debug(f"[reject_logger] write failed: {exc}")


def log_rejected_bet_obj(
    bet: "Bet",
    sport: str,
    slate_date: str,
    stage: str,
    reason_override: str | None = None,
) -> None:
    """
    Convenience wrapper — extracts all fields from a Bet dataclass and calls
    log_rejected_bet().  Works for gatekeeper discards, flagged near-misses,
    governance holds, and conflict holds.
    """
    rd: dict[str, Any] = bet.raw_result or {}
    proj = rd.get("weighted_projection")
    log_rejected_bet(
        sport=sport,
        slate_date=slate_date,
        stage=stage,
        reason=reason_override or bet.flag_reason or "below threshold",
        bet_id=bet.bet_id,
        market=bet.market,
        player=bet.player or "",
        team=bet.team,
        direction=bet.direction,
        sportsbook_line=float(bet.sportsbook_line) if bet.sportsbook_line is not None else None,
        edge=float(bet.edge_percentage) if bet.edge_percentage is not None else None,
        confidence=float(bet.confidence_score) if bet.confidence_score is not None else None,
        projection=float(proj) if proj is not None else None,
        game_id=bet.game_id or "",
        away_team=rd.get("away_team", ""),
        home_team=rd.get("home_team", ""),
        bookmaker=rd.get("bookmaker_source", ""),
        flag_reason=bet.flag_reason or "",
        minutes_stability=rd.get("minutes_stability"),
        minutes_range=rd.get("minutes_range"),
        blowout_level=rd.get("blowout_level"),
        ramp_flag=rd.get("ramp_flag"),
    )


def log_rejected_candidate(
    sport: str,
    candidate: dict[str, Any],
    stage: str,
    reason: str,
    slate_date: str,
    sigma: float | None = None,
    projection: float | None = None,
    rel_sigma_pct: float | None = None,
) -> None:
    """
    Convenience wrapper for raw candidate dicts (pre-simulation and stability
    rejections, where a Bet object hasn't been built yet).
    """
    mean = projection if projection is not None else candidate.get("weighted_projection")
    log_rejected_bet(
        sport=sport,
        slate_date=slate_date,
        stage=stage,
        reason=reason,
        bet_id=candidate.get("bet_id", ""),
        market=candidate.get("market", ""),
        player=candidate.get("player") or "",
        team=candidate.get("team", ""),
        direction=candidate.get("direction", ""),
        sportsbook_line=candidate.get("sportsbook_line"),
        edge=candidate.get("edge_percentage") or candidate.get("precomputed_edge"),
        confidence=candidate.get("confidence_score") or candidate.get("precomputed_confidence"),
        projection=float(mean) if mean is not None else None,
        sigma=float(sigma) if sigma is not None else None,
        rel_sigma_pct=float(rel_sigma_pct) if rel_sigma_pct is not None else None,
        game_id=candidate.get("game_id", ""),
        away_team=candidate.get("away_team", ""),
        home_team=candidate.get("home_team", ""),
        bookmaker=candidate.get("bookmaker_source", ""),
        minutes_stability=candidate.get("minutes_stability"),
        minutes_range=candidate.get("minutes_range"),
        blowout_level=candidate.get("blowout_level"),
        ramp_flag=candidate.get("ramp_flag"),
    )
