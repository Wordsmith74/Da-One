"""
Shared utilities for the historical grading engine.

Schema aligned to the `Bet` dataclass in core/decision_gatekeeper.py.
Each approved pick written to pick_history.jsonl is assumed to carry (at
least) these fields, matching Bet 1:1:

    bet_id             str   e.g. "prop_Davis_Martin_pitcher_strikeouts_over_6f7b9d64"
    team               str   team abbreviation/name the bet is tied to
    market             str   raw market string; normalize with
                              decision_gatekeeper.market_normalized() before dispatch
    direction          str   "over" / "under" (moneyline picks: see `team` instead)
    sportsbook_line     float
    edge_percentage     float
    confidence_score    float
    player              str | None
    game_id             str   "AWAY@HOME_SPORT_YYYY-MM-DD", e.g.
                              "PHX@MIN_WNBA_2026-06-01" -- sport/date/teams are
                              parsable directly from this, no separate fields needed
    american_odds       float
    data_reliability_score int
    mis_score           int
    tier                str   "Nuke" | "Diamond" | "Gold Standard"
    flagged             bool
    flag_reason         str
    raw_result          dict

stake_units is NOT part of Bet -- TODO: confirm how your pipeline assigns
stake sizing (flat 1u per tier? tier-scaled staking table?) before
historical_grader.py computes profit_units. Defaulting to 1.0u flat here.

After grading, we append:
    actual_result ("win"/"loss"/"push"),
    actual_stat (float or None),
    closing_line (float or None),
    closing_odds (int or None),
    graded_at (ISO8601),
    profit_units (float),
    roi (float, profit_units / stake_units)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger("historical_grader")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def american_odds_profit(odds: int, stake_units: float = 1.0) -> float:
    """Profit in units for a WIN at the given American odds, for a given stake."""
    if odds > 0:
        return stake_units * (odds / 100.0)
    else:
        return stake_units * (100.0 / abs(odds))


@dataclass
class GradeOutcome:
    actual_result: str  # "win" | "loss" | "push"
    actual_stat: Optional[float]
    profit_units: float
    roi: float


def settle(
    result: str,
    odds: int,
    stake_units: float,
    actual_stat: Optional[float] = None,
) -> GradeOutcome:
    """
    Convert a win/loss/push determination into profit/ROI given American odds.
    Pushes return the stake (0 profit).
    """
    if result == "win":
        profit = american_odds_profit(odds, stake_units)
    elif result == "loss":
        profit = -stake_units
    elif result == "push":
        profit = 0.0
    else:
        raise ValueError(f"Unknown grading result: {result!r}")

    roi = profit / stake_units if stake_units else 0.0
    return GradeOutcome(
        actual_result=result,
        actual_stat=actual_stat,
        profit_units=round(profit, 4),
        roi=round(roi, 4),
    )


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with open(path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    # Degrade gracefully: log and skip malformed lines rather
                    # than crashing the whole grading run.
                    logger.warning(
                        "malformed_jsonl_line",
                        extra={"path": path, "line_no": line_no, "error": str(e)},
                    )
    except FileNotFoundError:
        logger.warning("jsonl_not_found", extra={"path": path})
    return records


def write_jsonl(path: str, records: Iterable[dict[str, Any]]) -> None:
    """Full rewrite of the file. Caller is responsible for atomicity if needed."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")
    import os

    os.replace(tmp_path, path)


def log_reject(path: str, reject_record: dict[str, Any]) -> None:
    """
    Structured JSONL reject/error log (matches preference for discrete
    queryable fields over flat strings).
    """
    reject_record.setdefault("logged_at", utcnow_iso())
    with open(path, "a") as f:
        f.write(json.dumps(reject_record, default=str) + "\n")
