"""
shadow_logger.py — Records EVERY candidate pick the pipeline evaluates,
not just the ones that get published to picks.json.

Why this exists: picks.json / pick_history.jsonl only ever contain picks
that survived every filter (edge threshold, agreement_frac, contradiction
check, line movement, daily cap). That means you can never answer "was the
70% confidence floor too strict?" or "did we drop good picks on the daily
cap?" -- the rejected candidates simply vanish. This module writes a
permanent shadow record of every candidate, win or lose, published or not,
so recalibration (backtest.py) has the full population to work with.

Output: output/shadow_log.jsonl (one JSON object per line, append-only).

Each record captures the full decision trail:
  - raw model outputs (model_prob, edge_pct, side_agreement_frac)
  - confidence score actually computed
  - which filter (if any) rejected the candidate, and why
  - whether it was ultimately published
  - actual_result: null at write time, back-filled later by backtest.py's
    grading step once the game result is known

Usage from run_pipeline.py:
    from shadow_logger import log_candidate

    log_candidate(
        sport="MLB Ks", player=raw["player"], matchup=raw["matchup"],
        market_line=raw["market_k_line"], side=side,
        model_prob=pick["model_prob"], edge_pct=edge_pct,
        side_agreement_frac=robust["agreement_frac"],
        confidence=pick["confidence"],
        rejected_stage=None, rejected_reason=None,
        published=True, generated_at=output["generated_at"],
    )
    # ... and for anything that got filtered out, same call but with
    # rejected_stage="edge_threshold" / "contradiction_check" / etc.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

SHADOW_LOG_PATH = os.path.join(os.path.dirname(__file__), "output", "shadow_log.jsonl")


def _ensure_dir():
    os.makedirs(os.path.dirname(SHADOW_LOG_PATH), exist_ok=True)


def log_candidate(
    *,
    sport: str,
    player: str | None = None,
    matchup: str | None = None,
    market_line: float | None = None,
    side: str | None = None,
    model_prob: float | None = None,
    edge_pct: float | None = None,
    side_agreement_frac: float | None = None,
    confidence: float | None = None,
    rejected_stage: str | None = None,
    rejected_reason: str | None = None,
    published: bool = False,
    generated_at: str | None = None,
    extra: dict | None = None,
) -> str:
    """
    Append one candidate record to the shadow log. Returns the record's
    shadow_id (uuid4 hex) so the caller can correlate it with a published
    pick_id later if needed.

    rejected_stage examples: "edge_threshold", "agreement_frac",
    "min_confidence", "contradiction_check", "line_movement", "daily_cap".
    Leave both rejected_stage/rejected_reason as None if published=True.
    """
    _ensure_dir()
    shadow_id = uuid.uuid4().hex
    record = {
        "shadow_id": shadow_id,
        "logged_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "sport": sport,
        "player": player,
        "matchup": matchup,
        "market_line": market_line,
        "side": side,
        "model_prob": model_prob,
        "edge_pct": edge_pct,
        "side_agreement_frac": side_agreement_frac,
        "confidence": confidence,
        "published": published,
        "rejected_stage": rejected_stage,
        "rejected_reason": rejected_reason,
        # Back-filled later by backtest.py's grading step:
        "actual_result": None,   # "win" | "loss" | "push"
        "graded_at": None,
        "extra": extra or {},
    }
    with open(SHADOW_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return shadow_id


def load_shadow_log() -> list[dict]:
    """Read the full shadow log. Returns [] if the file doesn't exist yet."""
    if not os.path.exists(SHADOW_LOG_PATH):
        return []
    out = []
    with open(SHADOW_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def backfill_result(shadow_id: str, actual_result: str) -> bool:
    """
    Update one record's actual_result in place. Rewrites the whole file
    (shadow logs are small enough -- thousands of rows -- that this is
    fine; switch to a real DB if this ever exceeds ~100k rows).
    Returns True if a matching record was found and updated.
    """
    records = load_shadow_log()
    found = False
    for r in records:
        if r["shadow_id"] == shadow_id:
            r["actual_result"] = actual_result
            r["graded_at"] = datetime.now(timezone.utc).isoformat()
            found = True
            break
    if found:
        with open(SHADOW_LOG_PATH, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    return found
