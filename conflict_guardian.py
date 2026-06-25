"""
core/conflict_guardian.py

Locked-Pick Conflict Guardian

Before any new signal is published, this module checks whether an existing
LOCKED pick already covers the same (game_id, market_category).  If a
conflict exists, it applies the 5-condition Replacement Threshold:

  1. New confidence exceeds prior confidence by ≥ 5 percentage points
  2. New edge exceeds prior edge by ≥ 25 %
  3. New edge exceeds prior edge by ≥ 40 % (projection decisiveness proxy)
  4. Signal survived required confirmation cycles (verified upstream)
  5. Market Intelligence Score ≥ 40 (sharp data supports reversal)

Unless ALL five conditions are met the existing locked pick is retained
and the challenger is rejected (action = "hold").

Actual column layout (bets table):
  edge_percentage       REAL    — top-level column
  opening_confidence    REAL    — opening confidence score (nullable)
  slate_date            TEXT    — YYYY-MM-DD
  actual_outcome        TEXT    — NULL while open
  wager_details         TEXT    — JSON blob; contains "market", "direction",
                                  "confidence_score", etc.

Every conflict generates a Stability Warning logged to:
  data/stability_warnings/{SPORT}_{DATE}.jsonl

Exposed API:
  check_locked_conflict(bet, sport, date_str=None)  → ("clear"|"hold"|"replace", details)
  log_stability_warning(sport, date_str, details)
  get_stability_warnings(sport, date_str)           → list[dict]
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from core.time_utils import now_est
from core.decision_gatekeeper import market_normalized

_DB_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "results.db")
)

_WARNINGS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "stability_warnings")
)

# ── Replacement Threshold constants (spec §REPLACEMENT THRESHOLD) ─────────────
_MIN_CONF_GAIN_PP  = 5.0   # new_conf - old_conf ≥ 5 pp                (Cond 1)
_MIN_EDGE_GAIN     = 0.25  # (new_edge / old_edge) - 1 ≥ 25 %          (Cond 2)
_MIN_PROJ_GAIN     = 0.40  # (new_edge / old_edge) - 1 ≥ 40 %          (Cond 3)
_MIN_MIS_SUPPORT   = 40    # MIS ≥ 40 — some sharp backing required     (Cond 5)


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_locked_conflict(
    bet,
    sport: str,
    *,
    date_str: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Check whether an existing LOCKED pick conflicts with this new candidate.

    Returns:
        ("clear",   {})       — no conflict; proceed normally
        ("hold",    details)  — conflict; replacement threshold NOT met
        ("replace", details)  — conflict AND all 5 conditions met; supersede
    """
    today    = date_str or now_est().strftime("%Y-%m-%d")
    mkt_norm = market_normalized(getattr(bet, "market", "") or "")
    game_id  = getattr(bet, "game_id", "") or ""

    if not game_id:
        return "clear", {}

    conn = _conn()
    try:
        rows = conn.execute(
            """
            SELECT bet_id, edge_percentage, opening_confidence, wager_details
            FROM   bets
            WHERE  slate_date    = ?
              AND  sport         = ?
              AND  game_id       = ?
              AND  is_locked     = 1
              AND  actual_outcome IS NULL
            ORDER  BY timestamp DESC
            """,
            (today, sport.upper(), game_id),
        ).fetchall()
    finally:
        conn.close()

    # Narrow to same market category by normalising the JSON 'market' field.
    matched_row: sqlite3.Row | None = None
    matched_wd:  dict[str, Any]    = {}
    for r in rows:
        try:
            wd = json.loads(r["wager_details"] or "{}")
        except Exception:
            wd = {}
        if market_normalized(wd.get("market", "")) == mkt_norm:
            matched_row = r
            matched_wd  = wd
            break

    if matched_row is None:
        return "clear", {}

    # ── 5-condition Replacement Threshold ────────────────────────────────────
    old_edge = float(matched_row["edge_percentage"] or 0.0)
    old_conf = float(matched_row["opening_confidence"] or 0.0)
    old_dir  = matched_wd.get("direction", "").lower()

    new_edge = float(getattr(bet, "edge_percentage", 0.0) or 0.0)
    new_conf = float(getattr(bet, "confidence_score", 0.0) or 0.0)
    new_dir  = (getattr(bet, "direction", "") or "").lower()
    new_mis  = int(getattr(bet, "mis_score", 0) or 0)

    # Condition 1: confidence gain ≥ 5 pp
    cond1 = (new_conf - old_conf) >= _MIN_CONF_GAIN_PP
    # Condition 2: edge gain ≥ 25 %
    cond2 = old_edge > 0.0 and (new_edge / old_edge - 1.0) >= _MIN_EDGE_GAIN
    # Condition 3: projection decisiveness — edge gain ≥ 40 % (stricter gate)
    cond3 = old_edge > 0.0 and (new_edge / old_edge - 1.0) >= _MIN_PROJ_GAIN
    # Condition 4: upstream signal confirmation (gate_signals() already passed)
    cond4 = True
    # Condition 5: MIS ≥ 40 (sharp/steam data supports the reversal)
    cond5 = new_mis >= _MIN_MIS_SUPPORT

    all_pass = cond1 and cond2 and cond3 and cond4 and cond5

    details: dict[str, Any] = {
        "existing_bet_id":   matched_row["bet_id"],
        "existing_edge":     old_edge,
        "existing_conf":     old_conf,
        "existing_dir":      old_dir,
        "new_edge":          new_edge,
        "new_conf":          new_conf,
        "new_dir":           new_dir,
        "new_mis":           new_mis,
        "cond1_conf_gain":   cond1,
        "cond2_edge_gain":   cond2,
        "cond3_proj_gain":   cond3,
        "cond4_signal_conf": cond4,
        "cond5_sharp_mis":   cond5,
        "all_conditions":    all_pass,
        "game_id":           game_id,
        "market_norm":       mkt_norm,
    }

    log_stability_warning(sport, today, details)

    return ("replace" if all_pass else "hold"), details


# ---------------------------------------------------------------------------
# Stability Warning log
# ---------------------------------------------------------------------------

def log_stability_warning(
    sport: str,
    date_str: str,
    details: dict[str, Any],
) -> None:
    """Append a Stability Warning entry to data/stability_warnings/{SPORT}_{DATE}.jsonl."""
    os.makedirs(_WARNINGS_DIR, exist_ok=True)
    path = os.path.join(_WARNINGS_DIR, f"{sport.upper()}_{date_str}.jsonl")
    entry = {"timestamp": now_est().isoformat(), "sport": sport.upper(), **details}
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass   # non-fatal; never block the broadcast path


def get_stability_warnings(sport: str, date_str: str) -> list[dict[str, Any]]:
    """Return all stability warnings logged for a given sport + date."""
    path = os.path.join(_WARNINGS_DIR, f"{sport.upper()}_{date_str}.jsonl")
    if not os.path.exists(path):
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return out
