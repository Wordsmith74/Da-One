"""
Persistent pick-history storage.

This is the file SETUP.md referenced but never had: a durable log of every
pick the pipeline actually generated, with enough fields to later compute
REAL closing-line value and a REAL win rate -- the thing no synthetic
backtest can give you (see models/backtest.py's honesty caveats).

Storage format: JSON Lines (one JSON object per line) at
output/pick_history.jsonl. Chosen over a database deliberately:
  - No new infra/dependency -- this repo already commits output/*.json via
    the GitHub Action, and JSONL appends cleanly without read-modify-write
    races the way a single big JSON array would on concurrent runs.
  - Append-only by design -- grading a pick later UPDATES a record (adds
    closing_odds/actual_result fields), which this module does by reading
    the whole file, rewriting changed records, and never deleting history --
    you always have an audit trail of what the model said before knowing
    the outcome.

This module does NOT fetch results itself -- see models/grade_results.py
for that. This module only knows how to read/write/update the log.
"""
import json
import os
import uuid
from datetime import datetime, timezone

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "pick_history.jsonl")


def _ensure_dir():
    os.makedirs(os.path.dirname(os.path.abspath(HISTORY_PATH)), exist_ok=True)


def append_picks(picks, generated_at=None):
    """
    Call this once per pipeline run, right after writing picks.json, with the
    exact list of final picks that were published. Each pick gets a unique
    pick_id and a recorded snapshot of pick-time odds/line -- this snapshot
    is the whole point: closing_odds and actual_result get filled in LATER
    by grade_results.py, but pick_time values must be locked in NOW, before
    the market moves, or CLV becomes unmeasurable after the fact.
    """
    _ensure_dir()
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    records = []
    for p in picks:
        record = {
            "pick_id": str(uuid.uuid4()),
            "generated_at": generated_at,
            "sport": p.get("sport"),
            "market": p.get("market"),
            "tier": p.get("tier"),
            "matchup": p.get("matchup"),
            "player": p.get("player"),
            "pick": p.get("pick"),
            "side": p.get("side"),
            "pick_time_line": p.get("pick_time_line"),
            "pick_time_odds": p.get("pick_time_odds"),
            "model_prob": p.get("model_prob"),
            "edge_pct": p.get("edge_pct"),
            "side_agreement_frac": p.get("side_agreement_frac"),
            "confidence": p.get("confidence"),
            "stake_pct_bankroll": p.get("stake_pct_bankroll"),
            "season_phase": p.get("season_phase"),
            # Filled in later by grade_results.py -- None until graded.
            "closing_line": None,
            "closing_odds": None,
            "clv_pct": None,
            "actual_result": None,   # "win" | "loss" | "push" | None (ungraded)
            "graded_at": None,
        }
        records.append(record)

    with open(HISTORY_PATH, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return records


def _extract_odds(line_str):
    """pick['line'] is formatted like '4.5 (-110)' -- pull the odds back out
    so history doesn't depend on re-parsing display strings later. Returns
    None if the format doesn't match rather than guessing."""
    if not line_str or "(" not in line_str:
        return None
    try:
        inside = line_str.split("(")[-1].rstrip(")")
        return int(inside)
    except (ValueError, IndexError):
        return None


def load_history():
    """Returns a list of all pick records, oldest first. Empty list if the
    history file doesn't exist yet (first run)."""
    if not os.path.exists(HISTORY_PATH):
        return []
    records = []
    with open(HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def update_records(updated_records_by_id):
    """
    updated_records_by_id: {pick_id: {field: value, ...}} -- merges the given
    fields into existing records and rewrites the whole file. This is the
    only place the file gets rewritten rather than appended to; grading is
    inherently a read-all/write-all operation since JSONL has no in-place
    update primitive.
    """
    records = load_history()
    for r in records:
        if r["pick_id"] in updated_records_by_id:
            r.update(updated_records_by_id[r["pick_id"]])

    with open(HISTORY_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return records


def compute_track_record(sport_filter=None):
    """
    Real (not synthetic) performance summary from graded history only --
    ungraded picks (actual_result is None) are excluded, not counted as
    losses or pushes. THIS is the number that matters once enough real
    picks have been graded -- everything in models/backtest.py is a
    stand-in for this until real history accumulates.
    """
    records = load_history()
    if sport_filter:
        records = [r for r in records if r.get("sport") == sport_filter]

    graded = [r for r in records if r.get("actual_result") in ("win", "loss", "push")]
    if not graded:
        return {
            "n_total_picks": len(records), "n_graded": 0,
            "note": "No graded picks yet -- run models/grade_results.py after games complete.",
        }

    wins = sum(1 for r in graded if r["actual_result"] == "win")
    losses = sum(1 for r in graded if r["actual_result"] == "loss")
    pushes = sum(1 for r in graded if r["actual_result"] == "push")

    clv_values = [r["clv_pct"] for r in graded if r.get("clv_pct") is not None]
    avg_clv = sum(clv_values) / len(clv_values) if clv_values else None

    # ROI assuming flat 1-unit stakes for a simple sanity number; a real ROI
    # should use stake_pct_bankroll per pick, left as a documented simplification.
    units = 0.0
    for r in graded:
        if r["actual_result"] == "push":
            continue
        odds = r.get("pick_time_odds") or -110
        payout = (odds / 100) if odds > 0 else (100 / abs(odds))
        units += payout if r["actual_result"] == "win" else -1.0

    return {
        "n_total_picks": len(records), "n_graded": len(graded),
        "wins": wins, "losses": losses, "pushes": pushes,
        "win_rate_pct": round(wins / (wins + losses) * 100, 1) if (wins + losses) else None,
        "avg_clv_pct": round(avg_clv, 2) if avg_clv is not None else None,
        "flat_stake_units": round(units, 2),
    }
