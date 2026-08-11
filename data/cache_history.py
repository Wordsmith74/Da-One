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


def _pick_signature(p):
    """
    Identity of a "bet" independent of which pipeline run logged it and
    independent of small odds/edge drift between runs on the same slate.
    Used to de-duplicate re-runs (the pipeline can run many times a day as
    odds refresh) so the same bet doesn't get appended as a brand-new row
    every time, inflating win-rate/backtest numbers by counting one real
    bet several times over.
    """
    return (
        p.get("sport"), p.get("market"), p.get("player"), p.get("side"),
        p.get("pick_time_line"), p.get("matchup"),
    )


def _normalize_model_prob(prob):
    """
    Root-cause fix for the scale bug found while building
    core/probability_calibrator.py: output/pick_history.jsonl's model_prob
    field was being written on two different scales depending on which
    version of the caller wrote it -- some records 0-1 (0.7756), most
    0-100 (86.7) -- while output/shadow_log.jsonl's model_prob (written
    separately by shadow_logger.log_candidate, which the caller in
    run_pipeline.py divides by 100 before passing) has always been 0-1.
    Two logs, same field name, two different conventions -- a real
    footgun for anything that reads either file assuming a single scale.

    Standardizing on 0-1 (the normal probability convention, and what
    shadow_log.jsonl already uses) at this single write chokepoint, rather
    than trusting every current and future caller to remember to divide by
    100 before building the pick dict.
    """
    if prob is None:
        return None
    try:
        prob = float(prob)
    except (TypeError, ValueError):
        return None
    return prob / 100.0 if prob > 1.0 else prob


def append_picks(picks, generated_at=None):
    """
    Call this once per pipeline run, right after writing picks.json, with the
    exact list of final picks that were published. Each pick gets a unique
    pick_id and a recorded snapshot of pick-time odds/line -- this snapshot
    is the whole point: closing_odds and actual_result get filled in LATER
    by grade_results.py, but pick_time values must be locked in NOW, before
    the market moves, or CLV becomes unmeasurable after the fact.

    De-duplication: if a pick with the same (sport, market, player, side,
    line, matchup) was already logged for the same calendar day (by
    generated_at date), it's skipped rather than appended again. The
    pipeline can run multiple times per day as odds move; without this, the
    exact same bet gets a fresh pick_id and a fresh row every single run,
    and backtest.py silently treats each re-log as an independent graded
    result once the game settles -- inflating both sample size and win rate
    for whichever picks happened to survive the most re-runs.
    """
    _ensure_dir()
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    gen_date = generated_at[:10]

    existing = load_history()
    already_logged_today = {
        _pick_signature(r) for r in existing if (r.get("generated_at") or "")[:10] == gen_date
    }

    records = []
    skipped = 0
    for p in picks:
        sig = _pick_signature(p)
        if sig in already_logged_today:
            skipped += 1
            continue
        already_logged_today.add(sig)  # also de-dupe within this same batch
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
            "model_prob": _normalize_model_prob(p.get("model_prob")),
            "edge_pct": p.get("edge_pct"),
            "side_agreement_frac": p.get("side_agreement_frac"),
            "confidence": p.get("confidence"),
            "stake_pct_bankroll": p.get("stake_pct_bankroll"),
            "season_phase": p.get("season_phase"),
            # Fix #6: needed for core/book_exposure.py's per-book
            # concentration report. Previously computed upstream but never
            # persisted anywhere -- see run_pipeline.py's pick dict comment.
            "bookmaker_source": p.get("bookmaker_source"),
            # Sigma was previously only ever logged on stability-REJECTED
            # candidates (never on picks that got published) -- now carried
            # through from run_pipeline.py so it's on every graded pick too.
            "posterior_std": p.get("posterior_std"),
            "posterior_mean": p.get("posterior_mean"),
            "relative_sigma_pct": p.get("relative_sigma_pct"),
            # Fix #2 (probability calibration): both the pre-calibration
            # raw model output and the isotonic-calibrated probability
            # actually used for Kelly staking, so calibration quality can
            # itself be audited/backtested later (compare model_prob_raw
            # vs model_prob_calibrated vs actual_result over time) instead
            # of only ever seeing the post-calibration number.
            "model_prob_raw": p.get("model_prob_raw"),
            "model_prob_calibrated": p.get("model_prob_calibrated"),
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

    if skipped:
        print(f"  [cache_history] skipped {skipped} duplicate pick(s) already logged today")

    return records


def migrate_model_prob_scale():
    """
    One-time cleanup for existing rows written before _normalize_model_prob
    existed: rewrites every record's model_prob to the 0-1 convention in
    place. Idempotent -- safe to run more than once (values already <= 1.0
    pass through unchanged). Run this once after deploying the fix above;
    new rows never need it since append_picks() now normalizes at write
    time.
    """
    records = load_history()
    changed = 0
    for r in records:
        old = r.get("model_prob")
        new = _normalize_model_prob(old)
        if new != old:
            r["model_prob"] = new
            changed += 1
    with open(HISTORY_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return {"n_total": len(records), "n_rescaled": changed}


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
