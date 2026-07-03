"""
replay.py

Historical backtest driver. Loops `run_sport_pipeline(sport, as_of_date=...)`
over a range of past dates and writes the resulting picks to an isolated
replay output tree — never the live output/picks.json, output/pick_history.jsonl,
or data/results.db.

Isolation summary (why this is safe to run against live data)
----------------------------------------------------------------
  * odds_client.fetch_todays_candidates() / game_markets.fetch_expanded_game_candidates()
    use the Odds API's /v4/historical/ endpoints when as_of_date is set, and
    write/read slate_cache under data/slate_cache_replay/ instead of the live
    data/slate_cache/ (see core/slate_cache.py's injectable cache_dir).
  * player_props.get_player_prop_candidates() does the same for events/odds,
    and skips PropLine entirely in replay mode (PropLine has no historical
    endpoint — see its docstring).
  * All MLB/WNBA history lookups (data/game_logs.py, core/odds_client.py's
    get_mlb_game_totals_history, core/player_props.py's _mlb_player_stats /
    WNBA boxscore cache) filter to games on-or-before as_of_date and cache
    per-date, so a multi-date loop in one process can't leak one date's
    window into another's.
  * conflict_guardian.check_locked_conflict() is called with skip=True,
    bypassing the live results.db read entirely.
  * run_sport_pipeline() itself never writes to results.db, picks.json, or
    pick_history.jsonl — those all happen in run_pipeline()'s top-level
    run_pipeline() function, which replay.py deliberately does not call.

What replay.py does NOT attempt
--------------------------------
  * True point-in-time odds replay for markets/dates the Odds API's
    historical tier doesn't cover — if a historical snapshot call fails,
    that date/sport/market is skipped and logged, not silently faked.
  * Grading. Replay only generates picks as they *would have been* published;
    pair the output with core/historical_grader.py against final box scores
    to turn this into a scored backtest.

Usage
-----
    python replay.py 2026-05-01 2026-05-31
    python replay.py 2026-06-14                  # single date
    python replay.py 2026-05-01 2026-05-31 --sports MLB
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

from models.contradiction_check import filter_contradictions
from models.line_movement import apply_line_movement_filter
from models.sport_config import MLB, WNBA
from core.composite_confidence_score import compute_ccs

import run_pipeline as _rp   # reuse run_sport_pipeline, log(), _apply_daily_caps

_REPLAY_OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "replay")


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _rank_and_filter(all_results: list[tuple[dict, object, dict]]) -> list[dict]:
    """
    Same post-generation pipeline run_pipeline() applies after
    run_sport_pipeline(): global CCS ranking/tier assignment, contradiction
    check, line movement filter (a no-op in single-pass/replay mode — see
    models/line_movement.py), and per-sport daily caps.

    Deliberately does NOT call append_picks / log_bet_dict / mark_picks_published
    — those are live-DB and live-output writers with no place in a replay.
    """
    raw_picks: list[dict] = []
    if all_results:
        scored = []
        for pick, bd, ld in all_results:
            try:
                ccs, robustness = compute_ccs(bd, ld)
            except Exception as exc:
                _rp.log("warn", "replay", f"{pick.get('pick','?')}: CCS scoring failed ({exc}) -- using fallback")
                ccs = pick["edge_pct"] * 0.6 + pick["confidence"] * 0.4
                robustness = "unknown"
            scored.append((pick, ccs, robustness))

        scored.sort(key=lambda t: t[1], reverse=True)
        nuke_claimed = diamond_claimed = False
        for pick, ccs, robustness in scored:
            pick["ccs_score"] = round(ccs, 2)
            pick["robustness"] = robustness
            if not nuke_claimed:
                pick["tier"] = "Nuke"
                nuke_claimed = True
            elif not diamond_claimed:
                pick["tier"] = "Diamond"
                diamond_claimed = True
            else:
                pick["tier"] = "Gold Standard"
            raw_picks.append(pick)

    cleaned, _ = filter_contradictions(raw_picks)
    final, _ = apply_line_movement_filter(cleaned)
    final = _rp._apply_daily_caps(final)

    for p in final:
        p.setdefault("pick_id", uuid.uuid4().hex[:12])
        p.setdefault("actual_result", None)
        p.setdefault("closing_line", None)
        p.setdefault("clv_pct", None)

    return final


def replay_date(as_of_date: str, sports: list[str]) -> dict:
    """
    Run the full candidate-generation + gatekeeper pipeline for one
    historical date across *sports*. Returns the same {generated_at, picks}
    shape as run_pipeline()'s live output, plus a "replay_date" field and a
    per-sport candidate-count breakdown for calibration/debugging.
    """
    _rp.log("info", "replay", f"=== {as_of_date} ===")
    all_results = []
    sport_counts: dict[str, int] = {}
    for sport in sports:
        try:
            sport_results = _rp.run_sport_pipeline(sport, as_of_date=as_of_date)
            sport_counts[sport] = len(sport_results)
            all_results.extend(sport_results)
        except Exception as exc:
            _rp.log("error", sport, f"replay {as_of_date}: sport pipeline crashed: {type(exc).__name__}: {exc}")
            sport_counts[sport] = 0

    final = _rank_and_filter(all_results)

    for p in final:
        p["slate_date"] = as_of_date

    _rp.log("info", "replay", f"{as_of_date}: {len(final)} pick(s) after full pipeline "
                               f"(raw per-sport: {sport_counts}).")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "replay_date": as_of_date,
        "raw_candidate_counts": sport_counts,
        "picks": final,
    }


def run_replay(start_date: str, end_date: str, sports: list[str] | None = None) -> str:
    """
    Loop replay_date() over [start_date, end_date] inclusive, writing one
    JSON file per date under output/replay/{sport_scope}_{date}.json and an
    aggregate output/replay/summary_{start}_{end}.json index.

    Returns the path to the summary file.
    """
    sports = sports or list(_rp.ENABLED_SPORTS)
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")

    os.makedirs(_REPLAY_OUT_DIR, exist_ok=True)

    index: list[dict] = []
    total_picks = 0
    for d in _daterange(start, end):
        ds = d.isoformat()
        try:
            result = replay_date(ds, sports)
        except Exception as exc:
            _rp.log("error", "replay", f"{ds}: replay_date() crashed: {type(exc).__name__}: {exc}")
            result = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "replay_date": ds,
                "raw_candidate_counts": {},
                "picks": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

        out_path = os.path.join(_REPLAY_OUT_DIR, f"{ds}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        n_picks = len(result["picks"])
        total_picks += n_picks
        index.append({
            "date": ds,
            "n_picks": n_picks,
            "raw_candidate_counts": result.get("raw_candidate_counts", {}),
            "error": result.get("error"),
            "file": os.path.basename(out_path),
        })

    summary = {
        "start_date": start_date,
        "end_date": end_date,
        "sports": sports,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_picks": total_picks,
        "days": index,
    }
    summary_path = os.path.join(_REPLAY_OUT_DIR, f"summary_{start_date}_{end_date}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nReplay complete: {total_picks} pick(s) across {len(index)} date(s).")
    print(f"Per-date output: {_REPLAY_OUT_DIR}/*.json")
    print(f"Summary: {summary_path}")
    n_errors = sum(1 for e in index if e.get("error"))
    if n_errors:
        print(f"\n*** {n_errors} date(s) errored — see per-date files for details. ***")

    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Replay Da-One's pick-generation pipeline over historical dates.")
    parser.add_argument("start_date", help="ISO date, e.g. 2026-05-01")
    parser.add_argument("end_date", nargs="?", default=None, help="ISO date (defaults to start_date for a single day)")
    parser.add_argument("--sports", nargs="+", default=None, help=f"Subset of {_rp.ENABLED_SPORTS} (default: all enabled)")
    args = parser.parse_args()

    end_date = args.end_date or args.start_date
    run_replay(args.start_date, end_date, sports=args.sports)


if __name__ == "__main__":
    main()
