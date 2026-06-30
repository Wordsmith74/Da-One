"""
backtest.py — Grades historical picks against real outcomes and produces
calibration reports, so confidence/edge thresholds can be re-tuned based
on evidence instead of guesswork.

Two separate jobs, kept separate on purpose:

  1. grade_pending()   -- fills in actual_result for picks that don't have
                           one yet, using ESPN scoreboard data (MLB/WNBA).
  2. run_backtest()     -- reads ALL graded picks (from output/pick_history.jsonl
                           AND output/shadow_log.jsonl) and computes:
                             - reliability curve (confidence bucket -> actual win rate)
                             - Brier score (lower is better-calibrated)
                             - over/under split + win rate by side
                             - win rate by edge_pct bucket
                             - win rate by sport/market

Run directly:
    python3 backtest.py grade      # back-fill results for finished games
    python3 backtest.py report     # print the calibration report
    python3 backtest.py both       # grade, then report

IMPORTANT LIMITATION: grade_pending() grades WNBA player points (via ESPN's
scoreboard box-score leaders) and MLB pitcher strikeout props (via the
official MLB Stats API game log, matched to the pick's exact game date). It
does NOT yet grade MLB "F5" (first 5 innings) totals, which need
inning-by-inning linescore data that isn't in either source used here --
that needs ESPN's summary endpoint (event id -> /summary) or a different
box-score source. F5 picks are graded as "ungraded" (skipped, left null)
until that's wired in -- they are NOT silently scored wrong, they're just
left out of the report rather than counted incorrectly.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

PICK_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "output", "pick_history.jsonl")
SHADOW_LOG_PATH = os.path.join(os.path.dirname(__file__), "output", "shadow_log.jsonl")


def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _rewrite_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Grading: back-fill actual_result for picks where it's still null
# ---------------------------------------------------------------------------

def _grade_wnba_points(pick, scoreboard_events):
    """
    Looks up the player's final points from ESPN's boxscore leaders, if
    present in the scoreboard payload, and compares against the pick's
    line/side. Returns "win" / "loss" / "push" / None (can't determine).
    """
    player = (pick.get("player") or "").lower()
    line = pick.get("market_line") or pick.get("market_pts_line")
    side = (pick.get("side") or pick.get("pick") or "").lower()
    if not player or line is None or not side:
        return None

    for event in scoreboard_events:
        for comp in event.get("competitions", []):
            for leader_group in comp.get("leaders", []):
                if leader_group.get("name") != "points":
                    continue
                for leader in leader_group.get("leaders", []):
                    athlete = (leader.get("athlete", {}) or {}).get("displayName", "").lower()
                    if athlete and athlete == player:
                        try:
                            actual_pts = float(leader.get("value"))
                        except (TypeError, ValueError):
                            return None
                        if actual_pts == line:
                            return "push"
                        went_over = actual_pts > line
                        if "over" in side:
                            return "win" if went_over else "loss"
                        if "under" in side:
                            return "loss" if went_over else "win"
    return None  # player not found in this payload -- can't grade yet


def grade_pending(sport_filter: str | None = None) -> dict:
    """
    Scans pick_history.jsonl and shadow_log.jsonl for ungraded picks and
    attempts to grade them. Returns a summary dict of what was graded /
    skipped. Gradeable right now: WNBA player-points props, MLB pitcher
    strikeout props. F5 totals still need a different data source (see
    module docstring) and are skipped, not faked.
    """
    from fetch import get_espn_wnba_scoreboard, get_mlb_player_game_logs

    summary = {"graded": 0, "skipped_ungradeable_market": 0, "skipped_no_match": 0}

    history = _load_jsonl(PICK_HISTORY_PATH)
    shadow = _load_jsonl(SHADOW_LOG_PATH)

    def _date_of(p):
        ts = p.get("generated_at") or p.get("logged_at") or ""
        return ts[:10] if ts else None

    # WNBA points: grouped by date (one scoreboard fetch covers every pick
    # that day). MLB Ks: grouped by pitcher (one game-log fetch covers every
    # pending pick for that pitcher, regardless of date, since the log
    # itself is matched by date per-pick afterward).
    wnba_pending_by_date = defaultdict(list)
    mlb_k_pending_by_pitcher = defaultdict(list)

    for record_list, is_shadow in ((history, False), (shadow, True)):
        for p in record_list:
            if p.get("actual_result") is not None:
                continue
            sport = (p.get("sport") or "").upper()
            if sport_filter and sport_filter.upper() not in sport:
                continue

            if "WNBA" in sport:
                d = _date_of(p)
                if d:
                    wnba_pending_by_date[d].append((p, is_shadow))
            elif "MLB" in sport and "K" in sport:  # "MLB Ks"
                pitcher = p.get("player")
                if pitcher:
                    mlb_k_pending_by_pitcher[pitcher].append((p, is_shadow))
                else:
                    summary["skipped_no_match"] += 1
            else:
                summary["skipped_ungradeable_market"] += 1

    # ── WNBA points ──────────────────────────────────────────────────────────
    for date_str, pending in wnba_pending_by_date.items():
        espn_date = date_str.replace("-", "")
        try:
            events = get_espn_wnba_scoreboard(espn_date)
        except Exception as e:
            print(f"  [warn] couldn't fetch WNBA scoreboard for {date_str}: {e}")
            continue
        for p, is_shadow in pending:
            result = _grade_wnba_points(p, events)
            if result is None:
                summary["skipped_no_match"] += 1
                continue
            p["actual_result"] = result
            p["graded_at"] = datetime.now(timezone.utc).isoformat()
            summary["graded"] += 1

    # ── MLB pitcher strikeouts ────────────────────────────────────────────────
    for pitcher, pending in mlb_k_pending_by_pitcher.items():
        # Need a season per fetch call -- group this pitcher's pending picks
        # by year (almost always one year, but be correct across a
        # season boundary) and fetch each season's log once.
        seasons_needed = sorted({(_date_of(p) or "")[:4] for p, _ in pending if _date_of(p)})
        logs_by_season = {}
        for season in seasons_needed:
            if not season:
                continue
            try:
                # limit=30 -- enough to reach back through a normal rotation
                # turn count for a season-to-date pending backlog.
                logs_by_season[season] = get_mlb_player_game_logs(
                    player_id=None, season=int(season), limit=30, pitcher_name=pitcher
                )
            except Exception as e:
                print(f"  [warn] couldn't fetch MLB game log for {pitcher} ({season}): {e}")
                logs_by_season[season] = []

        for p, is_shadow in pending:
            season = (_date_of(p) or "")[:4]
            game_logs = logs_by_season.get(season, [])
            result = _grade_mlb_k_prop(p, game_logs)
            if result is None:
                summary["skipped_no_match"] += 1
                continue
            p["actual_result"] = result
            p["graded_at"] = datetime.now(timezone.utc).isoformat()
            summary["graded"] += 1

    _rewrite_jsonl(PICK_HISTORY_PATH, history)
    _rewrite_jsonl(SHADOW_LOG_PATH, shadow)
    return summary


def _grade_mlb_k_prop(pick, game_logs):
    """
    Looks up the pitcher's strikeout total for the specific game date the
    pick was generated for (matched via the "date" field added to
    get_mlb_player_game_logs in fetch.py), and compares against the pick's
    line/side. Returns "win" / "loss" / "push" / None (can't determine --
    e.g. game not found in the log window, postponed, or pitcher didn't
    actually appear that day).
    """
    line = pick.get("market_line") or pick.get("pick_time_line")
    side = (pick.get("side") or "").lower()
    pick_date = (pick.get("generated_at") or pick.get("logged_at") or "")[:10]
    if line is None or not side or not pick_date:
        return None

    for log_entry in game_logs:
        if log_entry.get("date") != pick_date:
            continue
        actual_ks = log_entry.get("strikeouts")
        if actual_ks is None:
            return None
        if actual_ks == line:
            return "push"
        went_over = actual_ks > line
        if "over" in side:
            return "win" if went_over else "loss"
        if "under" in side:
            return "loss" if went_over else "win"
        return None
    return None  # no game log entry on that exact date -- can't grade yet


# ---------------------------------------------------------------------------
# Calibration report: the actual "backtest"
# ---------------------------------------------------------------------------

def _confidence_bucket(conf):
    if conf is None:
        return "unknown"
    edges = [0, 60, 70, 80, 90, 95, 101]
    labels = ["<60", "60-70", "70-80", "80-90", "90-95", "95-100"]
    for i in range(len(edges) - 1):
        if edges[i] <= conf < edges[i + 1]:
            return labels[i]
    return "unknown"


def _edge_bucket(edge_pct):
    if edge_pct is None:
        return "unknown"
    e = abs(edge_pct)
    if e < 5:
        return "0-5"
    if e < 10:
        return "5-10"
    if e < 15:
        return "10-15"
    if e < 25:
        return "15-25"
    return "25+"


def run_backtest(min_sample_size: int = 5) -> dict:
    """
    Computes calibration metrics across every graded pick in
    pick_history.jsonl + shadow_log.jsonl. Buckets with fewer than
    min_sample_size graded picks are reported but flagged low-confidence
    so you don't over-react to a 2-pick "100% win rate" bucket.
    """
    history = _load_jsonl(PICK_HISTORY_PATH)
    shadow = _load_jsonl(SHADOW_LOG_PATH)
    all_records = history + shadow

    graded = [r for r in all_records if r.get("actual_result") in ("win", "loss", "push")]

    if not graded:
        return {"error": "No graded picks found. Run grade_pending() first, "
                          "or back-fill actual_result manually."}

    def win_rate(records):
        decided = [r for r in records if r["actual_result"] in ("win", "loss")]
        if not decided:
            return None, 0
        wins = sum(1 for r in decided if r["actual_result"] == "win")
        return round(wins / len(decided) * 100, 1), len(decided)

    # 1. Reliability curve: stated confidence vs actual win rate
    by_conf = defaultdict(list)
    for r in graded:
        by_conf[_confidence_bucket(r.get("confidence"))].append(r)
    reliability_curve = {}
    for bucket, records in sorted(by_conf.items()):
        wr, n = win_rate(records)
        reliability_curve[bucket] = {
            "actual_win_rate_pct": wr, "n": n,
            "low_sample_warning": n < min_sample_size,
        }

    # 2. Brier score (mean squared error between model_prob and outcome)
    brier_terms = []
    for r in graded:
        if r["actual_result"] not in ("win", "loss"):
            continue
        p = r.get("model_prob")
        if p is None:
            continue
        outcome = 1.0 if r["actual_result"] == "win" else 0.0
        brier_terms.append((float(p) - outcome) ** 2)
    brier_score = round(sum(brier_terms) / len(brier_terms), 4) if brier_terms else None

    # 3. Over/under split + win rate by side
    by_side = defaultdict(list)
    for r in graded:
        side = (r.get("side") or "").lower() or "unknown"
        by_side[side].append(r)
    side_report = {}
    for side, records in by_side.items():
        wr, n = win_rate(records)
        side_report[side] = {"count": len(records), "win_rate_pct": wr, "n_decided": n}

    # 4. Win rate by edge_pct bucket -- does higher stated edge actually win more?
    by_edge = defaultdict(list)
    for r in graded:
        by_edge[_edge_bucket(r.get("edge_pct"))].append(r)
    edge_report = {}
    for bucket, records in sorted(by_edge.items()):
        wr, n = win_rate(records)
        edge_report[bucket] = {"actual_win_rate_pct": wr, "n": n,
                                "low_sample_warning": n < min_sample_size}

    # 5. Win rate by sport/market
    by_sport = defaultdict(list)
    for r in graded:
        by_sport[r.get("sport", "unknown")].append(r)
    sport_report = {}
    for sport, records in by_sport.items():
        wr, n = win_rate(records)
        sport_report[sport] = {"count": len(records), "win_rate_pct": wr, "n_decided": n}

    return {
        "n_graded_total": len(graded),
        "reliability_curve": reliability_curve,
        "brier_score": brier_score,
        "brier_score_note": "0 = perfect calibration, 0.25 = no better than "
                             "always guessing 50%, closer to 0 is better.",
        "side_report": side_report,
        "edge_bucket_report": edge_report,
        "sport_report": sport_report,
    }


def print_report(report: dict):
    if "error" in report:
        print(report["error"])
        return

    print(f"\n=== Backtest / Calibration Report ({report['n_graded_total']} graded picks) ===\n")

    print(f"Brier score: {report['brier_score']}  ({report['brier_score_note']})\n")

    print("Reliability curve (stated confidence vs actual win rate):")
    print(f"  {'bucket':<10}{'actual win%':<14}{'n':<6}")
    for bucket, d in report["reliability_curve"].items():
        flag = "  (low sample)" if d["low_sample_warning"] else ""
        wr = d["actual_win_rate_pct"]
        print(f"  {bucket:<10}{(str(wr)+'%' if wr is not None else '-'):<14}{d['n']:<6}{flag}")

    print("\nWin rate by side:")
    for side, d in report["side_report"].items():
        wr = d["win_rate_pct"]
        print(f"  {side:<10} count={d['count']:<5} win%={wr if wr is not None else '-'} (n={d['n_decided']})")

    print("\nWin rate by edge_pct bucket (does bigger stated edge actually win more?):")
    for bucket, d in report["edge_bucket_report"].items():
        flag = "  (low sample)" if d["low_sample_warning"] else ""
        wr = d["actual_win_rate_pct"]
        print(f"  {bucket:<10}{(str(wr)+'%' if wr is not None else '-'):<10}n={d['n']}{flag}")

    print("\nWin rate by sport:")
    for sport, d in report["sport_report"].items():
        wr = d["win_rate_pct"]
        print(f"  {sport:<12} count={d['count']:<5} win%={wr if wr is not None else '-'} (n={d['n_decided']})")
    print()


def suggest_recalibration(min_sample_size: int = 8) -> dict:
    """
    Reads the calibration report and turns it into concrete, specific
    threshold suggestions -- NOT auto-applied. This file doesn't have
    access to sport_config.py (never uploaded/seen), so it can't safely
    edit MLB[...]/WNBA[...] thresholds directly without guessing at a file
    structure it hasn't verified. Instead it writes
    output/calibration_suggestions.json with the recommended changes and
    the evidence behind each one, for a human to apply.

    Logic (deliberately conservative -- only suggests a change when there's
    enough graded sample to trust it):
      - If actual win rate in a confidence bucket is >15 points below the
        bucket's own midpoint (e.g. "90-95" bucket actually winning 60%),
        suggest raising MIN_CONFIDENCE_PCT to the next bucket's floor.
      - If the lowest edge bucket (0-5%) has a graded win rate at or below
        52% (no real edge over a coin flip), suggest raising the edge
        threshold to exclude that bucket.
      - If a side (over/under) shows a lopsided split, flag it as a
        possible selection-bias signal worth investigating upstream (same
        shape as the Whiff%/SwStr% unit-mismatch bug found earlier) rather
        than a threshold to tune.
    """
    report = run_backtest(min_sample_size=min_sample_size)
    if "error" in report:
        return report

    suggestions = []

    _bucket_midpoints = {
        "<60": 55, "60-70": 65, "70-80": 75, "80-90": 85, "90-95": 92.5, "95-100": 97.5,
    }
    _bucket_order = ["<60", "60-70", "70-80", "80-90", "90-95", "95-100"]
    _bucket_floor = {"<60": 0, "60-70": 60, "70-80": 70, "80-90": 80, "90-95": 90, "95-100": 95}

    for bucket, d in report["reliability_curve"].items():
        if d["n"] < min_sample_size or d["actual_win_rate_pct"] is None:
            continue
        midpoint = _bucket_midpoints.get(bucket)
        if midpoint is None:
            continue
        gap = midpoint - d["actual_win_rate_pct"]
        if gap > 15:
            idx = _bucket_order.index(bucket) if bucket in _bucket_order else -1
            next_floor = _bucket_floor.get(_bucket_order[idx + 1]) if 0 <= idx < len(_bucket_order) - 1 else None
            suggestions.append({
                "type": "raise_confidence_floor",
                "evidence": f"Confidence bucket '{bucket}' (stated ~{midpoint}%) actually "
                            f"won {d['actual_win_rate_pct']}% over {d['n']} graded picks "
                            f"-- {gap:.0f}pt overconfidence gap.",
                "suggestion": (
                    f"Consider raising MIN_CONFIDENCE_PCT to {next_floor} in run_pipeline.py "
                    f"(currently 70.0) to exclude this bucket."
                    if next_floor else
                    "This is already the top bucket -- the overconfidence isn't fixable by "
                    "raising a floor; the confidence calculation itself needs review."
                ),
            })

    low_edge = report["edge_bucket_report"].get("0-5")
    if low_edge and low_edge["n"] >= min_sample_size and low_edge["actual_win_rate_pct"] is not None:
        if low_edge["actual_win_rate_pct"] <= 52:
            suggestions.append({
                "type": "raise_edge_threshold",
                "evidence": f"0-5% edge bucket won only {low_edge['actual_win_rate_pct']}% "
                            f"over {low_edge['n']} graded picks -- no real edge over a coin flip.",
                "suggestion": "Consider raising MLB['edge_threshold_pct'] / "
                              "WNBA['edge_threshold_pct'] above 5.0 to exclude this bucket.",
            })

    side_report = report["side_report"]
    over_n = side_report.get("over", {}).get("count", 0)
    under_n = side_report.get("under", {}).get("count", 0)
    total_sides = over_n + under_n
    if total_sides >= min_sample_size * 2:
        over_share = over_n / total_sides
        if over_share >= 0.80 or over_share <= 0.20:
            suggestions.append({
                "type": "investigate_selection_bias",
                "evidence": f"{over_n} over vs {under_n} under ({over_share:.0%} over) "
                            f"-- lopsided enough to suggest a systematic skew upstream, "
                            f"not just market conditions (same shape of issue as the "
                            f"Whiff%/SwStr% unit-mismatch bug found in strikeout_matchup.py "
                            f"earlier).",
                "suggestion": "Audit the projection layer's baseline constants and clamp "
                              "symmetry for the skewed side before touching any threshold.",
            })

    out_path = os.path.join(os.path.dirname(__file__), "output", "calibration_suggestions.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "n_graded": report["n_graded_total"], "suggestions": suggestions}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def print_suggestions(payload: dict):
    if "error" in payload:
        print(payload["error"])
        return
    sugg = payload["suggestions"]
    print(f"\n=== Recalibration Suggestions ({payload['n_graded']} graded picks) ===\n")
    if not sugg:
        print("No suggestions -- either everything is within tolerance, or there isn't "
              "enough graded sample yet to trust a change. Run `grade` again after more "
              "picks have settled.\n")
        return
    for s in sugg:
        print(f"[{s['type']}]")
        print(f"  Evidence:    {s['evidence']}")
        print(f"  Suggestion:  {s['suggestion']}\n")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "report"
    if action in ("grade", "both", "refine"):
        result = grade_pending()
        print(f"Graded {result['graded']} pick(s). "
              f"Skipped {result['skipped_ungradeable_market']} (ungradeable market type), "
              f"{result['skipped_no_match']} (no matching box score data yet).")
    if action in ("report", "both"):
        print_report(run_backtest())
    if action == "refine":
        print_suggestions(suggest_recalibration())
