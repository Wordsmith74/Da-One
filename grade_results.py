"""
Grades pending (ungraded) picks in output/pick_history.jsonl against real
game outcomes -- this is what turns the history log into a real track record.

Run this SEPARATELY from run_pipeline.py, and LATER -- typically the next
day, after games have finished. A second GitHub Actions job (not yet added
to daily-picks.yml) should run this on a delay; see the note at the bottom
of this file for the workflow snippet to add once this has been tested
against a real run's history.

HONESTY NOTE: this file's live-fetch grading logic is UNTESTED against real
data, same as the rest of the live paths -- this sandbox has no network
access. The structure and field-matching logic is real and consistent with
fetch.py's documented response shapes, but verify against an actual graded
run before trusting the numbers it produces.
"""
import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.cache_history import load_history, update_records, compute_track_record
from models.handicapper_rules import clv_pct


def grade_mlb_f5_pick(record, final_score_lookup):
    """
    final_score_lookup: dict like {"LAD @ SD": {"home_f5_runs": 3, "away_f5_runs": 2}}
    -- F5 runs specifically (through 5 innings), NOT the final game score.
    ESPN's boxscore linescore gives inning-by-inning runs, so this is
    buildable from get_espn_mlb_scoreboard()'s linescore data, but that
    wiring isn't built yet (see TODO below) -- this function takes the
    lookup as a plain argument so it can be tested/used independently of
    however that lookup gets populated.
    """
    game_data = final_score_lookup.get(record["matchup"])
    if game_data is None:
        return None  # can't grade without a result yet

    f5_total = game_data["home_f5_runs"] + game_data["away_f5_runs"]
    line = record["pick_time_line"]
    if f5_total == line:
        return "push"
    actual_over = f5_total > line
    picked_over = record["side"] == "over"
    return "win" if actual_over == picked_over else "loss"


def grade_k_prop_pick(record, actual_ks_lookup):
    """actual_ks_lookup: {pitcher_name: actual_strikeouts}"""
    actual_ks = actual_ks_lookup.get(record["player"])
    if actual_ks is None:
        return None
    line = record["pick_time_line"]
    if actual_ks == line:
        return "push"
    actual_over = actual_ks > line
    picked_over = record["side"] == "over"
    return "win" if actual_over == picked_over else "loss"


def grade_wnba_prop_pick(record, actual_pts_lookup):
    """actual_pts_lookup: {player_name: actual_points}"""
    actual_pts = actual_pts_lookup.get(record["player"])
    if actual_pts is None:
        return None
    line = record["pick_time_line"]
    if actual_pts == line:
        return "push"
    actual_over = actual_pts > line
    picked_over = record["side"] == "over"
    return "win" if actual_over == picked_over else "loss"


def run_grading(closing_odds_lookup=None, mlb_f5_results=None, mlb_k_results=None, wnba_pts_results=None,
                 max_age_days=3):
    """
    All four lookup dicts are OPTIONAL and default to empty -- this function
    is meant to be called with whatever real results you've actually fetched
    (live or pasted in manually); it grades only what it has data for and
    leaves the rest ungraded rather than guessing.

    closing_odds_lookup: {pick_id: closing_odds} -- separate from the result
    lookups because closing-line snapshots need capturing close to game time,
    independent of whether the game has finished yet.

    max_age_days: don't bother trying to grade picks older than this -- a
    pick that's still ungraded after 3 days almost certainly means the
    result-fetching step failed or was never run for that day, not that the
    game is still in progress. Surfacing that gap matters more than silently
    leaving it ungraded forever.
    """
    closing_odds_lookup = closing_odds_lookup or {}
    mlb_f5_results = mlb_f5_results or {}
    mlb_k_results = mlb_k_results or {}
    wnba_pts_results = wnba_pts_results or {}

    history = load_history()
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    updates = {}
    stale_ungraded = 0

    for r in history:
        if r["actual_result"] is not None:
            continue  # already graded, never re-grade

        result = None
        if r["sport"] == "MLB F5":
            result = grade_mlb_f5_pick(r, mlb_f5_results)
        elif r["sport"] == "MLB Ks":
            result = grade_k_prop_pick(r, mlb_k_results)
        elif r["sport"] == "WNBA":
            result = grade_wnba_prop_pick(r, wnba_pts_results)

        update = {}
        if result is not None:
            update["actual_result"] = result
            update["graded_at"] = datetime.now(timezone.utc).isoformat()

        closing_odds = closing_odds_lookup.get(r["pick_id"])
        if closing_odds is not None and r.get("pick_time_odds") is not None:
            update["closing_odds"] = closing_odds
            update["clv_pct"] = round(clv_pct(r["pick_time_odds"], closing_odds), 2)

        if update:
            updates[r["pick_id"]] = update
        elif datetime.fromisoformat(r["generated_at"]) < cutoff:
            stale_ungraded += 1

    if updates:
        update_records(updates)

    print(f"Graded {len(updates)} pick(s).")
    if stale_ungraded:
        print(f"[warn] {stale_ungraded} pick(s) are older than {max_age_days} days and still ungraded -- "
              f"this likely means result-fetching isn't actually running, not that games are still in progress. "
              f"Investigate rather than ignore.")

    print("\n=== Track record (real, graded picks only) ===")
    for sport in (None, "MLB F5", "MLB Ks", "WNBA"):
        label = sport or "ALL SPORTS"
        tr = compute_track_record(sport_filter=sport)
        print(f"[{label}] {tr}")

    return updates


if __name__ == "__main__":
    # No real result-fetching wired in yet -- running this standalone just
    # reports on whatever's already graded (nothing, on a fresh history) and
    # flags stale ungraded picks. Real use requires passing actual lookups,
    # e.g. from a script that calls fetch.get_espn_mlb_scoreboard() and builds
    # mlb_f5_results from the linescore field once that wiring is built --
    # that wiring is the next concrete gap, not yet implemented here.
    run_grading()

# ---- GitHub Actions wiring (NOT yet added to daily-picks.yml) ----
# A second job, run on a delay after games finish, e.g.:
#
#   grade-picks:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4
#       - uses: actions/setup-python@v5
#         with: { python-version: '3.11' }
#       - run: pip install requests pandas lxml pybaseball
#       - run: python3 models/grade_results.py
#       - run: |
#           git add output/pick_history.jsonl
#           git diff --quiet --cached || git commit -m "Grade picks $(date -u +%Y-%m-%d)"
#           git push
#
# Don't add this until grade_results.py's actual result-fetching (not just
# the grading math above) is built and tested against one real day's data.
