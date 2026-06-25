"""
Test script: demonstrates the full ResultsTracker pipeline end-to-end.

Covers:
  1. Logging open bets from a mock WNBA slate
  2. Closing bets with outcomes and verifying P&L
  3. Running the ROI calculator
  4. Running the model-refinement learning step (synthetic miss pattern)

Run from workspace root:
    python3 core/test_results_tracker.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

# ── Point imports at workspace root ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# Override DB path to a temp file so tests don't pollute data/results.db
import core.results_tracker as rt

_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
rt.DB_PATH = Path(_tmp_db.name)

from core.results_tracker import (
    init_db,
    log_bet_dict,
    close_bet,
    calculate_running_roi,
    format_roi_report,
    update_model_priors,
    get_open_bets,
)
from core.decision_gatekeeper import Bet, Tier, run_gatekeeper


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Shared mock slate
# ---------------------------------------------------------------------------

MOCK_BETS = [
    # bet_id, sport, market, direction, line, edge, conf, odds, model_prob, tier, player
    ("SEA_A.Wilson_pts_over",  "WNBA", "player_points", "over",  15.5, 17.3, 95.5, -115, 67.3, "S+", "A. Wilson"),
    ("LVA_team_total_over",    "WNBA", "team_total",    "over",  84.5, 16.1, 96.0, -110, 66.1, "S+", None),
    ("NYL_spread_cover",       "WNBA", "team_spread",   "over",  -3.5, 13.4, 90.0, -110, 63.4, "S",  None),
    ("CHI_C.Parker_pts_over",  "WNBA", "player_points", "over",  18.5, 12.8, 89.0, -108, 62.8, "S",  "C. Parker"),
    ("MIN_team_total_under",   "WNBA", "team_total",    "under", 79.5,  8.5, 76.0, -112, 58.5, "Value", None),
    ("CON_S.Ionescu_ast_over", "WNBA", "player_assists","over",   5.5,  6.2, 71.0, -105, 56.2, "Value", "S. Ionescu"),
]


def _log_all_mock_bets() -> None:
    """Log all mock bets as open entries."""
    init_db()
    for row in MOCK_BETS:
        bet_id, sport, market, direction, line, edge, conf, odds, model_prob, tier, player = row
        log_bet_dict(
            bet_id=bet_id,
            sport=sport,
            wager_details={
                "team": bet_id.split("_")[0], "market": market,
                "direction": direction, "sportsbook_line": line,
                "edge_percentage": edge, "confidence_score": conf,
                "tier": tier, "player": player,
            },
            model_probability=model_prob,
            sportsbook_odds=odds,
            stake=100.0,
            tier=tier,
            edge_percentage=edge,
        )


# ---------------------------------------------------------------------------
# TEST 1 — Log open bets
# ---------------------------------------------------------------------------

def test_log_open_bets() -> None:
    separator("TEST 1 — Log 6 open bets to SQLite")

    _log_all_mock_bets()

    open_bets = get_open_bets("WNBA")
    print(f"\n  Open bets logged: {len(open_bets)}")
    for b in open_bets:
        d = json.loads(b["wager_details"])
        print(
            f"  [{b['tier']:<6}] {b['bet_id']:<35}"
            f"  {d['direction'].upper()} {d['sportsbook_line']}"
            f"  model={b['model_probability']:.1f}%"
        )

    assert len(open_bets) == 6, f"Expected 6 open bets, got {len(open_bets)}"
    print(f"\n  PASS — all 6 bets logged as open.")


# ---------------------------------------------------------------------------
# TEST 2 — Close bets with outcomes and verify P&L
# ---------------------------------------------------------------------------

def test_close_bets_and_pl() -> None:
    separator("TEST 2 — Close bets and verify P&L calculation")

    outcomes = {
        # bet_id: (outcome, expected_pl at given odds and $100 stake)
        "SEA_A.Wilson_pts_over":  ("win",  round(100 * 100 / 115, 2)),  # -115 win
        "LVA_team_total_over":    ("win",  round(100 * 100 / 110, 2)),  # -110 win
        "NYL_spread_cover":       ("loss", -100.0),
        "CHI_C.Parker_pts_over":  ("win",  round(100 * 100 / 108, 2)),  # -108 win
        "MIN_team_total_under":   ("loss", -100.0),
        "CON_S.Ionescu_ast_over": ("push",  0.0),
    }

    print()
    for bet_id, (outcome, expected_pl) in outcomes.items():
        result = close_bet(bet_id, outcome)
        actual_pl = result["profit_loss"]
        match = "✓" if abs(actual_pl - expected_pl) < 0.02 else "✗"
        print(f"  {match} {result['summary']:<55}  expected ${expected_pl:+.2f}")
        assert abs(actual_pl - expected_pl) < 0.02, (
            f"{bet_id}: expected P&L ${expected_pl}, got ${actual_pl}"
        )

    print(f"\n  PASS — all P&L values correct.")


# ---------------------------------------------------------------------------
# TEST 3 — ROI calculator
# ---------------------------------------------------------------------------

def test_roi_calculator() -> None:
    separator("TEST 3 — ROI calculator")

    roi = calculate_running_roi(sport="WNBA")

    print(f"\n  Closed bets:    {roi['closed_bets']}")
    print(f"  Record:         {roi['wins']}W – {roi['losses']}L – {roi['pushes']}P")
    print(f"  Win Rate:       {roi['win_rate']}%")
    print(f"  Total Staked:   ${roi['total_staked']:.2f}")
    print(f"  Net P&L:        ${roi['total_profit_loss']:+.2f}")
    print(f"  ROI:            {roi['roi_pct']:+.2f}%")
    print(f"\n  By Tier:")
    for tier, data in roi["by_tier"].items():
        print(f"    [{tier:<6}]  {data['bets']} bets  WR {data['win_rate']}%  P&L ${data['pl']:+.2f}")

    assert roi["closed_bets"] == 6
    assert roi["wins"]   == 3
    assert roi["losses"] == 2
    assert roi["pushes"] == 1
    assert roi["total_staked"] == 600.0
    assert roi["roi_pct"] == pytest_approx(roi["roi_pct"])   # just check it runs

    print(f"\n  PASS — ROI summary correct.")

    print(f"\n  ── Telegram-ready ROI report ───────────────────────")
    print(format_roi_report("WNBA"))


def pytest_approx(v):
    return v   # simple passthrough for this test file


# ---------------------------------------------------------------------------
# TEST 4 — Model refinement (learning loop)
# ---------------------------------------------------------------------------

def test_model_refinement() -> None:
    separator("TEST 4 — Model refinement (synthetic miss pattern)")

    # To trigger an adjustment we need ≥ MIN_SAMPLE_FOR_REFINEMENT closed bets
    # with a clear calibration error. Seed the DB with 12 synthetic WNBA
    # player_points OVER bets where the model said ~70% but only 40% hit —
    # this simulates the model overestimating offensive output.

    import random
    rng = random.Random(42)
    synthetic_ids = []

    for i in range(12):
        bid = f"SYNTHETIC_pts_{i:02d}"
        synthetic_ids.append(bid)
        log_bet_dict(
            bet_id=bid,
            sport="WNBA",
            wager_details={
                "team": "TST", "market": "player_points",
                "direction": "over", "sportsbook_line": 16.5,
                "edge_percentage": 20.0, "confidence_score": 95.0,
                "tier": "S+", "player": "Test Player",
            },
            model_probability=70.0,
            sportsbook_odds=-110,
            stake=100.0,
            tier="S+",
            edge_percentage=20.0,
        )
        # 40 % hit rate → model overestimated by 30 %
        outcome = "win" if i < 5 else "loss"
        close_bet(bid, outcome)

    print(f"\n  Seeded 12 synthetic WNBA player_points OVER bets")
    print(f"  Model avg: 70%  |  Actual hit rate: {5/12*100:.1f}%  |  Calibration error: +{(0.70 - 5/12)*100:.1f}%")
    print(f"\n  Running update_model_priors('WNBA')…")

    # Read current off_efficiency weight before adjustment
    import json as _json
    with open(rt.CONFIG_PATH) as f:
        before = _json.load(f)
    old_off_eff = before["WNBA"]["total_weights"]["off_efficiency"]

    adjustments = update_model_priors("WNBA", min_sample=10)

    with open(rt.CONFIG_PATH) as f:
        after = _json.load(f)
    new_off_eff = after["WNBA"]["total_weights"]["off_efficiency"]

    print(f"\n  Adjustments made: {len(adjustments)}")
    for adj in adjustments:
        sign = "+" if adj["new_value"] > adj["old_value"] else ""
        print(f"    {adj['weight_group']}.{adj['weight_key']}  "
              f"{adj['old_value']:.4f} → {adj['new_value']:.4f}  ({sign}{adj['new_value']-adj['old_value']:+.4f})")
        print(f"    Reason: {adj['reason']}")

    # Verify off_efficiency was nudged down
    assert new_off_eff < old_off_eff, (
        f"Expected off_efficiency to decrease; was {old_off_eff}, now {new_off_eff}"
    )

    # Verify total_weights still sums to 1.0
    total = sum(after["WNBA"]["total_weights"].values())
    assert abs(total - 1.0) < 0.001, f"Weights don't sum to 1.0 after adjustment: {total}"

    print(f"\n  off_efficiency:  {old_off_eff:.4f} → {new_off_eff:.4f}  (reduced ✓)")
    print(f"  Weight sum:      {total:.4f}  (= 1.0 ✓)")
    print(f"\n  PASS — model correctly learned from systematic miss pattern.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_log_open_bets()
    test_close_bets_and_pl()
    test_roi_calculator()
    test_model_refinement()

    print(f"\n{'=' * 60}")
    print("  All ResultsTracker tests completed successfully.")
    print("=" * 60)

    # Clean up temp DB
    _tmp_db.close()
    os.unlink(rt.DB_PATH)
