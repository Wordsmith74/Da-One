"""
Test script: demonstrates the DecisionGatekeeper pipeline.

Key scenario: the 17.3% edge / 15.5 pts line from test_simulation_engine.py
is confirmed as a Tier S+ bet.

Run from workspace root:
    python3 core/test_decision_gatekeeper.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.decision_gatekeeper import (
    Bet,
    Tier,
    evaluate_tier,
    check_for_conflicts,
    run_gatekeeper,
)


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def print_bet(bet: Bet, indent: int = 2) -> None:
    pad = " " * indent
    tier_label = bet.tier.value if bet.tier else "DISCARDED"
    flag_label = f"  ⚑ {bet.flag_reason}" if bet.flagged else ""
    print(f"{pad}[{tier_label}] {bet.bet_id}")
    print(f"{pad}  Market:     {bet.market}  {bet.direction.upper()}  {bet.sportsbook_line}")
    print(f"{pad}  Edge:       {bet.edge_percentage:.2f}%   Confidence: {bet.confidence_score:.1f}")
    if flag_label:
        print(f"{pad}{flag_label}")


# ---------------------------------------------------------------------------
# TEST 1 — Tier S+ confirmation: 17.3% edge from simulation output
# ---------------------------------------------------------------------------

def test_tier_nuke_from_simulation() -> None:
    separator("TEST 1 — 17.3% edge / MLB → must classify as Tier Nuke")

    # Values taken directly from test_simulation_engine.py Test 3 output:
    #   OVER 15.5: 67.30%  →  edge_over = +17.30%
    # Confidence derived from: tight posterior (std 3.8), 10 obs → 95.5
    edge_percentage  = 17.30
    confidence_score = 95.5

    tier = evaluate_tier(edge_percentage, confidence_score, sport="MLB")

    print(f"\n  Input:")
    print(f"    edge_percentage:  {edge_percentage}%")
    print(f"    confidence_score: {confidence_score}")
    print(f"    sport:            MLB")
    print(f"\n  Result: Tier {tier.value if tier else 'DISCARDED'}")

    assert tier == Tier.NUKE, (
        f"Expected Nuke but got {tier}. "
        f"edge={edge_percentage}, confidence={confidence_score}"
    )
    print(f"\n  PASS — correctly classified as Tier Nuke")


# ---------------------------------------------------------------------------
# TEST 2 — Full tier boundary checks
# ---------------------------------------------------------------------------

def test_tier_boundaries() -> None:
    separator("TEST 2 — Tier boundary checks")

    cases = [
        # (label,                         edge,  conf,  expected_tier)   # MLB thresholds
        ("Nuke (exact minimum)",           16.0,  85.0,  Tier.NUKE),
        ("Nuke (well above)",              22.0,  95.0,  Tier.NUKE),
        ("Diamond (exact minimum)",        13.0,  78.0,  Tier.DIAMOND),
        ("Diamond (above Dia, below Nuke)",14.0,  82.0,  Tier.DIAMOND),
        ("Diamond fails: conf too low",    13.0,  77.9,  Tier.GOLD),
        ("Gold (exact minimum)",           10.0,  68.0,  Tier.GOLD),
        ("Gold (mid range)",               12.0,  72.0,  Tier.GOLD),
        ("Discard: edge too low",            0.8,  90.0,  None),
        ("Discard: conf too low",          10.0,  67.9,  None),
        ("Discard: both too low",           1.0,  50.0,  None),
    ]

    all_pass = True
    for label, edge, conf, expected in cases:
        result = evaluate_tier(edge, conf, sport="MLB")
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        expected_label = expected.value if expected else "DISCARD"
        result_label   = result.value   if result   else "DISCARD"
        print(f"  [{status}] {label:<35} → expected {expected_label:<8} got {result_label}")

    assert all_pass, "One or more tier boundary checks failed."
    print(f"\n  All boundary checks passed.")


# ---------------------------------------------------------------------------
# TEST 3 — Conflict detection: Over on team total + Under on player points
# ---------------------------------------------------------------------------

def test_conflict_detection_volume() -> None:
    separator("TEST 3 — Conflict: OVER team_total vs UNDER player_points (same team)")

    bet_a = Bet(
        bet_id="SEA_team_total_over",
        team="SEA",
        market="team_total",
        direction="over",
        sportsbook_line=84.5,
        edge_percentage=14.0,
        confidence_score=91.0,
    )
    bet_b = Bet(
        bet_id="SEA_player_pts_under",
        team="SEA",
        market="player_points",
        direction="under",
        sportsbook_line=15.5,
        edge_percentage=17.3,
        confidence_score=95.5,
    )

    # Stamp initial tiers before conflict check
    from core.decision_gatekeeper import stamp_tier
    stamp_tier(bet_a)
    stamp_tier(bet_b)

    print(f"\n  Before conflict check:")
    print_bet(bet_a)
    print_bet(bet_b)

    check_for_conflicts([bet_a, bet_b])

    print(f"\n  After conflict check:")
    print_bet(bet_a)
    print_bet(bet_b)

    assert bet_a.flagged, "bet_a (team_total OVER) should be flagged."
    assert bet_b.flagged, "bet_b (player_points UNDER) should be flagged."

    # Lower-edge bet (bet_a, edge 14.0) should have confidence downgraded
    assert bet_a.confidence_score == 81.0, (
        f"Expected bet_a confidence 81.0 after -10, got {bet_a.confidence_score}"
    )
    # bet_a was Nuke (edge 14 >> 3.5, conf 91 ≥ 85) → after downgrade conf=81
    # Nuke requires conf ≥ 85 → 81 < 85 → falls to Diamond (81 ≥ 78, 14 ≥ 2.0)
    assert bet_a.tier == Tier.DIAMOND, (
        f"Expected bet_a to be downgraded to Diamond, got {bet_a.tier}"
    )
    print(f"\n  PASS — both bets flagged; lower-edge bet confidence downgraded and tier revised.")


# ---------------------------------------------------------------------------
# TEST 4 — Conflict detection: same direction → no conflict
# ---------------------------------------------------------------------------

def test_no_conflict_same_direction() -> None:
    separator("TEST 4 — No conflict: OVER team_total + OVER player_points (same team)")

    bet_a = Bet(
        bet_id="CHI_team_total_over",
        team="CHI",
        market="team_total",
        direction="over",
        sportsbook_line=82.0,
        edge_percentage=9.0,
        confidence_score=80.0,
    )
    bet_b = Bet(
        bet_id="CHI_player_pts_over",
        team="CHI",
        market="player_points",
        direction="over",
        sportsbook_line=18.5,
        edge_percentage=11.0,
        confidence_score=85.0,
    )

    check_for_conflicts([bet_a, bet_b])

    assert not bet_a.flagged, "bet_a should NOT be flagged (same direction)."
    assert not bet_b.flagged, "bet_b should NOT be flagged (same direction)."
    print(f"\n  PASS — same-direction bets on same team correctly not flagged.")


# ---------------------------------------------------------------------------
# TEST 5 — run_gatekeeper full pipeline
# ---------------------------------------------------------------------------

def test_run_gatekeeper_pipeline() -> None:
    separator("TEST 5 — run_gatekeeper() full pipeline")

    # V3.0 note: WNBA game-market bets (Nuke/Diamond) must carry integrity
    # elements so the integrity filter passes — otherwise they are discarded
    # before conflict detection fires.
    _wnba_integrity = {
        "injury_score":            85,   # injury_projection ✓
        "pace":                    98.4, # pace_projection ✓
        "rotation_score":          77,   # rotation_projection ✓
        "market_agreement_score":  65,   # market_agreement ✓
        "rest_days":                2,   # travel_rest_analysis ✓
    }

    b_sea_pts = Bet(
        bet_id="WNBA_SEA_player_pts_over_15.5",
        team="SEA",
        market="player_points",
        direction="over",
        sportsbook_line=15.5,
        edge_percentage=17.3,
        confidence_score=95.5,
    )
    b_sea_pts.raw_result = {}   # prop → no game-market integrity needed

    # game-market bet that conflicts with the player-points OVER above
    b_sea_tt = Bet(
        bet_id="WNBA_SEA_team_total_under_84.5",
        team="SEA",
        market="team_total",
        direction="under",
        sportsbook_line=84.5,
        edge_percentage=12.5,
        confidence_score=89.0,
    )
    b_sea_tt.raw_result = _wnba_integrity   # pass integrity → conflict fires

    b_las = Bet(
        bet_id="WNBA_LAS_spread_over",
        team="LAS",
        market="team_spread",
        direction="over",
        sportsbook_line=-3.5,
        edge_percentage=7.2,
        confidence_score=74.0,
    )
    b_las.raw_result = {}   # Gold Standard → integrity not checked

    # Near-miss (below thresholds but within near-miss window) → goes to flagged
    b_min = Bet(
        bet_id="WNBA_MIN_player_pts_over",
        team="MIN",
        market="player_points",
        direction="over",
        sportsbook_line=12.5,
        edge_percentage=3.0,
        confidence_score=65.0,
    )
    b_min.raw_result = {}

    bets   = [b_sea_pts, b_sea_tt, b_las, b_min]
    result = run_gatekeeper(bets, sport="WNBA")

    approved  = result["approved"]
    flagged   = result["flagged"]
    discarded = result["discarded"]

    print(f"\n  APPROVED  ({len(approved)} bets):")
    for b in approved:
        print_bet(b)

    print(f"\n  FLAGGED   ({len(flagged)} bets) — manual review required:")
    for b in flagged:
        print_bet(b)

    print(f"\n  DISCARDED ({len(discarded)} bets) — suppressed:")
    for b in discarded:
        print_bet(b)

    # player-points OVER conflicts with team_total UNDER (same team, opposite
    # volume signal) → both end up in flagged
    flagged_ids = {b.bet_id for b in flagged}
    assert "WNBA_SEA_player_pts_over_15.5"   in flagged_ids, \
        f"SEA player_pts should be flagged (conflict); flagged={flagged_ids}"
    assert "WNBA_SEA_team_total_under_84.5"  in flagged_ids, \
        f"SEA team_total should be flagged (conflict); flagged={flagged_ids}"

    # LAS spread — different team, no conflict → approved as Gold Standard
    approved_ids = {b.bet_id for b in approved}
    assert "WNBA_LAS_spread_over" in approved_ids, \
        f"LAS spread should be approved; approved={approved_ids}"
    las_bet = next(b for b in approved if b.bet_id == "WNBA_LAS_spread_over")
    assert las_bet.tier == Tier.GOLD, \
        f"LAS spread should be Gold Standard, got {las_bet.tier}"

    # MIN player_pts — below threshold but within near-miss window → flagged
    assert "WNBA_MIN_player_pts_over" in flagged_ids, \
        f"MIN player_pts (near-miss) should be in flagged; flagged={flagged_ids}"

    print(f"\n  PASS — pipeline correctly partitioned all bets.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_tier_nuke_from_simulation()
    test_tier_boundaries()
    test_conflict_detection_volume()
    test_no_conflict_same_direction()
    test_run_gatekeeper_pipeline()

    print(f"\n{'=' * 60}")
    print("  All gatekeeper tests completed successfully.")
    print("=" * 60)
    print()
