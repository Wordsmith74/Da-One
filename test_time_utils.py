"""
Test script: validates all datetime / timezone changes across the engine.

Covers:
  1. core/time_utils.py  — conversion, formatting, comparison helpers
  2. core/api_connector.py — timestamp normalisation from every input format
  3. core/decision_orchestrator.py — UTC-aware game_time methods
  4. output/telegram_formatter.py — EST game time displayed on bet cards
  5. core/results_tracker.py — ROI report shows ET instead of UTC

Run from workspace root:
    python3 core/test_time_utils.py
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.time_utils import (
    UTC, EST,
    convert_to_est,
    localize_utc,
    now_utc,
    now_est,
    format_est,
    format_est_short,
    format_est_date,
    format_utc_iso,
    is_in_future,
    is_within_hours,
    utc_diff_minutes,
)
from core.api_connector import (
    normalize_api_timestamp,
    parse_game_from_api,
    filter_upcoming_games,
    GameInfo,
)
from core.decision_orchestrator import DecisionOrchestrator


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# TEST 1 — convert_to_est: UTC → EST / EDT
# ---------------------------------------------------------------------------

def test_convert_to_est() -> None:
    separator("TEST 1 — convert_to_est: UTC → America/New_York")

    cases = [
        # (utc_input,                              expected_hour_et, expected_abbr)
        (datetime(2026, 1, 15, 18, 0, tzinfo=UTC),  13, "EST"),   # winter −5
        (datetime(2026, 7, 15, 18, 0, tzinfo=UTC),  14, "EDT"),   # summer −4
        # 2026 DST starts Mar 8 (2nd Sun of Mar). Clocks jump 2 AM EST → 3 AM EDT,
        # which is 07:00 UTC. So 06:00 UTC is 1 AM EST (before spring-forward).
        (datetime(2026, 3,  8,  6, 0, tzinfo=UTC),   1, "EST"),   # 1 hr before spring-forward
        # At 07:00 UTC the jump already occurred → 3:00 AM EDT, not 2 AM.
        (datetime(2026, 3,  8,  7, 0, tzinfo=UTC),   3, "EDT"),   # exactly at spring-forward
        (datetime(2026, 3,  8, 12, 0, tzinfo=UTC),   8, "EDT"),   # well into DST, clearly EDT
    ]

    for utc_dt, expected_hour, abbr in cases:
        est_dt = convert_to_est(utc_dt)
        assert est_dt.hour == expected_hour, (
            f"Expected hour {expected_hour} ({abbr}), got {est_dt.hour} for {utc_dt}"
        )
        print(f"  UTC {utc_dt.strftime('%b %d %H:%M')} → {est_dt.strftime('%b %d %H:%M %Z')}  [{abbr}] ✓")

    # Naive datetime assumed UTC
    naive = datetime(2026, 6, 1, 20, 0)   # 8 PM UTC, summer → 4 PM EDT
    est_naive = convert_to_est(naive)
    assert est_naive.hour == 16, f"Naive dt: expected hour 16, got {est_naive.hour}"
    print(f"  Naive {naive} → {est_naive.strftime('%H:%M %Z')} (assumed UTC) ✓")

    print(f"\n  PASS — DST transitions handled correctly.")


# ---------------------------------------------------------------------------
# TEST 2 — localize_utc: stamp naive and re-normalise aware
# ---------------------------------------------------------------------------

def test_localize_utc() -> None:
    separator("TEST 2 — localize_utc: always produces UTC-aware datetime")

    naive  = datetime(2026, 5, 31, 23, 0)
    aware_est = datetime(2026, 5, 31, 19, 0, tzinfo=EST)

    result_naive = localize_utc(naive)
    result_aware = localize_utc(aware_est)

    assert result_naive.tzinfo == UTC
    assert result_naive.hour   == 23
    assert result_aware.tzinfo == UTC
    assert result_aware.hour   == 23   # 7 PM EDT = 11 PM UTC

    print(f"  Naive  {naive}           → {result_naive.isoformat()} ✓")
    print(f"  EST    {aware_est.isoformat()} → {result_aware.isoformat()} ✓")
    print(f"\n  PASS — localize_utc normalises both naive and aware inputs.")


# ---------------------------------------------------------------------------
# TEST 3 — normalize_api_timestamp: every inbound format
# ---------------------------------------------------------------------------

def test_normalize_api_timestamp() -> None:
    separator("TEST 3 — normalize_api_timestamp: all API timestamp formats")

    expected_utc = datetime(2026, 5, 31, 23, 0, 0, tzinfo=UTC)
    unix_ts = int(expected_utc.timestamp())   # derive from expected, never hardcode

    cases = [
        ("ISO with Z",         "2026-05-31T23:00:00Z"),
        ("ISO with offset",    "2026-05-31T19:00:00-04:00"),   # EDT → same UTC moment
        ("ISO naive (UTC)",    "2026-05-31T23:00:00"),
        ("Unix int",           unix_ts),
        ("Unix float",         float(unix_ts)),
        ("Aware datetime",     datetime(2026, 5, 31, 19, 0, tzinfo=EST)),
        ("Naive datetime",     datetime(2026, 5, 31, 23, 0)),
    ]

    for label, raw in cases:
        result = normalize_api_timestamp(raw)
        assert result.tzinfo is not None, f"{label}: result has no tzinfo"
        assert result.tzinfo == UTC or result.utcoffset() == timedelta(0), \
            f"{label}: not UTC — {result}"
        diff = abs((result - expected_utc).total_seconds())
        assert diff < 2, f"{label}: expected {expected_utc}, got {result} (diff {diff}s)"
        print(f"  {label:<25} → {result.isoformat()} ✓")

    print(f"\n  PASS — all timestamp formats normalised to UTC.")


# ---------------------------------------------------------------------------
# TEST 4 — parse_game_from_api: GameInfo with EST display
# ---------------------------------------------------------------------------

def test_parse_game_from_api() -> None:
    separator("TEST 4 — parse_game_from_api: GameInfo display in EST")

    raw = {
        "id":          "WNBA_20260531_SEA_LVA",
        "home_team":   "SEA",
        "away_team":   "LVA",
        "game_time":   "2026-05-31T23:00:00Z",   # 7 PM EDT
        "venue":       "Climate Pledge Arena",
    }

    game = parse_game_from_api(raw, "WNBA")

    assert game.game_time_utc.tzinfo == UTC
    assert game.game_time_utc.hour   == 23
    assert game.game_time_est.hour   == 19      # 7 PM EDT
    assert "PM ET" in game.display_time
    assert "2026" in game.display_date

    print(f"  game_time_utc:   {game.game_time_utc.isoformat()}")
    print(f"  game_time_est:   {game.game_time_est.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"  display_time:    {game.display_time}")
    print(f"  display_date:    {game.display_date}")
    print(f"  display_datetime:{game.display_datetime}")
    print(f"  is_upcoming:     {game.is_upcoming}  (game is in {'future' if game.is_upcoming else 'past'})")
    print(f"\n  PASS — GameInfo stores UTC, displays EST.")


# ---------------------------------------------------------------------------
# TEST 5 — DecisionOrchestrator: UTC-aware game_time methods
# ---------------------------------------------------------------------------

def test_orchestrator_time_methods() -> None:
    separator("TEST 5 — DecisionOrchestrator: UTC-aware game time comparison")

    orc = DecisionOrchestrator("WNBA")

    future_utc = now_utc() + timedelta(hours=2)
    past_utc   = now_utc() - timedelta(hours=2)
    far_future = now_utc() + timedelta(hours=30)

    assert orc.is_game_upcoming(future_utc) is True,  "Future game should be upcoming"
    assert orc.is_game_upcoming(past_utc)   is False, "Past game should not be upcoming"

    assert orc.is_game_within_window(future_utc, hours=24) is True
    assert orc.is_game_within_window(far_future, hours=24) is False
    assert orc.is_game_within_window(past_utc,   hours=24) is False

    # Naive datetime (assumed UTC) must also work without raising
    naive_future = (now_utc() + timedelta(hours=1)).replace(tzinfo=None)
    assert orc.is_game_upcoming(naive_future) is True

    # current_time_utc is always UTC-aware
    assert orc.current_time_utc.tzinfo == UTC

    print(f"  current_time_utc:               {orc.current_time_utc.isoformat()}")
    print(f"  is_game_upcoming(+2h):          True  ✓")
    print(f"  is_game_upcoming(-2h):          False ✓")
    print(f"  is_game_within_window(+2h,24h): True  ✓")
    print(f"  is_game_within_window(+30h,24h):False ✓")
    print(f"  is_game_upcoming(naive +1h):    True  ✓")
    print(f"\n  PASS — all UTC comparisons DST-safe.")


# ---------------------------------------------------------------------------
# TEST 6 — TelegramFormatter: game_time displays as EST on bet card
# ---------------------------------------------------------------------------

def test_formatter_game_time_est() -> None:
    separator("TEST 6 — TelegramFormatter: game_time shown in EST on bet card")

    from core.decision_gatekeeper import Bet, Tier
    from output.telegram_formatter import BetDisplay, format_daily_slate

    game_utc = datetime(2026, 5, 31, 23, 30, tzinfo=UTC)   # 7:30 PM EDT

    bet = Bet(
        bet_id="SEA_test_pts_over",
        team="SEA", market="player_points", direction="over",
        sportsbook_line=15.5, edge_percentage=17.3, confidence_score=95.5,
        player="A. Wilson",
    )
    bet.tier = Tier.S_PLUS

    bd = BetDisplay(
        bet=bet,
        american_odds=-115,
        model_probability=67.3,
        supporting_factor="Test factor sentence.",
        game_time_utc=game_utc,
    )

    report = format_daily_slate([bd], sport_type="WNBA", slate_date="Saturday, May 31 2026")

    print(f"\n  Report excerpt (bet card):")
    for line in report.split("\n"):
        if any(k in line for k in ("BET #", "Game Time", "Pick", "Odds", "Model")):
            print(f"    {line}")

    assert "7:30 PM ET" in report, f"Expected '7:30 PM ET' in report, not found.\n{report}"
    print(f"\n  PASS — game time '7:30 PM ET' correctly rendered in Telegram message.")


# ---------------------------------------------------------------------------
# TEST 7 — ROI report shows ET timestamp
# ---------------------------------------------------------------------------

def test_roi_report_shows_et() -> None:
    separator("TEST 7 — ROI report timestamp is in ET not UTC")

    import tempfile
    from pathlib import Path
    import core.results_tracker as rt

    orig_db = rt.DB_PATH
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    rt.DB_PATH = Path(tmp.name)

    try:
        from core.results_tracker import format_roi_report
        report = format_roi_report("WNBA")
        print(f"\n  ROI report header:")
        for line in report.split("\n")[:5]:
            print(f"    {line}")

        assert "ET" in report, f"Expected 'ET' in ROI report timestamp, got:\n{report[:300]}"
        assert "UTC" not in report, f"'UTC' should not appear in user-facing ROI report"
        print(f"\n  PASS — ROI report timestamp is in ET.")
    finally:
        rt.DB_PATH = orig_db
        tmp.close()
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_convert_to_est()
    test_localize_utc()
    test_normalize_api_timestamp()
    test_parse_game_from_api()
    test_orchestrator_time_methods()
    test_formatter_game_time_est()
    test_roi_report_shows_et()

    print(f"\n{'=' * 60}")
    print("  All time_utils / timezone tests passed.")
    print("=" * 60)
    print()
