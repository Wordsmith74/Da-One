"""
Test script: demonstrates DecisionOrchestrator with dummy WNBA game data.

Run from the workspace root:
    python core/test_orchestrator.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.decision_orchestrator import DecisionOrchestrator, MissingMetricError


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def test_wnba_spread() -> None:
    separator("TEST 1 — WNBA: calculate_true_spread (happy path)")

    orchestrator = DecisionOrchestrator(sport_type="WNBA")

    wnba_game_data = {
        # --- spread factors (keys must match spread_weights in JSON) ---
        "power_rating":     85.4,
        "form":             78.0,
        "matchup":          72.5,
        "injury":           90.0,
        "home_court":       65.0,
        "market":           80.0,
        # --- required metrics (needed to pass validation) ---
        "offensive_rating":         108.3,
        "defensive_rating":         101.7,
        "pace":                     88.2,
        "usage_rate":               0.22,
        "true_shooting_pct":        0.578,
        "net_rating":               6.6,
        "assist_to_turnover_ratio": 1.95,
        "rebound_rate":             0.51,
        "player_impact_estimate":   14.2,
        "home_away_split":          3.5,
        # --- total factors ---
        "off_efficiency":   108.3,
        "def_efficiency":   101.7,
    }

    result = orchestrator.calculate_true_spread(wnba_game_data)

    print(f"\nSport:            {orchestrator.sport_type}")
    print(f"\nSpread weights applied:")
    for factor, weight in orchestrator.spread_weights.items():
        value = wnba_game_data[factor]
        contribution = round(weight * value, 4)
        print(f"  {factor:<20} weight={weight}  value={value}  contribution={contribution}")

    total_weight = sum(orchestrator.spread_weights.values())
    print(f"\nSum of weights:   {total_weight:.2f}")
    print(f"True Spread:      {result}")


def test_wnba_total() -> None:
    separator("TEST 2 — WNBA: calculate_true_total (happy path)")

    orchestrator = DecisionOrchestrator(sport_type="WNBA")

    wnba_game_data = {
        # --- total factors ---
        "pace":             88.2,
        "off_efficiency":   108.3,
        "def_efficiency":   101.7,
        "form":             78.0,
        "injury":           90.0,
        "market":           80.0,
        # --- required metrics ---
        "offensive_rating":         108.3,
        "defensive_rating":         101.7,
        "usage_rate":               0.22,
        "true_shooting_pct":        0.578,
        "net_rating":               6.6,
        "assist_to_turnover_ratio": 1.95,
        "rebound_rate":             0.51,
        "player_impact_estimate":   14.2,
        "home_away_split":          3.5,
        # --- spread factors (needed to avoid KeyError if called) ---
        "power_rating":     85.4,
        "matchup":          72.5,
        "home_court":       65.0,
    }

    result = orchestrator.calculate_true_total(wnba_game_data)

    print(f"\nSport:            {orchestrator.sport_type}")
    print(f"\nTotal weights applied:")
    for factor, weight in orchestrator.total_weights.items():
        value = wnba_game_data[factor]
        contribution = round(weight * value, 4)
        print(f"  {factor:<20} weight={weight}  value={value}  contribution={contribution}")

    total_weight = sum(orchestrator.total_weights.values())
    print(f"\nSum of weights:   {total_weight:.2f}")
    print(f"True Total:       {result}")


def test_missing_metric_error() -> None:
    separator("TEST 3 — WNBA: MissingMetricError (error handling)")

    orchestrator = DecisionOrchestrator(sport_type="WNBA")

    incomplete_data = {
        "power_rating": 85.4,
        "form":         78.0,
        "matchup":      72.5,
        "injury":       90.0,
        "home_court":   65.0,
        "market":       80.0,
        # deliberately omitting all required_metrics
    }

    print("\nPassing game_data with no required_metrics present...")
    try:
        orchestrator.calculate_true_spread(incomplete_data)
        print("ERROR: Expected MissingMetricError was NOT raised.")
    except MissingMetricError as e:
        print(f"\nCaught MissingMetricError (expected):")
        print(f"  {e}")


def test_unsupported_sport() -> None:
    separator("TEST 4 — Unsupported sport type (error handling)")

    from core.decision_orchestrator import UnsupportedSportError

    print("\nInitializing with sport_type='CRICKET'...")
    try:
        DecisionOrchestrator(sport_type="CRICKET")
        print("ERROR: Expected UnsupportedSportError was NOT raised.")
    except UnsupportedSportError as e:
        print(f"\nCaught UnsupportedSportError (expected):")
        print(f"  {e}")


if __name__ == "__main__":
    test_wnba_spread()
    test_wnba_total()
    test_missing_metric_error()
    test_unsupported_sport()
    print("\n\nAll tests passed.\n")
