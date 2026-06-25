"""
Test script: demonstrates SimulationEngine with a dummy 'Player Points' O/U line.

Scenario
--------
- Sport: WNBA
- Metric: Player Points
- Sportsbook line: 15.5
- Historical data: last 10 games for a player who averages around 17 pts
- League mean (prior): 15.0

Run from the workspace root:
    python3 core/test_simulation_engine.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.decision_orchestrator import DecisionOrchestrator
from core.simulation_engine import (
    SimulationEngine,
    estimate_player_metric,
    run_monte_carlo,
    get_win_probability,
)


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def print_posterior(posterior: dict) -> None:
    print(f"\n  Posterior Mean:    {posterior['posterior_mean']:.3f} pts")
    print(f"  Posterior Std Dev: {posterior['posterior_std']:.3f} pts")
    print(f"  94% HDI:           [{posterior['hdi_low']:.3f}, {posterior['hdi_high']:.3f}]")
    print(f"  Observations used: {posterior['n_obs']}")


def print_probabilities(result: dict, line: float) -> None:
    wp = result if "over_probability" in result else result["win_probability"]
    print(f"\n  Sportsbook Line:   {line}")
    print(f"  Trials Run:        {wp['trials']:,}")
    print(f"  ─────────────────────────────────")
    print(f"  OVER  probability: {wp['over_probability']:>6.2f}%   (edge vs 50%: {wp['edge_over']:+.2f}%)")
    print(f"  UNDER probability: {wp['under_probability']:>6.2f}%   (edge vs 50%: {wp['edge_under']:+.2f}%)")
    if wp["push_probability"] > 0:
        print(f"  PUSH  probability: {wp['push_probability']:>6.2f}%")


# ---------------------------------------------------------------------------
# TEST 1 — Full pipeline via SimulationEngine (orchestrator-integrated)
# ---------------------------------------------------------------------------

def test_full_pipeline_via_engine() -> None:
    separator("TEST 1 — Full pipeline via SimulationEngine (WNBA)")

    orchestrator = DecisionOrchestrator(sport_type="WNBA")
    engine = SimulationEngine(orchestrator)

    # Last 10 games for a player who trends above the 15.5 line
    historical_points = [14.0, 18.0, 17.0, 16.0, 19.0, 15.0, 21.0, 17.0, 18.0, 16.0]
    league_mean = 15.0
    sportsbook_line = 15.5

    print(f"\nSport:            {engine.sport_type}")
    print(f"Historical data:  {historical_points}")
    print(f"Raw data mean:    {sum(historical_points)/len(historical_points):.2f} pts")
    print(f"League mean prior:{league_mean} pts")
    print(f"Sportsbook line:  {sportsbook_line} pts")
    print("\nRunning Bayesian inference (PyMC NUTS sampler)...")

    result = engine.analyze(
        historical_data=historical_points,
        league_mean=league_mean,
        sportsbook_line=sportsbook_line,
        trials=10_000,
        progressbar=False,
    )

    print("\n--- Bayesian Posterior ---")
    print_posterior(result["posterior"])

    print("\n--- Monte Carlo Win Probabilities (10,000 trials) ---")
    print_probabilities(result["win_probability"], sportsbook_line)

    recommendation = (
        "OVER" if result["win_probability"]["over_probability"] > 50
        else "UNDER"
    )
    edge = max(
        result["win_probability"]["edge_over"],
        result["win_probability"]["edge_under"],
    )
    print(f"\n  MODEL RECOMMENDATION: {recommendation}  (edge: {edge:+.2f}%)")


# ---------------------------------------------------------------------------
# TEST 2 — Standalone functions with a below-line player
# ---------------------------------------------------------------------------

def test_standalone_functions_under_scenario() -> None:
    separator("TEST 2 — Standalone functions: player trending UNDER 15.5")

    # A player who has been cold — data clusters below the line
    historical_points = [11.0, 13.0, 10.0, 14.0, 12.0, 9.0, 15.0, 11.0, 13.0, 10.0]
    league_mean = 15.0
    sportsbook_line = 15.5

    print(f"\nHistorical data:  {historical_points}")
    print(f"Raw data mean:    {sum(historical_points)/len(historical_points):.2f} pts")
    print(f"Sportsbook line:  {sportsbook_line} pts")
    print("\nRunning Bayesian inference...")

    posterior = estimate_player_metric(
        historical_data=historical_points,
        league_mean=league_mean,
        progressbar=False,
    )

    print("\n--- Bayesian Posterior ---")
    print_posterior(posterior)

    simulated = run_monte_carlo(
        mean=posterior["posterior_mean"],
        std_dev=posterior["posterior_std"],
        trials=10_000,
    )

    win_prob = get_win_probability(
        simulated_results=simulated,
        sportsbook_line=sportsbook_line,
    )

    print("\n--- Monte Carlo Win Probabilities (10,000 trials) ---")
    print_probabilities(win_prob, sportsbook_line)

    recommendation = "OVER" if win_prob["over_probability"] > 50 else "UNDER"
    edge = max(win_prob["edge_over"], win_prob["edge_under"])
    print(f"\n  MODEL RECOMMENDATION: {recommendation}  (edge: {edge:+.2f}%)")


# ---------------------------------------------------------------------------
# TEST 3 — Exact line of 15.5 with a mock distribution (no MCMC needed)
# ---------------------------------------------------------------------------

def test_mock_distribution_player_points() -> None:
    separator("TEST 3 — Mock Bayesian distribution, 15.5 pts line (10,000 trials)")

    # Directly supply a posterior mean/std — simulates the user's requested scenario
    mock_posterior_mean = 17.2
    mock_posterior_std  = 3.8
    sportsbook_line = 15.5

    print(f"\nMock posterior mean:  {mock_posterior_mean} pts")
    print(f"Mock posterior std:   {mock_posterior_std} pts")
    print(f"Sportsbook line:      {sportsbook_line} pts")

    simulated = run_monte_carlo(
        mean=mock_posterior_mean,
        std_dev=mock_posterior_std,
        trials=10_000,
    )

    print(f"\nSimulated sample stats:")
    print(f"  Mean:   {np.mean(simulated):.3f}")
    print(f"  Std:    {np.std(simulated):.3f}")
    print(f"  Min:    {np.min(simulated):.3f}")
    print(f"  Max:    {np.max(simulated):.3f}")

    win_prob = get_win_probability(
        simulated_results=simulated,
        sportsbook_line=sportsbook_line,
    )

    print("\n--- Model Probability (10,000 trials vs line 15.5) ---")
    print_probabilities(win_prob, sportsbook_line)

    recommendation = "OVER" if win_prob["over_probability"] > 50 else "UNDER"
    edge = max(win_prob["edge_over"], win_prob["edge_under"])
    print(f"\n  MODEL RECOMMENDATION: {recommendation}  (edge: {edge:+.2f}%)")
    print(f"\n  FINAL MODEL PROBABILITY OUTPUT:")
    print(f"    OVER  {sportsbook_line}: {win_prob['over_probability']}%")
    print(f"    UNDER {sportsbook_line}: {win_prob['under_probability']}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_full_pipeline_via_engine()
    test_standalone_functions_under_scenario()
    test_mock_distribution_player_points()
    print(f"\n{'=' * 60}")
    print("  All simulation tests completed.")
    print("=" * 60)
    print()
