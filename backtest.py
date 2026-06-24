"""
Backtest harness.

CRITICAL HONESTY NOTE: this sandbox has no network access and no cached real
historical odds/results data (see SETUP.md's instruction to run this against
`data/cache_history.py` real cached history once it exists). Everything this
script runs against below is SYNTHETIC data, generated with a known true edge
baked in -- its only purpose is to verify the PIPELINE MACHINERY is internally
consistent (a real edge gets detected, a fairly-priced or negative-edge game
gets correctly skipped, contradiction/line-movement filters fire when they
should). It proves nothing about whether the model would be profitable on
real markets. Do not present these numbers as a real track record.

Real backtesting requires real closing lines + real outcomes over a real
season, ideally sourced from data/cache_history.py once it's built out to
snapshot and store live API responses over time.
"""
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.monte_carlo import (
    simulate_f5_game, summarize_f5_total, f5_edge_with_uncertainty,
    simulate_pitcher_ks, k_prop_edge_with_uncertainty,
    simulate_wnba_stat, wnba_edge_with_uncertainty, summarize_over_under,
)
from models.advanced_metrics import pitcher_quality_factor, f5_park_factor
from models.sport_config import MLB, WNBA
from models.handicapper_rules import kelly_stake


def _american_to_prob(odds):
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def generate_synthetic_f5_slate(n_games, true_edge_games_fraction, seed=7):
    """
    Builds n_games synthetic MLB F5 matchups. true_edge_games_fraction of them
    are constructed with a deliberate, known mispricing (model should detect
    edge); the rest are constructed at a fair market price (model should NOT
    generate a pick). Also simulates a true outcome for each game so hit-rate
    can be checked against what the model predicted.
    """
    rng = random.Random(seed)
    games = []
    for i in range(n_games):
        true_home_mean = rng.uniform(1.8, 3.2)
        true_away_mean = rng.uniform(1.8, 3.2)
        true_total_mean = true_home_mean + true_away_mean

        has_edge = (i / n_games) < true_edge_games_fraction
        if has_edge:
            # Market line set deliberately off the true mean by 0.6-1.0 runs
            market_line = true_total_mean + rng.choice([-1, 1]) * rng.uniform(0.6, 1.0)
            market_line = round(market_line * 2) / 2  # books quote in .5 increments
        else:
            # Fair game: the market line should BE the nearest .5 to the true mean,
            # not the true mean plus noise then rounded -- rounding noise was
            # previously creating a structural baseline edge in "fair" games that
            # had nothing to do with the model, inflating the false-positive count
            # artificially. A genuinely fair market quotes the closest available
            # number to the true mean, period.
            market_line = round(true_total_mean * 2) / 2

        games.append({
            "matchup": f"SYN{i}_HOME @ SYN{i}_AWAY",
            "true_home_mean": true_home_mean,
            "true_away_mean": true_away_mean,
            "market_f5_total_line": market_line,
            "market_f5_total_odds": -110,
            "has_known_edge": has_edge,
        })
    return games


def simulate_true_outcome(game, seed):
    """Draws one 'actual' result from the TRUE generating distribution (not
    the model's estimate) -- this is what makes it a backtest rather than the
    model just grading its own homework."""
    rng = random.Random(seed)
    home_actual = max(0, round(rng.gauss(game["true_home_mean"], game["true_home_mean"] ** 0.5)))
    away_actual = max(0, round(rng.gauss(game["true_away_mean"], game["true_away_mean"] ** 0.5)))
    return home_actual + away_actual


def run_backtest(n_games=200, true_edge_games_fraction=0.35):
    print("=== BACKTEST (synthetic data -- see module docstring for honesty caveat) ===\n")
    games = generate_synthetic_f5_slate(n_games, true_edge_games_fraction)

    results = []
    for idx, g in enumerate(games):
        # Model only sees the market line + a noisy estimate of the true means
        # (simulating real-world projection error -- a model is never handed
        # the true mean directly), then runs the same Monte Carlo + edge logic
        # run_pipeline.py uses.
        rng = random.Random(idx * 13 + 1)
        noisy_home = max(0.5, g["true_home_mean"] + rng.gauss(0, 0.25))
        noisy_away = max(0.5, g["true_away_mean"] + rng.gauss(0, 0.25))

        robust = f5_edge_with_uncertainty(
            noisy_home, MLB["f5_mean_projection_std"],
            noisy_away, MLB["f5_mean_projection_std"],
            g["market_f5_total_line"], g["market_f5_total_odds"],
            seed=idx,
        )
        edge_pct = robust["mean_edge_pct"]
        side = robust["side"]
        pick_made = (abs(edge_pct) >= MLB["edge_threshold_pct"]
                     and robust["agreement_frac"] >= MLB["min_side_agreement_frac"])

        # Used only for grading win/loss probability bookkeeping below
        sims = simulate_f5_game(noisy_home, noisy_away, seed=idx)
        summary = summarize_f5_total(sims, g["market_f5_total_line"])

        actual_total = simulate_true_outcome(g, seed=idx + 9000)
        actual_side = "over" if actual_total > g["market_f5_total_line"] else "under"
        won = pick_made and (actual_side == side)
        lost = pick_made and (actual_side != side) and actual_total != g["market_f5_total_line"]

        stake = 0.0
        if pick_made:
            model_prob_for_side = summary["over_prob"] if side == "over" else summary["under_prob"]
            stake = kelly_stake(model_prob_for_side, g["market_f5_total_odds"], MLB["kelly_fraction"])

        results.append({
            "matchup": g["matchup"], "has_known_edge": g["has_known_edge"],
            "pick_made": pick_made, "edge_pct": round(edge_pct, 2), "side": side if pick_made else None,
            "actual_total": actual_total, "market_line": g["market_f5_total_line"],
            "won": won, "lost": lost, "stake_pct": round(stake * 100, 2),
        })

    # ---- Aggregate diagnostics ----
    picks_made = [r for r in results if r["pick_made"]]
    n_picks = len(picks_made)
    n_wins = sum(1 for r in picks_made if r["won"])
    n_losses = sum(1 for r in picks_made if r["lost"])
    n_pushes = n_picks - n_wins - n_losses

    detected_on_edge_games = sum(1 for r in results if r["pick_made"] and r["has_known_edge"])
    known_edge_games = sum(1 for r in results if r["has_known_edge"])
    false_picks_on_fair_games = sum(1 for r in results if r["pick_made"] and not r["has_known_edge"])
    fair_games = n_games - known_edge_games

    print(f"Synthetic slate: {n_games} games ({known_edge_games} built with a known mispricing, "
          f"{fair_games} built fairly-priced)")
    print(f"Pipeline generated {n_picks} picks total")
    print(f"  -> caught {detected_on_edge_games}/{known_edge_games} "
          f"({detected_on_edge_games/known_edge_games*100:.0f}%) of the deliberately-mispriced games")
    print(f"  -> incorrectly fired on {false_picks_on_fair_games}/{fair_games} "
          f"({false_picks_on_fair_games/max(fair_games,1)*100:.0f}%) of the fairly-priced games "
          f"(false-positive rate -- should be low; near-line noise can still trip the {MLB['edge_threshold_pct']}% threshold)")
    print(f"\nOf the {n_picks} picks actually made (graded against the TRUE simulated outcome, not the model's own estimate):")
    print(f"  Wins: {n_wins}  Losses: {n_losses}  Pushes: {n_pushes}")
    if n_wins + n_losses > 0:
        print(f"  Win rate (decided bets): {n_wins/(n_wins+n_losses)*100:.1f}%")
    print("\nReminder: this is a SYNTHETIC self-consistency check, not a real-market track record.")
    return results


def generate_synthetic_k_prop_slate(n_props, true_edge_fraction, seed=11):
    """Synthetic MLB pitcher K-prop slate. true K% per pitcher and a true
    batters-faced mean generate a true K distribution; market line is either
    fairly priced (rounded to the true median) or deliberately off."""
    rng = random.Random(seed)
    props = []
    for i in range(n_props):
        true_k_pct = rng.uniform(0.16, 0.32)
        true_bf = rng.uniform(20, 26)
        true_mean_ks = true_k_pct * true_bf

        has_edge = (i / n_props) < true_edge_fraction
        if has_edge:
            market_line = true_mean_ks + rng.choice([-1, 1]) * rng.uniform(1.2, 2.0)
        else:
            market_line = true_mean_ks
        market_line = round(market_line - 0.5) + 0.5  # books quote K props at X.5

        props.append({
            "pitcher": f"SynPitcher{i}", "true_k_pct": true_k_pct, "true_bf": true_bf,
            "market_k_line": market_line, "market_k_odds": -115, "has_known_edge": has_edge,
        })
    return props


def run_backtest_k_prop(n_props=200, true_edge_fraction=0.35):
    print("\n=== K-PROP BACKTEST (synthetic data) ===\n")
    props = generate_synthetic_k_prop_slate(n_props, true_edge_fraction)
    results = []
    for idx, p in enumerate(props):
        rng = random.Random(idx * 17 + 3)
        noisy_k_pct = max(0.05, p["true_k_pct"] + rng.gauss(0, 0.02))
        noisy_bf = max(10, p["true_bf"] + rng.gauss(0, 1.5))

        robust = k_prop_edge_with_uncertainty(
            noisy_k_pct, MLB["k_pct_projection_std"], noisy_bf, MLB["bf_mean_projection_std"],
            p["market_k_line"], p["market_k_odds"], seed=idx,
        )
        edge_pct, side = robust["mean_edge_pct"], robust["side"]
        pick_made = abs(edge_pct) >= MLB["edge_threshold_pct"] and robust["agreement_frac"] >= MLB["min_side_agreement_frac"]

        true_ks = max(0, round(random.Random(idx + 9000).gauss(p["true_k_pct"] * p["true_bf"], (p["true_k_pct"] * p["true_bf"]) ** 0.5)))
        actual_side = "over" if true_ks > p["market_k_line"] else "under"
        won = pick_made and actual_side == side
        lost = pick_made and actual_side != side

        results.append({"has_known_edge": p["has_known_edge"], "pick_made": pick_made, "won": won, "lost": lost})

    _print_summary(results, n_props, MLB["edge_threshold_pct"])
    return results


def generate_synthetic_wnba_slate(n_props, true_edge_fraction, seed=23):
    rng = random.Random(seed)
    props = []
    for i in range(n_props):
        true_rate_per_min = rng.uniform(0.4, 0.85)  # points per minute
        true_minutes = rng.uniform(24, 34)
        true_mean_pts = true_rate_per_min * true_minutes

        has_edge = (i / n_props) < true_edge_fraction
        if has_edge:
            market_line = true_mean_pts + rng.choice([-1, 1]) * rng.uniform(3.0, 5.0)
        else:
            market_line = true_mean_pts
        market_line = round(market_line - 0.5) + 0.5

        props.append({
            "player": f"SynPlayer{i}", "true_rate": true_rate_per_min, "true_minutes": true_minutes,
            "market_pts_line": market_line, "market_pts_odds": -115, "has_known_edge": has_edge,
        })
    return props


def run_backtest_wnba(n_props=200, true_edge_fraction=0.35):
    print("\n=== WNBA PROP BACKTEST (synthetic data) ===\n")
    props = generate_synthetic_wnba_slate(n_props, true_edge_fraction)
    results = []
    for idx, p in enumerate(props):
        rng = random.Random(idx * 19 + 5)
        noisy_rate = max(0.1, p["true_rate"] + rng.gauss(0, 0.05))
        noisy_minutes = max(10, p["true_minutes"] + rng.gauss(0, 3.0))

        robust = wnba_edge_with_uncertainty(
            noisy_rate, WNBA["rate_per_minute_projection_std"], noisy_minutes, WNBA["minutes_projection_std"],
            p["market_pts_line"], p["market_pts_odds"], seed=idx,
        )
        edge_pct, side = robust["mean_edge_pct"], robust["side"]
        pick_made = abs(edge_pct) >= WNBA["edge_threshold_pct"] and robust["agreement_frac"] >= WNBA["min_side_agreement_frac"]

        true_mean = p["true_rate"] * p["true_minutes"]
        true_pts = max(0, random.Random(idx + 9000).gauss(true_mean, max(2.5, true_mean * 0.35)))
        actual_side = "over" if true_pts > p["market_pts_line"] else "under"
        won = pick_made and actual_side == side
        lost = pick_made and actual_side != side

        results.append({"has_known_edge": p["has_known_edge"], "pick_made": pick_made, "won": won, "lost": lost})

    _print_summary(results, n_props, WNBA["edge_threshold_pct"])
    return results


def _print_summary(results, n_total, threshold_pct):
    picks_made = [r for r in results if r["pick_made"]]
    n_picks = len(picks_made)
    n_wins = sum(1 for r in picks_made if r["won"])
    n_losses = sum(1 for r in picks_made if r["lost"])

    known_edge = sum(1 for r in results if r["has_known_edge"])
    fair = n_total - known_edge
    detected = sum(1 for r in results if r["pick_made"] and r["has_known_edge"])
    false_pos = sum(1 for r in results if r["pick_made"] and not r["has_known_edge"])

    print(f"Slate: {n_total} ({known_edge} mispriced, {fair} fair)")
    print(f"Picks made: {n_picks}")
    print(f"  caught {detected}/{known_edge} ({detected/max(known_edge,1)*100:.0f}%) of mispriced")
    print(f"  false-positive on {false_pos}/{fair} ({false_pos/max(fair,1)*100:.0f}%) of fair "
          f"(threshold {threshold_pct}%)")
    if n_wins + n_losses > 0:
        print(f"  win rate (decided): {n_wins/(n_wins+n_losses)*100:.1f}% ({n_wins}W-{n_losses}L)")
    print("Reminder: synthetic self-consistency check only.")


if __name__ == "__main__":
    run_backtest()
    run_backtest_k_prop()
    run_backtest_wnba()
