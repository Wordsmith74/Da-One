"""
Central calibration registry -- one place for every sport-specific constant.

Why this file exists: MLB and WNBA are different sports with different season
lengths, variance profiles, and roster mechanics. A single global
EDGE_THRESHOLD_PCT or confidence curve applied to both is exactly the kind of
sloppiness that breaks a model quietly -- it'll look fine in code review and
then misprice one sport for an entire season. Every other module should pull
its constants from here instead of hardcoding sport-agnostic numbers.

If you only remember one rule from this file: never add a constant here
without naming which sport it's for.
"""

MLB = {
    # ---- Edge / confidence ----
    "edge_threshold_pct": 10.0,       # CALIBRATED via models/backtest.py sweep (see conversation/commit history):
                                        # at 4-6%, ~70-80% of "picks" on a fairly-priced synthetic slate were false
                                        # positives driven by projection noise, not real edge. 10% + 0.85 agreement
                                        # (below) brought that down to a realistic floor (~45%) without losing most
                                        # real-edge detection. Re-run the backtest sweep before changing this again.
    "min_edge_to_publish_pct": 10.0,
    "confidence_scale": 5.0,          # multiplier on abs(edge_pct) when converting to a display confidence
    "confidence_floor": 50,
    "confidence_cap": 95,             # MLB: cap below WNBA's cap -- single-game baseball variance is brutal,
                                       # even a real edge shouldn't display near-certain confidence

    # ---- Season structure ----
    "regular_season_start_month_day": (3, 20),
    "regular_season_end_month_day": (9, 28),
    "postseason_sample_size_games": 0,   # used by season_context to decide how much to trust early playoff samples

    # ---- Sample-size / shrinkage ----
    "min_batters_faced_for_k_prop": 15,   # below this, treat the pitcher prop as too thin to trust regardless of edge
    "min_innings_sample_starts": 3,
    "prior_weight_batters_faced": 60,
    "league_avg_k_pct": 0.225,
    "own_season_k_pct_weight": 0.65,   # weight given to a pitcher's own season-long K%
                                        # vs. league average when both are available
                                        # (used by bayesian.shrink_mlb_k_pct as the prior
                                        # blend -- own-season is a better personal prior
                                        # than league average once enough season data
                                        # exists). ADDED: this key was imported by
                                        # bayesian.py but never defined here, which would
                                        # raise a KeyError on import. Re-tune via
                                        # models/backtest.py before changing.
    "prior_weight_f5_games": 10,       # pseudo-count (in games) for shrinking a team's
                                        # recent F5 runs/game toward its season F5 average
                                        # (used by bayesian.shrink_mlb_f5_runs). ADDED:
                                        # same missing-key issue as above -- matches
                                        # WNBA's prior_weight_games value as a reasonable
                                        # starting point; re-tune via backtest.py.

    # ---- Workload / ramp-up ----
    "workload_metric": "innings_pitched",
    "ramp_lookback_starts": 5,
    "ramp_drop_threshold_pct": 20,     # % drop in IP vs baseline that triggers a ramp-up flag
    "il_return_discount_starts": 3,    # discount workload projection for this many starts after an IL return

    # ---- Park / environment ----
    "uses_park_factor": True,
    "f5_park_scale": 0.55,             # fraction of full-game park factor effect that applies to first 5 innings

    # ---- Line movement ----
    "steam_move_threshold_pct": 8.0,   # market move since pick generation that should void a total/prop pick
    "moneyline_steam_cents": 15,       # cents of moneyline movement that should void a side

    # ---- Backtest / bankroll ----
    "kelly_fraction": 0.25,            # quarter-Kelly -- MLB single-game variance argues for a smaller fraction
    "max_picks_per_day": 6,

    # ---- Uncertainty-robust edge gating ----
    # A pick must clear edge_threshold_pct on a MEAN edge averaged across
    # resampled projections (not one point estimate), AND have at least this
    # fraction of resampled draws agree on the side -- this is what actually
    # separates real edge from projection noise (see models/monte_carlo.py
    # f5_edge_with_uncertainty and models/backtest.py for why a naive
    # point-estimate threshold alone produced an ~80% false-positive rate).
    "min_side_agreement_frac": 0.90,  # CALIBRATED: F5 backtest preferred 0.85; K-prop backtest showed cleaner
                                        # signal/noise separation and tolerates 0.90 with only ~2% false-positive
                                        # rate vs ~51% detection of real mispricings -- using one shared MLB value
                                        # since F5 still works acceptably at 0.90 too (see models/backtest.py).
    # Mean-projection uncertainty (runs) assumed per team for a "typical"
    # small recent-game sample -- used by the backtest's robust check; a real
    # production version should derive this from actual sample size per team
    # rather than a fixed constant.
    "f5_mean_projection_std": 0.35,

    # K-prop uncertainty (used by models/monte_carlo.k_prop_edge_with_uncertainty)
    "k_pct_projection_std": 0.035,     # ~3.5 percentage points of K% uncertainty on a small recent sample
    "bf_mean_projection_std": 3.0,     # batters-faced start-to-start variance
}

WNBA = {
    # ---- Edge / confidence ----
    "edge_threshold_pct": 6.0,        # thinner books / lower liquidity on WNBA props -> require a bigger
                                        # edge before trusting the market line is even efficient
    "min_edge_to_publish_pct": 6.0,
    "confidence_scale": 4.0,
    "confidence_floor": 50,
    "confidence_cap": 90,              # cap lower than MLB-side cap would suggest, but WNBA prop markets
                                        # are softer AND thinner-sampled -- keep displayed confidence modest

    # ---- Season structure ----
    "regular_season_start_month_day": (5, 15),
    "regular_season_end_month_day": (9, 15),
    "postseason_sample_size_games": 0,

    # ---- Sample-size / shrinkage ----
    "min_minutes_sample_games": 4,     # below this many recent games, don't trust a minutes-based projection
    "prior_weight_games": 10,
    "min_games_for_player_prop": 4,

    # ---- Workload / ramp-up ----
    "workload_metric": "minutes",
    "ramp_lookback_games": 4,
    "ramp_drop_threshold_pct": 15,     # minutes are a tighter band than IP -- smaller drop is meaningful
    "injury_return_discount_games": 2,

    # ---- Park / environment ----
    "uses_park_factor": False,         # no ballpark effect in basketball; arenas don't move shooting/lines materially

    # ---- Line movement ----
    "steam_move_threshold_pct": 10.0,  # props move more on injury/news noise -- slightly wider tolerance than MLB
    "moneyline_steam_cents": 20,

    # ---- Backtest / bankroll ----
    "kelly_fraction": 0.20,            # even smaller fraction -- thinner books = worse closing-line liquidity
    "max_picks_per_day": 4,

    # ---- Uncertainty-robust edge gating (see MLB block above for rationale) ----
    "min_side_agreement_frac": 0.90,  # CALIBRATED: 0.90 gave 12% false-positive rate / 68% detection of real
                                        # mispricings vs 20-35% FP at lower settings -- see models/backtest.py
                                        # run_backtest_wnba sweep. edge_threshold_pct below turned out to not
                                        # be the binding constraint -- agreement_frac is doing the real work.
    "rate_per_minute_projection_std": 0.04,   # uncertainty on points-per-minute rate
    "minutes_projection_std": 4.0,            # minutes are the dominant uncertainty source for WNBA props
}


def get_config(sport):
    """sport: 'mlb' or 'wnba' (case-insensitive)."""
    key = sport.strip().lower()
    if key == "mlb":
        return MLB
    if key == "wnba":
        return WNBA
    raise ValueError(f"No calibration config for sport={sport!r}. Add one to sport_config.py "
                      f"rather than falling back to a generic default.")
    
