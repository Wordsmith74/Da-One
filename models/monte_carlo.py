"""
Monte Carlo simulation layer.

Deliberately NOT one generic "simulate a number" function reused across
sports -- the underlying distributions are genuinely different shapes:
  - MLB team runs (F5): low-mean count data -> Poisson is the standard,
    well-supported model for runs scored in a fixed number of innings.
  - MLB pitcher strikeouts: also count data over batters faced -> Binomial
    (n=batters faced, p=K%), not Poisson -- bounded by batters faced, which
    Poisson doesn't respect.
  - WNBA player points: continuous-ish, higher mean, more overdispersed than
    a simple Poisson/Binomial would capture -- modeled as Normal (Gaussian)
    around a rate*minutes mean with a variance term scaled to usage, which is
    the standard treatment for basketball box-score props in public models.

Mixing these up (e.g. using Poisson for basketball points) is a textbook
sport-mismatch bug, so each function below is named and shaped for exactly
one sport/stat.
"""
import random


def simulate_f5_game(home_runs_mean, away_runs_mean, n_sims=20000, seed=None):
    """MLB First-5-Innings runs: Poisson per team, simulated n_sims times."""
    rng = random.Random(seed)
    home_sims = [_poisson(rng, home_runs_mean) for _ in range(n_sims)]
    away_sims = [_poisson(rng, away_runs_mean) for _ in range(n_sims)]
    return {"home": home_sims, "away": away_sims, "total": [h + a for h, a in zip(home_sims, away_sims)]}


def summarize_f5_total(sims, line):
    totals = sims["total"]
    n = len(totals)
    mean_total = sum(totals) / n
    if line is None:
        return {"mean_total": mean_total, "over_prob": None, "under_prob": None}
    over = sum(1 for t in totals if t > line) / n
    push = sum(1 for t in totals if t == line) / n
    under = 1 - over - push
    return {"mean_total": mean_total, "over_prob": over, "under_prob": under, "push_prob": push}


def summarize_f5_moneyline(sims):
    home, away = sims["home"], sims["away"]
    n = len(home)
    home_wins = sum(1 for h, a in zip(home, away) if h > a) / n
    away_wins = sum(1 for h, a in zip(home, away) if a > h) / n
    tie = 1 - home_wins - away_wins
    return {"home_win_prob": home_wins, "away_win_prob": away_wins, "tie_prob": tie}


def simulate_pitcher_ks(k_pct, batters_faced_mean, n_sims=20000, seed=None):
    """
    MLB pitcher strikeouts: Binomial(n=batters_faced, p=k_pct), with batters_faced
    itself sampled from a tight Poisson around its projected mean (a pitcher
    doesn't face a fixed, deterministic number of batters -- that's also random).
    """
    rng = random.Random(seed)
    sims = []
    for _ in range(n_sims):
        bf = max(1, _poisson(rng, batters_faced_mean))
        ks = sum(1 for _ in range(bf) if rng.random() < k_pct)
        sims.append(ks)
    return sims


def simulate_wnba_stat(rate_per_minute, minutes_mean, stat_type="wnba_points",
                        n_sims=20000, seed=None):
    """
    WNBA player stat (points by default): Normal distribution around
    (rate * minutes), with standard deviation scaled relative to the mean --
    basketball box-score stats are well-approximated by a Gaussian for game
    totals, unlike the low-count Poisson/Binomial shapes used for baseball above.

    rate_per_minute: a TRUE per-minute rate (points per minute, already
    shrunk/adjusted upstream by run_pipeline.py's process_wnba_prop, which
    divides its per-30 shrinkage output by 30 before calling this). Do NOT
    divide by 30 again here -- that was a real double-conversion bug caught
    by running the pipeline end-to-end; the unit contract is per-minute, full stop.
    """
    rng = random.Random(seed)
    mean_stat = rate_per_minute * minutes_mean

    # Relative variance: WNBA scoring is noisier than a fixed-rate model would
    # suggest (hot/cold shooting nights) -- std dev set to ~35% of the mean,
    # a commonly-used heuristic for basketball point-prop modeling, floored so
    # low-usage projections don't get an unrealistically tiny spread.
    std_dev = max(2.5, mean_stat * 0.35)

    sims = [max(0.0, rng.gauss(mean_stat, std_dev)) for _ in range(n_sims)]
    return sims


def summarize_over_under(sims, line):
    """Generic over/under summary -- this part IS safely shared since once you
    have a list of simulated outcomes, computing P(over)/P(under) against a
    line is identical math regardless of sport."""
    n = len(sims)
    mean = sum(sims) / n
    if line is None:
        return {"mean": mean, "over_prob": None, "under_prob": None}
    over = sum(1 for s in sims if s > line) / n
    under = sum(1 for s in sims if s < line) / n
    push = 1 - over - under
    return {"mean": mean, "over_prob": over, "under_prob": under, "push_prob": push}


def f5_edge_with_uncertainty(home_mean, home_mean_std, away_mean, away_mean_std,
                              market_line, market_odds, n_outer=60, n_inner=1500, seed=None):
    """
    Robust edge check for MLB F5 totals -- THIS is the fix for the
    false-positive problem found in backtesting (see models/backtest.py
    history / conversation): a single point-estimate edge_pct treats the
    projected mean as if it were known exactly, when it's actually an
    estimate with real sampling error. Betting on a point estimate's edge
    without accounting for that error is exactly how a model fires on
    fairly-priced games whenever projection noise happens to push the
    estimate off the true mean.

    Method: instead of running ONE Monte Carlo sim from ONE mean estimate,
    redraw the mean itself n_outer times from its own uncertainty
    distribution (Normal(mean, mean_std) -- mean_std should reflect real
    sample-size-driven uncertainty, e.g. larger for a 5-game sample than a
    30-game one), run a smaller inner simulation for each redraw, and look
    at the DISTRIBUTION of edge_pct across outer draws rather than one number.

    Returns dict:
      mean_edge_pct: average edge across all outer draws
      agreement_frac: fraction of outer draws where edge has the same sign
                       AND magnitude >= the eventual side's required minimum
                       (caller compares against this, not just mean_edge_pct)
      side: "over"/"under" by majority vote across outer draws
    """
    rng = random.Random(seed)
    edges = []
    sides = []
    for _ in range(n_outer):
        h = max(0.3, rng.gauss(home_mean, home_mean_std))
        a = max(0.3, rng.gauss(away_mean, away_mean_std))
        sims = simulate_f5_game(h, a, n_sims=n_inner, seed=rng.randint(0, 10**9))
        summary = summarize_f5_total(sims, market_line)
        implied_over = (100 / (market_odds + 100)) if market_odds > 0 else (abs(market_odds) / (abs(market_odds) + 100))
        edge_pct = (summary["over_prob"] - implied_over) * 100
        edges.append(edge_pct)
        sides.append("over" if edge_pct > 0 else "under")

    mean_edge_pct = sum(edges) / len(edges)
    majority_side = "over" if sides.count("over") >= sides.count("under") else "under"
    agreement_frac = sides.count(majority_side) / len(sides)

    return {"mean_edge_pct": mean_edge_pct, "agreement_frac": agreement_frac, "side": majority_side,
            "edge_std": (sum((e - mean_edge_pct) ** 2 for e in edges) / len(edges)) ** 0.5}


def k_prop_edge_with_uncertainty(k_pct, k_pct_std, bf_mean, bf_std,
                                  market_line, market_odds, n_outer=60, n_inner=1500, seed=None):
    """
    MLB pitcher-strikeout analog of f5_edge_with_uncertainty -- redraws BOTH
    the projected K% and the projected batters-faced mean from their own
    uncertainty distributions (rather than trusting one point estimate of
    each), runs the Binomial-style simulation per redraw, and reports the
    distribution of edge_pct rather than one number. K% uncertainty should
    shrink with batters-faced sample size; bf_mean uncertainty reflects
    start-to-start variance in how deep a pitcher goes.
    """
    rng = random.Random(seed)
    edges, sides = [], []
    for _ in range(n_outer):
        k = min(0.6, max(0.02, rng.gauss(k_pct, k_pct_std)))
        bf = max(5, rng.gauss(bf_mean, bf_std))
        sims = simulate_pitcher_ks(k, bf, n_sims=n_inner, seed=rng.randint(0, 10**9))
        summary = summarize_over_under(sims, market_line)
        implied_over = (100 / (market_odds + 100)) if market_odds > 0 else (abs(market_odds) / (abs(market_odds) + 100))
        edge_pct = (summary["over_prob"] - implied_over) * 100
        edges.append(edge_pct)
        sides.append("over" if edge_pct > 0 else "under")

    mean_edge_pct = sum(edges) / len(edges)
    majority_side = "over" if sides.count("over") >= sides.count("under") else "under"
    agreement_frac = sides.count(majority_side) / len(sides)
    return {"mean_edge_pct": mean_edge_pct, "agreement_frac": agreement_frac, "side": majority_side}


def wnba_edge_with_uncertainty(rate_per_minute, rate_std, minutes_mean, minutes_std,
                                market_line, market_odds, n_outer=60, n_inner=1500, seed=None):
    """
    WNBA player-points analog -- redraws the per-minute scoring rate and
    projected minutes from their own uncertainty distributions. Minutes
    uncertainty matters a lot here: a role/workload surprise (blowout,
    foul trouble, minutes restriction) moves a points prop more than most
    rate-stat noise does, so minutes_std should usually be the dominant
    uncertainty term, not an afterthought.
    """
    rng = random.Random(seed)
    edges, sides = [], []
    for _ in range(n_outer):
        r = max(0.05, rng.gauss(rate_per_minute, rate_std))
        m = max(2.0, rng.gauss(minutes_mean, minutes_std))
        sims = simulate_wnba_stat(r, m, n_sims=n_inner, seed=rng.randint(0, 10**9))
        summary = summarize_over_under(sims, market_line)
        implied_over = (100 / (market_odds + 100)) if market_odds > 0 else (abs(market_odds) / (abs(market_odds) + 100))
        edge_pct = (summary["over_prob"] - implied_over) * 100
        edges.append(edge_pct)
        sides.append("over" if edge_pct > 0 else "under")

    mean_edge_pct = sum(edges) / len(edges)
    majority_side = "over" if sides.count("over") >= sides.count("under") else "under"
    agreement_frac = sides.count(majority_side) / len(sides)
    return {"mean_edge_pct": mean_edge_pct, "agreement_frac": agreement_frac, "side": majority_side}


def _poisson(rng, lam):
    """Knuth's algorithm -- avoids a numpy dependency for a single distribution."""
    if lam <= 0:
        return 0
    import math
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= l:
            return k - 1
