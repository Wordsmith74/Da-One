"""
simulation_engine.py

Handles prediction uncertainty via Bayesian inference (PyMC) and
Monte Carlo simulation. Designed to accept input from DecisionOrchestrator.
"""

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
import numpy as np
import pymc as pm
import arviz as az
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("betting_bot")

# Maximum seconds to wait for the NUTS sampler result before falling back
# to the fast analytical (conjugate-prior) estimate.
# Uses ThreadPoolExecutor — avoids SIGALRM which is Unix-only and
# can corrupt PyMC's C-extension state when fired mid-sample.
_SAMPLER_TIMEOUT_SECONDS = 30


def _analytical_fallback(
    data: np.ndarray,
    league_mean: float,
    league_std: float,
) -> dict[str, float]:
    """
    Fast closed-form Normal–Normal conjugate posterior.
    Used when the NUTS sampler times out or fails.
    """
    n       = len(data)
    tau_0   = 1.0 / (league_std ** 2)          # prior precision
    tau_lk  = n / max(float(np.var(data)), 1e-6) # likelihood precision
    tau_n   = tau_0 + tau_lk                    # posterior precision
    mu_n    = (tau_0 * league_mean + tau_lk * float(np.mean(data))) / tau_n
    std_n   = (1.0 / tau_n) ** 0.5
    return {
        "posterior_mean": mu_n,
        "posterior_std":  std_n,
        "used_fallback":  True,
        "hdi_low":        mu_n - 1.83 * std_n,  # ~94 % HDI approximation
        "hdi_high":       mu_n + 1.83 * std_n,
        "n_obs":          n,
    }


# ---------------------------------------------------------------------------
# Bayesian Posterior Estimation
# ---------------------------------------------------------------------------

def estimate_player_metric(
    historical_data: list[float],
    league_mean: float,
    league_std: float = 5.0,
    samples: int = 2000,
    chains: int = 2,
    progressbar: bool = False,
) -> dict[str, float]:
    """
    Estimate the posterior distribution of a player metric using Bayesian
    inference. Uses historical_data (e.g., last 10 games) to update a
    Normal prior centered on league_mean.

    Model
    -----
    Prior:       mu  ~ Normal(league_mean, league_std)
                 sigma ~ HalfNormal(league_std)
    Likelihood:  obs ~ Normal(mu, sigma)
    Posterior:   P(mu | obs)

    Args:
        historical_data: Observed metric values (e.g., last 10 game points).
        league_mean:     Prior mean — the league/position average for this metric.
        league_std:      Prior standard deviation capturing uncertainty in the mean.
                         Defaults to 5.0; increase for high-variance metrics.
        samples:         Number of MCMC posterior samples per chain.
        chains:          Number of independent MCMC chains.
        progressbar:     Whether to show PyMC's sampling progress bar.

    Returns:
        dict with keys:
            posterior_mean  — point estimate of the player's true metric mean
            posterior_std   — posterior standard deviation (spread of uncertainty)
            hdi_low         — 94% highest-density interval lower bound
            hdi_high        — 94% highest-density interval upper bound
            n_obs           — number of observations used
    """
    data = np.array(historical_data, dtype=float)

    # ------------------------------------------------------------------
    # Standardize onto a unit scale before sampling (2026-07-07 fix).
    #
    # Previously mu/sigma/obs were modeled directly in the raw data's
    # units. When league_mean/league_std/data were on a large or unusual
    # scale, the NUTS leapfrog integrator's step-size adaptation could
    # blow up during early tuning, producing
    # "RuntimeWarning: overflow encountered in dot" out of
    # pymc/step_methods/hmc/quadpotential.py. Sampling still completed
    # (the warning is non-fatal), but it's a symptom of a poorly scaled
    # model rather than something to silence.
    #
    # Z-scoring against league_std (the same scale already used for the
    # priors) reparameterizes the model onto an O(1) scale — mu_z ~
    # Normal(0, 1), sigma_z ~ HalfNormal(1) — which keeps the mass matrix
    # well-conditioned regardless of the metric's raw units. This is
    # mathematically equivalent to the original model; only the sampling
    # parameterization changes. The posterior is transformed back to the
    # original units before returning.
    # ------------------------------------------------------------------
    _scale = league_std if league_std > 1e-9 else 1.0
    data_z = (data - league_mean) / _scale

    def _run_nuts() -> dict[str, Any]:
        with pm.Model() as model:  # noqa: F841
            # Prior: player's true mean, centered on league average --
            # standardized, so this is just Normal(0, 1).
            mu_z = pm.Normal("mu_z", mu=0.0, sigma=1.0)

            # Prior: observation noise (half-normal keeps it positive) --
            # standardized, so this is HalfNormal(1).
            sigma_z = pm.HalfNormal("sigma_z", sigma=1.0)

            # Likelihood: what we actually observed, in standardized units.
            pm.Normal("obs", mu=mu_z, sigma=sigma_z, observed=data_z)

            # cores=1 forces sequential (in-process) execution, which is
            # required so the ThreadPoolExecutor timeout can stop waiting
            # without needing to interrupt a subprocess.
            trace = pm.sample(
                draws=samples,
                chains=chains,
                cores=1,
                progressbar=progressbar,
                return_inferencedata=True,
                target_accept=0.9,
            )

        # Transform the standardized posterior back to the metric's
        # original units before returning.
        posterior_mu_z = trace.posterior["mu_z"].values.flatten()
        posterior_mu = league_mean + posterior_mu_z * _scale
        hdi_z = az.hdi(trace, var_names=["mu_z"], hdi_prob=0.94)

        return {
            "posterior_mean": float(np.mean(posterior_mu)),
            "posterior_std":  float(np.std(posterior_mu)),
            "hdi_low":        float(league_mean + hdi_z["mu_z"].values[0] * _scale),
            "hdi_high":       float(league_mean + hdi_z["mu_z"].values[1] * _scale),
            "n_obs":          len(data),
            "used_fallback":  False,
        }

    # Run NUTS in a worker thread with a hard wall-clock timeout.
    # On timeout the thread is abandoned (PyMC cleans up on its own),
    # and we fall back to the fast conjugate-prior analytical estimate.
    with ThreadPoolExecutor(max_workers=1) as _pool:
        _future = _pool.submit(_run_nuts)
        try:
            return _future.result(timeout=_SAMPLER_TIMEOUT_SECONDS)
        except _FuturesTimeout:
            logger.warning(
                f"[SimEngine] NUTS timed out after {_SAMPLER_TIMEOUT_SECONDS}s "
                f"(n={len(data)}, league_mean={league_mean:.1f}) — "
                "using analytical conjugate-prior fallback"
            )
            result = _analytical_fallback(data, league_mean, league_std)
            result["used_fallback"] = True
            return result
        except Exception as _exc:
            logger.warning(
                f"[SimEngine] NUTS failed ({_exc}) — "
                "using analytical conjugate-prior fallback"
            )
            result = _analytical_fallback(data, league_mean, league_std)
            result["used_fallback"] = True
            return result


# ---------------------------------------------------------------------------
# Monte Carlo Simulator
# ---------------------------------------------------------------------------

def run_monte_carlo(
    mean: float,
    std_dev: float,
    trials: int = 10_000,
    rng_seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate game outcomes by drawing `trials` samples from a Normal
    distribution parameterised by the Bayesian posterior.

    Args:
        mean:     Center of the distribution — typically posterior_mean from
                  estimate_player_metric().
        std_dev:  Spread — typically posterior_std.
        trials:   Number of simulated outcomes. Defaults to 10,000.
        rng_seed: Random seed for reproducibility. Pass None for random results.

    Returns:
        np.ndarray of shape (trials,) containing simulated metric values.
    """
    rng = np.random.default_rng(rng_seed)
    return rng.normal(loc=mean, scale=std_dev, size=trials)


def run_monte_carlo_poisson(
    mean: float,
    trials: int = 10_000,
    rng_seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate game outcomes by drawing `trials` samples from a Poisson
    distribution parameterised by a single rate (lambda = mean).

    Intended for count-style full-game totals (combined runs/points), where
    the outcome is a non-negative integer and, absent overdispersion,
    variance == mean is a reasonable assumption -- unlike the continuous
    Normal draw run_monte_carlo() uses for box-score stats.

    This is independent of the Binomial(batters_faced, k_pct) path used for
    MLB pitcher strikeouts (see models/monte_carlo.py / core/player_props.py)
    -- that path never calls into this module.

    Args:
        mean:     Poisson rate (lambda). Must be > 0; values <= 0 are
                  floored to a small positive epsilon so np.random.poisson
                  doesn't raise.
        trials:   Number of simulated outcomes. Defaults to 10,000.
        rng_seed: Random seed for reproducibility. Pass None for random results.

    Returns:
        np.ndarray of shape (trials,) containing simulated integer totals
        (as floats, so downstream code that expects run_monte_carlo()'s
        dtype -- e.g. get_win_probability()'s comparisons -- works unchanged).
    """
    rng = np.random.default_rng(rng_seed)
    lam = max(float(mean), 1e-6)
    return rng.poisson(lam=lam, size=trials).astype(float)


def run_monte_carlo_negative_binomial(
    mean: float,
    std_dev: float,
    trials: int = 10_000,
    rng_seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate game outcomes by drawing `trials` samples from a Negative
    Binomial distribution matched to the given (mean, std_dev) via
    method-of-moments.

    Use this instead of run_monte_carlo_poisson() when the historical data
    is overdispersed for a count total (variance meaningfully exceeds the
    mean) -- a plain Poisson would then understate tail risk, the same
    failure mode documented for NRFI/runs-per-inning modeling elsewhere in
    this codebase (see models/monte_carlo.py's module docstring).

    Method of moments: for NB, variance = mean + mean^2 / n, so
        n = mean^2 / (variance - mean)
        p = n / (n + mean)
    If variance <= mean (no real overdispersion -- degenerate/undefined for
    NB, since that limit IS the Poisson), falls back to a Poisson draw at
    the same mean rather than raising or fabricating a bogus n.

    Args:
        mean:     Expected total (must be > 0; floored to a small epsilon).
        std_dev:  Standard deviation of the total, as already computed by
                  the Bayesian posterior / MC-sigma-floor step upstream.
        trials:   Number of simulated outcomes. Defaults to 10,000.
        rng_seed: Random seed for reproducibility. Pass None for random results.

    Returns:
        np.ndarray of shape (trials,) containing simulated integer totals
        (as floats, matching run_monte_carlo()'s dtype).
    """
    rng = np.random.default_rng(rng_seed)
    mean_f = max(float(mean), 1e-6)
    var = float(std_dev) ** 2
    if var <= mean_f:
        return rng.poisson(lam=mean_f, size=trials).astype(float)
    n = (mean_f ** 2) / (var - mean_f)
    p = n / (n + mean_f)
    return rng.negative_binomial(n=n, p=p, size=trials).astype(float)


# ---------------------------------------------------------------------------
# Distribution selection for count-style full-game totals
# ---------------------------------------------------------------------------
# Explicit allowlist of normalized market keys that represent a combined
# integer game total (runs/points), where Poisson/Negative-Binomial is the
# statistically appropriate shape instead of the default Normal draw.
#
# Deliberately scoped narrowly and matched only against this exact set --
# this must NEVER expand to player props (player_points, pitcher_strikeouts,
# etc.), which stay on the existing Normal Monte Carlo path in
# run_monte_carlo(), and it has no interaction at all with the separate
# Binomial(batters_faced, k_pct) pitcher-strikeout pipeline in
# models/monte_carlo.py / core/player_props.py -- that code doesn't import
# from or call into this module.
_COUNT_TOTAL_MARKETS: frozenset[str] = frozenset({
    "totals", "total", "game_total", "team_total",
    "totals_first_5_innings", "f5_total",
    "totals_q1", "totals_h1",
})


def _normalize_market_key(market_type: str) -> str:
    """Local, dependency-free normalization (lowercase + underscored).
    Deliberately does NOT import core/grading_utils.py's or
    core/decision_gatekeeper.py's market_normalized() -- those two modules
    alias "totals" and "game_total" in OPPOSITE directions for their own
    dispatch/publication purposes, and pulling either in here would risk a
    circular import. This function only needs to recognize the count-total
    markets in _COUNT_TOTAL_MARKETS above, not participate in that broader
    aliasing scheme.
    """
    return (market_type or "").strip().lower().replace(" ", "_").replace("-", "_")


def select_mc_distribution(market_type: str, mean: float, std_dev: float) -> str:
    """
    Decide which Monte Carlo shape to draw a candidate's simulation from.

    Returns "normal" for everything except the explicit count-total markets
    in _COUNT_TOTAL_MARKETS -- i.e. this is a no-op for every player prop,
    moneyline/spread (which bypass this file entirely), and the pitcher-K
    Binomial path. For a recognized game-total market, picks "negative_binomial"
    when the data shows meaningful overdispersion (variance > ~1.05x mean),
    else "poisson".
    """
    if _normalize_market_key(market_type) not in _COUNT_TOTAL_MARKETS:
        return "normal"
    mean_f = max(float(mean), 1e-9)
    var = float(std_dev) ** 2
    return "negative_binomial" if var > mean_f * 1.05 else "poisson"


# ---------------------------------------------------------------------------
# Probability Calculation
# ---------------------------------------------------------------------------

def get_win_probability(
    simulated_results: np.ndarray,
    sportsbook_line: float,
) -> dict[str, float]:
    """
    Compare Monte Carlo results against a sportsbook line to derive model
    probabilities for Over and Under.

    Args:
        simulated_results: Output of run_monte_carlo().
        sportsbook_line:   The sportsbook's published line for the metric
                           (e.g., 15.5 player points O/U).

    Returns:
        dict with keys:
            over_probability   — % of trials that exceeded the line (Over hits)
            under_probability  — % of trials that fell below the line (Under hits)
            push_probability   — % of trials that landed exactly on the line
            sportsbook_line    — the line used
            trials             — total number of simulated outcomes
            edge_over          — over_probability minus 50 % (positive = model favours Over)
            edge_under         — under_probability minus 50 % (positive = model favours Under)
    """
    n = len(simulated_results)
    over  = int(np.sum(simulated_results > sportsbook_line))
    under = int(np.sum(simulated_results < sportsbook_line))
    push  = int(np.sum(simulated_results == sportsbook_line))

    over_pct  = round(over  / n * 100, 2)
    under_pct = round(under / n * 100, 2)
    push_pct  = round(push  / n * 100, 2)

    return {
        "over_probability":  over_pct,
        "under_probability": under_pct,
        "push_probability":  push_pct,
        "sportsbook_line":   sportsbook_line,
        "trials":            n,
        "edge_over":         round(over_pct  - 50.0, 2),
        "edge_under":        round(under_pct - 50.0, 2),
    }


# ---------------------------------------------------------------------------
# SimulationEngine — orchestrator-aware wrapper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fix 3: Sport × market minimum MC sigma floors
# ---------------------------------------------------------------------------
# Prevents the posterior_std from collapsing to near-zero after Bayesian
# inference on synthetic or low-variance historical data, which would
# produce unrealistic 97-100% model probabilities.
# Keys are lowercase_underscore market names (matching market_normalized()).
_MC_SIGMA_FLOOR: dict[str, dict[str, float]] = {
    "MLB": {
        "hits":               0.85,   # batter hits: avg ~1.0, σ≥0.85
        "batter_hits":        0.85,
        "strikeouts":         1.80,   # pitcher K: avg ~5-7, σ≥1.8
        "pitcher_strikeouts": 1.80,
        "totals":             3.50,
        "team_total":         2.50,
        "default":            1.50,
    },
    "NBA": {
        "points":          5.0,
        "player_points":   5.0,
        "rebounds":        2.5,
        "player_rebounds": 2.5,
        "assists":         2.0,
        "player_assists":  2.0,
        "totals":          8.0,
        "team_total":      5.0,
        "default":         4.0,
    },
    "WNBA": {
        "points":          4.5,
        "player_points":   4.5,
        "rebounds":        2.5,
        "player_rebounds": 2.5,
        "assists":         2.0,
        "player_assists":  2.0,
        "totals":          7.0,
        "team_total":      4.5,
        "default":         3.5,
    },
    "default": {"default": 1.5},
}


class SimulationEngine:
    """
    High-level wrapper that accepts input from DecisionOrchestrator and
    runs the full Bayesian → Monte Carlo → probability pipeline.

    Supports two game contexts:
      'regular'  — full season historical data, standard prior width.
      'playoff'  — short-term series-specific data + compressed prior/MC
                   variance to reflect tighter coaching rotations and
                   reduced scoring variance in elimination games.

    Usage
    -----
    >>> from core.decision_orchestrator import DecisionOrchestrator
    >>> from core.simulation_engine import SimulationEngine
    >>>
    >>> orchestrator = DecisionOrchestrator("WNBA")
    >>> engine = SimulationEngine(orchestrator)
    >>> result = engine.analyze(
    ...     historical_data=[14, 17, 12, 19, 16, 15, 18, 13, 20, 16],
    ...     league_mean=15.0,
    ...     sportsbook_line=15.5,
    ... )
    """

    # Per-sport default volatility index for playoff games.
    # Higher = more defensive intensity → tighter posterior + MC variance.
    PLAYOFF_VOLATILITY: dict[str, float] = {
        "WNBA": 1.8,
        "NBA":  1.8,
        "MLB":  1.3,   # MLB playoffs have less tactical variance per game
    }

    def __init__(self, orchestrator: Any) -> None:
        """
        Args:
            orchestrator: A DecisionOrchestrator instance. Provides sport
                          context (sport_type, weights, required_metrics).
        """
        self.orchestrator = orchestrator
        self.sport_type: str = orchestrator.sport_type

    def analyze(
        self,
        historical_data: list[float],
        league_mean: float,
        sportsbook_line: float,
        league_std: float = 5.0,
        trials: int = 10_000,
        rng_seed: int | None = 42,
        progressbar: bool = False,
        context: str = "regular",
        recent_n: int = 5,
        volatility_index: float | None = None,
        market_type: str = "",
        distribution: str | None = None,
    ) -> dict[str, Any]:
        """
        Full pipeline: Bayesian posterior → Monte Carlo → win probability,
        with optional playoff context adjustments.

        Args:
            historical_data:   Last N game values for the metric.
            league_mean:       League/position prior mean.
            sportsbook_line:   Published O/U line to beat.
            league_std:        Prior standard deviation (full-season default).
            trials:            Monte Carlo trial count.
            rng_seed:          RNG seed for reproducibility.
            progressbar:       Show PyMC sampling bar.
            context:           'regular' (default) or 'playoff'.
            recent_n:          In playoff mode, only the last recent_n
                               observations are used (series-specific weight).
            volatility_index:  Playoff intensity multiplier (≥ 1.0).
                               None → uses PLAYOFF_VOLATILITY[sport] per
                               the sport-level lookup table.  Ignored when
                               context == 'regular'.
            distribution:      Explicit Monte Carlo shape override --
                               "normal" | "poisson" | "negative_binomial".
                               None (default) auto-selects via
                               select_mc_distribution(market_type, ...):
                               every market keeps the existing Normal draw
                               EXCEPT the explicit count-total markets (full
                               game totals), which get Poisson or Negative
                               Binomial depending on observed overdispersion.
                               Passing a value here forces that shape
                               regardless of market_type. Has no effect on,
                               and no interaction with, the separate
                               Binomial(batters_faced, k_pct) pitcher-K path
                               in models/monte_carlo.py / core/player_props.py.

        Playoff adjustments (active only when context == 'playoff')
        -----------------------------------------------------------
        Three sequential adjustments tighten the simulation to reflect
        the reduced-variance, high-intensity environment of playoff games:

          1. Data selection  — slice to last recent_n games (series-specific
             short-term form replaces full-season aggregates).
          2. Prior tightening — league_std /= volatility_index, compressing
             the Bayesian prior toward the league mean (coaching adjustments
             reduce player-to-player variance in elimination games).
          3. MC variance compression — posterior_std /= volatility_index
             before Monte Carlo sampling, capturing the tighter intra-game
             performance bands seen under playoff rotations.

        Returns:
            dict containing:
                sport_type        — active sport from orchestrator
                context           — 'regular' | 'playoff'
                volatility_index  — multiplier used (1.0 in regular mode)
                active_data_n     — number of observations used by the model
                posterior         — output of estimate_player_metric()
                win_probability   — output of get_win_probability()
                mc_distribution   — which Monte Carlo shape was actually used:
                                    "normal" | "poisson" | "negative_binomial"
        """
        _context = context.lower().strip()
        if _context not in ("regular", "playoff"):
            raise ValueError(
                f"context must be 'regular' or 'playoff', got {context!r}"
            )

        # Resolve volatility index
        if volatility_index is None:
            _vol = self.PLAYOFF_VOLATILITY.get(self.sport_type, 1.5)
        else:
            _vol = max(1.0, float(volatility_index))

        active_data  = list(historical_data)
        active_std   = float(league_std)

        if _context == "playoff":
            # ── Adjustment 1: series-specific short-term data ────────────────
            if len(active_data) > recent_n:
                active_data = active_data[-recent_n:]

            # ── Adjustment 2: prior tightening ────────────────────────────────
            active_std /= _vol

        # ── Bayesian posterior inference ─────────────────────────────────────
        posterior = estimate_player_metric(
            historical_data=active_data,
            league_mean=league_mean,
            league_std=active_std,
            progressbar=progressbar,
        )

        # ── Adjustment 3 (playoff only): MC variance compression ─────────────
        mc_std = posterior["posterior_std"]
        if _context == "playoff":
            mc_std /= _vol

        # ── Fix 3: Apply sport/market-specific minimum sigma floor ───────────
        # Prevents collapsed posteriors (e.g. synthetic data with low variance)
        # from producing unrealistic 97-100% win probabilities.
        _sport_floors = _MC_SIGMA_FLOOR.get(self.sport_type, _MC_SIGMA_FLOOR["default"])
        _mkt_key      = market_type.lower().replace(" ", "_").replace("-", "_")
        _sigma_floor  = _sport_floors.get(_mkt_key, _sport_floors.get("default", 1.5))
        if mc_std < _sigma_floor:
            logger.debug(
                f"[SimEngine] σ floor: {mc_std:.4f} → {_sigma_floor:.4f} "
                f"({self.sport_type}/{market_type or 'default'})"
            )
            mc_std = _sigma_floor

        # ── Monte Carlo simulation ────────────────────────────────────────────
        # _dist stays "normal" (identical behavior to before this feature was
        # added) for every market except the explicit count-total allowlist
        # in select_mc_distribution() -- player props, moneyline/spread, and
        # the separate pitcher-K Binomial pipeline are all unaffected.
        _dist = distribution or select_mc_distribution(
            market_type, posterior["posterior_mean"], mc_std
        )

        if _dist == "poisson":
            simulated = run_monte_carlo_poisson(
                mean=posterior["posterior_mean"],
                trials=trials,
                rng_seed=rng_seed,
            )
        elif _dist == "negative_binomial":
            simulated = run_monte_carlo_negative_binomial(
                mean=posterior["posterior_mean"],
                std_dev=max(mc_std, 1e-6),
                trials=trials,
                rng_seed=rng_seed,
            )
        else:
            simulated = run_monte_carlo(
                mean=posterior["posterior_mean"],
                std_dev=max(mc_std, 1e-6),   # floor avoids degenerate zero-std runs
                trials=trials,
                rng_seed=rng_seed,
            )

        win_prob = get_win_probability(
            simulated_results=simulated,
            sportsbook_line=sportsbook_line,
        )

        # NOTE: posterior["posterior_std"] is deliberately left as the RAW
        # (pre-floor, pre-playoff-compression) NUTS/analytical-fallback
        # value here -- core/stability_filter.py's relative-uncertainty
        # tiers (8/12/15% etc.) were calibrated against that raw fitted
        # value, not the floor. An earlier version of this fix made
        # posterior_std reflect the floored mc_std instead, on the theory
        # that the stability gate should see "what was actually simulated"
        # -- but confirmed by replay (2026-07-27 MLB slate): every single
        # pitcher_strikeouts candidate was rejected with relative σ pinned
        # at exactly the 1.80 floor value regardless of real data quality,
        # because that floor is large relative to typical low-mean K
        # projections (2-10), which alone exceeds the 15% ceiling almost
        # every time. The floor's job is bounding the Monte Carlo
        # win-probability calculation (mc_std, used just above) against an
        # unrealistically collapsed posterior -- it was never meant to also
        # feed the separately-calibrated stability gate. mc_std is exposed
        # here as posterior_std_mc so callers that specifically want the
        # floor-adjusted value (e.g. for debugging) can still get it.
        posterior["posterior_std_mc"] = mc_std

        return {
            "sport_type":       self.sport_type,
            "context":          _context,
            "volatility_index": _vol if _context == "playoff" else 1.0,
            "active_data_n":    len(active_data),
            "posterior":        posterior,
            "win_probability":  win_prob,
            "mc_distribution":  _dist,
        }
