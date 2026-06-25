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

    def _run_nuts() -> dict[str, Any]:
        with pm.Model() as model:  # noqa: F841
            # Prior: player's true mean, centered on league average
            mu = pm.Normal("mu", mu=league_mean, sigma=league_std)

            # Prior: observation noise (half-normal keeps it positive)
            sigma = pm.HalfNormal("sigma", sigma=league_std)

            # Likelihood: what we actually observed
            pm.Normal("obs", mu=mu, sigma=sigma, observed=data)

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

        posterior_mu = trace.posterior["mu"].values.flatten()
        hdi = az.hdi(trace, var_names=["mu"], hdi_prob=0.94)

        return {
            "posterior_mean": float(np.mean(posterior_mu)),
            "posterior_std":  float(np.std(posterior_mu)),
            "hdi_low":        float(hdi["mu"].values[0]),
            "hdi_high":       float(hdi["mu"].values[1]),
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

        return {
            "sport_type":       self.sport_type,
            "context":          _context,
            "volatility_index": _vol if _context == "playoff" else 1.0,
            "active_data_n":    len(active_data),
            "posterior":        posterior,
            "win_probability":  win_prob,
        }
