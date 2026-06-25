"""
main.py — Production entry point for the Multi-Sport Prediction Engine.

Operational modes
-----------------
    python3 main.py --mode run      Full pipeline: recap → 60 s → picks (all sports)
    python3 main.py --mode close    Record game results in the DB
    python3 main.py --mode refine   Auto-tune sports_metrics.json weights
    python3 main.py --mode recap    Morning recap only (sent_to_group bets)
    python3 main.py --mode picks    Picks broadcast only
    python3 main.py --mode open       List all open bets
    python3 main.py --mode calibrate  Side-by-side regular vs playoff tier comparison

Optional flags
--------------
    --sport   WNBA|NBA|MLB|ALL      Scope (default: ALL)
    --bet-id  <id>                  Required for --mode close (single bet)
    --outcome win|loss|push         Required for --mode close (single bet)
    --dry-run                       Print messages; never call Telegram API or sleep

Error handling
--------------
    Every mode runs inside a guarded executor.  On any unhandled exception:
      1. Full traceback written to betting_bot.log
      2. 🚨 CRITICAL ALERT pushed to the Telegram channel
      3. Process exits with code 1 (cron/systemd marks the run as failed)
    On clean exit the process exits with code 0.

Cron examples  (edit with `crontab -e`, adjust path as needed)
-------------------------------------------------------------
    # Full broadcast at 9:00 AM ET every day
    0 9 * * * /usr/bin/python3 /home/runner/workspace/main.py --mode run

    # Model refinement every Sunday at 2:00 AM ET
    0 2 * * 0 /usr/bin/python3 /home/runner/workspace/main.py --mode refine

Pipeline note
-------------
    _run_sport_pipeline() uses mock historical data until real API connectors
    are wired in via api_connector.py.  With idealized mock data the Bayesian
    posterior saturates (over_probability → ~100 %).  Real API data produces
    realistic spreads (edge 8–22 %, confidence 70–97 %).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths — always absolute so the script works from any cwd (important for cron)
# ---------------------------------------------------------------------------

ROOT_DIR  = Path(__file__).resolve().parent
LOG_PATH  = ROOT_DIR / "betting_bot.log"
# Rejection shadow-log: data/bet_rejects.jsonl (written by core.reject_logger)
# ---------------------------------------------------------------------------
# Logging — file (DEBUG+) and console (INFO+)
# Set up before any imports that might emit their own log messages.
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    _logger = logging.getLogger("betting_bot")
    if _logger.handlers:          # guard against double-init in test environments
        return _logger
    _logger.setLevel(logging.DEBUG)

    _fmt_full  = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _fmt_brief = logging.Formatter("%(message)s")

    fh = RotatingFileHandler(
        LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_fmt_full)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_fmt_brief)

    _logger.addHandler(fh)
    _logger.addHandler(ch)
    return _logger


logger = _setup_logging()

# ---------------------------------------------------------------------------
# Engine imports — all layers explicit for traceability
# ---------------------------------------------------------------------------

# ── Prediction layer ────────────────────────────────────────────────────────
from core.decision_orchestrator import (          # noqa: E402
    DecisionOrchestrator,
    MissingMetricError,
    UnsupportedSportError,
)
from core.simulation_engine import SimulationEngine  # noqa: E402
from core.decision_gatekeeper import (            # noqa: E402
    Bet,
    Tier,
    market_normalized,
    run_gatekeeper,
)
from core.market_governance import (              # noqa: E402
    is_publication_eligible,
    publication_priority,
)

try:
    from core.reject_logger import (             # noqa: E402
        log_rejected_bet_obj  as _log_reject_bet,
        log_rejected_candidate as _log_reject_cand,
    )
    _REJECT_LOGGER_AVAILABLE = True
except ImportError:
    _REJECT_LOGGER_AVAILABLE = False

# ── Data / results layer ────────────────────────────────────────────────────
from core.results_tracker import (               # noqa: E402
    close_bet,
    format_morning_recap,
    get_open_bets,
    init_db,
    log_bet_dict,
    update_model_priors,
)

# ── Time utilities ───────────────────────────────────────────────────────────
from core.time_utils import (                    # noqa: E402
    convert_to_est,
    format_est,
    format_est_date,
    format_est_short,
    localize_utc,
    now_est,
    now_utc,
)

# ── Output / delivery layer ──────────────────────────────────────────────────
from output.telegram_formatter import (          # noqa: E402
    BetDisplay,
    send_daily_picks,
    send_daily_recap,
)

# ── Intelligence layer ────────────────────────────────────────────────────────
try:
    from core.intelligence.rest_travel import get_rest_travel_factor   # noqa: E402
    from core.intelligence.lineup_intel import get_lineup_intel         # noqa: E402
    from core.intelligence.stat_model import get_stat_model_factor      # noqa: E402
    from core.intelligence.clv_tracker import snapshot_odds             # noqa: E402
    from core.intelligence.pitcher_intel import get_pitcher_intel       # noqa: E402
    from core.intelligence.rivalry_intel import get_rivalry_intel       # noqa: E402
    from core.intelligence.venue_intel import get_venue_factor          # noqa: E402
    _INTEL_AVAILABLE = True
except ImportError as _intel_import_err:
    logger.debug(f"Intelligence layer not available: {_intel_import_err}")
    _INTEL_AVAILABLE = False

try:
    from core.player_props import get_player_prop_candidates            # noqa: E402
    _PROPS_AVAILABLE = True
except ImportError as _props_import_err:
    logger.debug(f"Player props module not available: {_props_import_err}")
    _PROPS_AVAILABLE = False

try:
    from core.intelligence.line_movement import get_line_movement_signals  # noqa: E402
    _LINE_MOVEMENT_AVAILABLE = True
except ImportError as _lm_import_err:
    logger.debug(f"Line movement module not available: {_lm_import_err}")
    _LINE_MOVEMENT_AVAILABLE = False

# ── Data resilience layer ─────────────────────────────────────────────────────
try:
    from core.data_fetcher import check_connectivity   # noqa: E402
    from core import safe_state as _safe_state         # noqa: E402
    _RESILIENCE_AVAILABLE = True
except ImportError as _res_import_err:
    logger.debug(f"Data resilience layer not available: {_res_import_err}")
    _RESILIENCE_AVAILABLE = False

# ── V3.0 Calibration layer ───────────────────────────────────────────────────
try:
    from core.edge_calibrator import calibrate_edge as _v3_calibrate_edge  # noqa: E402
    from core.market_agreement import compute_market_agreement as _v3_compute_mas  # noqa: E402
    _V3_CALIBRATION_AVAILABLE = True
    logger.debug("V3.0 calibration layer loaded (edge_calibrator + market_agreement).")
except ImportError as _v3_import_err:
    logger.debug(f"V3.0 calibration layer not available: {_v3_import_err}")
    _V3_CALIBRATION_AVAILABLE = False

# ── Market Coverage Framework ────────────────────────────────────────────────
try:
    from core.game_markets import fetch_expanded_game_candidates  # noqa: E402
    _GAME_MARKETS_AVAILABLE = True
    logger.debug("Market coverage framework loaded (game_markets).")
except ImportError as _gm_import_err:
    logger.debug(f"game_markets not available: {_gm_import_err}")
    _GAME_MARKETS_AVAILABLE = False

try:
    from core.market_scanner import apply_per_game_caps as _apply_per_game_caps  # noqa: E402
    from core.market_scanner import log_market_audit as _log_market_audit  # noqa: E402
    from core.market_scanner import get_market_historical_roi as _get_market_roi  # noqa: E402
    _MARKET_SCANNER_AVAILABLE = True
    logger.debug("Market scanner loaded (composite ranking + per-game caps).")
except ImportError as _ms_import_err:
    logger.debug(f"market_scanner not available: {_ms_import_err}")
    _MARKET_SCANNER_AVAILABLE = False

try:
    from core.signal_confirmation import gate_signals as _gate_signals
    _SIGNAL_CONFIRMATION_AVAILABLE = True
    logger.debug("Signal confirmation loaded (Candidate → Confirmed → Locked lifecycle).")
except ImportError as _sc_import_err:
    logger.debug(f"signal_confirmation not available: {_sc_import_err}")
    _SIGNAL_CONFIRMATION_AVAILABLE = False

try:
    from core.stability_filter import check_stability as _check_stability
    _STABILITY_FILTER_AVAILABLE = True
    logger.debug("Stability filter loaded (posterior-variance gate).")
except ImportError as _sf_import_err:
    logger.debug(f"stability_filter not available: {_sf_import_err}")
    _STABILITY_FILTER_AVAILABLE = False

try:
    from core.conflict_guardian import check_locked_conflict as _check_locked_conflict
    _CONFLICT_GUARDIAN_AVAILABLE = True
    logger.debug("Conflict guardian loaded (locked-pick replacement threshold).")
except ImportError as _cg_import_err:
    logger.debug(f"conflict_guardian not available: {_cg_import_err}")
    _CONFLICT_GUARDIAN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SPORTS    = ["WNBA", "NBA", "MLB"]
RECAP_DELAY_S = 60     # seconds between recap and picks in --mode run


# ---------------------------------------------------------------------------
# Mock game-candidate data
# ---------------------------------------------------------------------------
# Each entry is one bet candidate.  historical_data drives engine.analyze();
# american_odds, factor, and game_time_utc drive BetDisplay formatting.
#
# Replace this function with real API connector calls (api_connector.py) once
# live data sources are wired in.

def _get_game_candidates(sport: str) -> list[dict[str, Any]]:
    """
    Fetch today's bet candidates from The Odds API (live data).

    Pulls game totals (O/U) for the requested sport, filters to today in ET,
    and returns one candidate per game in the format required by
    _run_sport_pipeline().  Key rotation is handled inside odds_client.
    """
    try:
        from core.odds_client import fetch_todays_candidates
        candidates = fetch_todays_candidates(sport)
        logger.info(f"  [{sport}] Odds API: {len(candidates)} candidate(s) for today.")
        return candidates
    except Exception as exc:
        logger.warning(f"  [{sport}] Odds API fetch failed — {exc}")
        return []


# ---------------------------------------------------------------------------
# Intelligence enrichment — Step 1.5 of the pipeline
# ---------------------------------------------------------------------------

def _enrich_candidate(c: dict[str, Any], sport: str) -> None:
    """
    Enrich a single game candidate with contextual intelligence signals.

    Calls rest_travel, lineup_intel, and stat_model in sequence; stores
    results in c["_intel"] and appends human-readable notes to c["factor"].
    All sub-calls are individually guarded — one failure never blocks the rest.
    """
    if not _INTEL_AVAILABLE:
        return

    team            = c.get("team", "")
    market          = c.get("market", "")
    direction       = c.get("direction", "over")
    game_time_utc   = c.get("game_time_utc")
    sportsbook_line = c.get("sportsbook_line")

    intel: dict[str, Any] = {}
    supplements: list[str] = []

    try:
        rt = get_rest_travel_factor(team, sport, game_time_utc)
        intel["rest_travel"] = rt
        if rt.factor_text:
            supplements.append(rt.factor_text)
    except Exception as exc:
        logger.debug(f"[intel] rest_travel failed for {team}: {exc}")

    try:
        li = get_lineup_intel(team, sport, bet_on_this_team=True)
        intel["lineup"] = li
        if li.factor_text:
            supplements.append(li.factor_text)
    except Exception as exc:
        logger.debug(f"[intel] lineup_intel failed for {team}: {exc}")

    try:
        sm = get_stat_model_factor(team, market, sport, sportsbook_line, direction)
        intel["stat_model"] = sm
        if sm.factor_text:
            supplements.append(sm.factor_text)
    except Exception as exc:
        logger.debug(f"[intel] stat_model failed for {team}: {exc}")

    # ── Pitcher ERA + Rivalry / H2H context (MLB only) ─────────────────────
    if sport.upper() == "MLB":
        game_id   = c.get("game_id", "")
        away_abbr = game_id.split("@")[0] if "@" in game_id else ""
        home_abbr = team
        away_name = c.get("away_team", "")
        home_name = c.get("home_team", "")

        # Pitcher ERA — shifts Bayesian prior in-place so starter quality
        # is reflected in the posterior (not just a post-hoc edge nudge).
        if "total" in market.lower() and home_abbr and away_abbr:
            try:
                pi = get_pitcher_intel(home_abbr, away_abbr)
                intel["pitcher"] = pi
                # Store in candidate so integrity filter can detect it via raw_result
                c["pitcher_intel"] = pi
                if pi.factor_text:
                    supplements.append(pi.factor_text)
                if pi.league_mean_adjustment != 0.0:
                    delta = pi.league_mean_adjustment
                    c["league_mean"] = round(c["league_mean"] + delta, 2)
                    c["historical_data"] = [
                        round(v + delta, 1) for v in c["historical_data"]
                    ]
                    logger.debug(
                        f"[intel] pitcher ERA adj {delta:+.2f} → "
                        f"league_mean now {c['league_mean']:.2f} for {home_abbr}"
                    )
            except Exception as exc:
                logger.debug(f"[intel] pitcher_intel failed for {team}: {exc}")

        # ── Pitcher Workload — expected IP/K/RA for both starters ────────────
        # Runs after pitcher_intel so both the FIP prior shift and the
        # workload projection are available on the candidate dict.
        # Stored as c["workload_home"] / c["workload_away"] for the
        # gatekeeper, formatter, and future downstream modules.
        if home_abbr and away_abbr:
            try:
                from core.pitcher_workload import get_game_workload_pair
                _wl_home, _wl_away = get_game_workload_pair(
                    home_abbr, away_abbr,
                )
                if _wl_home:
                    c["workload_home"] = _wl_home
                    if _wl_home.discrepancy_flag:
                        logger.info(
                            f"[intel] WORKLOAD DISCREPANCY (home) "
                            f"{home_abbr}: {_wl_home.discrepancy_detail}"
                        )
                if _wl_away:
                    c["workload_away"] = _wl_away
                    if _wl_away.discrepancy_flag:
                        logger.info(
                            f"[intel] WORKLOAD DISCREPANCY (away) "
                            f"{away_abbr}: {_wl_away.discrepancy_detail}"
                        )
                if _wl_home and _wl_away:
                    logger.debug(
                        f"[intel] workload {away_abbr}@{home_abbr}: "
                        f"home {_wl_home.pitcher_name} "
                        f"{_wl_home.expected_ip:.1f}ip/"
                        f"{_wl_home.expected_k:.1f}k  "
                        f"away {_wl_away.pitcher_name} "
                        f"{_wl_away.expected_ip:.1f}ip/"
                        f"{_wl_away.expected_k:.1f}k"
                    )
            except Exception as exc:
                logger.debug(f"[intel] pitcher_workload failed for {team}: {exc}")

        # Rivalry / Head-to-Head context — volatility penalty for
        # division/major rivalry games; modest H2H edge for competitive underdogs.
        try:
            ri = get_rivalry_intel(
                home_name  = home_name,
                away_name  = away_name,
                sport      = sport,
                home_abbr  = home_abbr,
                away_abbr  = away_abbr,
            )
            intel["rivalry"] = ri
            if ri.factor_text:
                supplements.append(ri.factor_text)
            if ri.tags:
                logger.debug(
                    f"[intel] rivalry {away_abbr}@{home_abbr}: "
                    f"{ri.rivalry_level}  tags={ri.tags}  adj={ri.edge_adjustment:+.1f}"
                )
        except Exception as exc:
            logger.debug(f"[intel] rivalry_intel failed for {team}: {exc}")

        # Venue environment — home/away splits, park factors, road competency.
        try:
            # Determine bet direction for park factor sign (over = positive, under = negative)
            _direction = "over"
            _market = c.get("market", "")
            if "under" in _market.lower():
                _direction = "under"

            vf = get_venue_factor(
                home_name  = home_name,
                away_name  = away_name,
                sport      = sport,
                home_abbr  = home_abbr,
                away_abbr  = away_abbr,
                direction  = _direction,
            )
            intel["venue"] = vf
            if vf.factor_text:
                supplements.append(vf.factor_text)
            if vf.tags:
                logger.debug(
                    f"[intel] venue {away_abbr}@{home_abbr}: "
                    f"park_adj={vf.park_adj:+.1f}  split_adj={vf.venue_split_adj:+.1f}  "
                    f"total={vf.edge_adjustment:+.1f}  tags={vf.tags}"
                )
        except Exception as exc:
            logger.debug(f"[intel] venue_intel failed for {team}: {exc}")

    c["_intel"] = intel

    if supplements:
        base = c.get("factor", "")
        c["factor"] = (base.rstrip() + " | " + " | ".join(supplements)).lstrip(" | ")


# ---------------------------------------------------------------------------
# Explicit per-sport prediction pipeline
# ---------------------------------------------------------------------------

def _derive_bet_params(
    sim: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[float, float, float]:
    """
    Extract edge_percentage, confidence_score, and model_probability
    from a SimulationEngine.analyze() result.

    edge_percentage
        (model_probability − implied_probability_from_odds) capped at 50 %.
        Positive = model favours the bet direction vs the book's price.

    confidence_score  (0–100)
        Derived from the posterior z-score: how many posterior standard
        deviations the sportsbook line sits from the posterior mean.
        Formula: min(99, 50 + z × 25).
        Maps → z=1.52 → 88 % (S threshold), z=1.80 → 95 % (S+ threshold).

        The z-score denominator uses the same sigma floor applied to the
        Monte Carlo simulation.  This prevents synthetic posteriors (n=15
        Gaussian draws) from collapsing posterior_std to ~0.3–0.5 and
        producing artificially high z-scores — and thus inflated confidence —
        even when the underlying signal is thin.

    model_probability
        Raw % (0–100) that the metric clears the line in the direction bet.
    """
    from core.simulation_engine import _MC_SIGMA_FLOOR
    from core.decision_gatekeeper import market_normalized

    posterior_mean = sim["posterior"]["posterior_mean"]
    posterior_std  = sim["posterior"]["posterior_std"]
    line           = candidate["sportsbook_line"]
    direction      = candidate["direction"].lower()
    odds           = candidate["american_odds"]

    # Model win probability
    if direction == "over":
        model_prob = sim["win_probability"]["over_probability"]
    else:
        model_prob = sim["win_probability"]["under_probability"]

    # Implied probability from American odds
    if odds < 0:
        implied = abs(odds) / (abs(odds) + 100) * 100
    else:
        implied = 100 / (odds + 100) * 100

    edge = round(min(50.0, model_prob - implied), 2)

    # Apply the same sigma floor the MC simulation uses so the confidence
    # z-score cannot be inflated by a collapsed synthetic posterior.
    _sport      = sim.get("sport_type", "default")
    _mkt        = market_normalized(candidate.get("market", ""))
    _floors     = _MC_SIGMA_FLOOR.get(_sport, _MC_SIGMA_FLOOR["default"])
    _sigma_floor = _floors.get(_mkt, _floors.get("default", 1.5))
    _conf_std   = max(posterior_std, _sigma_floor)

    # Posterior z-score → confidence
    z          = abs(posterior_mean - line) / max(_conf_std, 0.01)
    confidence = round(min(99.0, 50.0 + z * 25.0), 1)

    return edge, confidence, round(model_prob, 1)


def _calibrate_mlb_confidence(raw_prob: float) -> float:
    """
    Compress extreme Bayesian win-probability scores into realistic betting
    ranges for MLB display.  Applies log-odds compression (k=0.22).

    The compression factor k=0.22 produces:
        raw 99 % → ~73 %   raw 95 % → ~66 %
        raw 90 % → ~62 %   raw 85 % → ~59 %
        raw 80 % → ~58 %   raw 70 % → ~55 %
        raw 50 % → 50 %    (neutral — no distortion at coin-flip)

    Only applied to the *displayed* model_probability field.
    confidence_score (tier driver) is left untouched.
    """
    import math
    p = max(0.001, min(0.999, raw_prob / 100.0))
    log_odds  = math.log(p / (1.0 - p))
    compressed = log_odds * 0.22
    calibrated = 1.0 / (1.0 + math.exp(-compressed))
    return round(calibrated * 100.0, 1)


# ---------------------------------------------------------------------------
# Directive 2: Fail-fast candidate validation before Bayesian simulation
# ---------------------------------------------------------------------------

_REQUIRED_SIM_FIELDS = (
    "bet_id", "historical_data", "sportsbook_line",
    "american_odds", "direction", "market",
    "league_mean", "league_std",
)

# ---------------------------------------------------------------------------
# Directive 2a: Per-sport, per-market sanity bounds
# ---------------------------------------------------------------------------

# Sportsbook line must fall within these physical ranges.
# Any line outside bounds is either a mislabeled market (e.g. a game-total
# line filed under team_total) or a data error.  Reject before simulation.
_LINE_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "WNBA": {"totals": (135.0, 200.0), "team_total": (55.0, 100.0)},
    "NBA":  {"totals": (185.0, 275.0), "team_total": (85.0, 150.0)},
    "MLB":  {"totals": (5.0,   22.0),  "team_total": (1.5,  18.0)},
}

# Historical data values (individual observations) must fall within these
# ranges.  A majority violation signals a wrong-scale prior was applied
# (e.g. NBA-scale synthetic history used for a WNBA candidate).
_HIST_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "WNBA": {"totals": (130.0, 200.0), "team_total": (40.0, 115.0)},
    "NBA":  {"totals": (180.0, 275.0), "team_total": (80.0, 155.0)},
    "MLB":  {"totals": (5.0,   22.0),  "team_total": (1.5,  18.0)},
}

# Minimum closed graded non-prop bets required before full tier access.
# Below this count the sport is in early-calibration; team/game total bets
# are capped at Value tier regardless of edge and confidence.
_MIN_THIN_DATA_BETS = 15

# ---------------------------------------------------------------------------
# MLB sharp-signal discount multipliers
# MLB line movement is often lagging environmental noise (weather/bullpen/park),
# not genuine sharp action.  Discount non-steam signals heavily; full-steam
# (multi-book simultaneous move) retains 60% weight.
# ---------------------------------------------------------------------------
_MLB_SIGNAL_DISCOUNT: dict[str, float] = {
    "line_confirming":  0.30,
    "line_opposing":    0.40,
    "steam_confirming": 0.60,
    "steam_opposing":   0.50,
}
_WNBA_SIGNAL_DISCOUNT: dict[str, float] = {
    "line_confirming":  0.50,
    "line_opposing":    0.50,
    "steam_confirming": 0.60,
    "steam_opposing":   0.60,
}


def _validate_candidate_for_simulation(
    c: dict[str, Any],
    sport: str = "",
) -> tuple[bool, str]:
    """
    Fail-fast guard applied before engine.analyze().

    Returns (True, "") when the candidate is structurally sound, or
    (False, reason) to abort immediately.  Prevents malformed records from
    entering the NUTS sampler — a corrupt input can stall the sampler for
    the full tuning budget before raising an error deep in PyMC.

    Checks (in order)
    -----------------
    1. All required fields are present and non-empty/non-zero.
    2. historical_data is a list with ≥ 5 elements (minimum for NUTS).
    3. sportsbook_line is a positive number.
    4. american_odds is non-zero (zero = implied probability undefined).
    5. [Fix 2] sportsbook_line is within physical bounds for sport + market.
    6. [Fix 1] historical_data values are within range for sport + market.
       — Majority (>50 %) out-of-range → reject (wrong-scale prior).
       — Rolling-3 divergence >30 % from full mean → warn (non-fatal).
    """
    for field in _REQUIRED_SIM_FIELDS:
        val = c.get(field)
        if val is None or val == "" or (isinstance(val, list) and len(val) == 0):
            return False, f"missing required field '{field}'"
    hist = c.get("historical_data")
    if not isinstance(hist, list) or len(hist) < 5:
        n = len(hist) if isinstance(hist, list) else 0
        return False, f"historical_data too short (len={n}, need ≥5)"
    if c.get("sportsbook_line", 0.0) <= 0.0:
        return False, f"invalid sportsbook_line={c.get('sportsbook_line')}"
    if c.get("american_odds", 0) == 0:
        return False, "american_odds is zero (implied probability undefined)"

    # Phase 1: data quality gate — block NO PLAY candidates before simulation
    drs = c.get("data_reliability_score")
    if drs is not None and drs < 40:
        return False, (
            f"data reliability score {drs}/100 < 40 — "
            "NO PLAY: insufficient data (real stats unavailable)"
        )

    sport_up = sport.upper()
    mkt      = market_normalized(c.get("market", ""))

    # Fix 2: sportsbook line must be physically plausible for sport + market
    line_sport = _LINE_BOUNDS.get(sport_up, {})
    if mkt in line_sport:
        lo, hi = line_sport[mkt]
        line   = c.get("sportsbook_line", 0.0)
        if not (lo <= line <= hi):
            return False, (
                f"sportsbook_line {line} out of bounds for {sport_up}/{mkt} "
                f"(valid range {lo}–{hi}); likely mislabeled market or data error"
            )

    # Fix 1: historical data values must be in range for sport + market
    hist_sport = _HIST_BOUNDS.get(sport_up, {})
    if mkt in hist_sport and isinstance(hist, list):
        hlo, hhi   = hist_sport[mkt]
        out_of_rng = [v for v in hist if not (hlo <= v <= hhi)]
        if len(out_of_rng) > len(hist) // 2:
            return False, (
                f"historical_data majority out of range for {sport_up}/{mkt}: "
                f"{len(out_of_rng)}/{len(hist)} values outside [{hlo}, {hhi}] "
                f"— wrong-scale prior likely applied"
            )
        # Rolling consistency — non-fatal warning only
        if len(hist) >= 5:
            full_mean = sum(hist) / len(hist)
            recent3   = sum(hist[-3:]) / 3
            if full_mean and abs(recent3 - full_mean) / full_mean > 0.30:
                logger.warning(
                    f"  [{sport}] {c.get('bet_id', '?')}: rolling-3 mean "
                    f"{recent3:.1f} diverges >30 %% from full mean {full_mean:.1f} "
                    "(hot/cold streak — non-fatal, passing to simulation)"
                )

    return True, ""


# ── Per-run candidate cache ──────────────────────────────────────────────────
# Populated inside _run_sport_pipeline after all candidate sources are merged.
# Consumed by _refresh_open_bet_odds() to silently update live odds/lines in DB.
_FRESH_CANDIDATES: dict[str, list[dict[str, Any]]] = {}


def _run_sport_pipeline(
    sport: str,
    orchestrator: DecisionOrchestrator,
    engine: SimulationEngine,
) -> tuple[list[BetDisplay], list[dict[str, Any]]]:
    """
    Full prediction pipeline for one sport.

    Step 1  — Fetch game candidates (cache → live API).
    Step 1b — Fail-fast validation: skip structurally invalid candidates
               before they reach the Bayesian sampler (Directive 2).
    Step 2  — For each valid candidate: engine.analyze()
                  → Bayesian posterior (PyMC NUTS)
                  → Monte Carlo win probability
    Step 3  — Derive edge_percentage + confidence_score from posterior.
    Step 4  — Build Bet objects; convert game timestamps to ET.
    Step 5  — run_gatekeeper(): tier assignment + conflict detection.
    Step 6  — Wrap approved bets in BetDisplay for the broadcast layer.

    Returns
    -------
    (display_bets, log_dicts)
        display_bets : list ready for send_daily_picks()
        log_dicts    : list ready for log_bet_dict()
    """
    candidates = _get_game_candidates(sport)
    if not candidates:
        logger.info(f"  [{sport}] No game-total candidates — will still check player props.")

    # ── Step 1.5: Intelligence enrichment ───────────────────────────────────
    if _INTEL_AVAILABLE:
        logger.debug(
            f"  [{sport}] Enriching {len(candidates)} candidate(s) "
            "with rest/travel, lineup, and stat intelligence…"
        )
        for c in candidates:
            if not c.get("player"):   # props don't need team-level intel
                _enrich_candidate(c, sport)

    # ── Step 1.6: Player prop candidates ────────────────────────────────────
    if _PROPS_AVAILABLE:
        try:
            prop_cands = get_player_prop_candidates(sport)
            if prop_cands:
                candidates = candidates + prop_cands
                logger.info(
                    f"  [{sport}] {len(prop_cands)} player prop candidate(s) added."
                )
        except Exception as _pe:
            logger.warning(f"  [{sport}] player_props fetch failed: {_pe}")

    # ── Step 1.8: Expanded game market candidates ────────────────────────────
    # Fetches F5/team/Q1/H1 totals and ML/spread candidates in one API call.
    # Scaled-total candidates flow through engine.analyze(); ML/spread carry
    # precomputed_edge and bypass the NUTS sampler.
    if _GAME_MARKETS_AVAILABLE:
        try:
            _expanded = fetch_expanded_game_candidates(sport)
            if _expanded:
                candidates = candidates + _expanded
                logger.info(
                    f"  [{sport}] {len(_expanded)} expanded market candidate(s) added "
                    f"({sum(1 for c in _expanded if c.get('precomputed_edge') is not None)} precomputed, "
                    f"{sum(1 for c in _expanded if c.get('precomputed_edge') is None)} bayesian)."
                )
        except Exception as _gme:
            logger.warning(f"  [{sport}] game_markets fetch failed: {_gme}")

    # ── Step 1.9: Market scope gate ──────────────────────────────────────────
    # Enforce the System Scope Definition (core/market_gate.py).
    # Candidates outside scope are silently dropped before any modeling —
    # no simulation, no edge calculation, no confidence scoring.
    try:
        from core.market_gate import filter_candidates as _gate_filter, log_market_filter_summary as _gate_summary
        _total_before = len(candidates)
        candidates, _blocked = _gate_filter(candidates, sport)
        _blocked_reasons: dict[str, int] = {}
        for _bc in _blocked:
            _mkt_raw = _bc.get("market_key") or _bc.get("market") or "unknown"
            _blocked_reasons[_mkt_raw] = _blocked_reasons.get(_mkt_raw, 0) + 1
        _gate_summary(sport, _total_before, len(candidates), len(_blocked), _blocked_reasons)
        if _blocked and _REJECT_LOGGER_AVAILABLE:
            _today_str = now_est().strftime("%Y-%m-%d")
            for _bc in _blocked:
                _log_reject_cand(
                    sport, _bc, "market_gate",
                    f"REJECTED_MARKET_BLOCK: market '{_bc.get('market_key') or _bc.get('market')}' "
                    f"not in scope for {sport}",
                    _today_str,
                )
    except Exception as _gate_exc:
        logger.warning(f"  [{sport}] market_gate failed (non-fatal): {_gate_exc}")

    # ── Guard: all sources exhausted ────────────────────────────────────────
    if not candidates:
        logger.info(f"  [{sport}] No candidates available after all sources — skipping.")
        return [], []

    # Cache fresh candidates so _refresh_open_bet_odds() can update open bets
    # with current market data without an extra API call.
    _FRESH_CANDIDATES[sport] = list(candidates)

    # ── Step 1.7: Line movement signals ─────────────────────────────────────
    # Re-fetch current lines (bypasses slate cache) and compare against the
    # opening lines recorded at slate time.  A confirming move boosts edge;
    # an opposing move (market corrected against us) suppresses the pick.
    _lm_signals: dict[str, tuple[float, str, str]] = {}
    if _LINE_MOVEMENT_AVAILABLE:
        try:
            _lm_signals = get_line_movement_signals(sport, candidates)
            if _lm_signals:
                logger.info(
                    f"  [{sport}] Line movement: "
                    f"{sum(1 for adj,_,__ in _lm_signals.values() if adj>0)} confirming, "
                    f"{sum(1 for adj,_,__ in _lm_signals.values() if adj<0)} opposing."
                )
        except Exception as _lme:
            logger.debug(f"  [{sport}] line_movement failed: {_lme}")

    # ── Steps 2 & 3: Bayesian inference + edge derivation ───────────────────
    processed: list[tuple[dict[str, Any], float, float, float]] = []

    # Signal boosts (post-gate): sharp signals adjust ranking but never gate pass/fail.
    # Keyed by bet_id → total edge boost (accumulated from line movement + consensus).
    _signal_boosts: dict[str, float] = {}

    for c in candidates:
        bet_id = c["bet_id"]

        # ── Precomputed candidates (ML/spread — bypass NUTS sampler) ─────────
        # game_markets.py sets precomputed_edge on moneyline and spread
        # candidates that use a normal-approximation win probability model.
        # These skip validation + engine.analyze() + V3 calibration but still
        # compute effective edge and market agreement so the gatekeeper has the
        # full signal set.
        if c.get("precomputed_edge") is not None:
            _pc_edge  = float(c["precomputed_edge"])
            _pc_conf  = float(c.get("precomputed_confidence", 72.0))
            _pc_prob  = float(c.get("precomputed_model_prob", 0.5))
            try:
                from core.market_intelligence import compute_effective_edge as _cee_pc
                _eff_edge_pc = _cee_pc(
                    raw_edge           = _pc_edge,
                    sharp_signal       = c.get("sharp_signal", "no_sharp"),
                    rlm_detected       = c.get("rlm_detected", False),
                    steam_detected     = c.get("steam_detected", False),
                    mis_score          = c.get("mis_score", 0),
                    soft_line_detected = False,
                )
                c["_effective_edge"]      = _eff_edge_pc
                c["_soft_line_detected"]  = False
            except Exception:
                c["_effective_edge"]      = _pc_edge
                c["_soft_line_detected"]  = False
            if _V3_CALIBRATION_AVAILABLE:
                try:
                    c["market_agreement_score"] = int(_v3_compute_mas(
                        sharp_signal   = c.get("sharp_signal", "no_sharp"),
                        rlm_detected   = bool(c.get("rlm_detected", False)),
                        steam_detected = bool(c.get("steam_detected", False)),
                        mis_score      = int(c.get("mis_score", 0)),
                        line_move_dir  = "",
                    ))
                except Exception as _mas_exc:
                    logger.debug(f"[main] market_agreement_score failed: {_mas_exc}")
            logger.debug(
                f"  [{sport}] {bet_id}  [PRECOMPUTED] edge={_pc_edge:.2f}%  "
                f"conf={_pc_conf}  model_prob={_pc_prob:.1%}"
            )
            processed.append((c, _pc_edge, _pc_conf, _pc_prob))
            continue

        # ── Step 1b: Fail-fast — abort before NUTS sampler if data corrupt ──
        ok, reason = _validate_candidate_for_simulation(c, sport)
        if not ok:
            logger.warning(f"  [{sport}] {bet_id}: skipped (fail-fast) — {reason}")
            if _REJECT_LOGGER_AVAILABLE:
                _log_reject_cand(sport, c, "pre_sim", reason, now_est().strftime("%Y-%m-%d"))
            continue

        logger.debug(f"  [{sport}] Analyzing {bet_id}…")
        _ctx = c.get("context", "regular")
        logger.debug(
            f"  [{sport}] {bet_id}  context={_ctx}  "
            f"vol_idx={c.get('volatility_index') or 'default'}"
        )
        try:
            sim = engine.analyze(
                historical_data = c["historical_data"],
                league_mean     = c["league_mean"],
                league_std      = c.get("league_std", 5.0),
                sportsbook_line = c["sportsbook_line"],
                progressbar     = False,
                context         = _ctx,
                recent_n        = c.get("recent_n", 5),
                volatility_index= c.get("volatility_index"),
                market_type     = c.get("market", ""),
            )
        except Exception as exc:
            logger.warning(f"  [{sport}] {bet_id}: engine.analyze() failed — {exc}")
            continue

        # ── Model Stability Filter (spec §MODEL STABILITY FILTER) ────────────
        # Stability filter — relative uncertainty framework.
        # Primary: σ/|mean| < 15%  |  Secondary: absolute guardrails (emergency).
        if _STABILITY_FILTER_AVAILABLE:
            _post_std  = sim["posterior"].get("posterior_std",  0.0)
            _post_mean = sim["posterior"].get("posterior_mean", 0.0)
            _is_stable, _stab_reason = _check_stability(sport, _post_std, _post_mean)
            if not _is_stable:
                logger.info(f"  [{sport}] {bet_id}: STABILITY REJECT — {_stab_reason}")
                if _REJECT_LOGGER_AVAILABLE:
                    _rel = (_post_std / abs(_post_mean) * 100.0) if abs(_post_mean) > 1e-6 else None
                    _log_reject_cand(
                        sport, c, "stability", _stab_reason,
                        now_est().strftime("%Y-%m-%d"),
                        sigma=_post_std,
                        projection=_post_mean,
                        rel_sigma_pct=_rel,
                    )
                continue
            logger.debug(f"  [{sport}] {bet_id}: stability OK ({_stab_reason})")

        edge, confidence, model_prob = _derive_bet_params(sim, c)

        # V3.0: Calibrate raw simulation edge to realistic market-inefficiency range.
        # Game totals: [10%, 50%] raw → [1%, 10%] calibrated
        # Player props: [8%,  50%] raw → [2%, 15%] calibrated
        # This must happen BEFORE intel adjustments so intel signals apply
        # on the calibrated scale (their ± values are meaningful vs 1–10% edge).
        if _V3_CALIBRATION_AVAILABLE:
            _raw_edge_pre_cal = edge
            edge = _v3_calibrate_edge(edge, sport, c.get("market", "totals"))
            logger.debug(
                f"  [{sport}] {bet_id}  raw_edge={_raw_edge_pre_cal:.1f}%  "
                f"→ calibrated_edge={edge:.2f}%"
            )

        # Apply intelligence edge adjustments (rest/travel + lineup + stat model)
        _intel = c.get("_intel")
        if _intel:
            _intel_adj = (
                getattr(_intel.get("rest_travel"), "edge_adjustment", 0.0)
                + getattr(_intel.get("lineup"),      "edge_adjustment", 0.0)
                + getattr(_intel.get("stat_model"),  "edge_adjustment", 0.0)
                + getattr(_intel.get("rivalry"),     "edge_adjustment", 0.0)
                + getattr(_intel.get("venue"),       "edge_adjustment", 0.0)
            )
            if _intel_adj != 0.0:
                edge = round(min(15.0, max(-15.0, edge + _intel_adj)), 2)
                logger.debug(
                    f"  [{sport}] {bet_id}  intel_adj={_intel_adj:+.2f}  "
                    f"→ edge after intel={edge:.2f}%"
                )

        # ── Signal Layer 2: line movement (post-gate ranking boost only) ────────
        # Sharp signals do NOT modify the gate_edge used by the gatekeeper.
        # They accumulate in _signal_boosts[bet_id] and are applied to
        # approved bets after gatekeeper for ranking purposes only.
        # MLB signals are discounted (lagging noise); WNBA also discounted.
        _lm_bucket: str | None = None
        _lm = _lm_signals.get(bet_id)
        if _lm:
            _lm_adj_raw, _lm_desc, _lm_bucket = _lm
            # Apply sport-specific discount to line movement signal
            _sport_up = sport.upper()
            _lm_discount = (
                _MLB_SIGNAL_DISCOUNT.get(_lm_bucket, 1.0) if _sport_up == "MLB"
                else _WNBA_SIGNAL_DISCOUNT.get(_lm_bucket, 1.0) if _sport_up == "WNBA"
                else 1.0
            )
            _lm_adj = round(_lm_adj_raw * _lm_discount, 2)
            _signal_boosts[bet_id] = _signal_boosts.get(bet_id, 0.0) + _lm_adj
            # Append signal description to factor text so it surfaces in Telegram
            _existing_factor = c.get("factor", "")
            c["factor"] = (
                (_existing_factor.rstrip(" |") + " | " if _existing_factor else "")
                + f"📊 {_lm_desc}"
            )
            logger.info(
                f"  [{sport}] {bet_id}  lm_signal={_lm_adj_raw:+.1f} "
                f"× discount={_lm_discount:.2f} → ranking_boost={_lm_adj:+.2f}  "
                f"({_lm_desc}) [gate_edge unchanged={edge:.1f}%]"
            )

        # ── Signal Layer 2b: cross-book consensus boost (post-gate only) ─────
        _direction     = c.get("direction", "over").lower()
        _open_line     = float(c.get("opening_line") or c.get("sportsbook_line") or 0)
        _cons_line     = float(c.get("consensus_line") or 0)
        _dispersion    = float(c.get("line_dispersion") or 0)
        _cons_signal: str | None = None
        _signal_count  = 1 if _lm_bucket else 0  # track signals applied
        if _cons_line and _dispersion >= 0.5 and _signal_count < 2:
            # OVER: our line < consensus → easier to clear → value
            # UNDER: our line > consensus → easier to clear under → value
            _is_value = (
                (_direction == "over"  and _open_line < _cons_line - 0.3)
                or (_direction == "under" and _open_line > _cons_line + 0.3)
            )
            if _is_value:
                _cons_signal = "high" if _dispersion >= 1.0 else "medium"
                try:
                    from core.intelligence.signal_calibrator import get_adjustment as _gc
                    _cons_adj = _gc(
                        "consensus_high" if _cons_signal == "high" else "consensus_medium"
                    ).value
                except Exception:
                    _cons_adj = 1.5 if _cons_signal == "high" else 0.8
                _signal_boosts[bet_id] = _signal_boosts.get(bet_id, 0.0) + _cons_adj
                logger.debug(
                    f"  [{sport}] {bet_id}  consensus_boost=+{_cons_adj:.2f}  "
                    f"({_cons_signal}, dispersion={_dispersion}) "
                    f"[gate_edge unchanged={edge:.1f}%]"
                )

        # ── Phase 2: effective edge (raw edge ± market signal adjustments) ───
        from core.market_intelligence import compute_effective_edge as _cee
        from core.market_intelligence import detect_soft_line as _dsl
        _soft_line = _dsl(
            model_prob    = model_prob,
            american_odds = c.get("american_odds", -110),
            sport         = sport,
        )
        _eff_edge = _cee(
            raw_edge            = edge,
            sharp_signal        = c.get("sharp_signal", "no_sharp"),
            rlm_detected        = c.get("rlm_detected", False),
            steam_detected      = c.get("steam_detected", False),
            mis_score           = c.get("mis_score", 0),
            soft_line_detected  = _soft_line,
        )
        c["_soft_line_detected"] = _soft_line
        c["_effective_edge"]     = _eff_edge

        logger.debug(
            f"  [{sport}] {bet_id}  edge={edge:.2f}%  eff_edge={_eff_edge:.1f}%  "
            f"sharp={c.get('sharp_signal','no_sharp')}  steam={c.get('steam_detected')}  "
            f"rlm={c.get('rlm_detected')}  conf={confidence}  model={model_prob:.1f}%"
        )

        # V3.0: Compute market agreement score from available signals.
        # Stored back in c so the gatekeeper can access it via bet.raw_result.
        if _V3_CALIBRATION_AVAILABLE:
            c["market_agreement_score"] = int(_v3_compute_mas(
                sharp_signal   = c.get("sharp_signal", "no_sharp"),
                rlm_detected   = bool(c.get("rlm_detected", False)),
                steam_detected = bool(c.get("steam_detected", False)),
                mis_score      = int(c.get("mis_score", 0)),
                line_move_dir  = str(_lm_bucket or ""),
            ))
            logger.debug(
                f"  [{sport}] {bet_id}  "
                f"market_agreement={c['market_agreement_score']}"
            )

        processed.append((c, edge, confidence, model_prob))

    if not processed:
        logger.info(f"  [{sport}] All candidates failed simulation — skipping.")
        return [], []

    # ── Step 3.5: Game Truth Protocol ────────────────────────────────────────
    # Enforces a single Value Vector per game and applies sport-specific
    # volatility thresholds. Candidates that represent market noise (line
    # movement below threshold) or non-dominant markets are suppressed here
    # before Bet objects are constructed or the gatekeeper runs.
    from core.game_truth import apply_game_truth_protocol
    processed = apply_game_truth_protocol(processed, sport)

    if not processed:
        logger.info(f"  [{sport}] All candidates suppressed by Game Truth Protocol.")
        return [], []

    # ── Step 3.7: Thin-data calibration check ────────────────────────────────
    # Count closed, graded non-prop bets for this sport.  When the count is
    # below _MIN_THIN_DATA_BETS the model has no track record for team/game
    # totals in this sport yet.  Mark those candidates so the gatekeeper can
    # cap them at Value tier, preventing an untested model from broadcasting
    # Nuke or Diamond picks.  Player props are exempt — they are grounded in
    # individual player stats and do not require sport-wide calibration.
    _thin_data_ids: set[str] = set()
    try:
        from core.results_tracker import _connect as _rt_connect
        with _rt_connect() as _rtc:
            _closed_count = _rtc.execute(
                "SELECT COUNT(*) FROM bets "
                "WHERE sport = ? AND status = 'closed' AND player IS NULL",
                (sport,),
            ).fetchone()[0]
        if _closed_count < _MIN_THIN_DATA_BETS:
            logger.info(
                f"  [{sport}] Thin-data calibration: {_closed_count} closed "
                f"non-prop bets (need ≥{_MIN_THIN_DATA_BETS}) — "
                "team/game total picks capped at Value tier."
            )
            for _c, _, _, _ in processed:
                if not _c.get("player"):
                    _thin_data_ids.add(_c["bet_id"])
    except Exception as _td_exc:
        logger.debug(f"  [{sport}] thin_data check failed — {_td_exc}")

    # ── Step 4: Build Bet objects; convert game time to ET for display ───────
    bets: list[Bet] = []
    for c, edge, confidence, _ in processed:
        game_time_utc = c.get("game_time_utc")
        if game_time_utc and orchestrator.is_game_within_window(game_time_utc, hours=36.0):
            game_time_est = convert_to_est(localize_utc(game_time_utc))
            logger.debug(
                f"  [{sport}] {c['bet_id']} game time: "
                f"{format_est_short(game_time_est)} ET"
            )
        bets.append(
            Bet(
                bet_id                 = c["bet_id"],
                team                   = c["team"],
                market                 = market_normalized(c["market"]),
                direction              = c["direction"],
                sportsbook_line        = c["sportsbook_line"],
                edge_percentage        = edge,
                confidence_score       = confidence,
                player                 = c.get("player"),
                game_id                = c.get("game_id", ""),
                american_odds          = float(c.get("american_odds", 0)),
                data_reliability_score = c.get("data_reliability_score", 100),
                mis_score              = c.get("mis_score", 0),
                raw_result             = c,  # V3.0: full candidate for gatekeeper integrity/agreement checks
            )
        )

    # ── Step 5: Gatekeeper — tier assignment + conflict detection ────────────
    gk_result = run_gatekeeper(bets, sport=sport)
    approved  = gk_result["approved"]
    flagged   = gk_result["flagged"]
    discarded = gk_result["discarded"]

    # ── Step 5.1: Apply signal boosts for ranking (post-gate only) ───────────
    # Sharp signal boosts accumulated during processing are applied here to
    # approved bets so they affect pick ordering/ranking, not gate decisions.
    if _signal_boosts:
        for _bet in approved:
            _boost = _signal_boosts.get(_bet.bet_id, 0.0)
            if _boost != 0.0:
                _bet.edge_percentage = round(
                    min(50.0, max(-50.0, _bet.edge_percentage + _boost)), 2
                )
                logger.debug(
                    f"  [{sport}] {_bet.bet_id}: signal_boost={_boost:+.2f} "
                    f"→ ranking_edge={_bet.edge_percentage:.1f}%"
                )

    # ── Step 5.2: Per-game market caps ───────────────────────────────────────
    # max 1 recommendation per market type, max 3 per game.
    # Demoted bets are removed from the broadcast; their edge/confidence was
    # insufficient to rank in the top-3 across all markets for that game.
    if _MARKET_SCANNER_AVAILABLE and approved:
        try:
            _roi_map  = _get_market_roi()
            approved, _demoted = _apply_per_game_caps(
                approved,
                max_per_market=1,
                max_per_game=3,
                roi_lookup=_roi_map,
            )
            if _demoted:
                logger.info(
                    f"  [{sport}] Per-game caps: {len(_demoted)} pick(s) demoted "
                    f"({', '.join(b.bet_id for b in _demoted)})."
                )
        except Exception as _cap_err:
            logger.debug(f"  [{sport}] per_game_caps failed: {_cap_err}")

    # ── Step 5.3: Market coverage audit log ──────────────────────────────────
    if _MARKET_SCANNER_AVAILABLE:
        try:
            _audit_date = now_est().strftime("%Y-%m-%d")
            _candidates_by_mkt: dict[str, list] = {}
            for _c in candidates:
                _m = _c.get("market", "unknown")
                if _m not in _candidates_by_mkt:
                    _candidates_by_mkt[_m] = []
                _candidates_by_mkt[_m].append(_c)
            _log_market_audit(sport, _audit_date, _candidates_by_mkt, approved)
        except Exception as _audit_err:
            logger.debug(f"  [{sport}] market_audit log failed: {_audit_err}")

    logger.info(
        f"  [{sport}] Gatekeeper: "
        f"{len(approved)} approved / {len(flagged)} flagged / {len(discarded)} discarded"
    )
    for b in flagged:
        logger.debug(f"  [{sport}]   flagged  {b.bet_id}: {b.flag_reason}")
    for b in discarded:
        logger.debug(f"  [{sport}]   discarded {b.bet_id}: tier={b.tier}")

    if not approved:
        # ── Feature Variance Report — fired when gatekeeper approves 0 picks ─
        if processed:
            edges = [e for _, e, _, _ in processed]
            confs = [c for _, _, c, _ in processed]
            probs = [p for _, _, _, p in processed]
            lines = [c["sportsbook_line"] for c, _, _, _ in processed]
            dirs  = [c["direction"] for c, _, _, _ in processed]

            def _sd(vals: list[float]) -> float:
                n = len(vals)
                if n < 2:
                    return 0.0
                mu = sum(vals) / n
                return math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))

            logger.info(
                f"  [{sport}] ── Feature Variance Report ──────────────────────\n"
                f"  [{sport}]   Candidates: {len(processed)} "
                f"({dirs.count('over')} over / {dirs.count('under')} under)\n"
                f"  [{sport}]   Edge   : "
                f"min={min(edges):+.1f}%  max={max(edges):+.1f}%  σ={_sd(edges):.2f}\n"
                f"  [{sport}]   Conf   : "
                f"min={min(confs):.1f}  max={max(confs):.1f}  σ={_sd(confs):.2f}\n"
                f"  [{sport}]   Prob   : "
                f"min={min(probs):.1f}%  max={max(probs):.1f}%  σ={_sd(probs):.2f}\n"
                f"  [{sport}]   Lines  : "
                f"min={min(lines):.1f}  max={max(lines):.1f}  σ={_sd(lines):.2f}\n"
                f"  [{sport}]   → If edge σ ≈ 0, synthetic priors mirror market lines "
                f"(circular input). If conf σ ≈ 0, posterior std is too wide "
                f"(reduce league_std or increase n).\n"
                f"  [{sport}] ───────────────────────────────────────────────────"
            )
        return [], []

    # ── Step 5.5: Thin-data tier ceiling ────────────────────────────────────
    # Any approved bet flagged as thin_data is capped at Gold Standard tier.
    # The underlying edge/confidence scores are preserved — only the tier
    # label is clamped so that Nuke/Diamond cards are never sent from a
    # sport with no calibration history yet.
    if _thin_data_ids:
        for _bet in approved:
            if _bet.bet_id in _thin_data_ids and _bet.tier in (Tier.NUKE, Tier.DIAMOND):
                _orig_tier = _bet.tier.value
                _bet.tier  = Tier.GOLD
                logger.info(
                    f"  [{sport}] {_bet.bet_id}: thin_data ceiling "
                    f"{_orig_tier} → Gold Standard "
                    f"(conf={_bet.confidence_score:.1f}, edge={_bet.edge_percentage:.1f}%)"
                )

    # ── Step 5.5b: Data reliability tier ceiling ────────────────────────────
    # 40-59 → Gold Standard only (no Diamond or Nuke)
    # 60-74 → Diamond maximum (no Nuke)
    # 75+   → full tier access
    for _bet in approved:
        _drs = _bet.data_reliability_score
        if _drs < 60 and _bet.tier in (Tier.NUKE, Tier.DIAMOND):
            _orig_tier = _bet.tier.value
            _bet.tier  = Tier.GOLD
            logger.info(
                f"  [{sport}] {_bet.bet_id}: data_reliability {_drs} → "
                f"tier capped {_orig_tier} → Gold Standard"
            )
        elif _drs < 75 and _bet.tier == Tier.NUKE:
            _bet.tier = Tier.DIAMOND
            logger.info(
                f"  [{sport}] {_bet.bet_id}: data_reliability {_drs} → "
                "tier capped Nuke → Diamond"
            )

    # ── Step 5.6: Cross-run contradiction guard ──────────────────────────────
    # Before broadcasting, check the DB for any *already-open* bet on the
    # same game / team / market in the OPPOSITE direction.  If one exists,
    # accepting this pick guarantees one side will lose and the vig is paid
    # twice with zero expected net gain.  Drop the new pick and log it.
    from core.results_tracker import has_open_opposite_bet as _has_opposite
    _no_contradict: list = []
    for _bet in approved:
        if _bet.game_id and _has_opposite(
            sport     = sport,
            game_id   = _bet.game_id,
            team      = _bet.team,
            market    = _bet.market,
            direction = _bet.direction,
        ):
            logger.warning(
                f"  [{sport}] CONTRADICTION BLOCK — {_bet.bet_id} "
                f"({_bet.team} {_bet.market} {_bet.direction}) suppressed: "
                f"opposite side already open for game {_bet.game_id}"
            )
        else:
            _no_contradict.append(_bet)
    if len(_no_contradict) < len(approved):
        logger.info(
            f"  [{sport}] Contradiction guard removed "
            f"{len(approved) - len(_no_contradict)} pick(s) — "
            f"{len(_no_contradict)} remain."
        )
    approved = _no_contradict

    # ── CLV snapshot — record opening odds for every approved bet ────────────
    if _INTEL_AVAILABLE:
        proc_map_clv = {c["bet_id"]: c for c, _, _, _ in processed}
        for bet in approved:
            c_clv = proc_map_clv.get(bet.bet_id, {})
            if c_clv:
                snapshot_odds(
                    bet_id       = bet.bet_id,
                    opening_odds = c_clv.get("american_odds", 0),
                    opening_line = c_clv.get("sportsbook_line", 0.0),
                    sport        = sport,
                    market       = c_clv.get("market", ""),
                )

    # ── Step 6: Build BetDisplay and log dicts ───────────────────────────────
    proc_map = {c["bet_id"]: (c, mp) for c, _, _, mp in processed}
    display_list: list[BetDisplay]   = []
    log_dicts:    list[dict[str, Any]] = []

    for bet in approved:
        c, model_prob = proc_map[bet.bet_id]

        _is_thin = bet.bet_id in _thin_data_ids
        _factor  = c.get("factor", "")
        if _is_thin:
            _calib_note = "⚠️ Early calibration — fewer than 15 graded picks in this market"
            _factor = (_factor + (" | " if _factor else "") + _calib_note)

        # Apply confidence calibration for MLB to compress synthetic-prior
        # inflation.  Tier assignment (confidence_score) is NOT affected —
        # only the displayed model_probability value is compressed.
        display_model_prob = (
            _calibrate_mlb_confidence(model_prob)
            if sport == "MLB"
            else model_prob
        )

        display_list.append(
            BetDisplay(
                bet                    = bet,
                american_odds          = c["american_odds"],
                model_probability      = display_model_prob,
                supporting_factor      = _factor,
                game_time_utc          = c.get("game_time_utc"),
                away_team              = c.get("away_team", ""),
                home_team              = c.get("home_team", ""),
                full_team_name         = c.get("full_team_name", ""),
                bookmaker_source       = c.get("bookmaker_source", ""),
                book_count             = c.get("book_count", 0),
                verified_at            = c.get("verified_at"),
                opening_line           = c.get("opening_line"),
                consensus_line         = c.get("consensus_line"),
                mis_score              = c.get("mis_score"),
                data_reliability_score = c.get("data_reliability_score"),
            )
        )
        # ── Task B: raw_edge + edge_decay ────────────────────────────────────
        _raw_edge  = bet.edge_percentage
        _eff_edge_stored = c.get("_effective_edge", _raw_edge)
        _edge_decay = round(_raw_edge - _eff_edge_stored, 2)

        # ── Task D: liquidity score (0-10) + stability score (0-10) ──────────
        from core.market_intelligence import (
            SPORT_LIQUIDITY_THRESHOLDS as _SLT,
            compute_line_velocity as _clv,
        )
        _sport_thr  = _SLT.get(sport.upper(), _SLT["default"])
        _liq_score  = round(
            min(1.0, c.get("book_count", 0) / max(1, _sport_thr["full"])) * 10.0, 1
        )
        _open_ln    = float(c.get("opening_line") or bet.sportsbook_line or 0.0)
        _curr_ln    = float(bet.sportsbook_line or 0.0)
        _vel        = _clv(_open_ln, _curr_ln).get("velocity", 0.0)
        _stab_score = round(max(0.0, 10.0 - _vel * 10.0), 1)

        log_dicts.append({
            "bet_id":            bet.bet_id,
            "sport":             sport,
            "wager_details": {
                "team":                    bet.team,
                "market":                  bet.market,
                "direction":               bet.direction,
                "sportsbook_line":         bet.sportsbook_line,
                "opening_line":            c.get("opening_line", bet.sportsbook_line),
                "consensus_line":          c.get("consensus_line"),
                "verified_at":             c.get("verified_at"),
                "bookmaker_source":        c.get("bookmaker_source", ""),
                # Prop tracker matchup fields — used by /api/prop-tracker
                "away_team":               c.get("away_team", ""),
                "home_team":               c.get("home_team", ""),
                "game_id":                 c.get("game_id", ""),
                "edge_percentage":         bet.edge_percentage,
                "confidence_score":        bet.confidence_score,
                "tier":                    bet.tier.value if bet.tier else None,
                "player":                  bet.player,
                "thin_data":               _is_thin,
                "consensus_signal":        _cons_signal,
                "mis_score":               c.get("mis_score", 0),
                "data_reliability_score":  c.get("data_reliability_score", 100),
                "weighted_projection":     c.get("weighted_projection"),
                "l5_avg":                  c.get("l5_avg"),
                "l10_avg":                 c.get("l10_avg"),
                "data_available":          c.get("data_available", True),
                # Phase 2 — market intelligence signals
                "effective_edge":          _eff_edge_stored,
                "sharp_signal":            c.get("sharp_signal"),
                "rlm_detected":            c.get("rlm_detected", False),
                "steam_detected":          c.get("steam_detected", False),
                # Task B — edge transparency
                "raw_edge":                _raw_edge,
                "edge_decay":              _edge_decay,
                # Task D — market health scores
                "liquidity_score":         _liq_score,
                "stability_score":         _stab_score,
            },
            "model_probability":  model_prob,
            "sportsbook_odds":    c["american_odds"],
            "tier":               bet.tier.value if bet.tier else None,
            "edge_percentage":    bet.edge_percentage,
            "bookmaker_source":   c.get("bookmaker_source", ""),
            "line_move_dir":      _lm_bucket,
        })

    return display_list, log_dicts


# ---------------------------------------------------------------------------
# Mode executors
# ---------------------------------------------------------------------------

# ── Calibration helpers ─────────────────────────────────────────────────────

def _tier_from_params(edge: float, conf: float, sport: str = "MLB") -> str:
    """Mirror sport-specific gatekeeper thresholds for calibration display."""
    from core.decision_gatekeeper import SPORT_TIER_THRESHOLDS, _DEFAULT_SPORT_KEY
    thresholds = SPORT_TIER_THRESHOLDS.get(sport.upper(), SPORT_TIER_THRESHOLDS[_DEFAULT_SPORT_KEY])
    for tier, min_edge, min_conf in thresholds:
        if edge >= min_edge and conf >= min_conf:
            return tier.value
    return "DISCARD"


def _run_one_calibration(
    label: str,
    sport: str,
    hist: list[float],
    league_mean: float,
    league_std: float,
    line: float,
    odds: int,
    direction: str,
    context: str,
    vol_idx: float,
    recent_n: int = 5,
) -> dict[str, Any]:
    """
    Run a single Bayesian + Monte Carlo pass and return a result dict.
    Called twice per scenario (regular / playoff) by _mode_calibrate().
    """
    orchestrator = DecisionOrchestrator(sport)
    engine       = SimulationEngine(orchestrator)

    sim = engine.analyze(
        historical_data  = hist,
        league_mean      = league_mean,
        league_std       = league_std,
        sportsbook_line  = line,
        progressbar      = False,
        context          = context,
        recent_n         = recent_n,
        volatility_index = vol_idx,
    )

    edge, conf, model_prob = _derive_bet_params(
        sim,
        {"sportsbook_line": line, "american_odds": odds, "direction": direction},
    )

    return {
        "label":         label,
        "context":       sim["context"],
        "vol_idx":       sim["volatility_index"],
        "active_n":      sim["active_data_n"],
        "post_mean":     sim["posterior"]["posterior_mean"],
        "post_std":      sim["posterior"]["posterior_std"],
        "model_prob":    model_prob,
        "edge":          edge,
        "confidence":    conf,
        "tier":          _tier_from_params(edge, conf, sport),
    }


def _mode_calibrate() -> None:
    """
    --mode calibrate
    Run two parallel simulations using the SAME sportsbook line and the SAME
    underlying team, but label one 'regular' and one 'playoff'.

    Scenario A — Hot Finish  (early cold, recent hot):
        Regular  uses the full 10-game history → mixed signal → low tier
        Playoff  uses only the last 5 hot games → strong posterior → high tier

    Scenario B — Cold Finish  (early hot, recent cold):
        Regular  uses the full 10-game history → strong aggregate → high tier
        Playoff  uses only the last 5 cold games → weak posterior → low tier

    Both scenarios use NBA (volatility_index=1.8) so the compression effect
    is maximally visible.  Validation passes when every scenario shows a
    distinct tier between its regular and playoff columns.
    """
    LINE    = 14.5
    ODDS    = -110
    SPORT   = "NBA"
    VOL_IDX = SimulationEngine.PLAYOFF_VOLATILITY[SPORT]   # 1.8

    # ── Scenario A  (hot recent 5 games) ────────────────────────────────────
    HIST_A = [8.0, 9.0, 8.0, 7.0, 9.0,      # games 1-5: cold
              17.0, 18.0, 19.0, 17.0, 18.0]  # games 6-10: HOT

    # ── Scenario B  (cold recent 5 games) ────────────────────────────────────
    # First 5 games are clearly above the line; last 5 dip just below it.
    # Full-10 average (≈15.5) sits above the line → Regular sees positive edge.
    # Playoff mode slices to last-5 average (≈14.4, below line 14.5) → DISCARD.
    HIST_B = [17.0, 16.0, 17.0, 16.0, 17.0,  # games 1-5: HOT  (avg 16.6)
              14.0, 15.0, 14.0, 15.0, 14.0]   # games 6-10: cold (avg 14.4)

    LEAGUE_MEAN = 13.5
    LEAGUE_STD  = 5.0

    scenarios = [
        ("Scenario A · Hot Finish",   HIST_A),
        ("Scenario B · Cold Finish",  HIST_B),
    ]

    all_results: list[dict[str, Any]] = []

    for scenario_name, hist in scenarios:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"  {scenario_name}")
        logger.info(f"  Line={LINE}  Odds={ODDS:+d}  Sport={SPORT}  VI={VOL_IDX}")
        logger.info(f"  Full hist: {hist}")
        logger.info(f"  Last 5:   {hist[-5:]}")
        logger.info(f"{'─' * 60}")

        for ctx in ("regular", "playoff"):
            label = f"{scenario_name} [{ctx}]"
            logger.info(f"  Running {label}…")
            r = _run_one_calibration(
                label       = label,
                sport       = SPORT,
                hist        = hist,
                league_mean = LEAGUE_MEAN,
                league_std  = LEAGUE_STD,
                line        = LINE,
                odds        = ODDS,
                direction   = "over",
                context     = ctx,
                vol_idx     = VOL_IDX,
            )
            all_results.append(r)

    # ── Results table ────────────────────────────────────────────────────────
    header = (
        f"\n{'━' * 72}\n"
        f"  CALIBRATION RESULTS  —  Regular vs Playoff Context\n"
        f"{'━' * 72}\n"
        f"  {'Scenario':<28} {'ctx':>8} {'N':>3} {'mean':>6} "
        f"{'std':>5} {'model%':>7} {'edge%':>6} {'conf':>5}  {'Tier'}\n"
        f"{'━' * 72}"
    )
    logger.info(header)

    for r in all_results:
        row = (
            f"  {r['label']:<28} {r['context']:>8} "
            f"{r['active_n']:>3} {r['post_mean']:>6.2f} {r['post_std']:>5.2f} "
            f"{r['model_prob']:>7.1f} {r['edge']:>6.1f} {r['confidence']:>5.1f}"
            f"  {r['tier']}"
        )
        logger.info(row)

    logger.info("━" * 72)

    # ── Validation ───────────────────────────────────────────────────────────
    diverged: list[str] = []
    for scenario_name, _ in scenarios:
        pair = [r for r in all_results if scenario_name in r["label"]]
        reg_tier = next(r["tier"] for r in pair if r["context"] == "regular")
        ply_tier = next(r["tier"] for r in pair if r["context"] == "playoff")
        if reg_tier != ply_tier:
            diverged.append(
                f"  ✅  {scenario_name}: regular={reg_tier.strip()} "
                f"→ playoff={ply_tier.strip()}"
            )
        else:
            diverged.append(
                f"  ⚠️  {scenario_name}: SAME tier in both contexts "
                f"({reg_tier.strip()}) — check calibration data."
            )

    logger.info("\n  VALIDATION")
    for line_str in diverged:
        logger.info(line_str)
    logger.info("")

    all_diverged = all("✅" in d for d in diverged)
    if not all_diverged:
        raise RuntimeError(
            "Calibration failed: one or more scenarios show identical tiers "
            "across regular and playoff contexts. Adjust historical_data or "
            "league_mean before proceeding to the production pipeline."
        )
    logger.info(
        "  Calibration PASSED — playoff context produces distinct tier "
        "outcomes in all scenarios. Engine is context-aware."
    )


def _apply_global_tier_cap(
    sport_results: dict[str, tuple[list[BetDisplay], list[dict[str, Any]]]],
) -> None:
    """
    Filter-dominant tier assignment — Pick Ranking Governance Protocol.

    Identifies the single "dominant filter" (sport, market) group, then ranks
    all picks within that filter by their Composite Confidence Score (CCS)
    instead of the legacy edge×0.6 + conf×0.4 formula.  Tier labels are
    assigned exclusively through pool ranking — never by raw edge size,
    confidence thresholds, or arbitrary score cutoffs.

    Dominant filter selection (tie-breaking in order)
    --------------------------------------------------
    1. Highest count of Nuke+Diamond picks in the group
    2. Highest average CCS in the group
    3. Highest individual CCS in the group
    4. Tier-1 market priority (publication priority, lower = better)

    Within the dominant filter (Nuke Pool)
    ---------------------------------------
        Rank 1  →  Nuke         (sole global Nuke — highest CCS from Nuke pool)
        Rank 2  →  Diamond      (runner-up from Nuke pool; never forced)
        Rank 3+ →  Gold Standard

    Diamond-pool picks (gatekeeper tier = Diamond) enter Gold Standard directly.
    All picks from non-dominant filters → Gold Standard.

    CCS factors per Pick Ranking Governance Protocol:
        1. Projection Reliability  35%
        2. Signal Agreement        25%
        3. Edge Strength           20%
        4. Volatility Adjustment   10%
        5. Market Efficiency       10%
    Multiplied by a sensitivity robustness factor (0.78–1.00).

    DRS / restricted-market ceilings from the per-pick gatekeeper are respected
    exactly as before.

    Mutations applied in-place to BetDisplay.bet.tier, log_dict["tier"], and
    log_dict["wager_details"] so DB and Telegram stay in sync.
    """
    from core.composite_confidence_score import compute_ccs

    def _sync(
        bd: BetDisplay,
        ld: dict[str, Any],
        ccs: float | None = None,
        robustness: str | None = None,
    ) -> None:
        new_val = bd.bet.tier.value if bd.bet.tier else None
        ld["tier"] = new_val
        wd = ld.get("wager_details")
        if isinstance(wd, dict):
            wd["tier"] = new_val
            if ccs is not None:
                wd["ccs_score"]   = round(ccs, 2)
            if robustness is not None:
                wd["robustness"]  = robustness

    # ── 1. Build flat list tagged with sport ─────────────────────────────────
    flat: list[tuple[BetDisplay, dict[str, Any], str]] = []
    for sport_key, (display_bets, log_dicts) in sport_results.items():
        id_to_log = {d["bet_id"]: d for d in log_dicts}
        for bd in display_bets:
            ld = id_to_log.get(bd.bet.bet_id)
            if ld:
                flat.append((bd, ld, sport_key))

    if not flat:
        return

    # ── 2. Pre-compute CCS for every pick (avoids redundant calls) ───────────
    # Stored as {bet_id: (ccs_score, robustness_label)}
    _ccs_cache: dict[str, tuple[float, str]] = {}
    for bd, ld, _ in flat:
        try:
            _ccs_cache[bd.bet.bet_id] = compute_ccs(bd, ld)
        except Exception as _exc:
            # CCS failure must never block publication — fall back to legacy score
            legacy = bd.bet.edge_percentage * 0.6 + bd.bet.confidence_score * 0.4
            _ccs_cache[bd.bet.bet_id] = (legacy, "unknown")
            logger.warning("[TierCap] CCS fallback for %s: %s", bd.bet.bet_id, _exc)

    def _ccs(bd: BetDisplay) -> float:
        return _ccs_cache.get(bd.bet.bet_id, (0.0, "unknown"))[0]

    def _robustness(bd: BetDisplay) -> str:
        return _ccs_cache.get(bd.bet.bet_id, (0.0, "unknown"))[1]

    # ── 3. Publication-only policy ────────────────────────────────────────────
    # Research-market picks graded Nuke/Diamond by the gatekeeper are demoted
    # to Gold Standard — they are never eligible for a premium tier slot because
    # they may not appear in any public-facing output.
    for bd, ld, sp in flat:
        if bd.bet.tier in (Tier.NUKE, Tier.DIAMOND) and not is_publication_eligible(sp, bd.bet.market):
            bd.bet.tier = Tier.GOLD
            _sync(bd, ld, _ccs(bd), _robustness(bd))

    # ── 4. Partition into Nuke+Diamond eligible vs. the rest ─────────────────
    eligible   = [(bd, ld, sp) for bd, ld, sp in flat if bd.bet.tier in (Tier.NUKE, Tier.DIAMOND)]
    ineligible = [(bd, ld, sp) for bd, ld, sp in flat if bd.bet.tier not in (Tier.NUKE, Tier.DIAMOND)]

    if not eligible:
        # Nothing qualified — sync Gold picks through and return
        for bd, ld, _ in ineligible:
            _sync(bd, ld, _ccs(bd), _robustness(bd))
        return

    # ── 5. Group eligible picks by (sport, market) filter ────────────────────
    groups: dict[tuple[str, str], list[tuple[BetDisplay, dict[str, Any]]]] = {}
    for bd, ld, sp in eligible:
        fkey = (sp.upper(), market_normalized(bd.bet.market))
        groups.setdefault(fkey, []).append((bd, ld))

    # ── 6. Score each filter by CCS and elect the dominant one ───────────────
    def _filter_score(
        items: list[tuple[BetDisplay, dict[str, Any]]],
    ) -> tuple[int, float, float]:
        scores = [_ccs(bd) for bd, _ in items]
        return (len(scores), sum(scores) / len(scores), max(scores))

    dominant_key = max(
        groups,
        # Primary: filter score (count, avg CCS, max CCS)
        # Tiebreaker: Tier-1 market priority (lower number = higher priority)
        key=lambda k: (*_filter_score(groups[k]), -publication_priority(k[0], k[1])),
    )
    # Rank dominant pool by CCS descending — this IS the governance ranking
    dominant_pool = sorted(
        groups[dominant_key],
        key=lambda t: _ccs(t[0]),
        reverse=True,
    )
    non_dominant = [
        (bd, ld)
        for fkey, items in groups.items()
        if fkey != dominant_key
        for bd, ld in items
    ]

    _dom_fs = _filter_score(dominant_pool)
    logger.info(
        "[TierCap] Dominant filter: %s/%s | %d eligible | "
        "avg_ccs=%.1f max_ccs=%.1f | %d non-dominant → Gold Standard",
        dominant_key[0], dominant_key[1],
        _dom_fs[0], _dom_fs[1], _dom_fs[2],
        len(non_dominant),
    )
    for _rank, (bd, _) in enumerate(dominant_pool, 1):
        logger.info(
            "[TierCap]   Rank %d — %s  CCS=%.1f (%s)  edge=%.2f%%  conf=%.1f",
            _rank, bd.bet.bet_id,
            _ccs(bd), _robustness(bd),
            bd.bet.edge_percentage, bd.bet.confidence_score,
        )

    # ── 7. Assign Nuke / Diamond within the dominant pool ────────────────────
    # Tier labels come exclusively from ranking position, not thresholds.
    # Only Nuke-pool picks (gatekeeper tier = Nuke) fill the Nuke/Diamond slots.
    nuke_claimed    = False
    diamond_claimed = False

    for bd, ld in dominant_pool:
        gk_tier   = bd.bet.tier
        ccs_score = _ccs(bd)
        robust    = _robustness(bd)

        if not nuke_claimed and gk_tier == Tier.NUKE:
            bd.bet.tier  = Tier.NUKE
            nuke_claimed = True
        elif not diamond_claimed and gk_tier == Tier.NUKE:
            # Runner-up Nuke-pool pick becomes the daily Diamond.
            # Diamond-pool picks never fill this slot per the protocol.
            bd.bet.tier     = Tier.DIAMOND
            diamond_claimed = True
        else:
            bd.bet.tier = Tier.GOLD

        _sync(bd, ld, ccs_score, robust)

    # ── 8. Demote non-dominant eligible picks to Gold Standard ───────────────
    for bd, ld in non_dominant:
        bd.bet.tier = Tier.GOLD
        _sync(bd, ld, _ccs(bd), _robustness(bd))

    # ── 9. Pass-through: ineligible (already Gold) picks — sync log dicts ────
    for bd, ld, _ in ineligible:
        _sync(bd, ld, _ccs(bd), _robustness(bd))


def _apply_governance_gates(
    pipeline_results: dict[str, tuple],
    *,
    dry_run: bool = False,
) -> None:
    """
    Apply signal confirmation + conflict guardian gates to _pipeline_results.

    Mutates pipeline_results in-place — removes sports / picks that don't pass:
      1. Signal confirmation gate (Candidate → Confirmed lifecycle)
      2. Locked-pick conflict guardian (5-condition replacement threshold)

    dry_run=True  →  signal confirmation bypassed; all picks pass immediately.

    Spec: §PICK LIFECYCLE SYSTEM, §SIGNAL CONFIRMATION REQUIREMENTS,
          §CONFLICT REVIEW MODE, §REPLACEMENT THRESHOLD
    """
    if not _SIGNAL_CONFIRMATION_AVAILABLE and not _CONFLICT_GUARDIAN_AVAILABLE:
        return

    today = now_est().strftime("%Y-%m-%d")

    for sport in list(pipeline_results.keys()):
        display_bets, log_dicts = pipeline_results[sport]
        bets = [bd.bet for bd in display_bets]

        # ── 1. Signal confirmation gate ───────────────────────────────────────
        if _SIGNAL_CONFIRMATION_AVAILABLE:
            try:
                ready_bets, held_bets = _gate_signals(bets, sport, dry_run=dry_run)
                ready_ids = {b.bet_id for b in ready_bets}
                if held_bets:
                    logger.info(
                        f"  [{sport}] Signal gate: {len(ready_bets)} confirmed, "
                        f"{len(held_bets)} held for next confirmation cycle"
                    )
            except Exception as _sg_exc:
                logger.debug(f"  [{sport}] gate_signals failed (non-fatal): {_sg_exc}")
                ready_ids = {bd.bet.bet_id for bd in display_bets}
        else:
            ready_ids = {bd.bet.bet_id for bd in display_bets}

        # ── 2. Conflict guardian: locked-pick replacement threshold ───────────
        final_display: list = []
        final_logs:    list = []
        for bd, ld in zip(display_bets, log_dicts):
            if bd.bet.bet_id not in ready_ids:
                if _REJECT_LOGGER_AVAILABLE:
                    _log_reject_bet(bd.bet, sport, today, "governance_hold",
                                    "Signal confirmation: fewer than 3 confirmed cycles")
                continue   # held for confirmation — do not publish yet

            if _CONFLICT_GUARDIAN_AVAILABLE:
                try:
                    action, cg = _check_locked_conflict(bd.bet, sport, date_str=today)
                    if action == "hold":
                        logger.info(
                            f"  [{sport}] {bd.bet.bet_id}: CONFLICT HOLD — "
                            f"replacement threshold not met "
                            f"(existing locked: {cg.get('existing_bet_id', '?')})"
                        )
                        if _REJECT_LOGGER_AVAILABLE:
                            _log_reject_bet(
                                bd.bet, sport, today, "conflict_hold",
                                f"Conflict guardian: replacement threshold not met "
                                f"(existing: {cg.get('existing_bet_id', '?')})",
                            )
                        continue
                    if action == "replace":
                        logger.info(
                            f"  [{sport}] {bd.bet.bet_id}: CONFLICT REPLACE — "
                            f"all 5 conditions met; superseding "
                            f"{cg.get('existing_bet_id', '?')}"
                        )
                except Exception as _cg_exc:
                    logger.debug(f"  [{sport}] conflict_guardian failed (non-fatal): {_cg_exc}")

            final_display.append(bd)
            final_logs.append(ld)

        if final_display:
            pipeline_results[sport] = (final_display, final_logs)
        else:
            del pipeline_results[sport]


def _refresh_open_bet_odds() -> None:
    """
    After each picks run, silently update sportsbook_odds, current_line (in
    wager_details JSON), and current_confidence for today's open bets by
    matching them against fresh candidate data collected this run.

    This lets the MiniApp always display live market data without polling the
    Odds API separately — the data is already in _FRESH_CANDIDATES.
    """
    if not _FRESH_CANDIDATES:
        return

    import sqlite3 as _sqlite3
    import json as _json
    from pathlib import Path as _Path

    _db = _Path(__file__).parent / "data" / "results.db"
    today = now_est().strftime("%Y-%m-%d")

    # Build bet_id → candidate lookup from all sports processed this run
    cand_by_id: dict[str, dict[str, Any]] = {}
    for _cands in _FRESH_CANDIDATES.values():
        for c in _cands:
            bid = c.get("bet_id", "")
            if bid:
                cand_by_id[bid] = c

    if not cand_by_id:
        return

    try:
        conn = _sqlite3.connect(_db)
        conn.row_factory = _sqlite3.Row
        rows = conn.execute(
            "SELECT bet_id, wager_details FROM bets "
            "WHERE slate_date = ? AND status = 'open'",
            (today,),
        ).fetchall()

        if not rows:
            conn.close()
            return

        updated = 0
        for row in rows:
            bid  = row["bet_id"]
            cand = cand_by_id.get(bid)
            if cand is None:
                continue

            new_odds = cand.get("american_odds")
            new_line = cand.get("sportsbook_line")
            new_conf = cand.get("confidence_score") or cand.get("precomputed_confidence")

            if new_odds is None:
                continue

            try:
                wd = _json.loads(row["wager_details"] or "{}")
            except Exception:
                wd = {}

            if new_line is not None:
                wd["current_line"] = float(new_line)

            conn.execute(
                """
                UPDATE bets
                SET sportsbook_odds    = ?,
                    current_confidence = COALESCE(?, current_confidence),
                    wager_details      = ?
                WHERE bet_id     = ?
                  AND status     = 'open'
                  AND slate_date = ?
                """,
                (
                    int(new_odds),
                    round(float(new_conf), 2) if new_conf is not None else None,
                    _json.dumps(wd),
                    bid,
                    today,
                ),
            )
            updated += 1

        conn.commit()
        conn.close()

        if updated:
            logger.info(f"[OddsRefresh] Updated market data for {updated} open bet(s).")

    except Exception as exc:
        logger.warning(f"[OddsRefresh] Non-fatal: {exc}")


def _mode_run(sports: list[str], dry_run: bool) -> None:
    """
    --mode run
    Priority 1 → Morning recap (sent_to_group=True bets from yesterday)
    60-second pause
    Priority 2 → Daily picks (DecisionOrchestrator → SimulationEngine → Gatekeeper)
    """
    logger.info("━" * 50)
    logger.info(f"STARTING full broadcast  (sports: {', '.join(sports)})")
    logger.info("━" * 50)

    # ── Priority 1: Morning recap ────────────────────────────────────────────
    logger.info("PRIORITY 1 — Morning Recap")
    recap_text = format_morning_recap()
    logger.info(recap_text)

    logger.info("Recap logged internally — Telegram group send disabled.")

    # ── 60-second pause ──────────────────────────────────────────────────────
    if not dry_run:
        logger.info(f"Waiting {RECAP_DELAY_S} s before picks broadcast…")
        time.sleep(RECAP_DELAY_S)

    # ── Priority 2: Picks — data status check + Safe State gate ─────────────
    logger.info("PRIORITY 2 — Daily Picks")

    # Rule 4: report data-pull status at the start of every picks run.
    # Rule 2: if no source is reachable, enter Safe State and halt the engine.
    if _RESILIENCE_AVAILABLE:
        logger.info("Checking data source connectivity…")
        connectivity = check_connectivity()
        sources_ok = _safe_state.report_and_evaluate(
            connectivity,
            alert_fn=None,   # Telegram reserved for slate releases + reversals only
        )
        if not sources_ok:
            logger.error(
                "Safe State active — all data sources unavailable. "
                "Picks broadcast HALTED. No picks will be generated until "
                "connectivity is restored."
            )
            return
        logger.info("Data sources confirmed — proceeding to picks pipeline.")
    else:
        logger.debug("Data resilience layer not available; skipping connectivity check.")

    # ── Collect all sports first (no DB writes yet) ──────────────────────────
    _pipeline_results: dict[str, tuple[list[BetDisplay], list[dict[str, Any]]]] = {}

    for sport in sports:
        logger.info(f"[{sport}] Initializing DecisionOrchestrator…")
        try:
            orchestrator = DecisionOrchestrator(sport)
        except UnsupportedSportError as exc:
            logger.warning(f"[{sport}] Unsupported sport — {exc}")
            continue

        logger.info(f"[{sport}] Initializing SimulationEngine…")
        engine = SimulationEngine(orchestrator)

        logger.info(f"[{sport}] Running prediction pipeline…")
        try:
            display_bets, log_dicts = _run_sport_pipeline(sport, orchestrator, engine)
        except MissingMetricError as exc:
            logger.warning(f"[{sport}] Missing metric in game data — {exc}")
            continue
        except Exception as exc:
            logger.error(f"[{sport}] Pipeline error — {exc}", exc_info=True)
            continue

        if not display_bets:
            logger.info(f"[{sport}] No approved picks after gating — skipped.")
            continue

        logger.info(f"[{sport}] {len(display_bets)} approved pick(s) ready.")
        _pipeline_results[sport] = (display_bets, log_dicts)

    # ── Ranked tier assignment: Nuke #1, Diamond #2, Gold Standard rest ──────
    if _pipeline_results:
        _apply_global_tier_cap(_pipeline_results)
        nuke_ct    = sum(
            1 for bds, _ in _pipeline_results.values()
            for bd in bds if bd.bet.tier == Tier.NUKE
        )
        diamond_ct = sum(
            1 for bds, _ in _pipeline_results.values()
            for bd in bds if bd.bet.tier == Tier.DIAMOND
        )
        gold_ct    = sum(
            1 for bds, _ in _pipeline_results.values()
            for bd in bds if bd.bet.tier == Tier.GOLD
        )
        logger.info(
            f"Ranked tier assignment — Nuke: {nuke_ct}  Diamond: {diamond_ct}  "
            f"Gold Standard: {gold_ct}"
        )

    # ── Signal confirmation + conflict guardian (spec §SIGNAL GOVERNANCE) ────
    _apply_governance_gates(_pipeline_results, dry_run=dry_run)

    # ── Persist to DB + assemble broadcast list ───────────────────────────────
    bets_by_sport: dict[str, list[BetDisplay]] = {}
    for sport, (display_bets, log_dicts) in _pipeline_results.items():
        # Publication gate: only Tier-1 approved markets enter DB + broadcast.
        # Research-market picks are evaluated internally but never logged or sent.
        pub_display   = [bd for bd in display_bets
                         if is_publication_eligible(sport, bd.bet.market)]
        pub_log_dicts = [d for d in log_dicts
                         if is_publication_eligible(
                             d.get("sport", sport),
                             (d.get("wager_details") or {}).get("market", ""),
                         )]
        held = len(display_bets) - len(pub_display)
        if held:
            logger.info(
                f"[{sport}] Publication gate: {held} research-market pick(s) "
                "held internally — not logged or broadcast."
            )
        for d in pub_log_dicts:
            log_bet_dict(
                bet_id            = d["bet_id"],
                sport             = d["sport"],
                wager_details     = d["wager_details"],
                model_probability = d["model_probability"],
                sportsbook_odds   = d["sportsbook_odds"],
                tier              = d["tier"],
                edge_percentage   = d["edge_percentage"],
                sent_to_group     = False,
                bookmaker_source  = d.get("bookmaker_source", ""),
                line_move_dir     = d.get("line_move_dir"),
            )
        if pub_display:
            bets_by_sport[sport] = pub_display

    if not bets_by_sport:
        logger.info("No active sports with approved picks today — broadcast silent.")
        return

    # ── Pre-publish line accuracy verification (Rules 4, 7, 8) ───────────────
    try:
        from core.line_validator import pre_publish_verify
        bets_by_sport, _failed = pre_publish_verify(bets_by_sport, dry_run=dry_run)
        if _failed:
            logger.warning(
                f"LINE VALIDATION: {len(_failed)} prop pick(s) removed — "
                "line moved since analysis."
            )
            for fp in _failed:
                logger.warning(
                    f"  ✗ {fp.get('player','?')} {fp.get('market','')} "
                    f"{fp.get('direction','')} — {fp['reason']}"
                )
    except Exception as _lv_exc:
        logger.warning(f"Pre-publish line verification failed (non-fatal): {_lv_exc}")

    if not bets_by_sport:
        logger.info(
            "All picks removed by pre-publish line validation gate — broadcast silent."
        )
        return

    send_results = send_daily_picks(
        approved_bets_by_sport = bets_by_sport,
        date_str               = format_est_date(now_utc()),
        dry_run                = dry_run,
    )
    sent_count = sum(1 for r in send_results if r.get("sent"))
    logger.info(
        f"Picks broadcast complete — {sent_count}/{len(send_results)} message(s) sent."
    )

    # ── Mark published games in Game Truth state so intraday re-runs suppress ──
    if sent_count > 0 and not dry_run:
        try:
            from core.game_truth import mark_picks_published
            _published_gids = list({
                bd.bet.game_id
                for bds in bets_by_sport.values()
                for bd in bds
                if getattr(bd.bet, "game_id", "")
            })
            mark_picks_published(_published_gids)
        except Exception as _gtp_exc:
            logger.warning(f"mark_picks_published failed (non-fatal): {_gtp_exc}")

    # ── Snapshot this slate as the official opening record (v1) ──────────────
    try:
        from core.slate_versioner import snapshot_slate
        snap = snapshot_slate(bets_by_sport, trigger_reason="scheduled", dry_run=dry_run)
        logger.info(
            f"Slate versioner: Official Slate v{snap['version']} saved "
            f"({'locked — opening record' if snap.get('is_v1') else 'rerun'})."
        )
        if snap.get("alert_sent"):
            logger.info(f"Change alert sent to Telegram ({len(snap['changes'])} changes).")
    except Exception as exc:
        logger.warning(f"Slate versioner snapshot failed (non-fatal): {exc}")

    # Silently refresh sportsbook_odds / current_line / current_confidence in DB
    # for all open bets from today, using the fresh candidates from this run.
    _refresh_open_bet_odds()


def _mode_close(bet_id: str | None, outcome: str | None) -> None:
    """
    --mode close
    Resolve one open bet against the DB and compute profit/loss.

    For nightly batch closes via cron, loop this mode in a shell script:
        while IFS=, read -r id result; do
            python3 main.py --mode close --bet-id "$id" --outcome "$result"
        done < results.csv
    """
    if not bet_id or not outcome:
        logger.error("--mode close requires both --bet-id and --outcome.")
        sys.exit(1)

    logger.info(f"Closing bet {bet_id!r}  outcome={outcome}")
    result = close_bet(bet_id, outcome)
    logger.info(result["summary"])


def _mode_refine(sports: list[str]) -> None:
    """
    --mode refine
    Learning loop: compare model_probability vs actual_outcome for each
    closed bet, then auto-adjust weights in config/sports_metrics.json
    when the model is miscalibrated by > 8 % over at least 10 resolved bets.
    """
    logger.info("━" * 50)
    logger.info("STARTING model refinement")
    logger.info("━" * 50)

    total_adjustments = 0
    for sport in sports:
        logger.info(f"Refining weights for {sport}…")
        adjustments = update_model_priors(sport)
        if adjustments:
            for adj in adjustments:
                logger.info(
                    f"  [{adj['sport']}] {adj['weight_key']}: "
                    f"{adj['old_value']:.4f} → {adj['new_value']:.4f}  "
                    f"({adj['reason']})"
                )
            total_adjustments += len(adjustments)
        else:
            logger.info(
                f"  {sport}: no adjustments — insufficient data or "
                "within ±8 % calibration threshold."
            )

    logger.info(
        f"Refinement complete — {total_adjustments} weight(s) updated "
        f"across {len(sports)} sport(s)."
    )


def _mode_recap(sport: str | None, dry_run: bool) -> None:
    """--mode recap  — morning recap only."""
    logger.info("Morning Recap — started")
    recap_text = format_morning_recap(sport=sport)
    logger.info(recap_text)

    logger.info("Recap logged internally — Telegram group send disabled.")


def _mode_picks(sports: list[str], dry_run: bool) -> None:
    """--mode picks  — picks broadcast only (no recap, no sleep)."""
    logger.info("Daily Picks — building slates")

    # ── Collect all sports first (no DB writes yet) ──────────────────────────
    _pipeline_results: dict[str, tuple[list[BetDisplay], list[dict[str, Any]]]] = {}

    for sport in sports:
        logger.info(f"[{sport}] Initializing DecisionOrchestrator…")
        try:
            orchestrator = DecisionOrchestrator(sport)
        except UnsupportedSportError as exc:
            logger.warning(f"[{sport}] Unsupported sport — {exc}")
            continue

        logger.info(f"[{sport}] Initializing SimulationEngine…")
        engine = SimulationEngine(orchestrator)

        logger.info(f"[{sport}] Running prediction pipeline…")
        try:
            display_bets, log_dicts = _run_sport_pipeline(sport, orchestrator, engine)
        except Exception as exc:
            logger.error(f"[{sport}] Pipeline error — {exc}", exc_info=True)
            continue

        if not display_bets:
            logger.info(f"[{sport}] No approved picks after gating — skipped.")
            continue

        logger.info(f"[{sport}] {len(display_bets)} approved pick(s) ready.")
        _pipeline_results[sport] = (display_bets, log_dicts)

    # ── Ranked tier assignment: Nuke #1, Diamond #2, Gold Standard rest ──────
    if _pipeline_results:
        _apply_global_tier_cap(_pipeline_results)
        nuke_ct    = sum(
            1 for bds, _ in _pipeline_results.values()
            for bd in bds if bd.bet.tier == Tier.NUKE
        )
        diamond_ct = sum(
            1 for bds, _ in _pipeline_results.values()
            for bd in bds if bd.bet.tier == Tier.DIAMOND
        )
        gold_ct    = sum(
            1 for bds, _ in _pipeline_results.values()
            for bd in bds if bd.bet.tier == Tier.GOLD
        )
        logger.info(
            f"Ranked tier assignment — Nuke: {nuke_ct}  Diamond: {diamond_ct}  "
            f"Gold Standard: {gold_ct}"
        )

    # ── Signal confirmation + conflict guardian (spec §SIGNAL GOVERNANCE) ────
    _apply_governance_gates(_pipeline_results, dry_run=dry_run)

    # ── Persist to DB + assemble broadcast list ───────────────────────────────
    bets_by_sport: dict[str, list[BetDisplay]] = {}
    for sport, (display_bets, log_dicts) in _pipeline_results.items():
        # Publication gate: only Tier-1 approved markets enter DB + broadcast.
        # Research-market picks are evaluated internally but never logged or sent.
        pub_display   = [bd for bd in display_bets
                         if is_publication_eligible(sport, bd.bet.market)]
        pub_log_dicts = [d for d in log_dicts
                         if is_publication_eligible(
                             d.get("sport", sport),
                             (d.get("wager_details") or {}).get("market", ""),
                         )]
        held = len(display_bets) - len(pub_display)
        if held:
            logger.info(
                f"[{sport}] Publication gate: {held} research-market pick(s) "
                "held internally — not logged or broadcast."
            )
        for d in pub_log_dicts:
            log_bet_dict(
                bet_id            = d["bet_id"],
                sport             = d["sport"],
                wager_details     = d["wager_details"],
                model_probability = d["model_probability"],
                sportsbook_odds   = d["sportsbook_odds"],
                tier              = d["tier"],
                edge_percentage   = d["edge_percentage"],
                sent_to_group     = False,
                bookmaker_source  = d.get("bookmaker_source", ""),
                line_move_dir     = d.get("line_move_dir"),
            )
        if pub_display:
            bets_by_sport[sport] = pub_display

    if not bets_by_sport:
        logger.info("No active sports with approved picks — broadcast silent.")
        return

    # ── Pre-publish line accuracy verification (Rules 4, 7, 8) ───────────────
    try:
        from core.line_validator import pre_publish_verify
        bets_by_sport, _failed = pre_publish_verify(bets_by_sport, dry_run=dry_run)
        if _failed:
            logger.warning(
                f"LINE VALIDATION: {len(_failed)} prop pick(s) removed — "
                "line moved since analysis."
            )
            for fp in _failed:
                logger.warning(
                    f"  ✗ {fp.get('player','?')} {fp.get('market','')} "
                    f"{fp.get('direction','')} — {fp['reason']}"
                )
    except Exception as _lv_exc:
        logger.warning(f"Pre-publish line verification failed (non-fatal): {_lv_exc}")

    if not bets_by_sport:
        logger.info(
            "All picks removed by pre-publish line validation gate — broadcast silent."
        )
        return

    send_results = send_daily_picks(
        approved_bets_by_sport = bets_by_sport,
        date_str               = format_est_date(now_utc()),
        dry_run                = dry_run,
    )
    sent_count = sum(1 for r in send_results if r.get("sent"))
    logger.info(
        f"Picks broadcast complete — {sent_count}/{len(send_results)} message(s) sent."
    )

    # ── Mark published games in Game Truth state so intraday re-runs suppress ──
    if sent_count > 0 and not dry_run:
        try:
            from core.game_truth import mark_picks_published
            _published_gids = list({
                bd.bet.game_id
                for bds in bets_by_sport.values()
                for bd in bds
                if getattr(bd.bet, "game_id", "")
            })
            mark_picks_published(_published_gids)
        except Exception as _gtp_exc:
            logger.warning(f"mark_picks_published failed (non-fatal): {_gtp_exc}")

    # ── Snapshot this rerun — creates v2/v3/… and diffs vs v1 ────────────────
    try:
        from core.slate_versioner import snapshot_slate
        snap = snapshot_slate(bets_by_sport, trigger_reason="manual", dry_run=dry_run)
        logger.info(
            f"Slate versioner: Official Slate v{snap['version']} saved."
        )
        if snap.get("alert_sent"):
            logger.info(f"Change alert sent to Telegram ({len(snap['changes'])} changes).")
    except Exception as exc:
        logger.warning(f"Slate versioner snapshot failed (non-fatal): {exc}")

    # Silently refresh sportsbook_odds / current_line / current_confidence in DB
    # for all open bets from today, using the fresh candidates from this run.
    _refresh_open_bet_odds()


def _mode_open(sport: str | None) -> None:
    """--mode open  — list all open bets."""
    bets = get_open_bets(sport)
    if not bets:
        logger.info("No open bets found.")
        return
    logger.info(f"Open bets ({len(bets)}):")
    for b in bets:
        logger.info(
            f"  [{b['sport']}] {b['bet_id']:<42} "
            f"tier={b['tier'] or '?':5}  "
            f"odds={b['sportsbook_odds']:+d}  "
            f"stake=${b['stake']:.0f}"
        )


def _mode_calibrate_odds(sports: list[str], dry_run: bool = False) -> None:
    """
    --mode calibrate-odds  — Side-by-side A/B calibration run.

    Fetches today's live slate, runs the full Bayesian engine on every candidate
    (both OVER and UNDER), and prints a ranked breakdown showing which picks
    would be approved/near-miss/discarded under the current calibration.

    Output Goal: ≥ 4–6 candidates per day clearing the Nuke threshold.
    If fewer than 4 Nuke picks are found, a Feature Variance Report is printed
    identifying which variables limit signal.
    """
    from core.odds_client import fetch_todays_candidates
    from core.decision_gatekeeper import Tier, evaluate_tier

    for sport in sports:
        candidates = fetch_todays_candidates(sport)
        if not candidates:
            logger.info(f"[{sport}] No candidates available today — skipping calibration.")
            continue

        logger.info(f"\n{'═' * 68}")
        logger.info(f"  [{sport}] CALIBRATION RUN — {len(candidates)} candidates")
        logger.info(f"{'═' * 68}")

        orchestrator = DecisionOrchestrator(sport)
        sim_engine   = SimulationEngine(orchestrator)
        results: list[tuple[dict, float, float, float, Tier | None]] = []

        for c in candidates:
            try:
                sim = sim_engine.analyze(
                    historical_data  = c["historical_data"],
                    league_mean      = c["league_mean"],
                    league_std       = c.get("league_std", 5.0),
                    sportsbook_line  = c["sportsbook_line"],
                    progressbar      = False,
                    context          = c.get("context", "regular"),
                    recent_n         = c.get("recent_n", 5),
                    volatility_index = c.get("volatility_index"),
                    market_type      = c.get("market", ""),
                )
                edge, conf, prob = _derive_bet_params(sim, c)
                tier = evaluate_tier(edge, conf)
                results.append((c, edge, conf, prob, tier))
            except Exception as exc:
                logger.warning(f"  [{sport}] {c['bet_id']}: engine failed — {exc}")

        # Sort by edge descending
        results.sort(key=lambda r: r[1], reverse=True)

        for c, edge, conf, prob, tier in results:
            tier_tag = tier.value if tier else (
                "Near-miss" if edge >= 2.5 and conf >= 65.0 else "Discard"
            )
            logger.info(
                f"  {c['direction']:5s} {c['sportsbook_line']:5.1f}  "
                f"edge={edge:+6.1f}%  conf={conf:5.1f}  "
                f"prob={prob:5.1f}%  → {tier_tag:10s}  "
                f"[{c['bet_id'][:35]}]"
            )

        nuke_count  = sum(1 for _, _, _, _, t in results if t == Tier.NUKE)
        diam_count  = sum(1 for _, _, _, _, t in results if t == Tier.DIAMOND)
        gold_count  = sum(1 for _, _, _, _, t in results if t == Tier.GOLD)
        miss_count  = sum(1 for _, e, c_, _, t in results if t is None and e >= 2.5 and c_ >= 65)
        disc_count  = len(results) - nuke_count - diam_count - gold_count - miss_count

        logger.info(f"\n  Summary: {nuke_count} Nuke / {diam_count} Diamond / "
                    f"{gold_count} Gold Standard / {miss_count} Near-miss / {disc_count} Discard")

        if nuke_count == 0 and diam_count == 0:
            edges_all = [e for _, e, _, _, _ in results]
            confs_all = [c_ for _, _, c_, _, _ in results]

            def _sd(v: list[float]) -> float:
                n = len(v)
                if n < 2:
                    return 0.0
                mu = sum(v) / n
                return math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1))

            logger.info(
                f"\n  ℹ Feature Variance Report (0 Nuke, 0 Diamond — thresholds not met today):\n"
                f"    Edge σ={_sd(edges_all):.2f} — low σ → "
                f"synthetic priors may mirror market (circular input).\n"
                f"    Conf σ={_sd(confs_all):.2f} — low σ → "
                f"posterior too wide (increase n or reduce league_std).\n"
                f"    Note: 0 Nuke / 0 Diamond is valid — thresholds must not be relaxed to force output."
            )

        logger.info(f"{'═' * 68}\n")


def _mode_grade(sports: list[str]) -> None:
    """--mode grade  — auto-grade settled bets and player props via ESPN."""
    graded = 0
    for sport in sports:
        # Full-game totals, spreads, and in-game markets (NRFI/YRFI/F5/Q1/H1)
        try:
            from core.score_grader import grade_settled_bets
            n = grade_settled_bets(sport, days_from=5)
            graded += n
            if n:
                logger.info(f"  [{sport}] score_grader: {n} bet(s) graded.")
        except ImportError:
            logger.debug(f"  [{sport}] score_grader not available — skipping.")
        except Exception as exc:
            logger.warning(f"  [{sport}] score_grader error: {exc}")

        # Player props (pts/reb/ast/3pm/etc.)
        try:
            from core.prop_grader import grade_player_props
            p = grade_player_props(sport, days_from=5)
            graded += p
            if p:
                logger.info(f"  [{sport}] prop_grader: {p} prop(s) graded.")
        except ImportError:
            logger.debug(f"  [{sport}] prop_grader not available — skipping.")
        except Exception as exc:
            logger.warning(f"  [{sport}] prop_grader error: {exc}")

    logger.info(f"Auto-grade complete: {graded} bet(s) settled across {len(sports)} sport(s).")

    # Rebuild cumulative performance cache whenever any bets are graded
    if graded > 0:
        try:
            from core.performance_tracker import rebuild_stats
            stats = rebuild_stats()
            logger.info(
                f"  [perf] rebuilt: {stats['total_wins']}W-{stats['total_losses']}L "
                f"({stats['win_rate']}% WR, {stats['net_units']:+.2f}u, {stats['roi_pct']:+.2f}% ROI)"
            )
        except Exception as exc:
            logger.warning(f"  [perf] rebuild_stats failed: {exc}")


def _mode_reconcile(sports: list[str]) -> None:
    """
    --mode reconcile — catch-all scan for stuck open bets (7-day lookback).

    Runs with a wider days_from window than the regular grade so it catches
    any bet that slipped through due to scheduler downtime, a missed grade
    window, or a late-posting ESPN scoreboard.  Logs every step so problems
    are visible in scheduler.log.
    """
    logger.info("Reconcile scan started — checking open bets up to 7 days back.")
    graded = 0
    for sport in sports:
        try:
            from core.score_grader import grade_settled_bets
            n = grade_settled_bets(sport, days_from=7)
            graded += n
            logger.info(f"  [{sport}] score_grader (reconcile): {n} bet(s) graded.")
        except ImportError:
            logger.debug(f"  [{sport}] score_grader not available — skipping.")
        except Exception as exc:
            logger.warning(f"  [{sport}] score_grader reconcile error: {exc}")

        try:
            from core.prop_grader import grade_player_props
            p = grade_player_props(sport, days_from=7)
            graded += p
            logger.info(f"  [{sport}] prop_grader (reconcile): {p} prop(s) graded.")
        except ImportError:
            logger.debug(f"  [{sport}] prop_grader not available — skipping.")
        except Exception as exc:
            logger.warning(f"  [{sport}] prop_grader reconcile error: {exc}")

    logger.info(
        f"Reconcile complete: {graded} bet(s) settled across {len(sports)} sport(s)."
    )

    if graded > 0:
        try:
            from core.performance_tracker import rebuild_stats
            stats = rebuild_stats()
            logger.info(
                f"  [perf] rebuilt: {stats['total_wins']}W-{stats['total_losses']}L "
                f"({stats['win_rate']}% WR, {stats['net_units']:+.2f}u, {stats['roi_pct']:+.2f}% ROI)"
            )
        except Exception as exc:
            logger.warning(f"  [perf] rebuild_stats failed: {exc}")


# ---------------------------------------------------------------------------
# Critical-error Telegram alert
# ---------------------------------------------------------------------------

def _send_critical_alert(mode: str, exc: Exception) -> None:
    """
    Format a 🚨 CRITICAL ALERT and push it to Telegram.
    Called after the full traceback is already in betting_bot.log.
    Never raises — a broken alert must not mask the original error.
    """
    tb_lines   = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_excerpt = "".join(tb_lines[-8:]).strip()
    timestamp  = format_est(now_utc(), "%A, %B %d %Y  %I:%M %p ET")

    alert = (
        f"🚨 CRITICAL ERROR — betting_bot\n"
        f"{'━' * 42}\n"
        f"  ⚙️  Mode:     --mode {mode}\n"
        f"  📅 Time:     {timestamp}\n"
        f"  ❌ Error:    {type(exc).__name__}: {exc}\n"
        f"{'━' * 42}\n"
        f"  Traceback (last lines):\n"
        f"{tb_excerpt}\n"
        f"{'━' * 42}\n"
        f"  📋 Full log: {LOG_PATH}"
    )

    # Telegram is reserved for slate releases and pick reversals only.
    # Critical errors are logged to file — not broadcast to Telegram.
    logger.error(f"Critical error in mode={mode}: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Guarded mode executor
# ---------------------------------------------------------------------------

def _run_mode(mode: str, fn, *args, **kwargs) -> None:
    """
    Execute fn(*args, **kwargs).  On any unhandled exception:
      • log full traceback to betting_bot.log at CRITICAL level
      • send a 🚨 Telegram alert
      • sys.exit(1) so cron / systemd marks the run as failed
    On clean completion: sys.exit(0).
    """
    logger.info(f"[betting_bot] --mode {mode} started  (PID {os.getpid()})")
    try:
        fn(*args, **kwargs)
        logger.info(f"[betting_bot] --mode {mode} completed successfully.\n")
        sys.exit(0)
    except SystemExit:
        raise      # propagate clean exits (lock contention, missing args, etc.)
    except Exception as exc:
        logger.critical(
            f"[betting_bot] --mode {mode} FAILED: {type(exc).__name__}: {exc}",
            exc_info=True,
        )
        _send_critical_alert(mode, exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _mode_clv() -> None:
    """
    --mode clv  — Closing Line Value snapshot (run nightly at 10 PM ET).

    Iterates all open bets that have a CLV snapshot, fetches the current
    market price from The Odds API, and records the closing line + CLV %
    in both clv_snapshots and the bets table.

    CLV > 0  → opening price was sharper than close (model beating the market).
    CLV < 0  → market moved away from us (soft side).
    Target: CLV > 0 on ≥ 55 % of picks.
    """
    import json
    import os as _os
    import requests as _requests

    API_KEY = _os.getenv("THE_ODDS_API_KEY", "")
    if not API_KEY:
        logger.warning("[CLV] THE_ODDS_API_KEY not set — CLV update skipped.")
        return

    _SPORT_KEY_MAP = {
        "WNBA": "basketball_wnba",
        "NBA":  "basketball_nba",
        "MLB":  "baseball_mlb",
    }

    open_bets = get_open_bets()
    if not open_bets:
        logger.info("[CLV] No open bets to snapshot.")
        return

    logger.info(f"[CLV] Snapshotting closing lines for {len(open_bets)} open bet(s)...")

    # Cache per-sport odds API responses to avoid duplicate calls
    _odds_cache: dict[str, list[dict]] = {}

    def _fetch_odds(sport: str, market_param: str) -> list[dict]:
        key = f"{sport}:{market_param}"
        if key in _odds_cache:
            return _odds_cache[key]
        sport_key = _SPORT_KEY_MAP.get(sport.upper())
        if not sport_key:
            return []
        url = (
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
            f"?apiKey={API_KEY}&regions=us&markets={market_param}"
            f"&oddsFormat=american&bookmakers=draftkings,fanduel,betmgm"
        )
        try:
            resp = _requests.get(url, timeout=15)
            if resp.status_code == 200:
                _odds_cache[key] = resp.json()
                return _odds_cache[key]
            logger.warning(f"[CLV] Odds API {resp.status_code} for {sport}/{market_param}")
        except Exception as exc:
            logger.warning(f"[CLV] Odds API fetch failed ({sport}/{market_param}): {exc}")
        return []

    try:
        from core.intelligence.clv_tracker import update_closing_line as _update_clv
    except ImportError:
        logger.warning("[CLV] clv_tracker not available — skipping.")
        return

    # Inline DB update for bets.closing_price / bets.clv_pct
    import sqlite3 as _sqlite3
    _DB = str(
        __import__("pathlib").Path(__file__).parent / "data" / "results.db"
    )

    def _write_bet_closing(bet_id: str, closing_price: int, clv: float | None) -> None:
        try:
            con = _sqlite3.connect(_DB)
            con.execute(
                "UPDATE bets SET closing_price=?, clv_pct=? WHERE bet_id=?",
                (closing_price, clv, bet_id),
            )
            con.commit()
        except Exception as exc:
            logger.debug(f"[CLV] bets table write failed for {bet_id}: {exc}")
        finally:
            con.close()

    updated = 0
    skipped = 0
    failed  = 0

    for bet_row in open_bets:
        try:
            bet_id   = bet_row.get("bet_id") or ""
            sport    = (bet_row.get("sport") or "").upper()
            wd_raw   = bet_row.get("wager_details") or "{}"
            wd       = json.loads(wd_raw) if isinstance(wd_raw, str) else wd_raw
            team     = (wd.get("team") or "").upper()
            market   = (wd.get("market") or "").lower()
            direction= (wd.get("direction") or "over").lower()
            line     = float(wd.get("line") or 0)

            if not bet_id or not sport or sport not in _SPORT_KEY_MAP:
                skipped += 1
                continue

            # Choose market endpoint
            is_prop = any(market.startswith(p) for p in ("player_", "pitcher_", "batter_"))
            if is_prop:
                mkt_param = "player_props"
            elif "spread" in market:
                mkt_param = "spreads"
            else:
                mkt_param = "totals"

            events = _fetch_odds(sport, mkt_param)

            # Scan events for a matching game + outcome price
            closing_odds: int | None = None
            closing_line: float      = line

            for event in events:
                home = (event.get("home_team") or "").upper()
                away = (event.get("away_team") or "").upper()
                # Flexible team match: abbreviation substring or full-name substring
                if team and not (
                    team in home or team in away
                    or any(team in w for w in home.split()) or any(team in w for w in away.split())
                ):
                    continue

                for bookmaker in event.get("bookmakers", []):
                    for mkt_obj in bookmaker.get("markets", []):
                        for outcome in mkt_obj.get("outcomes", []):
                            out_name = (outcome.get("name") or "").lower()
                            out_dir  = (
                                "over"  if out_name == "over"  else
                                "under" if out_name == "under" else ""
                            )
                            if out_dir and out_dir == direction:
                                try:
                                    closing_odds = int(round(float(outcome["price"])))
                                    closing_line = float(outcome.get("point", line))
                                except (TypeError, ValueError):
                                    pass
                                break
                        if closing_odds is not None:
                            break
                    if closing_odds is not None:
                        break
                if closing_odds is not None:
                    break

            if closing_odds is None:
                logger.debug(f"[CLV] No current line found for {bet_id} — skipped.")
                skipped += 1
                continue

            clv = _update_clv(bet_id, closing_odds, closing_line)
            _write_bet_closing(bet_id, closing_odds, clv)
            updated += 1
            label = "SHARP ✓" if (clv or 0) > 0 else "soft ✗"
            logger.info(
                f"[CLV] {bet_id}  closing={closing_odds:+d}  "
                f"CLV={clv:+.2f}%  [{label}]"
            )

        except Exception as exc:
            logger.warning(f"[CLV] Error processing {bet_row.get('bet_id','?')}: {exc}")
            failed += 1

    logger.info(
        f"[CLV] Complete: {updated} updated, {skipped} no-match/skipped, {failed} error(s)."
    )

    # Print aggregate CLV report
    try:
        from core.intelligence.clv_tracker import print_clv_report
        print_clv_report()
    except Exception as exc:
        logger.debug(f"[CLV] print_clv_report failed: {exc}")


def _mode_revalidate(sports: list[str], dry_run: bool) -> None:
    """--mode revalidate  — pregame revalidation of today's open picks."""
    logger.info("Pregame Revalidation — evaluating today's open picks.")
    try:
        from core.revalidation_engine import run_revalidation
        changes = run_revalidation(sports=sports, dry_run=dry_run)
    except Exception as exc:
        logger.error(f"Revalidation engine error: {exc}", exc_info=True)
        return

    notable = [c for c in changes if c.get("revalidation_status") != "confirmed"]
    if not notable:
        logger.info("Revalidation complete — all picks confirmed, no alerts sent.")
        return

    # Reversal alerts (≥50 ppt shift) are sent inline by run_revalidation()
    # via _send_regrade_telegram_alert.  No secondary broadcast needed here.
    logger.info(
        f"Revalidation complete — {len(notable)} notable change(s) "
        f"(reversal alerts fired inline if ≥50 ppt shift detected)."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["run", "close", "refine", "recap", "picks", "open",
                 "calibrate", "calibrate-odds", "grade", "reconcile", "revalidate", "clv"],
        default="run",
        metavar="MODE",
        help=(
            "run            – full pipeline: recap → 60 s → picks  [default]\n"
            "close          – record a bet result (--bet-id + --outcome required)\n"
            "refine         – auto-tune sports_metrics.json weights\n"
            "recap          – morning recap only\n"
            "picks          – picks broadcast only\n"
            "open           – list open bets\n"
            "calibrate      – regular vs playoff tier comparison (no DB / Telegram)\n"
            "calibrate-odds – A/B live-slate calibration with Feature Variance Report\n"
            "grade          – auto-grade settled bets via Odds API scores\n"
            "revalidate     – pregame revalidation of open picks (30-60 min before game)\n"
            "clv            – snapshot closing lines for all open bets (10 PM ET daily)"
        ),
    )
    parser.add_argument(
        "--sport",
        default="ALL",
        choices=["WNBA", "NBA", "MLB", "ALL"],
        help="Sport scope (default: ALL)",
    )
    parser.add_argument(
        "--bet-id",
        dest="bet_id",
        metavar="ID",
        help="Bet ID to close (--mode close only)",
    )
    parser.add_argument(
        "--outcome",
        choices=["win", "loss", "push"],
        help="Game result (--mode close only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print messages without sending to Telegram or sleeping",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args   = _parse_args(argv)
    sports = ALL_SPORTS if args.sport == "ALL" else [args.sport]
    sport  = None       if args.sport == "ALL" else args.sport

    init_db()   # ensure DB schema is current; _migrate_db() is idempotent

    if args.mode == "run":
        _run_mode("run",    _mode_run,    sports,   args.dry_run)
    elif args.mode == "close":
        _run_mode("close",  _mode_close,  args.bet_id, args.outcome)
    elif args.mode == "refine":
        _run_mode("refine", _mode_refine, sports)
    elif args.mode == "recap":
        _run_mode("recap",  _mode_recap,  sport,    args.dry_run)
    elif args.mode == "picks":
        _run_mode("picks",  _mode_picks,  sports,   args.dry_run)
    elif args.mode == "open":
        _run_mode("open",      _mode_open,      sport)
    elif args.mode == "calibrate":
        _run_mode("calibrate", _mode_calibrate)
    elif args.mode == "calibrate-odds":
        _run_mode("calibrate-odds", _mode_calibrate_odds, sports, args.dry_run)
    elif args.mode == "grade":
        _run_mode("grade", _mode_grade, sports)
    elif args.mode == "reconcile":
        _run_mode("reconcile", _mode_reconcile, sports)
    elif args.mode == "revalidate":
        _run_mode("revalidate", _mode_revalidate, sports, args.dry_run)
    elif args.mode == "clv":
        _run_mode("clv", _mode_clv)


if __name__ == "__main__":
    main()
