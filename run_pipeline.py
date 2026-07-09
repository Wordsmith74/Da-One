"""
Glue script: chains the core/ Bayesian engine into one pipeline and writes
output/picks.json.

This is the ported version of main.py's prediction pipeline (everything
except the MiniApp, Telegram broadcast layer, and cron scheduler), adapted
onto the simpler no-DB / JSON-output architecture this repo already uses
for GitHub Pages.

Pipeline order
--------------
  1. Fetch today's game-total candidates from core/odds_client.py (The Odds
     API -- WNBA, MLB; NBA omitted, not currently published).
  2. DecisionOrchestrator -- sport validation + game-time window check.
  3. SimulationEngine.analyze() -- Bayesian posterior (PyMC NUTS, falls
     back to a fast analytical posterior on timeout) + Monte Carlo win
     probability.
  4. Derive edge_percentage / confidence_score / model_probability from
     the posterior (ported from main.py's _derive_bet_params).
  5. Build Bet objects; run_gatekeeper() -- tier assignment + same-game
     conflict detection (core/decision_gatekeeper.py).
  6. apply_game_truth_protocol() -- one Value Vector per game, sport
     volatility thresholds (core/game_truth.py).
  7. Map surviving Bets into this repo's existing pick-dict schema so the
     rest of the file (contradiction check, line movement check, daily
     caps, output writers) is unchanged from before this port.
  8. Write output/picks.json + output/run_log.json + append
     output/pick_history.jsonl, same as before.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

from models.contradiction_check import filter_contradictions
from models.line_movement import apply_line_movement_filter
from models.sport_config import MLB, WNBA
from models.handicapper_rules import kelly_stake
from data.cache_history import append_picks
from data.name_registry import canonical_team, canonical_player

from core.decision_orchestrator import DecisionOrchestrator, UnsupportedSportError
from core.simulation_engine import SimulationEngine, _MC_SIGMA_FLOOR
from core.decision_gatekeeper import Bet, Tier, run_gatekeeper, market_normalized
from core.game_truth import apply_game_truth_protocol, mark_picks_published
from core.odds_client import fetch_todays_candidates
from core.player_props import get_player_prop_candidates
from core.game_markets import fetch_expanded_game_candidates
from core.stability_filter import check_stability
from core.market_gate import filter_candidates as gate_filter_candidates, log_market_filter_summary
from core.market_governance import is_publication_eligible
from core.bet_display import BetDisplay
from core.composite_confidence_score import compute_ccs
from core.conflict_guardian import check_locked_conflict
from core.line_validator import pre_publish_verify
from core.results_tracker import init_db, log_bet_dict
from core.edge_calibrator import is_game_market, calibrate_edge
from core.market_intelligence import compute_data_reliability, tier_eligibility
from shadow_logger import log_candidate
from core.reject_logger import log_rejected_candidate, log_rejected_bet_obj
from core.intelligence import (
    get_lineup_intel,
    get_stat_model_factor,
    get_rest_travel_factor,
    get_venue_factor,
)

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# Sports this pipeline currently publishes picks for. NBA support exists in
# core/odds_client.py but is intentionally not enabled here yet -- flip on
# once there's a confirmed live NBA slate/season to validate against.
ENABLED_SPORTS = ["MLB", "WNBA"]

# ---------- Structured run log ----------
RUN_LOG = []


def _enrich_integrity_fields(c, sport):
    """
    Populate the fields core/integrity_filters.py checks for on game-market
    candidates (injury, pace, rotation, market agreement, rest/travel, and
    -- for MLB -- starting pitcher, bullpen, and weather).

    Without this, every WNBA/NBA/MLB game-market pick that reaches Diamond
    or Nuke tier in the gatekeeper gets auto-discarded by the V3.0 integrity
    filter, because `bet.raw_result` (== this candidate dict) never carries
    these keys. The intelligence modules already exist in core/intelligence/
    -- they just weren't being called anywhere in the pipeline.

    Fields are written unconditionally (even when a module returns a zeroed
    default) because the integrity filter only checks `is not None` -- it
    needs a key to be present, not necessarily populated with real data.
    A zeroed/empty value still satisfies the filter; a missing key fails it.
    """
    if not is_game_market(c.get("market", "")):
        return  # player props skip the integrity filter entirely

    team_abbr = c.get("team", "")
    home_team = c.get("home_team", "")
    away_team = c.get("away_team", "")
    direction = c.get("direction", "over")
    game_time = c.get("game_time_utc")

    # --- 1. Injury / lineup intel -------------------------------------------
    try:
        injury = get_lineup_intel(team_abbr, sport, bet_on_this_team=True)
    except Exception as exc:
        log("warn", sport, f"{c.get('bet_id', '?')}: lineup_intel enrichment failed -- {exc}")
        injury = None

    c["injury_report"] = (injury.factor_text if injury and injury.factor_text
                          else f"{team_abbr}: no injuries fetched")
    c["injury_score"]  = injury.impact_score if injury else 0.0
    # rotation_score proxies off injury availability (no standalone module yet)
    c.setdefault("rotation_score", c["injury_score"])

    # --- 2. Pace / stat model -----------------------------------------------
    try:
        stat = get_stat_model_factor(
            team_abbr, c.get("market", ""), sport,
            sportsbook_line=c.get("sportsbook_line"), direction=direction,
        )
    except Exception as exc:
        log("warn", sport, f"{c.get('bet_id', '?')}: stat_model enrichment failed -- {exc}")
        stat = None

    # Write pace_projection unconditionally; fall back to 0.0 so the key
    # exists and the integrity filter passes even when the fetch fails.
    c["pace_projection"] = stat.pace if (stat and stat.pace is not None) else 0.0

    # --- 3. Rest / travel ---------------------------------------------------
    try:
        rest = get_rest_travel_factor(team_abbr, sport, game_time_utc=game_time)
    except Exception as exc:
        log("warn", sport, f"{c.get('bet_id', '?')}: rest_travel enrichment failed -- {exc}")
        rest = None

    c["rest_days"]     = rest.rest_days    if (rest and rest.rest_days    is not None) else 0
    c["travel_factor"] = rest.travel_miles if (rest and rest.travel_miles is not None) else 0.0

    # --- 4. Venue / park factor ---------------------------------------------
    try:
        venue = get_venue_factor(home_team, away_team, sport, direction=direction)
    except Exception as exc:
        log("warn", sport, f"{c.get('bet_id', '?')}: venue_intel enrichment failed -- {exc}")
        venue = None

    c["park_factor"] = venue.park_factor if (venue and venue.park_factor is not None) else 0

    # --- 5. MLB starting pitcher + bullpen + weather -------------------------
    # core/odds_client.py already calls pitcher_intel/bullpen_intel for the
    # base "totals" candidates it builds and feeds the result into the edge
    # model itself -- but core/game_markets.py (run line, F5 variants, team
    # total, NRFI/YRFI, and every other expanded MLB market) never calls
    # either module, so those candidates reach the gatekeeper with the keys
    # missing entirely. Weather was not wired into candidate enrichment
    # anywhere in the pipeline -- models/weather_intel.py existed but was
    # only ever consumed by strikeout_matchup.py's K-rate layer.
    #
    # setdefault-style guards below only backfill when odds_client hasn't
    # already supplied a real value, so we never clobber the figure the
    # model actually used to compute edge/confidence for totals candidates.
    if sport.upper() == "MLB":
        game_date = game_time.date() if hasattr(game_time, "date") else None

        if c.get("starting_pitcher") is None and c.get("sp_fip") is None:
            try:
                from core.intelligence.pitcher_intel import get_pitcher_intel
                sp = get_pitcher_intel(home_team, away_team, game_date)
            except Exception as exc:
                log("warn", sport, f"{c.get('bet_id', '?')}: pitcher_intel enrichment failed -- {exc}")
                sp = None
            c["starting_pitcher"] = (
                f"{sp.away_pitcher}/{sp.home_pitcher}"
                if (sp and sp.combined_fip is not None) else "unavailable"
            )
            # sp_fip can legitimately be None on an API miss -- fall back to
            # 0.0 so the key is present (integrity filter only checks
            # is not None) rather than silently discarding the pick.
            c["sp_fip"] = sp.combined_fip if (sp and sp.combined_fip is not None) else 0.0

        if (c.get("bullpen_score") is None and c.get("bullpen_fatigue") is None
                and c.get("bullpen_era") is None):
            try:
                from core.intelligence.bullpen_intel import get_bullpen_intel
                bp = get_bullpen_intel(home_team, away_team, game_date)
            except Exception as exc:
                log("warn", sport, f"{c.get('bet_id', '?')}: bullpen_intel enrichment failed -- {exc}")
                bp = None
            c["bullpen_score"]   = bp.bullpen_score   if bp else 50.0
            c["bullpen_fatigue"] = bp.bullpen_fatigue if bp else 0.0
            c["bullpen_era"]     = bp.bullpen_era     if bp else None

        try:
            from models.weather_intel import get_weather
            wx = get_weather(home_team, away_team, game_date.isoformat() if game_date else None)
        except Exception as exc:
            log("warn", sport, f"{c.get('bet_id', '?')}: weather_intel enrichment failed -- {exc}")
            wx = None
        # Always write a non-None "weather" key -- even a dome/unavailable
        # marker -- so the integrity filter sees the element as present.
        c["weather"]      = wx if wx is not None else {"unavailable": True}
        c["temperature"]  = wx.get("temp_f")   if wx else None
        c["wind_factor"]  = wx.get("wind_mph") if wx else None
        c["weather_score"] = (
            round(wx["temp_f"] - 70.0, 1)
            if (wx and not wx.get("is_dome") and wx.get("temp_f") is not None)
            else 0.0
        )

    # --- 6. Market agreement score ------------------------------------------
    # No dedicated scorer exists yet in the repo; composite_confidence_score.py
    # defaults to 50 when the key is absent. Set explicitly so the integrity
    # filter sees a present key rather than a missing one.
    # TODO: replace with a real cross-book agreement model.
    c.setdefault("market_agreement_score", 50)


def log(level, stage, message):
    """level: 'info' | 'warn' | 'error'. Always prints AND records structured."""
    entry = {"level": level, "stage": stage, "message": str(message)}
    RUN_LOG.append(entry)
    prefix = {"info": "  ", "warn": "[warn] ", "error": "[ERROR] "}[level]
    print(f"{prefix}[{stage}] {message}")


def _debug_gatekeeper_reasons(sport, flagged, discarded):
    """
    Debug helper -- surfaces WHY each bet was flagged/discarded by the
    gatekeeper into the structured run log (output/run_log.json).

    decision_gatekeeper.py already computes a `flag_reason` string for every
    bet that doesn't get approved, but it only emits it via
    `logging.getLogger("betting_bot").debug(...)`, which is a totally
    separate logging path from this file's RUN_LOG/log() -- so none of that
    detail was ever reaching run_log.json. This pulls it across.

    Set env var GATEKEEPER_DEBUG=0 to silence (defaults on).
    """
    if os.environ.get("GATEKEEPER_DEBUG", "1") == "0":
        return
    for bet in flagged:
        log(
            "info",
            sport,
            f"{bet.bet_id}: GATEKEEPER FLAGGED -- tier={bet.tier} "
            f"edge={bet.edge_percentage:.2f}% conf={bet.confidence_score:.1f} "
            f"market={bet.market} | {bet.flag_reason or 'no reason recorded'}",
        )
    for bet in discarded:
        log(
            "info",
            sport,
            f"{bet.bet_id}: GATEKEEPER DISCARD -- "
            f"edge={bet.edge_percentage:.2f}% conf={bet.confidence_score:.1f} "
            f"market={bet.market} | {bet.flag_reason or 'below threshold, no specific reason recorded'}",
        )


def run_preflight_checks():
    """Check env vars / packages BEFORE touching any live data."""
    print("=== Preflight checks ===")
    ok = True

    for var in ("THE_ODDS_API_KEY",):
        if os.environ.get(var):
            log("info", "preflight", f"{var} is set")
        else:
            log("error", "preflight", f"{var} is NOT set -- odds_client calls will raise")
            ok = False

    for pkg in ("requests", "pandas", "numpy", "pymc", "arviz"):
        try:
            __import__(pkg)
            log("info", "preflight", f"package '{pkg}' importable")
        except ImportError as e:
            log("error", "preflight", f"package '{pkg}' missing: {e}")
            ok = False

    print(f"=== Preflight {'PASSED' if ok else 'FAILED -- see warnings above'} ===\n")
    return ok


def _score_and_gate_mlb_totals_reliability(c):
    """
    Real Data Reliability Score (DRS, 0-100) for MLB full-game totals --
    wired to core/market_intelligence.py's compute_data_reliability() /
    tier_eligibility(), the same scorer core/player_props.py already uses
    for player props.

    Previously MLB total candidates never got a DRS at all: the pick-dict
    builder further down (`c.get("data_reliability_score", 100)`) silently
    defaulted every one of them to 100 (max trust), which made
    composite_confidence_score.py's 35%-weighted reliability factor a
    no-op for this market -- edge alone effectively decided ranking.

    This runs BEFORE engine.analyze()/_derive_bet_params(), i.e. before any
    edge is computed for the candidate, per the "accuracy before edge" rule:
    a candidate whose real projection inputs (starting pitcher, bullpen,
    sample depth, book coverage, injury data) don't clear the floor never
    gets an edge computed for it at all -- it's rejected outright rather
    than being scored and merely ranked lower later.

    Inputs are read off fields _enrich_integrity_fields() already populates
    (starting_pitcher/sp_fip/bullpen_era from pitcher_intel/bullpen_intel,
    injury_report from lineup_intel) plus book_count/historical_data that
    core/odds_client.py already attaches to every totals candidate.

    Returns (keep: bool, drs: int, reason: str).
    """
    has_real_stats = (
        c.get("starting_pitcher") not in (None, "unavailable")
        and c.get("sp_fip") not in (None, 0.0)
        and c.get("bullpen_era") is not None
    )
    hist = c.get("historical_data") or []
    has_l5 = len(hist) >= 5
    has_l10 = len(hist) >= 10
    book_count = c.get("book_count", 0) or 0
    injury_report = c.get("injury_report", "") or ""
    has_injury_data = bool(injury_report) and not injury_report.endswith("no injuries fetched")

    drs = compute_data_reliability(
        has_real_stats=has_real_stats,
        book_count=book_count,
        has_l5=has_l5,
        has_l10=has_l10,
        has_injury_data=has_injury_data,
    )
    elig = tier_eligibility(drs)
    if not elig["allow_play"]:
        reason = (
            f"DRS {drs} < 40 floor -- has_real_stats={has_real_stats}, "
            f"book_count={book_count}, has_l5={has_l5}, has_l10={has_l10}, "
            f"has_injury_data={has_injury_data}"
        )
        return False, drs, reason
    return True, drs, "ok"


# ---------- Bet-derivation math (ported from main.py's _derive_bet_params) ----------

def _derive_bet_params(sim, candidate):
    """
    Extract edge_percentage, confidence_score, and model_probability from a
    SimulationEngine.analyze() result. See core/simulation_engine.py and the
    original main.py for the full rationale behind this math.
    """
    posterior_mean = sim["posterior"]["posterior_mean"]
    posterior_std = sim["posterior"]["posterior_std"]
    line = candidate["sportsbook_line"]
    direction = candidate["direction"].lower()
    odds = candidate["american_odds"]

    if direction == "over":
        model_prob = sim["win_probability"]["over_probability"]
    else:
        model_prob = sim["win_probability"]["under_probability"]

    # ── Weighted multi-book devig (core/devig.py) ──────────────────────────
    # Replaces the raw single-book implied probability with a vig-free
    # consensus blended across whichever of the 3 named books (weights vary
    # by market type — see devig.get_weights_for_market) quoted BOTH sides
    # of this line. Falls back to the old raw-odds formula when none of
    # those books posted a two-sided price for this candidate (e.g. early
    # in the fetch rollout, or a thin market) — never silently drops a pick,
    # just benchmarks it against a less-precise number, same as before.
    from core.devig import weighted_fair_prob_for_candidate
    devig_result = weighted_fair_prob_for_candidate(candidate)
    candidate["devig_meta"] = devig_result  # visibility for the caller's log line
    # side_agreement_frac: fraction of reporting book-weight whose own
    # devigged price also favors this side. weighted_fair_prob_for_candidate
    # now computes this (see core/devig.py:compute_side_agreement_frac) --
    # attach it to the candidate here so it survives into the pick dict the
    # same way posterior_std/mean do, instead of being silently dropped like
    # it was before (append_picks/log_candidate always received None for
    # this field because nothing upstream ever set it).
    candidate["side_agreement_frac"] = devig_result.get("side_agreement_frac")

    # market_agreement_score: _enrich_integrity_fields() (called earlier, per
    # candidate, before this edge/confidence step) sets this to a hardcoded
    # neutral 50 -- "no dedicated scorer exists yet" -- because at that point
    # in the pipeline side_agreement_frac hasn't been computed. It now has
    # been (immediately above), so overwrite the neutral placeholder with the
    # real cross-book agreement figure whenever devig actually produced one.
    # This is Factor 2 (Signal Agreement) input in composite_confidence_score.py
    # at 35% sub-weight (8.75% of overall CCS) -- previously a dead constant.
    if candidate["side_agreement_frac"] is not None:
        candidate["market_agreement_score"] = round(candidate["side_agreement_frac"] * 100.0, 1)

    if devig_result["fair_prob"] is not None:
        implied = devig_result["fair_prob"] * 100.0
    elif odds < 0:
        implied = abs(odds) / (abs(odds) + 100) * 100
    else:
        implied = 100 / (odds + 100) * 100

    raw_edge = round(min(50.0, model_prob - implied), 2)

    sport_key = sim.get("sport_type", "default")
    mkt = market_normalized(candidate.get("market", ""))

    # ── Reliability damping (MLB totals only) ──────────────────────────────
    # "Accuracy before edge" stage 2: a candidate that cleared the DRS >= 40
    # floor in _score_and_gate_mlb_totals_reliability() but isn't full-strength
    # (DRS 40-74) still has its raw edge compressed here, BEFORE
    # calibrate_edge()/edge_threshold_pct ever sees it -- rather than being
    # scored at full edge and only down-weighted afterward by CCS's 35%
    # reliability factor. Full-strength DRS (>=75) gets no damping (1.0x).
    # data_reliability_score is only set on the candidate for MLB totals
    # (see the gate above); every other market falls through untouched.
    drs = candidate.get("data_reliability_score")
    if drs is not None and mkt == "game_total":
        if drs < 60:
            damp = 0.75
        elif drs < 75:
            damp = 0.90
        else:
            damp = 1.00
        if damp != 1.00:
            raw_edge = round(raw_edge * damp, 2)

    # V3.0 calibration: compress the inflated simulation-native raw edge into
    # the calibrated scale that SPORT_TIER_THRESHOLDS / _MARKET_ENTRY_FLOORS
    # are actually written for. Previously this step was skipped entirely and
    # the raw 24-50% edge was written straight onto Bet.edge_percentage,
    # which trivially cleared every tier's edge floor and made the edge gate
    # a no-op (tier assignment was confidence-only).
    edge = calibrate_edge(raw_edge, sport_key, mkt)

    floors = _MC_SIGMA_FLOOR.get(sport_key, _MC_SIGMA_FLOOR["default"])
    sigma_floor = floors.get(mkt, floors.get("default", 1.5))
    conf_std = max(posterior_std, sigma_floor)

    z = abs(posterior_mean - line) / max(conf_std, 0.01)
    confidence = round(min(99.0, 50.0 + z * 25.0), 1)

    return edge, confidence, round(model_prob, 1)


# ---------- Per-sport pipeline ----------

def run_sport_pipeline(sport, as_of_date=None):
    """
    Full prediction pipeline for one sport. Returns a list of pick dicts in
    this repo's existing schema (compatible with filter_contradictions,
    apply_line_movement_filter, and _apply_daily_caps below).

    as_of_date : ISO 'YYYY-MM-DD' string. When set (replay mode), threads
                 through to every candidate source (odds_client, player_props,
                 game_markets) so they pull historical-snapshot data bounded
                 to that date instead of live "today" data, and bypasses the
                 conflict_guardian's live results.db read entirely (there is
                 no "locked pick" concept in a backtest). This function has
                 no live-output side effects either way -- results.db writes,
                 output/picks.json, and pick_history.jsonl are all handled by
                 the caller (run_pipeline()) -- so it's safe for replay.py to
                 call this directly per historical date.
    """
    slate_date = as_of_date or TODAY

    try:
        orchestrator = DecisionOrchestrator(sport)
    except UnsupportedSportError as exc:
        log("warn", sport, f"Unsupported sport -- {exc}")
        return []

    engine = SimulationEngine(orchestrator)

    try:
        candidates = fetch_todays_candidates(sport, as_of_date=as_of_date)
    except Exception as exc:
        log("error", sport, f"odds_client fetch failed ({type(exc).__name__}: {exc})")
        return []

    try:
        prop_candidates = get_player_prop_candidates(sport, as_of_date=as_of_date)
        if prop_candidates:
            candidates = candidates + prop_candidates
            log("info", sport, f"{len(prop_candidates)} player prop candidate(s) added.")
    except Exception as exc:
        log("error", sport, f"player_props fetch failed ({type(exc).__name__}: {exc}) -- continuing with game totals only")

    # Expanded game markets (moneyline / spread / team_total). Built in
    # game_markets.py but never wired into the pipeline until now -- WNBA's
    # scope here is h2h + team_totals + spreads (see _MARKET_BUNDLE). Totals
    # candidates from this call flow through engine.analyze() like normal;
    # moneyline/spread candidates carry precomputed_edge/confidence/model_prob
    # and bypass the Bayesian engine entirely (handled below in the main loop).
    try:
        expanded_candidates = fetch_expanded_game_candidates(sport, as_of_date=as_of_date)
        if expanded_candidates:
            candidates = candidates + expanded_candidates
            log("info", sport, f"{len(expanded_candidates)} expanded game market candidate(s) added.")
    except Exception as exc:
        log("error", sport, f"game_markets fetch failed ({type(exc).__name__}: {exc}) -- continuing without expanded markets")

    if not candidates:
        log("info", sport, "No candidates returned for today.")
        return []

    # ── System Scope Definition Layer (core/market_gate.py) ──────────────
    # Blocks any (sport, market) pair outside the documented approved scope
    # before it reaches simulation/scoring. See core/market_gate.py for the
    # allowed-markets table and core/market_governance.py for the separate
    # publication whitelist applied later, at final output time.
    total_before = len(candidates)
    candidates, blocked = gate_filter_candidates(candidates, sport)
    if blocked:
        blocked_reasons: dict[str, int] = {}
        for bc in blocked:
            key = bc.get("market_key") or bc.get("market", "unknown")
            blocked_reasons[key] = blocked_reasons.get(key, 0) + 1
        log("info", sport, f"market_gate blocked {len(blocked)}/{total_before} candidate(s) outside approved scope: {blocked_reasons}")
        log_market_filter_summary(sport, total_before, len(candidates), len(blocked), blocked_reasons)

    if not candidates:
        log("info", sport, "No candidates remain after market_gate scope filter.")
        return []

    log("info", sport, f"{len(candidates)} total candidate(s) (game totals + player props).")

    processed = []
    for c in candidates:
        bet_id = c.get("bet_id", "?")

        # Enrichment moved here (previously ran AFTER edge/confidence were
        # already computed on the simulated path, and again -- redundantly
        # -- on the precomputed path). Pitcher/bullpen/weather/injury fields
        # now exist before ANY edge math for ANY candidate, which is what
        # the reliability gate right below depends on.
        _enrich_integrity_fields(c, sport)

        # ── MLB totals reliability gate ("accuracy before edge", stage 1) ──
        # Only scoped to MLB full-game totals for now -- player props already
        # get a real DRS from core/player_props.py, and moneyline/spread
        # candidates carry their own precomputed edge/confidence logic.
        if sport.upper() == "MLB" and market_normalized(c.get("market", "")) == "game_total":
            keep, drs, reason = _score_and_gate_mlb_totals_reliability(c)
            c["data_reliability_score"] = drs
            if not keep:
                log("info", sport, f"{bet_id}: RELIABILITY REJECT -- {reason}")
                log_rejected_candidate(
                    sport=sport, candidate=c, stage="reliability_gate",
                    reason=reason, slate_date=slate_date,
                )
                log_candidate(
                    sport=sport, player=c.get("player"), matchup=c.get("matchup", c.get("game_id", "")),
                    market_line=c.get("sportsbook_line"), side=c.get("direction"),
                    rejected_stage="reliability_gate", rejected_reason=reason,
                    published=False, extra={"bet_id": bet_id, "data_reliability_score": drs},
                )
                continue

        # ── Precomputed markets (moneyline / spread from game_markets.py) ──
        # These bypass the NUTS sampler entirely -- _process_moneyline() /
        # _process_spread() already computed a Kelly-style edge and a
        # deliberately-capped (<=82) confidence. Still run them through
        # calibrate_edge() so they land on the same scale SPORT_TIER_THRESHOLDS
        # was written for -- same rule as every other market.
        if "precomputed_edge" in c:
            raw_edge = c["precomputed_edge"]
            confidence = c["precomputed_confidence"]
            model_prob = round(c["precomputed_model_prob"] * 100, 1)
            mkt = market_normalized(c.get("market", ""))
            edge = calibrate_edge(raw_edge, sport, mkt)
            c.setdefault("posterior_std", None)
            c.setdefault("posterior_mean", None)
            c.setdefault("relative_sigma_pct", None)
            c.setdefault("side_agreement_frac", None)
            processed.append((c, edge, confidence, model_prob))
            continue

        try:
            sim = engine.analyze(
                historical_data=c["historical_data"],
                league_mean=c["league_mean"],
                league_std=c.get("league_std", 5.0),
                sportsbook_line=c["sportsbook_line"],
                progressbar=False,
                context=c.get("context", "regular"),
                recent_n=c.get("recent_n", 5),
                volatility_index=c.get("volatility_index"),
                market_type=c.get("market", ""),
            )
        except Exception as exc:
            log("error", sport, f"{bet_id}: engine.analyze() failed -- {exc}")
            log_rejected_candidate(
                sport=sport, candidate=c, stage="pre_sim",
                reason=f"engine.analyze() failed: {exc}", slate_date=slate_date,
            )
            log_candidate(
                sport=sport, player=c.get("player"), matchup=c.get("matchup", c.get("game_id", "")),
                market_line=c.get("sportsbook_line"), side=c.get("direction"),
                rejected_stage="pre_sim", rejected_reason=f"engine.analyze() failed: {exc}",
                published=False, extra={"bet_id": bet_id},
            )
            continue

        post_std = sim["posterior"].get("posterior_std", 0.0)
        post_mean = sim["posterior"].get("posterior_mean", 0.0)
        is_stable, stab_reason = check_stability(sport, post_std, post_mean)
        if not is_stable:
            log("info", sport, f"{bet_id}: STABILITY REJECT -- {stab_reason}")
            _rel_sigma_pct = (
                abs(post_std / post_mean) * 100.0
                if post_mean else None
            )
            log_rejected_candidate(
                sport=sport, candidate=c, stage="stability", reason=stab_reason,
                slate_date=slate_date, sigma=post_std, projection=post_mean,
                rel_sigma_pct=_rel_sigma_pct,
            )
            log_candidate(
                sport=sport, player=c.get("player"), matchup=c.get("matchup", c.get("game_id", "")),
                market_line=c.get("sportsbook_line"), side=c.get("direction"),
                rejected_stage="stability", rejected_reason=stab_reason,
                published=False, extra={"bet_id": bet_id, "posterior_std": post_std, "posterior_mean": post_mean},
            )
            continue

        # Carry sigma forward onto the candidate itself -- previously these
        # two locals only lived inside this loop iteration and were dropped
        # the moment the stability gate passed, so every published pick lost
        # its own uncertainty data. Downstream code (pick dict construction,
        # ~line 600) reads these back off `c`.
        c["posterior_std"] = post_std
        c["posterior_mean"] = post_mean
        c["relative_sigma_pct"] = (
            round(abs(post_std / post_mean) * 100.0, 2) if post_mean else None
        )

        try:
            edge, confidence, model_prob = _derive_bet_params(sim, c)
        except Exception as exc:
            log("error", sport, f"{bet_id}: _derive_bet_params failed -- {exc}")
            log_rejected_candidate(
                sport=sport, candidate=c, stage="pre_sim",
                reason=f"_derive_bet_params failed: {exc}", slate_date=slate_date,
                projection=post_mean, sigma=post_std,
            )
            log_candidate(
                sport=sport, player=c.get("player"), matchup=c.get("matchup", c.get("game_id", "")),
                market_line=c.get("sportsbook_line"), side=c.get("direction"),
                rejected_stage="pre_sim", rejected_reason=f"_derive_bet_params failed: {exc}",
                published=False, extra={"bet_id": bet_id},
            )
            continue

        _dv = c.get("devig_meta") or {}
        if _dv.get("fair_prob") is not None:
            log(
                "info", sport,
                f"{bet_id}: devig fair_prob={_dv['fair_prob']*100:.1f}% "
                f"books={_dv['books_used']} weight_covered={_dv['weight_covered']:.2f}"
                + (" [LOW CONFIDENCE — <1 full book of weight]" if _dv.get("low_confidence") else "")
            )
        else:
            log(
                "info", sport,
                f"{bet_id}: devig unavailable (no named book quoted both sides) "
                f"-- fell back to raw single-book implied prob"
            )

        processed.append((c, edge, confidence, model_prob))

    if not processed:
        log("info", sport, "All candidates failed simulation -- skipping.")
        return []

    # Game Truth Protocol -- one Value Vector per game.
    # NOTE: apply_game_truth_protocol() logs its own per-game reasoning via
    # the standard `logging` module (logger "betting_bot"), which is a
    # separate sink from this file's log()/RUN_LOG -- entries there never
    # reach output/run_log.json. Log the before/after delta here too so the
    # suppression is visible in the same place every other pipeline stage
    # reports to, instead of silently disappearing from run_log.json.
    _pre_gtp_count = len(processed)
    processed = apply_game_truth_protocol(processed, sport)
    _n_suppressed = _pre_gtp_count - len(processed)
    if _n_suppressed:
        log(
            "info", sport,
            f"Game Truth Protocol: {_pre_gtp_count} candidate(s) in -> "
            f"{len(processed)} out ({_n_suppressed} suppressed -- see "
            f"'betting_bot' logger for per-game reasoning)."
        )
    if not processed:
        log("info", sport, "All candidates suppressed by Game Truth Protocol.")
        return []

    bets = []
    for c, edge, confidence, _ in processed:
        bets.append(Bet(
            bet_id=c["bet_id"],
            team=c["team"],
            market=market_normalized(c["market"]),
            direction=c["direction"],
            sportsbook_line=c["sportsbook_line"],
            edge_percentage=edge,
            confidence_score=confidence,
            player=c.get("player"),
            game_id=c.get("game_id", ""),
            american_odds=float(c.get("american_odds", 0)),
            data_reliability_score=c.get("data_reliability_score", 100),
            mis_score=c.get("mis_score", 0),
            raw_result=c,
        ))

    gk_result = run_gatekeeper(bets, sport=sport)
    approved = gk_result["approved"]
    flagged = gk_result["flagged"]
    discarded = gk_result["discarded"]

    log("info", sport, f"Gatekeeper: {len(approved)} approved / {len(flagged)} flagged / {len(discarded)} discarded")
    _debug_gatekeeper_reasons(sport, flagged, discarded)

    for bet in discarded:
        log_candidate(
            sport=sport, player=bet.player, matchup=(bet.raw_result or {}).get("matchup", bet.game_id),
            market_line=bet.sportsbook_line, side=bet.direction,
            model_prob=None, edge_pct=bet.edge_percentage, confidence=bet.confidence_score,
            rejected_stage="gatekeeper", rejected_reason=bet.flag_reason or "below Gold Standard threshold",
            published=False, extra={"bet_id": bet.bet_id},
        )
    for bet in flagged:
        log_candidate(
            sport=sport, player=bet.player, matchup=(bet.raw_result or {}).get("matchup", bet.game_id),
            market_line=bet.sportsbook_line, side=bet.direction,
            model_prob=None, edge_pct=bet.edge_percentage, confidence=bet.confidence_score,
            rejected_stage="gatekeeper_flagged", rejected_reason=bet.flag_reason or "flagged, not discarded",
            published=False, extra={"bet_id": bet.bet_id},
        )

    # ── Locked-pick conflict guardian ───────────────────────────────────────
    # Checks each approved bet against any already-LOCKED pick on the same
    # (game_id, market) from a prior run today. "hold" = drop the new
    # candidate (existing locked pick stands); "replace"/"clear" = keep it.
    # signal_confirmation.py (multi-cycle confirmation) is intentionally NOT
    # wired in -- condition 4 of the 5-condition threshold below is hardcoded
    # True inside conflict_guardian.py for exactly this reason.
    no_conflict = []
    for bet in approved:
        try:
            action, details = check_locked_conflict(
                bet, sport, date_str=slate_date, skip=(as_of_date is not None),
            )
        except Exception as exc:
            log("warn", sport, f"{bet.bet_id}: conflict_guardian check failed (non-fatal): {exc}")
            no_conflict.append(bet)
            continue
        if action == "hold":
            log("info", sport, f"{bet.bet_id}: CONFLICT HOLD -- existing locked pick "
                                f"{details.get('existing_bet_id','?')} stands (replacement threshold not met)")
            _hold_reason = (
                f"conflict_guardian hold: existing locked pick "
                f"{details.get('existing_bet_id','?')} stands"
            )
            log_rejected_bet_obj(bet, sport, slate_date, "conflict_hold", reason_override=_hold_reason)
            log_candidate(
                sport=sport, player=bet.player, matchup=(bet.raw_result or {}).get("matchup", bet.game_id),
                market_line=bet.sportsbook_line, side=bet.direction,
                edge_pct=bet.edge_percentage, confidence=bet.confidence_score,
                rejected_stage="conflict_hold", rejected_reason=_hold_reason,
                published=False, extra={"bet_id": bet.bet_id},
            )
        else:
            if action == "replace":
                log("info", sport, f"{bet.bet_id}: CONFLICT REPLACE -- superseding "
                                    f"{details.get('existing_bet_id','?')}")
            no_conflict.append(bet)
    approved = no_conflict

    if not approved:
        return []

    proc_map = {c["bet_id"]: (c, mp) for c, _, _, mp in processed}

    results = []
    for bet in approved:
        c, model_prob = proc_map[bet.bet_id]
        side = bet.direction
        line = bet.sportsbook_line

        team_raw = c.get("team", "")
        team_canon = canonical_team(team_raw, sport) if team_raw else None
        team_name = team_canon["full"] if team_canon else team_raw
        player_raw = c.get("player")
        player_canon = canonical_player(player_raw) if player_raw else None
        player_name = player_canon["display"] if player_canon else None

        pick_text = (
            f"{player_name or team_name} {bet.market} {side} {line}"
        )

        pick = {
            # BUGFIX: this used to read
            #   "MLB F5" if (...) else f"{sport} Totals" if not player_name else sport
            # The middle branch fired for EVERY team-level market (moneyline,
            # run_line, game_total -- anything without a player_name), mislabeling
            # them "MLB Totals" / "WNBA Totals" instead of "MLB" / "WNBA". Both
            # is_publication_eligible() (below, keyed on plain "MLB"/"WNBA" in
            # core/market_governance.py's PUBLICATION_MARKETS) and
            # _apply_daily_caps()'s sport_to_cfg lookup only recognize the real
            # sport name, so every one of those picks silently failed the
            # whitelist check after already passing the gatekeeper -- confirmed
            # in output/run_log.json: "dropped 1 pick(s) not in publication
            # whitelist: [('WNBA Totals', 'moneyline')]".
            "sport": "MLB F5" if (sport == "MLB" and "first_5" in bet.market) else sport,
            # Needed downstream by pre_publish_verify() (core/line_validator.py)
            # to map each surviving pick dict back to the BetDisplay it came
            # from -- picks lose their BetDisplay wrapper after CCS scoring,
            # so bet_id is the only handle left to reconnect the two.
            "bet_id": bet.bet_id,
            "player": player_name,
            "team": team_name,
            "matchup": c.get("matchup", c.get("game_id", "")),
            # away_team/home_team: needed downstream by the line_movement
            # live-current-line lookup (see run_pipeline()'s call site for
            # apply_line_movement_filter) -- previously absent from this
            # dict entirely, so that filter had no way to key into a
            # fresh odds fetch even if one existed.
            "away_team": c.get("away_team", ""),
            "home_team": c.get("home_team", ""),
            "market": bet.market,
            "market_type": bet.market,
            "side": side,
            "pick": pick_text,
            "pick_time_line": line,
            "pick_time_odds": bet.american_odds,
            "current_line": line,
            "current_odds": bet.american_odds,
            "line": line,
            "model_number": model_prob,
            "edge_pct": bet.edge_percentage,
            "confidence": bet.confidence_score,
            "model_prob": model_prob,
            "tier": bet.tier.value if bet.tier else None,
            "game_id": c.get("game_id", ""),
            "steam_move_threshold_pct": (MLB if sport == "MLB" else WNBA)["steam_move_threshold_pct"],
            "moneyline_steam_cents": (MLB if sport == "MLB" else WNBA)["moneyline_steam_cents"],
            # Previously never set on this dict at all -- append_picks() has
            # always read side_agreement_frac via p.get(...), so every
            # published pick silently logged null here, and posterior_std/
            # mean were never logged on accepted picks anywhere (only on
            # stability-rejected candidates, which never become bets).
            "side_agreement_frac": c.get("side_agreement_frac"),
            "posterior_std": c.get("posterior_std"),
            "posterior_mean": c.get("posterior_mean"),
            "relative_sigma_pct": c.get("relative_sigma_pct"),
        }

        # ── Stake sizing (fractional Kelly) ──────────────────────────────
        # Previously never computed anywhere in this pipeline -- every pick
        # dict was missing "stake_pct_bankroll" entirely, which grade()/
        # settle() silently default to a flat 1% of bankroll. That means
        # every published pick was being flat-staked regardless of edge or
        # confidence, with no error or warning anywhere in the run.
        # kelly_stake() and per-sport kelly_fraction already existed in
        # handicapper_rules.py / sport_config.py -- they just weren't wired
        # in. Deliberately keyed off model_prob (not edge_pct), so stake
        # size tracks the calibrated win-probability estimate rather than
        # the raw/calibrated edge number -- backtesting showed edge
        # magnitude does NOT track actual win rate monotonically in this
        # engine's history, so it should never drive bet size.
        _kelly_fraction = (MLB if sport == "MLB" else WNBA)["kelly_fraction"]
        _stake_frac = kelly_stake(
            model_prob=(model_prob / 100.0) if model_prob is not None else 0.0,
            american_odds=bet.american_odds,
            kelly_fraction=_kelly_fraction,
        )
        pick["stake_pct_bankroll"] = round(_stake_frac * 100, 2)

        bd = BetDisplay(
            bet=bet,
            american_odds=bet.american_odds,
            model_probability=model_prob,
            supporting_factor="",
            away_team=c.get("away_team", ""),
            home_team=c.get("home_team", ""),
        )
        log_dict = {"wager_details": {}}

        log_candidate(
            sport=sport, player=player_name, matchup=pick["matchup"],
            market_line=line, side=side, model_prob=model_prob / 100.0 if model_prob is not None else None,
            edge_pct=bet.edge_percentage, confidence=bet.confidence_score,
            side_agreement_frac=pick["side_agreement_frac"],
            rejected_stage=None, rejected_reason=None, published=True,
            extra={
                "bet_id": bet.bet_id,
                "posterior_std": pick["posterior_std"],
                "posterior_mean": pick["posterior_mean"],
                "relative_sigma_pct": pick["relative_sigma_pct"],
            },
        )

        results.append((pick, bd, log_dict))

    return results


def _apply_daily_caps(picks):
    """Caps total published picks per sport per day (models.sport_config -> max_picks_per_day)."""
    sport_to_cfg = {"MLB": MLB, "MLB F5": MLB, "WNBA": WNBA}
    by_cfg_group = {}
    for p in picks:
        cfg = sport_to_cfg.get(p.get("sport"))
        group_key = "MLB" if cfg is MLB else "WNBA" if cfg is WNBA else "OTHER"
        by_cfg_group.setdefault(group_key, []).append(p)

    capped = []
    for group_key, group_picks in by_cfg_group.items():
        cap = MLB["max_picks_per_day"] if group_key == "MLB" else (
            WNBA["max_picks_per_day"] if group_key == "WNBA" else len(group_picks)
        )
        group_picks.sort(key=lambda p: abs(p.get("edge_pct", 0)), reverse=True)
        kept = group_picks[:cap]
        dropped_n = len(group_picks) - len(kept)
        if dropped_n > 0:
            print(f"  [{group_key}] capped at {cap}/day -- dropped {dropped_n} lower-edge pick(s)")
        capped.extend(kept)
    return capped


def run_pipeline():
    init_db()
    preflight_ok = run_preflight_checks()
    if not preflight_ok:
        log("error", "pipeline", "Preflight failed -- aborting before any live calls.")
        all_results = []
    else:
        all_results = []
        for sport in ENABLED_SPORTS:
            log("info", "pipeline", f"=== Running {sport} ===")
            try:
                sport_results = run_sport_pipeline(sport)
                all_results.extend(sport_results)
            except Exception as exc:
                log("error", sport, f"sport pipeline crashed: {type(exc).__name__}: {exc}")

    # ── Global CCS ranking across all sports (ported from main.py's
    # _apply_global_tier_cap, simplified: rank everyone by CCS, top score
    # gets Nuke, runner-up gets Diamond, the rest fall to Gold Standard.
    # The original's "dominant filter group" logic is not reproduced here --
    # this is a straightforward global ranking, not a per-market-group one.
    raw_picks = []
    # Keep a bet_id -> BetDisplay handle alive past this point -- raw_picks
    # below only keeps the plain pick dict, but pre_publish_verify() (the
    # final gate right before output, added further down) needs the actual
    # BetDisplay/Bet objects to re-fetch and compare live lines.
    bd_by_bet_id: dict[str, "BetDisplay"] = {
        bd.bet.bet_id: bd for _pick, bd, _ld in all_results
    }
    if all_results:
        scored = []
        for pick, bd, ld in all_results:
            try:
                ccs, robustness = compute_ccs(bd, ld)
            except Exception as exc:
                log("warn", "pipeline", f"{pick.get('pick','?')}: CCS scoring failed ({exc}) -- using edge*0.6+conf*0.4 fallback")
                ccs = pick["edge_pct"] * 0.6 + pick["confidence"] * 0.4
                robustness = "unknown"
            scored.append((pick, ccs, robustness))

        scored.sort(key=lambda t: t[1], reverse=True)
        nuke_claimed = False
        diamond_claimed = False
        for pick, ccs, robustness in scored:
            pick["ccs_score"] = round(ccs, 2)
            pick["robustness"] = robustness
            if not nuke_claimed:
                pick["tier"] = "Nuke"
                nuke_claimed = True
            elif not diamond_claimed:
                pick["tier"] = "Diamond"
                diamond_claimed = True
            else:
                pick["tier"] = "Gold Standard"
            raw_picks.append(pick)

        log("info", "pipeline", f"CCS ranking: {len(raw_picks)} pick(s) scored -- top assigned Nuke, runner-up Diamond, rest Gold Standard.")

    log("info", "pipeline", f"{len(raw_picks)} raw picks generated. Running contradiction check...")
    cleaned, dropped_contradiction = filter_contradictions(raw_picks)
    if dropped_contradiction:
        log("warn", "pipeline", f"dropped {len(dropped_contradiction)} pick(s) on contradiction check")

    # ── Live line-movement refresh (core/intelligence/line_movement.py) ────
    # models/line_movement.py's apply_line_movement_filter() has always been
    # a documented no-op: current_line is set identically to pick_time_line
    # at pick-generation time (see the pick dict above) and nothing upstream
    # ever overwrote it -- its own docstring calls this out as "the hook
    # point... not a live-odds fetcher itself." core/intelligence/
    # line_movement.py IS that fetcher; it was fully built (API-key
    # rotation, 5-min in-process cache, steam/RLM bucket classification) but
    # never called from anywhere in the pipeline either.
    #
    # Wired here narrowly and defensively:
    #  - Off by default (ENABLE_LIVE_LINE_CHECK env var) because it makes
    #    one extra live Odds API call per sport per run outside the normal
    #    slate cache -- a real, if small (2 calls/run for MLB+WNBA), credit
    #    cost the person should opt into deliberately rather than discover
    #    on their bill.
    #  - Only used to feed the *existing* protective drop-filter (movement
    #    against the pick since generation -> drop it), not to add positive
    #    edge from the richer confirming/opposing signal the module also
    #    computes. That signal (+2.5/-3.0 style edge deltas) touches live
    #    edge/staking math the same way rivalry_intel/stat_model do, and
    #    deserves its own reviewed pass rather than being folded in here.
    #  - Game-total picks only (side is "over"/"under"), matching
    #    line_movement.py's own scope -- it fetches a "totals" market
    #    snapshot, so applying it to a moneyline/spread pick's line would
    #    compare unrelated numbers.
    if os.getenv("ENABLE_LIVE_LINE_CHECK", "").strip().lower() in ("1", "true", "yes"):
        try:
            from core.intelligence.line_movement import fetch_current_lines
            _live_lines = {sp: fetch_current_lines(sp) for sp in ("MLB", "WNBA")}
            _live_updates = 0
            for p in cleaned:
                if p.get("player") or str(p.get("side", "")).lower() not in ("over", "under"):
                    continue
                _sp_key = "MLB" if str(p.get("sport", "")).upper().startswith("MLB") else "WNBA"
                _game_key = f'{p.get("away_team", "")}||{p.get("home_team", "")}'
                _cur_game = _live_lines.get(_sp_key, {}).get(_game_key)
                if not _cur_game:
                    continue
                _cur_line = (
                    _cur_game["over_line"] if p["side"].lower() == "over"
                    else _cur_game["under_line"]
                )
                p["current_line"] = _cur_line
                _live_updates += 1
            log("info", "pipeline", f"live line check: refreshed current_line for {_live_updates}/{len(cleaned)} pick(s)")
        except Exception as exc:
            log("warn", "pipeline", f"live line movement refresh failed (non-fatal, filter runs as no-op): {exc}")

    log("info", "pipeline", f"{len(cleaned)} picks after contradiction check. Running line movement check...")
    final, dropped_line_move = apply_line_movement_filter(cleaned)
    if dropped_line_move:
        log("warn", "pipeline", f"dropped {len(dropped_line_move)} pick(s) on line movement check")

    log("info", "pipeline", f"{len(final)} picks after line movement check. Applying per-sport daily caps...")
    final = _apply_daily_caps(final)
    log("info", "pipeline", f"{len(final)} picks after daily caps.")

    # ── Publication whitelist (core/market_governance.py) ────────────────
    # Separate from market_gate's simulation-scope filter above: a market can
    # be modeled but still not be approved for public output. Runs last, on
    # the final picks list, so it's the last thing that can drop a pick
    # before it's written anywhere.
    _pub_eligible, _pub_blocked = [], []
    for _p in final:
        (_pub_eligible if is_publication_eligible(_p.get("sport", ""), _p.get("market", "")) else _pub_blocked).append(_p)
    if _pub_blocked:
        log("warn", "pipeline", f"dropped {len(_pub_blocked)} pick(s) not in publication whitelist: "
                                 f"{[(p.get('sport'), p.get('market')) for p in _pub_blocked]}")
    final = _pub_eligible
    log("info", "pipeline", f"{len(final)} picks after publication whitelist.")

    # ── Pre-publish line verification (core/line_validator.py) ───────────
    # Independent final safety net: re-fetches live sportsbook lines for
    # every surviving pick right before publish and drops any pick whose
    # line has drifted beyond threshold since analysis. Deliberately the
    # LAST gate before output -- catches a stale in-memory line (the exact
    # class of bug a mismatched run_line was) one layer later than every
    # earlier filter, independent of all of them.
    if final:
        _bets_by_sport: dict[str, list[BetDisplay]] = {}
        _unmapped = 0
        for _p in final:
            _bd = bd_by_bet_id.get(_p.get("bet_id"))
            if _bd is None:
                # Shouldn't happen -- every pick this pipeline builds is
                # stamped with bet_id above. Fail open (pass the pick
                # through unverified) rather than silently dropping
                # something we have no way to re-check.
                _unmapped += 1
                continue
            _bets_by_sport.setdefault(_p.get("sport", ""), []).append(_bd)

        if _unmapped:
            log("warn", "pipeline", f"{_unmapped} pick(s) missing a bet_id -> BetDisplay "
                                     "mapping -- passing through pre-publish verification unverified.")

        _skip_verify = os.getenv("SKIP_PRE_PUBLISH_VERIFY", "").strip().lower() in ("1", "true", "yes")
        if _skip_verify:
            log("warn", "pipeline", "SKIP_PRE_PUBLISH_VERIFY set -- running in dry-run mode, "
                                     "no picks will be dropped by this gate.")

        try:
            _cleaned_bd, _removed = pre_publish_verify(_bets_by_sport, dry_run=_skip_verify)
            if _removed:
                _dropped_bet_ids = {r.get("bet_id") for r in _removed}
                for _r in _removed:
                    log("warn", "pipeline", f"pre_publish_verify DROPPED "
                                             f"{_r.get('player') or _r.get('team')} {_r.get('market')} "
                                             f"{_r.get('direction')}: {_r.get('reason')}")
                    _rem_bd = bd_by_bet_id.get(_r.get("bet_id"))
                    if _rem_bd is not None:
                        log_rejected_bet_obj(
                            _rem_bd.bet, _r.get("sport", ""), TODAY,
                            "pre_publish_verify", reason_override=_r.get("reason"),
                        )
                final = [p for p in final if p.get("bet_id") not in _dropped_bet_ids]
            log("info", "pipeline", f"{len(final)} picks after pre-publish line verification "
                                     f"({len(_removed)} dropped).")
        except Exception as exc:
            log("error", "pipeline", f"pre_publish_verify crashed (non-fatal -- all picks "
                                      f"pass through unverified this run): {type(exc).__name__}: {exc}")

    import uuid as _uuid
    for _p in final:
        _p.setdefault("pick_id", _uuid.uuid4().hex[:12])
        _p.setdefault("actual_result", None)
        _p.setdefault("closing_line", None)
        _p.setdefault("clv_pct", None)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "picks": final,
    }

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "picks.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    if final:
        appended = append_picks(final, generated_at=output["generated_at"])
        log("info", "pipeline", f"appended {len(appended)} pick(s) to output/pick_history.jsonl "
                                 f"({len(final) - len(appended)} skipped as same-day duplicates)")
        for _p in final:
            try:
                log_bet_dict(
                    bet_id=_p["pick_id"],
                    sport=_p["sport"],
                    wager_details={
                        "market": _p.get("market", ""),
                        "direction": _p.get("side", ""),
                        "game_id": _p.get("game_id", ""),
                    },
                    model_probability=_p.get("model_prob", 0.0),
                    sportsbook_odds=int(_p.get("pick_time_odds", 0) or 0),
                    tier=_p.get("tier"),
                    edge_percentage=_p.get("edge_pct", 0.0),
                )
            except Exception as exc:
                log("warn", "pipeline", f"{_p.get('pick_id','?')}: failed to lock pick in results.db (non-fatal): {exc}")
        try:
            mark_picks_published(list({p["game_id"] for p in final if p.get("game_id")}))
        except Exception as exc:
            log("warn", "pipeline", f"mark_picks_published failed (non-fatal): {exc}")

    log_path = os.path.join(out_dir, "run_log.json")
    n_errors = sum(1 for e in RUN_LOG if e["level"] == "error")
    n_warnings = sum(1 for e in RUN_LOG if e["level"] == "warn")
    with open(log_path, "w") as f:
        json.dump({
            "run_at": datetime.now(timezone.utc).isoformat(),
            "n_errors": n_errors, "n_warnings": n_warnings,
            "n_picks_generated": len(final),
            "entries": RUN_LOG,
        }, f, indent=2)

    print(f"\nWrote {len(final)} final picks to {out_path}")
    print(f"Wrote run log ({n_errors} errors, {n_warnings} warnings) to {log_path}")
    if n_errors > 0:
        print("\n*** This run had errors -- paste output/run_log.json back for debugging. ***")
    return output


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as fatal:
        log("error", "fatal", f"uncaught exception: {type(fatal).__name__}: {fatal}")
        out_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run_log.json"), "w") as f:
            json.dump({
                "run_at": datetime.now(timezone.utc).isoformat(),
                "fatal_error": True,
                "entries": RUN_LOG,
            }, f, indent=2)
        raise
