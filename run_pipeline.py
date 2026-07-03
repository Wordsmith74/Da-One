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
from core.bet_display import BetDisplay
from core.composite_confidence_score import compute_ccs
from core.conflict_guardian import check_locked_conflict
from core.results_tracker import init_db, log_bet_dict
from core.edge_calibrator import is_game_market, calibrate_edge
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
    candidates (injury, pace, rotation, market agreement, rest/travel).

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

    # --- 5. Market agreement score ------------------------------------------
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

    if odds < 0:
        implied = abs(odds) / (abs(odds) + 100) * 100
    else:
        implied = 100 / (odds + 100) * 100

    raw_edge = round(min(50.0, model_prob - implied), 2)

    sport_key = sim.get("sport_type", "default")
    mkt = market_normalized(candidate.get("market", ""))

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

def run_sport_pipeline(sport):
    """
    Full prediction pipeline for one sport. Returns a list of pick dicts in
    this repo's existing schema (compatible with filter_contradictions,
    apply_line_movement_filter, and _apply_daily_caps below).
    """
    try:
        orchestrator = DecisionOrchestrator(sport)
    except UnsupportedSportError as exc:
        log("warn", sport, f"Unsupported sport -- {exc}")
        return []

    engine = SimulationEngine(orchestrator)

    try:
        candidates = fetch_todays_candidates(sport)
    except Exception as exc:
        log("error", sport, f"odds_client fetch failed ({type(exc).__name__}: {exc})")
        return []

    try:
        prop_candidates = get_player_prop_candidates(sport)
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
        expanded_candidates = fetch_expanded_game_candidates(sport)
        if expanded_candidates:
            candidates = candidates + expanded_candidates
            log("info", sport, f"{len(expanded_candidates)} expanded game market candidate(s) added.")
    except Exception as exc:
        log("error", sport, f"game_markets fetch failed ({type(exc).__name__}: {exc}) -- continuing without expanded markets")

    if not candidates:
        log("info", sport, "No candidates returned for today.")
        return []

    log("info", sport, f"{len(candidates)} total candidate(s) (game totals + player props).")

    processed = []
    for c in candidates:
        bet_id = c.get("bet_id", "?")

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
            _enrich_integrity_fields(c, sport)
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
            continue

        post_std = sim["posterior"].get("posterior_std", 0.0)
        post_mean = sim["posterior"].get("posterior_mean", 0.0)
        is_stable, stab_reason = check_stability(sport, post_std, post_mean)
        if not is_stable:
            log("info", sport, f"{bet_id}: STABILITY REJECT -- {stab_reason}")
            continue

        try:
            edge, confidence, model_prob = _derive_bet_params(sim, c)
        except Exception as exc:
            log("error", sport, f"{bet_id}: _derive_bet_params failed -- {exc}")
            continue

        _enrich_integrity_fields(c, sport)

        processed.append((c, edge, confidence, model_prob))

    if not processed:
        log("info", sport, "All candidates failed simulation -- skipping.")
        return []

    # Game Truth Protocol -- one Value Vector per game.
    processed = apply_game_truth_protocol(processed, sport)
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
            action, details = check_locked_conflict(bet, sport, date_str=TODAY)
        except Exception as exc:
            log("warn", sport, f"{bet.bet_id}: conflict_guardian check failed (non-fatal): {exc}")
            no_conflict.append(bet)
            continue
        if action == "hold":
            log("info", sport, f"{bet.bet_id}: CONFLICT HOLD -- existing locked pick "
                                f"{details.get('existing_bet_id','?')} stands (replacement threshold not met)")
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
            "sport": "MLB F5" if (sport == "MLB" and "first_5" in bet.market) else f"{sport} Totals" if not player_name else sport,
            "player": player_name,
            "team": team_name,
            "matchup": c.get("matchup", c.get("game_id", "")),
            "market": bet.market,
            "market_type": bet.market,
            "side": side,
            "pick": pick_text,
            "pick_time_line": line,
            "pick_time_odds": bet.american_odds,
            "current_line": line,
            "current_odds": bet.american_odds,
            "edge_pct": bet.edge_percentage,
            "confidence": bet.confidence_score,
            "model_prob": model_prob,
            "tier": bet.tier.value if bet.tier else None,
            "game_id": c.get("game_id", ""),
            "steam_move_threshold_pct": (MLB if sport == "MLB" else WNBA)["steam_move_threshold_pct"],
            "moneyline_steam_cents": (MLB if sport == "MLB" else WNBA)["moneyline_steam_cents"],
        }

        bd = BetDisplay(
            bet=bet,
            american_odds=bet.american_odds,
            model_probability=model_prob,
            supporting_factor="",
            away_team=c.get("away_team", ""),
            home_team=c.get("home_team", ""),
        )
        log_dict = {"wager_details": {}}

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

    log("info", "pipeline", f"{len(cleaned)} picks after contradiction check. Running line movement check...")
    final, dropped_line_move = apply_line_movement_filter(cleaned)
    if dropped_line_move:
        log("warn", "pipeline", f"dropped {len(dropped_line_move)} pick(s) on line movement check")

    log("info", "pipeline", f"{len(final)} picks after line movement check. Applying per-sport daily caps...")
    final = _apply_daily_caps(final)
    log("info", "pipeline", f"{len(final)} picks after daily caps.")

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
        append_picks(final, generated_at=output["generated_at"])
        log("info", "pipeline", f"appended {len(final)} pick(s) to output/pick_history.jsonl")
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
