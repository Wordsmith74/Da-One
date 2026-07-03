"""
historical_grader.py -- Main grading engine.

Pipeline:
  1. Read output/pick_history.jsonl
  2. Skip picks that already have actual_result
  3. Group ungraded picks by sport + game date (parsed from `game_id`)
  4. Fetch official results (MLB Stats API / WNBA stats endpoints)
  5. Grade each pick: win / loss / push
  6. Record actual_result, actual_stat, closing_line, closing_odds,
     graded_at, profit_units, roi
  7. Write updated records back to pick_history.jsonl
  8. Generate summary reports (overall, by market/confidence/edge/tier)

Schema: see grading_utils.py docstring -- fields match the `Bet` dataclass
in core/decision_gatekeeper.py (bet_id, team, market, direction,
sportsbook_line, edge_percentage, confidence_score, player, game_id,
american_odds, tier, ...).

game_id format: "AWAY@HOME_SPORT_YYYY-MM-DD" (e.g. "PHX@MIN_WNBA_2026-06-01").
We parse sport/date/teams directly from it instead of requiring separate
fields on the pick record.

Run as a script:
    python -m core.historical_grader --pick-history output/pick_history.jsonl
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from typing import Any, Callable, Optional

from .grading_utils import log_reject, read_jsonl, utcnow_iso, write_jsonl
from .grading import assists, moneyline, rebounds, strikeouts, totals
from .reports.calibration import generate_summary, print_summary
from .stat_fetchers import mlb, wnba

logger = logging.getLogger("historical_grader")

DEFAULT_PICK_HISTORY_PATH = "output/pick_history.jsonl"
DEFAULT_REJECT_LOG_PATH = "output/grading_rejects.jsonl"

# Market dispatch table. Keys are the *normalized* market string --
# reuse decision_gatekeeper.market_normalized() at call sites so aliases
# ("assists" -> "player_assists", "strikeouts" -> "pitcher_strikeouts", etc)
# land here correctly. TODO: import market_normalized directly from
# core.decision_gatekeeper once that module is importable in this
# environment, rather than duplicating alias logic.
_ALIAS_MAP = {
    "assists": "player_assists",
    "rebounds": "player_rebounds",
    "strikeouts": "pitcher_strikeouts",
    "player_strikeouts": "pitcher_strikeouts",
}

_MARKET_GRADERS: dict[str, Any] = {
    "pitcher_strikeouts": strikeouts,
    "player_rebounds": rebounds,
    "player_assists": assists,
    "totals": totals,
    "first_5_total": totals,
    "moneyline": moneyline,
    "h2h": moneyline,
}


def market_normalized(market: str) -> str:
    norm = market.strip().lower().replace(" ", "_")
    return _ALIAS_MAP.get(norm, norm)


def parse_game_id(game_id: str) -> Optional[dict[str, str]]:
    """
    Parse "AWAY@HOME_SPORT_YYYY-MM-DD" into components.
    Returns None if the format doesn't match (logged, not raised, so one
    malformed game_id doesn't kill the whole grading run).
    """
    try:
        teams_part, sport, game_date = game_id.rsplit("_", 2)
        # game_date and sport were split off the end; teams_part may still
        # contain more underscores for multi-word team names, so re-check.
        # Expected: "AWAY@HOME"
        away, home = teams_part.split("@")
        return {"away": away, "home": home, "sport": sport, "game_date": game_date}
    except (ValueError, AttributeError):
        logger.warning("game_id_parse_failed", extra={"game_id": game_id})
        return None


def fetch_actual_stat(pick: dict, parsed_game: dict) -> tuple[Optional[float], Optional[str]]:
    """
    Returns (actual_stat_or_winner, error_reason).
    Dispatches to the right stat_fetchers module by sport + market.
    """
    sport = parsed_game["sport"].upper()
    market = market_normalized(pick["market"])
    game_date = parsed_game["game_date"]
    team = pick.get("team") or parsed_game["home"]
    player = pick.get("player")

    try:
        if sport == "MLB":
            game_pk = mlb.find_game_pk(team, game_date)
            if game_pk is None:
                return None, "mlb_game_not_found"
            if market == "pitcher_strikeouts":
                if not player:
                    return None, "missing_player_for_prop"
                return mlb.get_pitcher_strikeouts(game_pk, player), None
            elif market == "totals":
                return mlb.get_game_total_runs(game_pk), None
            elif market == "first_5_total":
                return mlb.get_f5_total_runs(game_pk), None
            elif market in ("moneyline", "h2h"):
                return mlb.get_moneyline_winner(game_pk), None
            else:
                return None, f"unsupported_market:{market}"

        elif sport == "WNBA":
            game_id_wnba = wnba.find_game_id(team, game_date)
            if game_id_wnba is None:
                return None, "wnba_game_not_found"
            if market == "player_rebounds":
                if not player:
                    return None, "missing_player_for_prop"
                return wnba.get_player_rebounds(game_id_wnba, player), None
            elif market == "player_assists":
                if not player:
                    return None, "missing_player_for_prop"
                return wnba.get_player_assists(game_id_wnba, player), None
            elif market == "totals":
                return wnba.get_game_total_points(game_id_wnba), None
            elif market in ("moneyline", "h2h"):
                return wnba.get_moneyline_winner(game_id_wnba), None
            else:
                return None, f"unsupported_market:{market}"
        else:
            return None, f"unsupported_sport:{sport}"
    except Exception as e:  # noqa: BLE001 -- grading must degrade gracefully, never crash the run
        logger.warning(
            "stat_fetch_exception",
            extra={"bet_id": pick.get("bet_id"), "error": str(e)},
        )
        return None, f"exception:{e}"


def grade_pick(pick: dict, reject_log_path: str) -> Optional[dict]:
    """
    Returns the updated pick record with grading fields populated, or None
    if it couldn't be graded this run (left ungraded for a future pass).
    """
    game_id = pick.get("game_id", "")
    parsed_game = parse_game_id(game_id)
    if parsed_game is None:
        log_reject(
            reject_log_path,
            {"bet_id": pick.get("bet_id"), "reason": "unparseable_game_id", "game_id": game_id},
        )
        return None

    market = market_normalized(pick.get("market", ""))
    grader_module = _MARKET_GRADERS.get(market)
    if grader_module is None:
        log_reject(
            reject_log_path,
            {"bet_id": pick.get("bet_id"), "reason": "no_grader_for_market", "market": market},
        )
        return None

    actual_stat_or_winner, error_reason = fetch_actual_stat(pick, parsed_game)
    if error_reason is not None:
        log_reject(
            reject_log_path,
            {
                "bet_id": pick.get("bet_id"),
                "reason": error_reason,
                "market": market,
                "game_id": game_id,
            },
        )
        return None

    outcome = grader_module.grade(pick, actual_stat_or_winner)
    if outcome is None:
        # e.g. player DNP, game postponed -- not gradeable this run, no error
        return None

    updated = dict(pick)
    updated["actual_result"] = outcome.actual_result
    updated["actual_stat"] = outcome.actual_stat
    updated["profit_units"] = outcome.profit_units
    updated["roi"] = outcome.roi
    updated["graded_at"] = utcnow_iso()
    # TODO: closing_line / closing_odds require a snapshot of the market at
    # game start, which isn't available from either stat_fetcher (those
    # only return final results, not pregame closing odds). If your
    # pipeline logs closing lines elsewhere (e.g. a separate closing-odds
    # snapshot job), merge them in here. Left as None until that's wired up.
    updated["closing_line"] = pick.get("closing_line")
    updated["closing_odds"] = pick.get("closing_odds")
    return updated


def run(pick_history_path: str = DEFAULT_PICK_HISTORY_PATH,
        reject_log_path: str = DEFAULT_REJECT_LOG_PATH) -> dict:
    all_picks = read_jsonl(pick_history_path)

    ungraded_idx = [
        i for i, p in enumerate(all_picks) if not p.get("actual_result")
    ]
    logger.info(
        "grading_run_start",
        extra={"total_picks": len(all_picks), "ungraded": len(ungraded_idx)},
    )

    # Group ungraded picks by (sport, game_date) purely for logging /
    # potential future batching of fetcher calls per date -- grading itself
    # still proceeds pick-by-pick below.
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in ungraded_idx:
        parsed = parse_game_id(all_picks[i].get("game_id", ""))
        key = (parsed["sport"], parsed["game_date"]) if parsed else ("unknown", "unknown")
        groups[key].append(i)

    graded_count = 0
    for (sport, game_date), idxs in groups.items():
        logger.info("grading_group", extra={"sport": sport, "game_date": game_date, "n": len(idxs)})
        for i in idxs:
            updated = grade_pick(all_picks[i], reject_log_path)
            if updated is not None:
                all_picks[i] = updated
                graded_count += 1

    write_jsonl(pick_history_path, all_picks)
    logger.info("grading_run_complete", extra={"graded_this_run": graded_count})

    summary = generate_summary(all_picks)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Grade completed picks against official results.")
    parser.add_argument("--pick-history", default=DEFAULT_PICK_HISTORY_PATH)
    parser.add_argument("--reject-log", default=DEFAULT_REJECT_LOG_PATH)
    args = parser.parse_args()

    summary = run(args.pick_history, args.reject_log)
    print_summary(summary)


if __name__ == "__main__":
    main()
