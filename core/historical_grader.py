"""
historical_grader.py -- Main grading engine.

Pipeline:
  1. Read output/pick_history.jsonl
  2. Skip picks that already have actual_result
  3. Group ungraded picks by sport + game date (parsed from `matchup`)
  4. Fetch official results (MLB Stats API / WNBA stats endpoints)
  5. Grade each pick: win / loss / push
  6. Record actual_result, actual_stat, closing_line, closing_odds,
     graded_at, profit_units, roi
  7. Write updated records back to pick_history.jsonl
  8. Generate summary reports (overall, by market/confidence/edge/tier)

Schema: as actually persisted to pick_history.jsonl (NOT the old Bet
dataclass field names -- those never made it to disk). Real fields:
pick_id, matchup, side, pick_time_line, pick_time_odds, edge_pct,
confidence, stake_pct_bankroll, player, sport, market, generated_at.
No game_id, no tier, no bet_id, no game_date field exists on the record.

`matchup` has taken three different shapes over time in the data, all of
which we parse defensively (see parse_matchup):
  - "Texas Rangers @ Cleveland Guardians"           (full names, spaced)
  - "Detroit_Tigers@New_York_Yankees_MLB"            (underscored + sport)
  - "SD@CHC_MLB_2026-07-01"                          (abbrev + sport + date)
`game_date` is not a stored field: we take it from a trailing
_YYYY-MM-DD on `matchup` when present, else fall back to the date portion
of `generated_at`. `sport` also has drifted ("MLB Ks" / "MLB Totals" /
"MLB") -- we only ever take the first whitespace-delimited token.

Run as a script:
    python -m core.historical_grader --pick-history output/pick_history.jsonl
"""
from __future__ import annotations

import argparse
import logging
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .grading_utils import log_reject, market_normalized, read_jsonl, utcnow_iso, write_jsonl
from . import assists, moneyline, nrfi, rebounds, strikeouts, totals
from .calibration import generate_summary, print_summary
from . import mlb, wnba
from .time_utils import convert_to_est

logger = logging.getLogger("historical_grader")

DEFAULT_PICK_HISTORY_PATH = "output/pick_history.jsonl"
DEFAULT_REJECT_LOG_PATH = "output/grading_rejects.jsonl"

# Same results.db path convention as core/slate_versioner.py and
# core/performance_tracker.py -- this is the single shared SQLite DB.
DB_PATH = Path(__file__).parent.parent / "data" / "results.db"


def _close_bet_in_db(bet_id: str, actual_result: str, profit_loss: float) -> bool:
    """
    Mirror a grading result into the `bets` SQL table (results.db), if a
    matching row exists there.

    Why this exists: run_pipeline.py opens a `bets` row for every published
    pick (log_bet_dict(bet_id=pick["pick_id"], ...), status='open') but
    nothing was ever closing it -- score_grader.py / prop_grader.py /
    core/results_tracker.py can all close a `bets` row, but none of them
    are called by anything that runs. Meanwhile this module (the grading
    engine CI actually runs) grades a completely separate store
    (pick_history.jsonl) and never touched `bets` at all, so every
    downstream `bets`-table consumer (core/performance_tracker.py,
    core/slate_versioner.py) always saw zero closed bets.

    This closes that loop from the one grading engine that's actually live,
    rather than wiring in a second, independent grading path that could
    disagree with this one. Best-effort: if `bet_id` has no matching row
    (e.g. picks graded from before this was wired in, or the row was never
    opened), this is a no-op, not an error -- pick_history.jsonl remains the
    source of truth for grading either way.
    """
    if not bet_id:
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            cur = conn.execute(
                """
                UPDATE bets
                SET actual_outcome = ?,
                    profit_loss    = ?,
                    status         = 'closed'
                WHERE bet_id = ? AND status != 'closed'
                """,
                (actual_result, profit_loss, bet_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("bets_table_close_failed", extra={"bet_id": bet_id, "error": str(exc)})
        return False

# Market dispatch table. Keys are the *normalized* market string, produced
# by grading_utils.market_normalized() -- the same function calibration.py
# uses for ROI-by-market reporting, so dispatch and reporting can't drift
# apart into separate buckets for the same market again.
_MARKET_GRADERS: dict[str, Any] = {
    "pitcher_strikeouts": strikeouts,
    "player_rebounds": rebounds,
    "player_assists": assists,
    "totals": totals,
    "first_5_total": totals,
    "moneyline": moneyline,
    "h2h": moneyline,
    "nrfi": nrfi,
    "yrfi": nrfi,
}


_DATE_SUFFIX_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})$")


def parse_matchup(pick: dict) -> Optional[dict[str, str]]:
    """
    Parse the real `matchup` field into components, defensively handling all
    three shapes seen in pick_history.jsonl:
      - "Texas Rangers @ Cleveland Guardians"   (full names, spaced, no
        sport/date embedded -- sport comes from the `sport` field)
      - "Detroit_Tigers@New_York_Yankees_MLB"   (underscored names + a
        trailing "_SPORT" suffix, no date)
      - "SD@CHC_MLB_2026-07-01"                 (abbreviated codes + a
        trailing "_SPORT_YYYY-MM-DD" suffix)

    `sport` is read from the record's own `sport` field, not parsed out of
    `matchup` -- that field has also drifted ("MLB Ks", "MLB Totals",
    "MLB"), so we only take its first whitespace-delimited token.

    game_date: taken from a trailing _YYYY-MM-DD on `matchup` when present
    (most accurate), else falls back to the date portion of `generated_at`.
    There is no dedicated game_date field on the record.

    Returns None if matchup/sport can't be resolved at all (logged, not
    raised, so one bad row doesn't kill the whole grading run).
    """
    matchup = pick.get("matchup", "") or ""
    raw_sport = (pick.get("sport") or "").strip()
    sport = raw_sport.split()[0].upper() if raw_sport else ""

    parts = matchup.split("@", 1)
    if len(parts) != 2:
        logger.warning("matchup_parse_failed", extra={"matchup": matchup})
        return None
    away_raw, home_raw = parts[0].strip(), parts[1].strip()
    if not away_raw or not home_raw:
        logger.warning("matchup_parse_failed", extra={"matchup": matchup})
        return None

    game_date = None
    m = _DATE_SUFFIX_RE.search(home_raw)
    if m:
        game_date = m.group(1)
        home_raw = home_raw[: m.start()]

    # Strip a trailing "_SPORT" suffix off the home side, if present. Only
    # done when we already know `sport` (from the `sport` field) so we
    # don't accidentally chew into a team name.
    if sport and home_raw.upper().endswith("_" + sport):
        home_raw = home_raw[: -(len(sport) + 1)]

    away = away_raw.replace("_", " ").strip()
    home = home_raw.replace("_", " ").strip()

    if game_date is None:
        gen = pick.get("generated_at") or ""
        if gen:
            try:
                # Game slates are ET-based -- a UTC-date slice can land on
                # the wrong calendar day for runs close to the UTC
                # midnight boundary (e.g. 03:00 UTC is still last night in
                # ET). Convert first, per time_utils.py's own rule that all
                # display/scheduling-relevant dates go through EST.
                game_date = convert_to_est(datetime.fromisoformat(gen)).strftime("%Y-%m-%d")
            except ValueError:
                game_date = gen[:10] if len(gen) >= 10 else None

    if not sport or not game_date or not away or not home:
        logger.warning(
            "matchup_parse_failed",
            extra={"matchup": matchup, "sport": raw_sport, "generated_at": pick.get("generated_at")},
        )
        return None

    return {"away": away, "home": home, "sport": sport, "game_date": game_date}


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
            elif market in ("nrfi", "yrfi"):
                return mlb.get_first_inning_runs(game_pk), None
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
            extra={"pick_id": pick.get("pick_id"), "error": str(e)},
        )
        return None, f"exception:{e}"


def grade_pick(pick: dict, reject_log_path: str) -> Optional[dict]:
    """
    Returns the updated pick record with grading fields populated, or None
    if it couldn't be graded this run (left ungraded for a future pass).
    """
    matchup = pick.get("matchup", "")
    parsed_game = parse_matchup(pick)
    if parsed_game is None:
        log_reject(
            reject_log_path,
            {"pick_id": pick.get("pick_id"), "reason": "unparseable_matchup", "matchup": matchup},
        )
        return None

    market = market_normalized(pick.get("market", ""))
    grader_module = _MARKET_GRADERS.get(market)
    if grader_module is None:
        log_reject(
            reject_log_path,
            {"pick_id": pick.get("pick_id"), "reason": "no_grader_for_market", "market": market},
        )
        return None

    actual_stat_or_winner, error_reason = fetch_actual_stat(pick, parsed_game)
    if error_reason is not None:
        log_reject(
            reject_log_path,
            {
                "pick_id": pick.get("pick_id"),
                "reason": error_reason,
                "market": market,
                "matchup": matchup,
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

    # Mirror this result into the `bets` SQL table (see _close_bet_in_db
    # docstring) -- best-effort, never blocks pick_history.jsonl grading.
    bet_id = pick.get("pick_id")
    if outcome.profit_units is not None:
        _close_bet_in_db(bet_id, outcome.actual_result, outcome.profit_units * 100.0)

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
        parsed = parse_matchup(all_picks[i])
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
