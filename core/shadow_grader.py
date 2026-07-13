"""
shadow_grader.py -- Grade REJECTED picks from output/shadow_log.jsonl.

Purpose: historical_grader.py only ever grades picks that were published
(output/pick_history.jsonl). Every candidate the gatekeeper rejected
(edge/confidence too low, stability check failed, etc.) is logged to
output/shadow_log.jsonl with published=false and rejected_stage /
rejected_reason -- but nothing ever grades those. That means we've never
been able to answer "would this rejected pick have won?", which is
exactly the question needed to sanity-check whether a threshold is too
strict, too loose, or about right -- an optimizer than only ever sees
*published* picks can't tell you what it's missing.

This module reuses the SAME grader dispatch (_MARKET_GRADERS) and stat
fetchers (fetch_actual_stat / mlb.py / wnba.py) as historical_grader.py,
so a rejected pick and a published pick are graded identically -- no
second, potentially-diverging grading path.

Schema translation (shadow_log.jsonl -> the shape fetch_actual_stat/
grade_pick expect, which matches pick_history.jsonl's Bet-derived
fields):
    market_line   -> pick_time_line
    (no odds field -- these were never priced/published, so profit_units
     and roi come back None; actual_result is still fully meaningful)
    market        -> parsed from extra.bet_id, since shadow_log.jsonl
                     never stored it directly:
        prop bets:  "prop_{Player_Name}_{market}_{side}_{hash}"
        game bets:  "{TEAM}_{market}_{side}_{hash}"

Only rejected_stage in ("gatekeeper", "gatekeeper_flagged") records are
graded by default -- these are picks the MODEL still generated edge/
confidence numbers for, just below the bar. "stability" rejects (edge_pct/
confidence are null -- see shadow_log.jsonl's own stability-stage rows)
never got that far and have nothing to compare against a threshold.

Run as a script:
    python -m core.shadow_grader --shadow-log output/shadow_log.jsonl
Writes output/shadow_log_graded.jsonl (full records + actual_result) and
prints a summary of win rate by market for graded rejects, so you can see
directly which rejected picks would have won.
"""
from __future__ import annotations

import argparse
import logging
import re
from typing import Any, Optional

from .grading_utils import market_normalized, read_jsonl, write_jsonl, utcnow_iso
from .historical_grader import _MARKET_GRADERS, fetch_actual_stat, parse_matchup

logger = logging.getLogger("shadow_grader")

DEFAULT_SHADOW_LOG_PATH = "output/shadow_log.jsonl"
DEFAULT_OUTPUT_PATH = "output/shadow_log_graded.jsonl"

# Only grade rejects from these stages by default -- see module docstring.
GRADEABLE_STAGES = ("gatekeeper", "gatekeeper_flagged")


_HEX_SUFFIX_RE = re.compile(r"^[0-9a-f]{6,10}$")


def _parse_market_from_bet_id(record: dict) -> Optional[str]:
    """
    shadow_log.jsonl never stored a `market` field directly -- it's only
    recoverable from extra.bet_id. Three shapes seen in the data:
      prop bets:      "prop_{Player_Name_With_Underscores}_{market}_{side}_{hash}"
      game (total):   "{TEAM}_total_{side}_{hash}"              (has a hash)
      game (other):   "{TEAM}_{market}_{side}"                  (NO hash --
                       e.g. "PIT_run_line_home", "DAL_moneyline_away")
    The hash isn't consistently present, so it's detected (pure lowercase
    hex, 6-10 chars) and stripped if found, rather than assumed by a fixed
    token count -- an earlier version of this assumed every game bet_id
    ended in team_market_side_hash, which silently mis-parsed
    "PIT_run_line_home" as market="run" (dropping "_line") since there's
    no hash token to account for.
    """
    bet_id = ((record.get("extra") or {}).get("bet_id")) or ""
    side = record.get("side") or ""
    if not bet_id:
        return None

    if bet_id.startswith("prop_") and record.get("player"):
        player_slug = record["player"].replace(" ", "_")
        prefix = f"prop_{player_slug}_"
        if not bet_id.startswith(prefix):
            return None
        remainder = bet_id[len(prefix):]
        marker = f"_{side}_"
        if marker not in remainder:
            return None
        return remainder.split(marker)[0]

    tokens = bet_id.split("_")
    if len(tokens) < 3:
        return None
    if _HEX_SUFFIX_RE.match(tokens[-1]):
        tokens = tokens[:-1]
    if len(tokens) < 3:
        return None
    # tokens is now [TEAM, ...market_tokens..., side]
    market = "_".join(tokens[1:-1])
    return market or None


def _to_gradeable_pick(record: dict) -> Optional[dict]:
    """
    Translate a shadow_log.jsonl record into the field shape grade_pick()
    (via fetch_actual_stat + the market's grade()) expects. Returns None
    if there isn't enough info to even attempt grading (e.g. market
    couldn't be parsed from bet_id).
    """
    market = _parse_market_from_bet_id(record)
    if not market:
        return None

    return {
        "pick_id": record.get("shadow_id"),
        "matchup": record.get("matchup"),
        "sport": record.get("sport"),
        "market": market,
        "player": record.get("player"),
        "side": record.get("side"),
        "pick_time_line": record.get("market_line"),
        # parse_matchup() falls back to this when `matchup` has no
        # trailing _YYYY-MM-DD suffix (true for the "prop_" bet_id shape,
        # e.g. "St._Louis_Cardinals@Chicago_Cubs_MLB") -- without it every
        # such record fails matchup parsing outright.
        "generated_at": record.get("logged_at"),
        # Rejected picks were never priced -- no pick_time_odds exists.
        # grade() modules already treat odds=None as "return win/loss/push
        # with profit_units=None" (see rebounds.py/assists.py/strikeouts.py),
        # so actual_result is still meaningful even without odds.
        "pick_time_odds": None,
        "stake_pct_bankroll": None,
    }


def grade_shadow_record(record: dict, reject_log_path: str) -> Optional[dict]:
    """
    Returns `record` with actual_result/actual_stat/graded_at added, or
    None if it couldn't be graded (unparseable market, game not found,
    game not final yet, player DNP, etc. -- same "leave for a future
    pass" contract as historical_grader.grade_pick).
    """
    pick = _to_gradeable_pick(record)
    if pick is None:
        return None

    parsed_game = parse_matchup(pick)
    if parsed_game is None:
        return None

    norm_market = market_normalized(pick["market"])
    grader_module = _MARKET_GRADERS.get(norm_market)
    if grader_module is None:
        return None

    actual_stat_or_winner, error_reason = fetch_actual_stat(pick, parsed_game)
    if error_reason is not None:
        return None

    outcome = grader_module.grade(pick, actual_stat_or_winner)
    if outcome is None:
        return None

    updated = dict(record)
    updated["_market"] = norm_market
    updated["actual_result"] = outcome.actual_result
    updated["actual_stat"] = outcome.actual_stat
    updated["graded_at"] = utcnow_iso()
    return updated


def run(shadow_log_path: str = DEFAULT_SHADOW_LOG_PATH,
        output_path: str = DEFAULT_OUTPUT_PATH,
        stages: tuple = GRADEABLE_STAGES) -> dict[str, Any]:
    all_records = read_jsonl(shadow_log_path)
    candidates = [
        r for r in all_records
        if not r.get("published") and r.get("rejected_stage") in stages
    ]
    logger.info(
        "shadow_grading_start",
        extra={"total_shadow_records": len(all_records), "candidates": len(candidates)},
    )

    graded = []
    for r in candidates:
        g = grade_shadow_record(r, output_path + ".rejects.jsonl")
        if g is not None:
            graded.append(g)

    write_jsonl(output_path, graded)

    # Summary: win rate by market, among rejects that WOULD have cleared
    # the pick's own edge/confidence numbers (i.e. all of them here, since
    # these are exactly the ones the model scored but the gate blocked).
    from collections import defaultdict
    by_market = defaultdict(lambda: {"win": 0, "loss": 0, "push": 0})
    for g in graded:
        by_market[g["_market"]][g["actual_result"]] += 1

    summary = {"n_candidates": len(candidates), "n_graded": len(graded), "by_market": dict(by_market)}
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Grade rejected picks from shadow_log.jsonl.")
    parser.add_argument("--shadow-log", default=DEFAULT_SHADOW_LOG_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    summary = run(args.shadow_log, args.output)
    print(f"Candidates (gatekeeper-rejected, edge/confidence scored): {summary['n_candidates']}")
    print(f"Graded this run: {summary['n_graded']}")
    for market, rec in summary["by_market"].items():
        total = rec["win"] + rec["loss"] + rec["push"]
        win_rate = rec["win"] / total * 100 if total else 0.0
        print(f"  {market:20s} {rec['win']:3d}W-{rec['loss']:3d}L-{rec['push']:3d}P  ({win_rate:.1f}% win)")


if __name__ == "__main__":
    main()
