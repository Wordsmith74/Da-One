"""
core/output_writer.py

Replaces the broadcast half of output/telegram_formatter.py (send_daily_picks
/ send_daily_recap), which is excluded from this build. This writes approved
picks to output/picks.json -- the same file run_pipeline-1.py's pipeline
already produces -- instead of pushing to Telegram, so both code paths now
converge on one shared output contract.

send_daily_recap was imported in main.py but never actually called anywhere,
so it has no replacement here -- nothing depended on it.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from core.bet_display import BetDisplay

_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
_PICKS_PATH = os.path.join(_OUTPUT_DIR, "picks.json")


def _bet_display_to_dict(bd: BetDisplay) -> dict[str, Any]:
    bet = bd.bet
    return {
        "bet_id": bet.bet_id,
        "team": bet.team,
        "player": bet.player,
        "market": bet.market,
        "direction": bet.direction,
        "sportsbook_line": bet.sportsbook_line,
        "american_odds": bd.american_odds,
        "edge_percentage": bet.edge_percentage,
        "confidence_score": bet.confidence_score,
        "model_probability": bd.model_probability,
        "tier": bet.tier.value if bet.tier else None,
        "supporting_factor": bd.supporting_factor,
        "game_id": bet.game_id,
        "game_time_utc": bd.game_time_utc,
        "away_team": bd.away_team,
        "home_team": bd.home_team,
        "full_team_name": bd.full_team_name,
        "bookmaker_source": bd.bookmaker_source,
        "book_count": bd.book_count,
        "verified_at": bd.verified_at,
        "opening_line": bd.opening_line,
        "consensus_line": bd.consensus_line,
        "mis_score": bd.mis_score,
        "data_reliability_score": bd.data_reliability_score,
    }


def send_daily_picks(
    *,
    approved_bets_by_sport: dict[str, list[BetDisplay]],
    date_str: str,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """
    Drop-in replacement for the old Telegram send_daily_picks(). Same
    call signature and same general "list of per-sport send results" return
    shape (so existing `sent_count = sum(1 for r in send_results if
    r.get("sent"))` call sites keep working) -- but writes to
    output/picks.json instead of broadcasting to a chat channel.

    dry_run=True still computes/logs what would be written but does not
    touch disk, matching the old function's dry-run semantics.
    """
    results: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "date": date_str,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sports": {},
    }

    for sport, bets in approved_bets_by_sport.items():
        payload["sports"][sport] = [_bet_display_to_dict(bd) for bd in bets]
        results.append({
            "sport": sport,
            "count": len(bets),
            "sent": True if not dry_run else False,
            "dry_run": dry_run,
        })

    if dry_run:
        return results

    try:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        with open(_PICKS_PATH, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception:
        # Surface the failure as "not sent" for every sport rather than
        # raising -- mirrors the old broadcast function's per-sport
        # failure-tolerant behavior.
        for r in results:
            r["sent"] = False

    return results
