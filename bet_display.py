"""
core/bet_display.py

BetDisplay — the engine-internal wrapper around a gatekeeper-approved Bet
that carries the extra display/context fields (odds, model probability,
team names, line-verification metadata, etc.) needed downstream by tiering,
CCS scoring, conflict resolution, and output writing.

This was previously defined inside output/telegram_formatter.py, bundled
together with the Telegram broadcast functions (send_daily_picks /
send_daily_recap). It has been extracted here because BetDisplay itself is
pure data with no broadcast-channel dependency -- multiple core modules
(decision logic, CCS scoring, line validation) need the dataclass without
needing anything Telegram-related. Broadcast/output formatting now lives in
core/output_writer.py instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.decision_gatekeeper import Bet


@dataclass
class BetDisplay:
    """
    Wraps an approved Bet with the additional fields needed to render or
    publish it: market odds, the model's (possibly calibrated-for-display)
    win probability, team/game context, and line-integrity metadata.

    bet                    : The underlying gatekeeper-evaluated Bet (tier,
                              edge, confidence, market, etc. live here).
    american_odds          : Market odds at time of publication.
    model_probability      : Display-facing win probability for the picked
                              side (may be calibration-compressed for MLB --
                              see main.py:_calibrate_mlb_confidence -- while
                              bet.confidence_score, the tier driver, is not).
    supporting_factor      : Free-text rationale / calibration note shown
                              alongside the pick.
    game_time_utc          : Scheduled start time (UTC), if known.
    away_team / home_team  : Short team identifiers.
    full_team_name         : Full display name for the relevant side.
    bookmaker_source       : Which book the published odds came from.
    book_count             : Number of books that agreed on this line.
    verified_at            : Timestamp of last line-verification pass.
    opening_line            : Line value when the pick was first generated.
    consensus_line         : Current market consensus line.
    mis_score               : Market-integrity-score snapshot (duplicated
                              from bet.mis_score for convenience at display
                              time; bet.mis_score remains the source of
                              truth for any logic that needs it).
    data_reliability_score  : Data-source reliability snapshot, same
                              duplication rationale as mis_score above.
    """

    bet: Bet
    american_odds: float
    model_probability: float
    supporting_factor: str = ""
    game_time_utc: Any = None
    away_team: str = ""
    home_team: str = ""
    full_team_name: str = ""
    bookmaker_source: str = ""
    book_count: int = 0
    verified_at: Any = None
    opening_line: float | None = None
    consensus_line: float | None = None
    mis_score: int | None = None
    data_reliability_score: int | None = None
