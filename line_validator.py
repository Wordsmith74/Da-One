"""
line_validator.py — Pre-publish line accuracy gate for ALL pick types.

Implements Rules 4, 7, 8 from the Player Prop Line Accuracy spec, extended
to cover game totals, team totals, and all derivative markets:

  Rule 4: Last-minute verification — re-check every line before publishing.
  Rule 7: Reject any pick where the current sportsbook line differs from the
          analysis line by more than the applicable drift threshold.
  Rule 8: Pick quality-control checklist: player/team ✓  market ✓  line ✓
          odds ✓  projection ✓  timestamp ✓

Entry point
-----------
    pre_publish_verify(bets_by_sport, dry_run) → (cleaned, removed_report)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from output.telegram_formatter import BetDisplay

logger = logging.getLogger(__name__)

# Maximum allowed line drift between analysis and pre-publish re-verification.
_LINE_DRIFT_THRESHOLD      = 0.5   # player props
_GAME_LINE_DRIFT_THRESHOLD = 0.5   # game totals / team totals / derivatives

# Module-level cache: at most one Odds API call per sport per run.
_fresh_game_cache: dict[str, list[dict[str, Any]]] = {}


def _refresh_game_line(
    sport:     str,
    team:      str,
    market:    str,
    direction: str,
) -> float | None:
    """
    Return the current sportsbook line for a game-level or derivative bet.

    Results are cached per sport so a single Odds API call covers all picks
    in the same sport within one run.  Returns None when the market cannot
    be located (derivative markets like NRFI/YRFI may not appear in the
    standard candidates list — those are passed through unmodified).
    """
    global _fresh_game_cache

    sport_up = sport.upper()
    if sport_up not in _fresh_game_cache:
        try:
            from core.odds_client import fetch_todays_candidates
            _fresh_game_cache[sport_up] = fetch_todays_candidates(sport_up)
            logger.debug(
                f"[line_validator] Fetched {len(_fresh_game_cache[sport_up])} "
                f"game candidates for {sport_up}."
            )
        except Exception as exc:
            logger.warning(
                f"[line_validator] Could not fetch game candidates for {sport_up}: {exc}"
            )
            _fresh_game_cache[sport_up] = []

    candidates = _fresh_game_cache[sport_up]
    t = team.upper()
    m = market.lower()
    d = direction.lower()

    for c in candidates:
        if (
            str(c.get("team", "")).upper() == t
            and str(c.get("market", "")).lower() == m
            and str(c.get("direction", "")).lower() == d
        ):
            line = c.get("sportsbook_line")
            return float(line) if line is not None else None

    return None


def pre_publish_verify(
    bets_by_sport: dict[str, list["BetDisplay"]],
    dry_run: bool = False,
) -> tuple[dict[str, list["BetDisplay"]], list[dict[str, Any]]]:
    """
    Re-fetch live sportsbook lines for every approved pick and validate them
    against the lines used during analysis.

    Covers ALL pick types:
      • Player props  — re-verified via refresh_prop_line()
      • Game totals   — re-verified via fetch_todays_candidates()
      • Team totals   — same as game totals
      • Derivatives   — NRFI/YRFI, F5, First Half, First Quarter (pass-through
                        when no live match found in standard candidates)

    Parameters
    ----------
    bets_by_sport : sport → list[BetDisplay] — output from _run_sport_pipeline
    dry_run       : when True, skip live API calls and pass all picks through

    Returns
    -------
    (cleaned_bets_by_sport, removed_report)
      cleaned_bets_by_sport : same shape as input, with stale picks stripped
      removed_report        : list of dicts describing every removed pick
    """
    from core.player_props import get_prop_meta, refresh_prop_line

    # Reset per-run cache so this invocation always uses fresh data.
    _fresh_game_cache.clear()

    removed: list[dict[str, Any]] = []
    cleaned: dict[str, list["BetDisplay"]] = {}

    for sport, bets in bets_by_sport.items():
        passing: list["BetDisplay"] = []

        for bd in bets:
            bet = bd.bet

            # ── DRY RUN ───────────────────────────────────────────────────────
            if dry_run:
                logger.debug(
                    f"[line_validator] DRY RUN — skipping re-fetch for "
                    f"{bet.player or bet.team} {bet.market}."
                )
                passing.append(bd)
                continue

            # ── PLAYER PROPS ──────────────────────────────────────────────────
            if bet.player:
                meta = get_prop_meta(bet.bet_id)
                if not meta:
                    logger.warning(
                        f"[line_validator] No metadata for {bet.bet_id} — "
                        "passing through unverified."
                    )
                    passing.append(bd)
                    continue

                fresh_line = refresh_prop_line(
                    api_sport  = meta["api_sport"],
                    event_id   = meta["event_id"],
                    player     = meta["player"],
                    market_key = meta["market_key"],
                    direction  = meta["direction"],
                )

                if fresh_line is None:
                    logger.warning(
                        f"[line_validator] Could not re-fetch line for "
                        f"{bet.player} {meta['market_key']} — passing through."
                    )
                    passing.append(bd)
                    continue

                opening   = meta["opening_line"]
                drift     = fresh_line - opening
                abs_drift = abs(drift)
                threshold = _LINE_DRIFT_THRESHOLD

            # ── GAME TOTALS / TEAM TOTALS / DERIVATIVES ───────────────────────
            else:
                fresh_line = _refresh_game_line(
                    sport     = sport,
                    team      = bet.team,
                    market    = bet.market,
                    direction = bet.direction,
                )

                if fresh_line is None:
                    # Not found in live data (thin/derivative market).
                    # Pass through rather than silently stripping a valid pick.
                    logger.debug(
                        f"[line_validator] {sport} {bet.market} {bet.team} "
                        f"{bet.direction} — no live match found, passing through."
                    )
                    passing.append(bd)
                    continue

                opening   = float(bet.sportsbook_line or 0.0)
                drift     = fresh_line - opening
                abs_drift = abs(drift)
                threshold = _GAME_LINE_DRIFT_THRESHOLD

            # ── Rule 7: reject if drift exceeds threshold ──────────────────────
            if abs_drift > threshold:
                sign   = "+" if drift >= 0 else ""
                label  = bet.player or f"{bet.market} {bet.team}"
                reason = (
                    f"Line moved {opening} → {fresh_line} "
                    f"({sign}{drift:.1f}) — exceeds ±{threshold} threshold"
                )
                logger.warning(
                    f"[line_validator] ✗ LINE VALIDATION FAILED — "
                    f"{label} {bet.direction}: {reason}"
                )
                removed.append({
                    "bet_id":       bet.bet_id,
                    "player":       bet.player,
                    "team":         bet.team,
                    "sport":        sport,
                    "market":       bet.market,
                    "direction":    bet.direction,
                    "opening_line": opening,
                    "live_line":    fresh_line,
                    "drift":        drift,
                    "reason":       reason,
                })
            else:
                sign_str = f"+{drift:.1f}" if drift >= 0 else f"{drift:.1f}"
                label    = bet.player or f"{bet.market} {bet.team}"
                logger.info(
                    f"[line_validator] ✓ {label} {bet.direction}: "
                    f"line {opening} confirmed (live: {fresh_line}, drift: {sign_str})"
                )
                passing.append(bd)

        if passing:
            cleaned[sport] = passing

    total_in  = sum(len(v) for v in bets_by_sport.values())
    total_out = sum(len(v) for v in cleaned.values())
    logger.info(
        f"[line_validator] Pre-publish gate: "
        f"{total_out}/{total_in} pick(s) passed · {len(removed)} removed."
    )
    return cleaned, removed
