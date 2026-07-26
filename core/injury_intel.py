"""
injury_intel.py

Converts live injury reports into an edge adjustment for a player/team prop
or moneyline pick. Ported from Wordsmith74's core/intelligence/lineup_intel.py
(WNBA ML/props engine) and adapted to sports-engine's sport_config.py style.

Why this exists: run_pipeline.py's live_fetch_wnba_player_prop() has had a
literal "TODO: wire injury status if available" on status_history since this
file didn't exist yet. This fills that gap using data/fetch.py's
get_wnba_team_injuries() (RotoWire-free-scrape primary, ESPN fallback --
see data/rotowire_injuries.py and data/fetch.py).

Severity / impact model (same shape as Wordsmith's, WNBA-only for now):
  status severity:  Out/IR/Suspended -> 1.00, Doubtful -> 0.70,
                     Questionable -> 0.35, Day-to-day -> 0.15, Probable -> 0.10
  position impact:  G/PG/SG -> 1.5x, F/SF -> 1.4x, C -> 1.2x, bench/unknown -> 0.8x
  total penalty = sum(severity * impact), capped at 6.0
  direction: hurts the edge if the injured player IS the prop subject /
             IS on the team we're backing; helps if it's the opponent.

Fail-safe: returns a zero-adjustment result on any error, exactly like
Wordsmith's get_lineup_intel() -- a missing/broken injury source should never
crash a pick, just leave it unadjusted.

check_pregame_availability() (added 2026-07-26) is a separate, MLB-focused
helper: a tri-state pregame status check (confirmed_active /
confirmed_out / unconfirmed), NOT a boolean. The reason it's tri-state:
lineups and last-minute starter scratches routinely aren't posted until
1-2 hours before first pitch, so a pipeline run earlier in the day will
often have zero positive-or-negative signal on a given player -- that is
a completely normal "unconfirmed" state, not equivalent to "confirmed
out," and must never be treated as such by a caller. Collapsing this to a
boolean either discards good picks all day (treating "no data yet" as
"assume out") or hides real late scratches (treating it as "assume
fine"). See the function's own docstring for the caller-side contract.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("sports_engine")

_STATUS_SEVERITY = {
    "out": 1.00,
    "injured reserve": 1.00,
    "ir": 1.00,
    "suspended": 1.00,
    "doubtful": 0.70,
    "questionable": 0.35,
    "day-to-day": 0.15,
    "probable": 0.10,
}

_WNBA_POSITION_IMPACT = {
    "G": 1.5, "PG": 1.5, "SG": 1.4,
    "F": 1.4, "SF": 1.3, "PF": 1.2,
    "C": 1.2,
}

_MAX_IMPACT = 6.0


def _severity(status: str) -> float:
    return _STATUS_SEVERITY.get((status or "").strip().lower(), 0.0)


def _position_impact(position: str) -> float:
    return _WNBA_POSITION_IMPACT.get((position or "").strip().upper(), 0.8)


def compute_injury_adjustment(injury_data: dict | None, subject_team_is_backed: bool = True) -> dict:
    """
    injury_data: shape returned by data/fetch.get_wnba_team_injuries(), i.e.
      {"injuries": [{"athlete": {"shortName": ..., "position": {"abbreviation": ...}},
                     "status": "Out" | "Doubtful" | ...}, ...]}
    subject_team_is_backed:
      True  -> these injuries are on the team/player the pick is FOR (hurts edge)
      False -> these injuries are on the OPPONENT (helps edge -- weaker opponent)

    Returns {"edge_adjustment": float, "injury_count": int, "star_out": bool,
             "factor_text": str}. Zero/empty on any missing or malformed data.
    """
    empty = {"edge_adjustment": 0.0, "injury_count": 0, "star_out": False, "factor_text": ""}
    if not injury_data:
        return empty

    rows = injury_data.get("injuries") or []
    if not rows:
        return empty

    total_impact = 0.0
    names = []
    star_out = False

    for inj in rows:
        try:
            athlete = inj.get("athlete", {})
            name = athlete.get("shortName") or "Unknown"
            position = (athlete.get("position") or {}).get("abbreviation") or "?"
            status = inj.get("status") or ""

            sev = _severity(status)
            mult = _position_impact(position)
            score = sev * mult
            if score > 0:
                total_impact += score
                names.append(f"{name} ({status})")
                if mult >= 1.4 and sev >= 0.70:
                    star_out = True
        except Exception as exc:
            logger.debug("[injury_intel] malformed injury row skipped: %s", exc)
            continue

    capped = min(_MAX_IMPACT, total_impact)
    direction = -1.0 if subject_team_is_backed else 1.0
    adj = round(direction * capped, 2)

    factor_text = ""
    if names:
        top = names[:3]
        factor_text = ", ".join(top) + (f" +{len(names)-3} more" if len(names) > 3 else "")

    return {
        "edge_adjustment": adj,
        "injury_count": len(names),
        "star_out": star_out,
        "factor_text": factor_text,
    }


# ---------------------------------------------------------------------------
# Pregame availability check (MLB pitcher_strikeouts, tri-state)
# ---------------------------------------------------------------------------

_OUT_SEVERITY_FLOOR = 0.70   # "doubtful" and worse count as confirmed_out;
                             # "questionable" and below stay unconfirmed --
                             # not a hard scratch signal on its own.

# Below this many hours-to-first-pitch, a still-unconfirmed status is itself
# mildly informative (lineups/starter news is normally out by then) rather
# than the routine, expected gap it is earlier in the day.
_LATE_UNCONFIRMED_WINDOW_HOURS = 1.5


def check_pregame_availability(
    player_name: str,
    team_abbr: str,
    game_time_utc: str | None = None,
    injury_data: dict | None = None,
    now_utc: datetime | None = None,
) -> dict:
    """
    Tri-state pregame availability check for an MLB player (built for
    pitcher_strikeouts, where a late scratch voids the whole prop; usable
    for any MLB player prop).

    Returns {"status": ..., "reason": str, "hours_to_game": float | None}
    status is one of:
      "confirmed_out"    -- found in the injury feed at >= _OUT_SEVERITY_FLOOR
                             severity (doubtful/out/IR/suspended). Treat as a
                             hard signal -- safe to discard the pick on this.
      "unconfirmed"       -- no disqualifying signal found. This is the
                             ROUTINE state for most of the day, since
                             lineups/starter-scratch news typically isn't
                             posted until 1-2 hours before first pitch --
                             it means "nothing bad reported YET," not
                             "confirmed to play." Do not discard on this
                             alone. hours_to_game tells the caller how much
                             weight to put on that gap: normal earlier in
                             the day, mildly notable inside
                             _LATE_UNCONFIRMED_WINDOW_HOURS of first pitch
                             (see reason string for which applies).
      "unknown"           -- injury source itself failed/unavailable (fetch
                             error, parse failure, etc.) or game_time_utc
                             wasn't parseable. Same fail-open contract as
                             compute_injury_adjustment() above: never crash,
                             never silently treat as confirmed_out.

    injury_data: pass a pre-fetched result in the same shape
    data/rotowire_injuries_mlb.get_recent_mlb_injuries() returns, i.e.
    {"injuries": [{"athlete": {"shortName": ...}, "status": "Out" | ...}]}.
    Caller is responsible for fetching it (this function does not import
    the network layer itself, to stay testable / dependency-light, same
    reasoning as compute_injury_adjustment() above taking injury_data as
    a parameter rather than fetching it internally).

    game_time_utc: ISO 8601 first-pitch time, if known. Optional -- when
    omitted, hours_to_game is None and the late-window escalation in
    "reason" is skipped entirely (caller just gets a plain "unconfirmed").
    """
    now = now_utc or datetime.now(timezone.utc)

    hours_to_game: float | None = None
    if game_time_utc:
        try:
            gt = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
            if gt.tzinfo is None:
                gt = gt.replace(tzinfo=timezone.utc)
            hours_to_game = round((gt - now).total_seconds() / 3600.0, 2)
        except (ValueError, TypeError) as exc:
            logger.debug("[injury_intel] unparseable game_time_utc=%r: %s", game_time_utc, exc)

    if not injury_data:
        reason = "no injury source data available for this check"
        return {"status": "unknown", "reason": reason, "hours_to_game": hours_to_game}

    rows = injury_data.get("injuries") or []
    target = (player_name or "").strip().lower()

    for inj in rows:
        try:
            athlete = inj.get("athlete", {})
            name = (athlete.get("shortName") or "").strip().lower()
            if not name or name != target:
                continue
            status = inj.get("status") or ""
            sev = _severity(status)
            if sev >= _OUT_SEVERITY_FLOOR:
                reason = f"confirmed via injury feed: status={status!r} (severity={sev:.2f})"
                return {"status": "confirmed_out", "reason": reason, "hours_to_game": hours_to_game}
        except Exception as exc:
            logger.debug("[injury_intel] malformed row in pregame check, skipped: %s", exc)
            continue

    # Nothing disqualifying found. Distinguish routine-gap from late-and-still-quiet
    # purely for the caller's benefit (informational -- both are "unconfirmed").
    if hours_to_game is not None and hours_to_game <= _LATE_UNCONFIRMED_WINDOW_HOURS:
        reason = (
            f"no disqualifying status found, but only {hours_to_game:.2f}h to first "
            f"pitch and still no positive confirmation -- lineups/starter news is "
            f"normally out by this point, treat with mild extra caution"
        )
    else:
        reason = (
            "no disqualifying status found -- routine gap, lineup/starter "
            "confirmation is not expected to be posted yet this far from first pitch"
        )
    return {"status": "unconfirmed", "reason": reason, "hours_to_game": hours_to_game}
