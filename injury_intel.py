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
"""

from __future__ import annotations

import logging

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
