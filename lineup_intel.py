"""
lineup_intel.py

Fetches live injury and lineup status from the ESPN public injuries API
and converts it into an edge adjustment for the prediction engine.

Adjustment logic
----------------
Each injured player is scored by severity:
  Out / Suspended / IR         → 1.00
  Doubtful                     → 0.70
  Questionable                 → 0.35
  Day-to-day / Probable        → 0.15

Impact multiplier per position group (sport-specific):
  NBA/WNBA : G/F/C starters → 1.5×;  bench → 0.6×
  MLB      : SP (starting pitcher) → 2.0×; hitter → 1.0×; RP → 0.4×

Total edge penalty = Σ(severity × impact_multiplier), capped at -6.0.
The penalty is applied against the BET DIRECTION:
  - If the injured team is the one we are betting ON  → full penalty
  - If the injured team is the opponent              → penalty becomes a bonus (+)

Data resilience
---------------
Uses data_fetcher.fetch_espn() — strict 3-second timeout, ESPN primary →
ESPN fallback → RotoWire waterfall (Rule 1 + Rule 2).  Every source
failure is logged as a "Source Unavailable" event (Rule 4).
Response structure is validated before use (Rule 3).

Fail-safe: returns LineupIntelFactor() with zero adjustment on any error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("betting_bot")

from core.data_fetcher import fetch_espn, fetch_wnba_injuries  # Rule 1+2 — fetch with timeout + waterfall
from core.data_validator import validate_injuries  # Rule 3 — structural integrity check

_SPORT_PATHS: dict[str, str] = {
    "WNBA": "basketball/wnba",
    "NBA":  "basketball/nba",
    "MLB":  "baseball/mlb",
}


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

_STATUS_SEVERITY: dict[str, float] = {
    "out":             1.00,
    "injured reserve": 1.00,
    "ir":              1.00,
    "suspended":       1.00,
    "doubtful":        0.70,
    "questionable":    0.35,
    "day-to-day":      0.15,
    "probable":        0.10,
}

# Position → impact multiplier
_POSITION_IMPACT: dict[str, dict[str, float]] = {
    "NBA": {
        "PG": 1.5, "SG": 1.4, "SF": 1.4, "PF": 1.2, "C": 1.2,
        "G": 1.3, "F": 1.3,
    },
    "WNBA": {
        "G": 1.5, "F": 1.4, "C": 1.2,
        "PG": 1.5, "SG": 1.4, "SF": 1.3,
    },
    "MLB": {
        "SP": 2.0, "P": 1.0,
        "C": 1.0, "1B": 0.9, "2B": 0.9, "3B": 0.9, "SS": 1.1,
        "LF": 0.8, "CF": 1.0, "RF": 0.8, "DH": 0.9,
        "RP": 0.4, "CP": 0.5,
    },
}


def _severity(status_str: str) -> float:
    key = status_str.lower().strip()
    return _STATUS_SEVERITY.get(key, 0.0)


def _impact_mult(position: str, sport: str) -> float:
    table = _POSITION_IMPACT.get(sport.upper(), {})
    return table.get(position.upper(), 0.8)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class LineupIntelFactor:
    injury_count:    int   = 0
    impact_score:    float = 0.0
    star_out:        bool  = False
    edge_adjustment: float = 0.0
    factor_text:     str   = ""
    injuries:        list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def get_lineup_intel(
    team_abbr: str,
    sport: str,
    bet_on_this_team: bool = True,
) -> LineupIntelFactor:
    """
    Fetch injury data for *team_abbr* and return an edge adjustment.

    Parameters
    ----------
    team_abbr        : Team abbreviation used in the engine (e.g. 'GSW', 'NYY').
    sport            : 'NBA' | 'WNBA' | 'MLB'.
    bet_on_this_team : If True, injuries HURT the edge (we're betting on them).
                       If False, injuries HELP the edge (we're betting against them).
    """
    try:
        return _compute(team_abbr, sport, bet_on_this_team)
    except Exception as exc:
        logger.debug(f"[lineup_intel] Error for {team_abbr}: {exc}")
        return LineupIntelFactor()


def _compute(team_abbr: str, sport: str, bet_on_this_team: bool) -> LineupIntelFactor:
    sport_path = _SPORT_PATHS.get(sport.upper())
    if not sport_path:
        return LineupIntelFactor()

    # Rule 1+2: fetch with strict timeout + waterfall failover.
    # WNBA tries the FREE RotoWire scrape first (off-ESPN diversification);
    # other sports keep the original ESPN-primary waterfall.
    if sport.upper() == "WNBA":
        result = fetch_wnba_injuries(team_abbr)
    else:
        result = fetch_espn(f"{sport_path}/teams/{team_abbr}/injuries")
    if not result.ok:
        logger.debug(
            f"[lineup_intel] All sources failed for {team_abbr} injuries "
            f"(last error: {result.error})"
        )
        return LineupIntelFactor()

    # Rule 3: validate structure before using data
    validation = validate_injuries(result.data)
    if not validation.valid:
        logger.debug(
            f"[lineup_intel] Data integrity: {validation.reason} for {team_abbr}"
            + (f"  missing={validation.missing_fields}" if validation.missing_fields else "")
        )
        return LineupIntelFactor()

    injuries_raw = result.data.get("injuries", [])  # type: ignore[union-attr]
    if not injuries_raw:
        return LineupIntelFactor(factor_text=f"{team_abbr}: no reported injuries")

    total_impact = 0.0
    injury_names: list[str] = []
    star_out     = False

    for inj in injuries_raw:
        athlete  = inj.get("athlete", {})
        name     = athlete.get("shortName") or athlete.get("displayName") or "Unknown"
        pos_obj  = athlete.get("position", {})
        position = pos_obj.get("abbreviation") or pos_obj.get("name") or "?"
        status   = inj.get("status") or inj.get("type", {}).get("description") or ""

        sev  = _severity(status)
        mult = _impact_mult(position, sport)
        score = sev * mult

        if score > 0:
            total_impact += score
            injury_names.append(f"{name} ({status})")
            if mult >= 1.4 and sev >= 0.70:
                star_out = True

        logger.debug(
            f"[lineup_intel] {team_abbr} — {name} [{position}] {status}: "
            f"sev={sev:.2f} × mult={mult:.1f} = {score:.2f}"
        )

    # Cap impact at 6.0, convert to edge adjustment
    capped_impact = min(6.0, total_impact)
    # Negative when betting on the injured team, positive when betting against
    direction     = -1.0 if bet_on_this_team else +1.0
    adj           = round(direction * capped_impact, 2)

    # Build factor text
    if injury_names:
        top = injury_names[:3]
        factor_text = f"{team_abbr} injuries: {', '.join(top)}"
        if len(injury_names) > 3:
            factor_text += f" +{len(injury_names)-3} more"
    else:
        factor_text = ""

    logger.debug(
        f"[lineup_intel] {team_abbr} {sport}: "
        f"injuries={len(injury_names)}  impact={capped_impact:.2f}  adj={adj:+.2f}"
    )

    return LineupIntelFactor(
        injury_count    = len(injury_names),
        impact_score    = capped_impact,
        star_out        = star_out,
        edge_adjustment = adj,
        factor_text     = factor_text,
        injuries        = injury_names,
    )
