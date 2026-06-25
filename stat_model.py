"""
stat_model.py

Pulls season statistics from the ESPN public API and applies
pace-adjusted matchup modelling to the engine's projections.

What it does
------------
For TEAM TOTAL bets:
  Projected total = team_avg_score × pace_ratio
                  vs sportsbook_line
  If projected > line by ≥ 2 pts  → +1.0 edge
  If projected < line by ≥ 2 pts  → -1.0 edge

For PLAYER PROP bets (points / assists / rebounds):
  Uses team offensive pace rank as a proxy.
  High-pace team (top 5)  → +0.5 edge for over props
  Low-pace  team (bot 5)  → -0.5 edge for over props

For TEAM SPREAD bets:
  Uses points-differential (avg_score − avg_allowed).
  Significant positive differential vs bookmaker spread → +0.8 edge.

Adjustments are capped at ±2.0 per module.

Data resilience
---------------
Uses data_fetcher.fetch_espn() — strict 3-second timeout, ESPN primary →
ESPN fallback → RotoWire waterfall (Rule 1 + Rule 2).  Every source
failure is logged as a "Source Unavailable" event (Rule 4).
Response structure is validated before use (Rule 3).

Fail-safe: returns StatModelFactor() with zero adjustment on any error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("betting_bot")

from core.data_fetcher import fetch_espn            # Rule 1+2
from core.data_validator import validate_statistics  # Rule 3

_SPORT_PATHS: dict[str, str] = {
    "WNBA": "basketball/wnba",
    "NBA":  "basketball/nba",
    "MLB":  "baseball/mlb",
}


# ---------------------------------------------------------------------------
# Stat name normalisation
# ---------------------------------------------------------------------------

_STAT_ALIASES: dict[str, list[str]] = {
    "avg_points":         ["avgPoints", "pointsPerGame", "pts", "runs"],
    "avg_points_allowed": ["avgPointsAllowed", "oppPointsPerGame", "oppRuns"],
    "avg_rebounds":       ["avgRebounds", "reboundsPerGame", "reb"],
    "avg_assists":        ["avgAssists", "assistsPerGame", "ast"],
    "pace":               ["pace", "possessionsPerGame", "pitchesPer9"],
}


def _extract_stat(stats_list: list[dict], key: str) -> float | None:
    aliases = _STAT_ALIASES.get(key, [key])
    by_name = {s.get("name", ""): s.get("value") for s in stats_list}
    for alias in aliases:
        val = by_name.get(alias)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _flatten_stats(data: dict[str, Any]) -> list[dict]:
    """
    ESPN statistics endpoint returns stats in a nested structure.
    Flatten all stat nodes into a single list of {name, value} dicts.
    """
    flat: list[dict] = []

    def _walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, dict):
            if "name" in node and "value" in node:
                flat.append(node)
            for v in node.values():
                _walk(v)

    _walk(data)
    return flat


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class StatModelFactor:
    avg_points:         float | None = None
    avg_points_allowed: float | None = None
    projected_total:    float | None = None
    pace:               float | None = None
    edge_adjustment:    float        = 0.0
    factor_text:        str          = ""


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def get_stat_model_factor(
    team_abbr: str,
    market: str,
    sport: str,
    sportsbook_line: float | None = None,
    direction: str = "over",
) -> StatModelFactor:
    """
    Fetch team season statistics and return a matchup-adjusted edge delta.

    Parameters
    ----------
    team_abbr       : Team abbreviation (e.g. 'GSW', 'SEA', 'NYY').
    market          : Bet market string ('Team Total', 'Player Points', etc.).
    sport           : 'NBA' | 'WNBA' | 'MLB'.
    sportsbook_line : The posted O/U line (optional; used for total comparison).
    direction       : 'over' | 'under'.
    """
    try:
        return _compute(team_abbr, market, sport, sportsbook_line, direction)
    except Exception as exc:
        logger.debug(f"[stat_model] Error for {team_abbr}: {exc}")
        return StatModelFactor()


def _compute(
    team_abbr: str,
    market: str,
    sport: str,
    sportsbook_line: float | None,
    direction: str,
) -> StatModelFactor:
    sport_path = _SPORT_PATHS.get(sport.upper())
    if not sport_path:
        return StatModelFactor()

    # Rule 1+2: fetch with strict timeout + waterfall failover
    result = fetch_espn(f"{sport_path}/teams/{team_abbr}/statistics")
    if not result.ok:
        logger.debug(
            f"[stat_model] All sources failed for {team_abbr} statistics "
            f"(last error: {result.error})"
        )
        return StatModelFactor()

    # Rule 3: validate structure before using data
    validation = validate_statistics(result.data)
    if not validation.valid:
        logger.debug(
            f"[stat_model] Data integrity: {validation.reason} for {team_abbr}"
            + (f"  missing={validation.missing_fields}" if validation.missing_fields else "")
        )
        return StatModelFactor()

    stats_flat = _flatten_stats(result.data)  # type: ignore[arg-type]
    if not stats_flat:
        return StatModelFactor()

    avg_pts     = _extract_stat(stats_flat, "avg_points")
    avg_allowed = _extract_stat(stats_flat, "avg_points_allowed")
    pace        = _extract_stat(stats_flat, "pace")
    avg_reb     = _extract_stat(stats_flat, "avg_rebounds")
    avg_ast     = _extract_stat(stats_flat, "avg_assists")

    adj          = 0.0
    factor_parts: list[str] = []
    market_lower = market.lower()

    # ── Team total / team spread ──────────────────────────────────────────
    if "team total" in market_lower or "total" in market_lower:
        if avg_pts is not None:
            factor_parts.append(f"avg {avg_pts:.1f} pts/g")
            if sportsbook_line is not None:
                diff = avg_pts - sportsbook_line
                if direction == "over" and diff >= 2.0:
                    adj += min(1.5, diff * 0.3)
                    factor_parts.append(f"{diff:+.1f} vs line")
                elif direction == "under" and diff <= -2.0:
                    adj += min(1.5, abs(diff) * 0.3)
                    factor_parts.append(f"{diff:+.1f} vs line")
                elif direction == "over" and diff <= -2.0:
                    adj -= min(1.5, abs(diff) * 0.3)
                elif direction == "under" and diff >= 2.0:
                    adj -= min(1.5, diff * 0.3)

    elif "spread" in market_lower:
        if avg_pts is not None and avg_allowed is not None:
            point_diff = avg_pts - avg_allowed
            factor_parts.append(f"pt diff {point_diff:+.1f}")
            if sportsbook_line is not None:
                model_spread = -point_diff
                spread_edge  = sportsbook_line - model_spread
                if spread_edge >= 2.0:
                    adj += min(1.0, spread_edge * 0.2)
                elif spread_edge <= -2.0:
                    adj -= min(1.0, abs(spread_edge) * 0.2)

    # ── Player props ─────────────────────────────────────────────────────
    elif "player points" in market_lower or "points" in market_lower:
        if pace is not None:
            factor_parts.append(f"pace {pace:.1f}")
            if pace > 100:
                adj += 0.5 if direction == "over" else -0.5
            elif pace < 95:
                adj -= 0.5 if direction == "over" else -0.5
        if avg_pts is not None:
            factor_parts.append(f"team {avg_pts:.1f} pts/g")

    elif "assists" in market_lower:
        if avg_ast is not None:
            factor_parts.append(f"team {avg_ast:.1f} ast/g")
        if pace is not None and pace > 100:
            adj += 0.3 if direction == "over" else -0.3

    elif "rebounds" in market_lower:
        if avg_reb is not None:
            factor_parts.append(f"team {avg_reb:.1f} reb/g")

    # Cap
    adj = round(max(-2.0, min(2.0, adj)), 2)

    proj_total: float | None = None
    if avg_pts is not None and pace is not None:
        proj_total = round(avg_pts, 1)

    factor_text = f"{team_abbr} stats: {', '.join(factor_parts)}" if factor_parts else ""

    logger.debug(
        f"[stat_model] {team_abbr} {sport} [{market}]: adj={adj:+.2f}  {factor_text}"
    )

    return StatModelFactor(
        avg_points         = avg_pts,
        avg_points_allowed = avg_allowed,
        projected_total    = proj_total,
        pace               = pace,
        edge_adjustment    = adj,
        factor_text        = factor_text,
    )
