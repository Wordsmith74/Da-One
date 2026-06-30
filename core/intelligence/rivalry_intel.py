"""
rivalry_intel.py

Rivalry and Head-to-Head Context Engine for MLB.

What it does
------------
1. Classifies rivalry level: NONE / MINOR / DIVISION / MAJOR
   - Division rivals  → DIVISION
   - Curated historic pairs → MAJOR (Yankees–Red Sox, Cubs–Cardinals, etc.)

2. Fetches current-season H2H results from the MLB Stats API (free, no key).
   One API call per home team, shared across H2H + season win-rate lookups
   via a process-level schedule cache.

3. Flags MATCHUP_OVERPERFORMANCE / MATCHUP_UNDERPERFORMANCE when the
   H2H win rate deviates ≥ 10 pp from the team's overall season win rate.

4. Applies a volatility/uncertainty penalty to the edge:
      MAJOR:    −3.0 pts  (acknowledges unpredictability, not an upset pick)
      DIVISION: −2.0 pts
      MINOR:    −0.5 pts

5. Partial underdog recovery (up to +2.0) when the rival underdog has
   ≥ 40% H2H win rate AND a positive recent margin trend.

Safety rules
------------
- Maximum total influence: ±5.0 edge points.
- Never overrides pitcher, injury, bullpen, or market-efficiency signals.
- MLB-only for now; returns zero-adjustment for WNBA/NBA.
- Fail-safe: returns RivalryFactor(edge_adjustment=0.0) on any error.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from urllib import request as urllib_req

logger = logging.getLogger("betting_bot")


# ---------------------------------------------------------------------------
# MLB division membership  (abbreviation → division, for same-div detection)
# ---------------------------------------------------------------------------

_MLB_DIVISIONS: dict[str, frozenset[str]] = {
    "AL East":    frozenset({"NYY", "BOS", "TOR", "BAL", "TB"}),
    "AL Central": frozenset({"CLE", "MIN", "CWS", "KC",  "DET"}),
    "AL West":    frozenset({"HOU", "TEX", "SEA", "LAA", "OAK"}),
    "NL East":    frozenset({"ATL", "NYM", "PHI", "MIA", "WSH"}),
    "NL Central": frozenset({"MIL", "CHC", "STL", "PIT", "CIN"}),
    "NL West":    frozenset({"LAD", "SF",  "SD",  "COL", "ARI"}),
}

# ---------------------------------------------------------------------------
# Curated MAJOR rivalries (abbreviation pair frozensets)
# ---------------------------------------------------------------------------

_MAJOR_RIVALRIES: set[frozenset[str]] = {
    frozenset({"NYY", "BOS"}),  # Yankees–Red Sox
    frozenset({"CHC", "STL"}),  # Cubs–Cardinals
    frozenset({"LAD", "SF"}),   # Dodgers–Giants
    frozenset({"NYY", "NYM"}),  # Subway Series
    frozenset({"OAK", "SF"}),   # Bay Bridge Series
    frozenset({"ATL", "NYM"}),  # Braves–Mets
    frozenset({"HOU", "TEX"}),  # Texas rivalry
    frozenset({"CHC", "CWS"}),  # Chicago rivalry
    frozenset({"LAD", "SD"}),   # NL West border rivalry
    frozenset({"STL", "CIN"}),  # NL Central rivalry
    frozenset({"NYY", "HOU"}),  # Modern ALCS rivalry
    frozenset({"BOS", "TB"}),   # AL East rivalry
}

# ---------------------------------------------------------------------------
# Abbreviation → MLB Stats API teamId
# ---------------------------------------------------------------------------

_ABBREV_TO_ID: dict[str, int] = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111,
    "CHC": 112, "CWS": 145, "CIN": 113, "CLE": 114,
    "COL": 115, "DET": 116, "HOU": 117, "KC":  118,
    "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158,
    "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD":  135, "SF":  137,
    "SEA": 136, "STL": 138, "TB":  139, "TEX": 140,
    "TOR": 141, "WSH": 120,
}

# ---------------------------------------------------------------------------
# Process-level caches
# ---------------------------------------------------------------------------

# abbr → list of completed regular-season games this season (raw API dicts)
_SCHEDULE_CACHE: dict[str, list[dict[str, Any]]] = {}

# frozenset({abbr_a, abbr_b}) → _H2HRecord
_H2H_CACHE: dict[frozenset[str], "_H2HRecord"] = {}


# ---------------------------------------------------------------------------
# Internal data holders
# ---------------------------------------------------------------------------

@dataclass
class _H2HRecord:
    """Current-season head-to-head record from home team's perspective."""
    wins:        int              = 0
    losses:      int              = 0
    game_totals: list[float]      = field(default_factory=list)
    margins:     list[float]      = field(default_factory=list)  # + = home wins

    @property
    def total_games(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.total_games if self.total_games >= 3 else None

    @property
    def avg_total(self) -> float | None:
        return sum(self.game_totals) / len(self.game_totals) if self.game_totals else None

    @property
    def avg_margin(self) -> float | None:
        return sum(self.margins) / len(self.margins) if self.margins else None


# ---------------------------------------------------------------------------
# MLB Stats API helpers
# ---------------------------------------------------------------------------

def _fetch_mlb_schedule(team_id: int) -> list[dict[str, Any]]:
    """Return all completed regular-season games for *team_id* this season."""
    season = date.today().year
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&teamId={team_id}&season={season}&gameType=R"
    )
    with urllib_req.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    return [
        g
        for d in data.get("dates", [])
        for g in d.get("games", [])
        if g.get("status", {}).get("statusCode") == "F"
    ]


def _get_schedule(abbr: str) -> list[dict[str, Any]]:
    """
    Return the team's completed schedule for the current season.
    Cached at process level — one fetch per team per run.
    """
    if abbr in _SCHEDULE_CACHE:
        return _SCHEDULE_CACHE[abbr]

    team_id = _ABBREV_TO_ID.get(abbr)
    if team_id is None:
        _SCHEDULE_CACHE[abbr] = []
        return []

    try:
        games = _fetch_mlb_schedule(team_id)
        _SCHEDULE_CACHE[abbr] = games
        return games
    except Exception as exc:
        logger.debug(f"[rivalry] schedule fetch failed for {abbr}: {exc}")
        _SCHEDULE_CACHE[abbr] = []
        return []


def _get_h2h_record(home_abbr: str, away_abbr: str) -> _H2HRecord:
    """
    Return current-season H2H record (home team's perspective).
    Cached by sorted pair to avoid duplicate fetches.
    """
    key = frozenset({home_abbr, away_abbr})
    if key in _H2H_CACHE:
        # Re-orient to home_abbr perspective
        cached = _H2H_CACHE[key]
        return cached

    home_id = _ABBREV_TO_ID.get(home_abbr)
    away_id = _ABBREV_TO_ID.get(away_abbr)
    if home_id is None or away_id is None:
        rec = _H2HRecord()
        _H2H_CACHE[key] = rec
        return rec

    games = _get_schedule(home_abbr)
    rec   = _H2HRecord()

    for g in games:
        h = g["teams"]["home"]
        a = g["teams"]["away"]
        gm_home_id = h.get("team", {}).get("id")
        gm_away_id = a.get("team", {}).get("id")

        # Only keep games between these two specific teams
        if {gm_home_id, gm_away_id} != {home_id, away_id}:
            continue

        h_score = h.get("score")
        a_score = a.get("score")
        if h_score is None or a_score is None:
            continue

        total  = float(h_score + a_score)
        rec.game_totals.append(total)

        if gm_home_id == home_id:
            # home team in this game = our home_abbr
            margin = float(h_score - a_score)
            won    = bool(h.get("isWinner", h_score > a_score))
        else:
            # home team in this game = away_abbr (visiting today's home)
            margin = float(a_score - h_score)
            won    = bool(a.get("isWinner", a_score > h_score))

        rec.margins.append(margin)
        if won:
            rec.wins   += 1
        else:
            rec.losses += 1

    _H2H_CACHE[key] = rec
    avg_str = f"{rec.avg_total:.1f}" if rec.avg_total is not None else "n/a"
    logger.debug(
        f"[rivalry] H2H {away_abbr}@{home_abbr}: "
        f"{rec.wins}W-{rec.losses}L  avg_total={avg_str}"
    )
    return rec


def _get_season_win_rate(abbr: str) -> float | None:
    """
    Return the team's overall current-season win rate (≥ 5 games required).
    Reuses the schedule cache — no extra API call.
    """
    team_id = _ABBREV_TO_ID.get(abbr)
    if team_id is None:
        return None

    games = _get_schedule(abbr)
    if len(games) < 5:
        return None

    wins = sum(
        1 for g in games
        if (
            g["teams"]["home"].get("team", {}).get("id") == team_id
            and bool(g["teams"]["home"].get("isWinner", False))
        ) or (
            g["teams"]["away"].get("team", {}).get("id") == team_id
            and bool(g["teams"]["away"].get("isWinner", False))
        )
    )
    return wins / len(games)


# ---------------------------------------------------------------------------
# Rivalry classification
# ---------------------------------------------------------------------------

def _classify_rivalry(abbr_a: str, abbr_b: str) -> str:
    """Return 'NONE', 'MINOR', 'DIVISION', or 'MAJOR'."""
    pair = frozenset({abbr_a, abbr_b})

    if pair in _MAJOR_RIVALRIES:
        return "MAJOR"

    for div_teams in _MLB_DIVISIONS.values():
        if abbr_a in div_teams and abbr_b in div_teams:
            return "DIVISION"

    return "NONE"


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class RivalryFactor:
    rivalry_level:   str               = "NONE"
    h2h_wins:        int               = 0
    h2h_losses:      int               = 0
    h2h_avg_total:   float | None      = None
    h2h_avg_margin:  float | None      = None
    h2h_win_rate:    float | None      = None   # actual (current season)
    season_win_rate: float | None      = None   # expected (full season)
    performance_tag: str               = ""     # MATCHUP_OVER/UNDERPERFORMANCE
    edge_adjustment: float             = 0.0
    tags:            list[str]         = field(default_factory=list)
    factor_text:     str               = ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_rivalry_intel(
    home_name:  str,
    away_name:  str,
    sport:      str,
    home_abbr:  str = "",
    away_abbr:  str = "",
) -> RivalryFactor:
    """
    Evaluate rivalry and H2H context for a matchup.

    Parameters
    ----------
    home_name / away_name : Full team names (used for fallback abbrev lookup).
    sport                 : 'MLB', 'WNBA', or 'NBA'.  Non-MLB returns neutral.
    home_abbr / away_abbr : Short abbreviations (preferred over name lookup).

    Returns
    -------
    RivalryFactor with edge_adjustment, tags, and factor_text populated.
    Zero-adjustment RivalryFactor on any error or unsupported sport.
    """
    if sport.upper() != "MLB":
        return RivalryFactor()
    try:
        return _compute(home_name, away_name, home_abbr, away_abbr)
    except Exception as exc:
        logger.debug(
            f"[rivalry] unexpected error ({home_name} vs {away_name}): {exc}"
        )
        return RivalryFactor()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute(
    home_name: str,
    away_name: str,
    home_abbr: str,
    away_abbr: str,
) -> RivalryFactor:
    # Fallback abbreviation resolution from team name
    if not home_abbr:
        home_abbr = home_name[:3].upper()
    if not away_abbr:
        away_abbr = away_name[:3].upper()

    rivalry_level = _classify_rivalry(home_abbr, away_abbr)

    # ── Head-to-Head record (home team's perspective) ────────────────────────
    h2h = _get_h2h_record(home_abbr, away_abbr)

    # ── Season win rate (home team) ──────────────────────────────────────────
    season_rate = _get_season_win_rate(home_abbr)

    # ── Performance deviation ────────────────────────────────────────────────
    performance_tag = ""
    h2h_rate = h2h.win_rate   # None when < 3 H2H meetings this season
    if h2h_rate is not None and season_rate is not None:
        deviation = h2h_rate - season_rate
        if deviation <= -0.10:
            performance_tag = "MATCHUP_UNDERPERFORMANCE"
        elif deviation >= 0.10:
            performance_tag = "MATCHUP_OVERPERFORMANCE"

    # ── Rivalry volatility penalty ───────────────────────────────────────────
    base_adj: float = 0.0
    if rivalry_level == "MAJOR":
        base_adj = -3.0
    elif rivalry_level == "DIVISION":
        base_adj = -2.0
    elif rivalry_level == "MINOR":
        base_adj = -0.5

    # ── Underdog recovery ────────────────────────────────────────────────────
    # If the home team (perspective target) wins ≥ 40 % of their H2H meetings
    # AND their recent margin trend is positive, restore up to half the penalty.
    # Caps at +2.0 per spec, never net-positive relative to a zero-rivalry game.
    underdog_adj: float = 0.0
    if rivalry_level in ("MAJOR", "DIVISION") and h2h_rate is not None:
        if h2h_rate >= 0.40:
            margins = h2h.margins
            if len(margins) >= 4:
                earlier_avg = sum(margins[:-3]) / max(1, len(margins) - 3)
                recent_avg  = sum(margins[-3:]) / 3
                if recent_avg > earlier_avg:
                    underdog_adj = min(2.0, abs(base_adj) * 0.50)

    total_adj = round(max(-5.0, min(5.0, base_adj + underdog_adj)), 2)

    # ── Tags ─────────────────────────────────────────────────────────────────
    tags: list[str] = []
    if rivalry_level != "NONE":
        tags.append("RIVALRY DETECTED")
    if rivalry_level == "MAJOR":
        tags.append("HIGH-VARIANCE RIVALRY GAME")
    elif rivalry_level == "DIVISION":
        tags.append("HIGH-VARIANCE DIVISION GAME")
    if performance_tag:
        tags.append(performance_tag)
    if h2h_rate is not None and h2h_rate >= 0.60:
        tags.append("HEAD-TO-HEAD EDGE")

    # ── Factor text ───────────────────────────────────────────────────────────
    parts: list[str] = []
    if rivalry_level != "NONE":
        parts.append(f"{rivalry_level} rivalry ({away_abbr}@{home_abbr})")
    if h2h.total_games >= 3:
        avg_str = f" avg total {h2h.avg_total:.1f}" if h2h.avg_total else ""
        parts.append(f"H2H {h2h.wins}W-{h2h.losses}L{avg_str}")
    if performance_tag:
        parts.append(performance_tag.replace("_", " ").title())
    if total_adj != 0.0:
        parts.append(f"rivalry adj {total_adj:+.1f}")

    factor_text = " | ".join(parts)

    return RivalryFactor(
        rivalry_level   = rivalry_level,
        h2h_wins        = h2h.wins,
        h2h_losses      = h2h.losses,
        h2h_avg_total   = h2h.avg_total,
        h2h_avg_margin  = h2h.avg_margin,
        h2h_win_rate    = h2h_rate,
        season_win_rate = season_rate,
        performance_tag = performance_tag,
        edge_adjustment = total_adj,
        tags            = tags,
        factor_text     = factor_text,
    )
