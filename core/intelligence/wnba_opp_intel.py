"""
wnba_opp_intel.py — WNBA Opponent & Game-Environment Intelligence
=================================================================

Five multiplicative projection-adjustment layers for WNBA assists and
rebounds props.  All adjustments compound on top of the base per-minute
× projected-minutes projection in _wnba_prop_projection().

Layer 1 — Shooting Environment  (assists only)
    Assists require made baskets.  Games where both teams allow more
    scoring create a richer assist environment; stingy defences suppress it.
    Proxy: average of both teams' points-allowed vs WNBA league average.
    Multiplier range: [0.92, 1.08]

Layer 2 — Rebounding Environment  (rebounds only)
    Missed shots create rebounds.  An opponent that dominates the boards
    leaves fewer opportunities for our player.
    Proxy: opponent's average rebounds vs WNBA league average, inverted.
    Multiplier range: [0.92, 1.08]

Layer 3 — Blowout Risk  (all markets via minutes dampener)
    Large point spreads (>= 10) indicate meaningful garbage-time risk.
    Blowouts compress the playing time of starters — especially the
    favoured team's in the final quarter.  Applied conservatively (70 %
    of full dampener) to all players because we cannot always identify
    which team the player is on.
    Multiplier range: [0.92, 1.00]  — only reduces, never inflates.

Layer 4 — Teammate-Out Reallocation  (own team, both markets)
    When a teammate at the position that normally "owns" this stat is
    out/doubtful, the opportunity doesn't disappear — it gets redistributed
    to whoever is left on the floor. This is the single most-requested
    signal for these two props: a missing starting center pushes boards to
    the remaining bigs/wings; a missing point guard pushes assist chances
    to the remaining ball-handlers.
    Uses core.intelligence.lineup_intel.get_lineup_intel()'s structured
    injuries_detail (position + severity) for the player's OWN team.
    Multiplier range: [1.00, 1.08] — only ever redistributes upward, since
    we can't identify which specific teammate a given player will absorb
    minutes/touches from.

Layer 5 — Positional Matchup  (rebounds only, opponent team)
    A missing opposing frontcourt player means a smaller opponent lineup on
    the floor, which typically means fewer contested boards league-wide and
    more second-chance opportunities for whoever is rebounding against them.
    Uses the opponent's own injuries_detail the same way Layer 4 uses the
    player's own team's.
    Multiplier range: [1.00, 1.05].
    NOTE: the inverse case from the article ("center vs a dominant
    individual rebounder → downgrade") is NOT modeled here — that needs a
    per-opponent-player rebound-rate feed this codebase doesn't have. Rather
    than fake it off a team-level number, this layer only fires on the
    injury-driven upside case, which the data can actually support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("betting_bot")

# ── WNBA league averages (2025 season estimates) ──────────────────────────────
_LG_AVG_PTS_ALLOWED: float = 82.0   # points allowed per game, league mean
_LG_AVG_REB:         float = 33.0   # total rebounds per game, league mean
_LG_AVG_FG_PCT:      float = 0.445  # league-average team FG%, 2025 estimate

# ── Layer 4/5: teammate-out reallocation thresholds ───────────────────────────
# Only a genuinely-out-or-worse absence (doubtful/out/IR/suspended, severity
# >= 0.70 per lineup_intel._STATUS_SEVERITY) triggers reallocation — a
# "questionable" tag is too noisy/likely-to-play to redistribute opportunity on.
_REALLOC_SEVERITY_FLOOR: float = 0.70
_REALLOC_FRONTCOURT_POS: set[str] = {"C", "F", "PF"}
_REALLOC_BALLHANDLER_POS: set[str] = {"G", "PG"}

# ── Blowout thresholds ────────────────────────────────────────────────────────
# Two-tier model: moderate (spread 10–17) / heavy (spread ≥ 17).
# Multipliers target projected-minutes × 0.85 (moderate) / 0.70 (heavy) but
# are applied conservatively at 70 % weight since we can't always identify
# which team the player is on.
#   moderate: 1.0 − (1.0−0.85) × 0.70 = 0.895
#   heavy:    1.0 − (1.0−0.70) × 0.70 = 0.79
_BLOWOUT_MODERATE: float = 10.0   # |spread| ≥ this → "moderate" tier
_BLOWOUT_HEAVY:    float = 17.0   # |spread| ≥ this → "heavy" tier
_BLOWOUT_MULT: dict[str, float] = {
    "moderate": 0.895,
    "heavy":    0.790,
}

# ── Process-level cache: one ESPN call per team per engine run ────────────────
_TEAM_STATS_CACHE: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _espn_team_stats(team_abbr: str) -> list[dict]:
    """
    Fetch and flatten ESPN team statistics for a WNBA team.
    Results are cached for the lifetime of the process.
    """
    if team_abbr in _TEAM_STATS_CACHE:
        return _TEAM_STATS_CACHE[team_abbr]

    flat: list[dict] = []
    try:
        from core.data_fetcher import fetch_espn

        result = fetch_espn(f"basketball/wnba/teams/{team_abbr}/statistics")
        if result.ok and result.data:

            def _walk(node: object) -> None:
                if isinstance(node, list):
                    for item in node:
                        _walk(item)
                elif isinstance(node, dict):
                    if "name" in node and "value" in node:
                        flat.append({"name": node["name"], "value": node["value"]})
                    for v in node.values():
                        _walk(v)

            _walk(result.data)
    except Exception as exc:
        logger.debug(f"[wnba_opp_intel] ESPN stats failed for {team_abbr}: {exc}")

    _TEAM_STATS_CACHE[team_abbr] = flat
    return flat


def _stat(flat: list[dict], *aliases: str) -> float | None:
    """Extract the first matching stat from a flat ESPN stat list."""
    by_name = {s.get("name", ""): s.get("value") for s in flat}
    for alias in aliases:
        val = by_name.get(alias)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


# ── Odds API full team name → ESPN team abbreviation ─────────────────────────
# Keyword-based so minor name variants ("LA Sparks" vs "Los Angeles Sparks") still match.
#
# "golden state" and "washington" below used to map to "GOL" and "WAS" --
# neither is ESPN's actual slug (confirmed live 2026-07 against ESPN's
# /sports/basketball/wnba/teams listing: Golden State is "GS", Washington
# is "WSH"; the same drift data.game_logs._WNBA_TEAM_ID already flagged in
# its own comments for these two teams, just never fixed here). This map's
# output is used two ways, and both were silently broken by the wrong
# values: it's passed straight into fetch_espn() below (a 400, same
# failure mode as MLB's OAK/ATH), and it's used unmodified as the
# matchup-key token core.player_props._prop_matchup_key() compares against
# ESPN's own event data -- that comparison doesn't go through
# core.data_fetcher's URL-rewriting alias table at all, so it needed the
# correct value here directly, not just a URL-level patch.
_NAME_KEYWORDS: list[tuple[str, str]] = [
    ("atlanta",      "ATL"),
    ("chicago",      "CHI"),
    ("connecticut",  "CON"),
    ("dallas",       "DAL"),
    ("golden state", "GS"),
    ("indiana",      "IND"),
    ("las vegas",    "LV"),
    ("los angeles",  "LA"),
    ("minnesota",    "MIN"),
    ("new york",     "NY"),
    ("phoenix",      "PHX"),
    ("portland",     "POR"),
    ("san antonio",  "SA"),
    ("seattle",      "SEA"),
    ("washington",   "WSH"),
]


def _to_abbr(full_name: str) -> str | None:
    lower = full_name.lower()
    for keyword, abbr in _NAME_KEYWORDS:
        if keyword in lower:
            return abbr
    return None


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class WNBAOppIntel:
    """
    Multiplicative adjustments for a single WNBA player prop projection.

    shooting_mult      — Layer 1: game shooting-environment factor (assists)
    rebound_mult       — Layer 2: opponent rebounding-environment factor (rebounds)
    blowout_mult       — Layer 3: projected-minutes dampener (0.895 moderate / 0.79 heavy)
    blowout_level      — Tier classification: "none" | "moderate" | "heavy"
    reallocation_mult  — Layer 4: own-team teammate-out opportunity shift
    matchup_mult       — Layer 5: opponent-frontcourt-out positional matchup (rebounds only)
    reallocation_note  — human-readable note when Layer 4 fires (surfaced in pick factor text)
    matchup_note       — human-readable note when Layer 5 fires (surfaced in pick factor text)
    diag               — human-readable breakdown for logger.debug
    """
    shooting_mult:     float = 1.0
    rebound_mult:      float = 1.0
    blowout_mult:      float = 1.0
    blowout_level:     str   = "none"
    reallocation_mult: float = 1.0
    matchup_mult:      float = 1.0
    reallocation_note: str   = ""
    matchup_note:      str   = ""
    diag:              str   = ""


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def get_wnba_opp_intel(
    home_team: str,
    away_team: str,
    market: str,
    spread: float | None = None,
    own_team_abbr: str | None = None,
    opp_team_abbr: str | None = None,
    role_label: str | None = None,
) -> WNBAOppIntel:
    """
    Compute five multiplicative projection adjustments for a WNBA prop.

    Parameters
    ----------
    home_team : Full team name from the Odds API, e.g. "Los Angeles Sparks".
    away_team : Full team name from the Odds API, e.g. "Portland Fire".
    market    : "player_assists" or "player_rebounds".
    spread    : Home-team point spread (negative = home favoured).
                None disables the blowout layer.
    own_team_abbr : The prop player's own team abbreviation, if resolved
                (see core.player_props._wnba_resolve_player_team). None
                disables Layer 4 (teammate-out reallocation).
    opp_team_abbr : The opponent's team abbreviation. None disables Layer 5
                (positional matchup, rebounds only).
    role_label : Output of core.player_props._wnba_classify_role() for this
                player/market — needed so Layer 4 only boosts the players
                who'd plausibly absorb the vacated opportunity.
    """
    parts: list[str] = []
    shoot_mult   = 1.0
    reb_mult     = 1.0
    blowout_mult = 1.0
    realloc_mult = 1.0
    realloc_note = ""
    matchup_mult = 1.0
    matchup_note = ""

    home_abbr = _to_abbr(home_team)
    away_abbr = _to_abbr(away_team)

    try:
        # ── Layer 1: Shooting Environment → assist projection ──────────────
        # ESPN exposes each team's own offensive stats, not their defensive
        # pts-allowed.  We use each team's avgPoints as the shooting-environment
        # signal: a game between two high-scoring teams creates more made baskets
        # → more assist opportunities for ball-handlers on either side.
        # Combined avg vs league baseline drives the multiplier.
        if market == "player_assists":
            avg_pts_vals: list[float] = []
            fg_pct_vals:  list[float] = []
            for abbr in filter(None, [home_abbr, away_abbr]):
                flat = _espn_team_stats(abbr)
                v = _stat(flat, "avgPoints", "pointsPerGame", "pts")
                if v is not None:
                    avg_pts_vals.append(v)
                # Teammate shooting quality: assists require made shots, so a
                # team's own field-goal percentage is a more direct proxy for
                # "will this player's passes turn into assists" than points
                # scored (which conflates volume with accuracy).
                fg = _stat(flat, "fieldGoalPct", "fgPct", "FG%")
                if fg is not None:
                    # ESPN reports some of these as 0-100, others as 0-1.
                    fg_pct_vals.append(fg / 100.0 if fg > 1.0 else fg)

            if avg_pts_vals:
                combined_avg_pts = sum(avg_pts_vals) / len(avg_pts_vals)
                ratio  = combined_avg_pts / _LG_AVG_PTS_ALLOWED
                # +6 % max boost in a high-scoring game; −6 % when both teams struggle
                shoot_mult = round(1.0 + (ratio - 1.0) * 0.6, 3)
                shoot_mult = max(0.92, min(1.08, shoot_mult))
                parts.append(
                    f"combined_avg_pts={combined_avg_pts:.1f} "
                    f"(lg={_LG_AVG_PTS_ALLOWED:.0f}) shoot×{shoot_mult:.3f}"
                )
            else:
                parts.append("avg_pts=N/A shoot×1.0")

            if fg_pct_vals:
                combined_fg = sum(fg_pct_vals) / len(fg_pct_vals)
                fg_ratio = combined_fg / _LG_AVG_FG_PCT
                # Smaller ±4% band layered on top of shoot_mult — a poor-shooting
                # team turns fewer of a passer's assist opportunities into
                # actual assists, independent of how many shots get hoisted.
                fg_mult = round(1.0 + (fg_ratio - 1.0) * 0.5, 3)
                fg_mult = max(0.96, min(1.04, fg_mult))
                shoot_mult = round(max(0.90, min(1.10, shoot_mult * fg_mult)), 3)
                parts.append(
                    f"combined_fg_pct={combined_fg:.3f} "
                    f"(lg={_LG_AVG_FG_PCT:.3f}) teammate_shooting×{fg_mult:.3f}"
                )
            else:
                parts.append("fg_pct=N/A teammate_shooting×1.0")

        # ── Layer 2: Rebounding Environment → rebound projection ───────────
        if market == "player_rebounds":
            reb_vals: list[float] = []
            for abbr in filter(None, [home_abbr, away_abbr]):
                flat = _espn_team_stats(abbr)
                v = _stat(flat,
                          "avgRebounds",
                          "reboundsPerGame",
                          "reb")
                if v is not None:
                    reb_vals.append(v)

            if reb_vals:
                avg_reb = sum(reb_vals) / len(reb_vals)
                ratio   = avg_reb / _LG_AVG_REB
                # High combined rebounding → every board is more contested
                # Low combined rebounding → more available boards
                reb_mult = round(1.0 + (1.0 - ratio) * 0.5, 3)
                reb_mult = max(0.92, min(1.08, reb_mult))
                parts.append(
                    f"avg_reb={avg_reb:.1f} "
                    f"(lg={_LG_AVG_REB:.0f}) reb×{reb_mult:.3f}"
                )
            else:
                parts.append("avg_reb=N/A reb×1.0")

        # ── Layer 3: Blowout Risk → tiered minutes dampener ───────────────
        blowout_level = "none"
        if spread is not None:
            abs_sp = abs(spread)
            if abs_sp >= _BLOWOUT_HEAVY:
                blowout_level = "heavy"
                blowout_mult  = _BLOWOUT_MULT["heavy"]
                parts.append(
                    f"spread={spread:+.1f} (≥{_BLOWOUT_HEAVY:.0f}) "
                    f"HEAVY blowout×{blowout_mult:.3f}"
                )
            elif abs_sp >= _BLOWOUT_MODERATE:
                blowout_level = "moderate"
                blowout_mult  = _BLOWOUT_MULT["moderate"]
                parts.append(
                    f"spread={spread:+.1f} (≥{_BLOWOUT_MODERATE:.0f}) "
                    f"MODERATE blowout×{blowout_mult:.3f}"
                )
            else:
                parts.append(f"spread={spread:+.1f} no blowout risk")
        else:
            parts.append("spread=None blowout skipped")

        # ── Layer 4: Teammate-Out Reallocation → own-team injury/lineup ────
        if own_team_abbr and role_label:
            try:
                from core.intelligence.lineup_intel import get_lineup_intel
                own_intel = get_lineup_intel(own_team_abbr, "WNBA", bet_on_this_team=True)
                out_positions = {
                    d["position"] for d in own_intel.injuries_detail
                    if d.get("severity", 0.0) >= _REALLOC_SEVERITY_FLOOR
                }
                if market == "player_rebounds" and out_positions & _REALLOC_FRONTCOURT_POS:
                    if role_label == "frontcourt":
                        realloc_mult = 1.08
                    elif role_label == "wing":
                        realloc_mult = 1.05
                    # guards: opportunity doesn't meaningfully redistribute to them
                    if realloc_mult > 1.0:
                        realloc_note = (
                            f"{own_team_abbr} missing frontcourt teammate "
                            f"({', '.join(sorted(out_positions & _REALLOC_FRONTCOURT_POS))}) "
                            f"— rebound reallocation ×{realloc_mult:.2f}"
                        )
                elif market == "player_assists" and out_positions & _REALLOC_BALLHANDLER_POS:
                    if role_label == "ballhandler":
                        realloc_mult = 1.04  # already the primary creator; smaller marginal bump
                    elif role_label == "secondary":
                        realloc_mult = 1.07
                    elif role_label == "off_ball":
                        realloc_mult = 1.02
                    if realloc_mult > 1.0:
                        realloc_note = (
                            f"{own_team_abbr} missing ball-handler "
                            f"({', '.join(sorted(out_positions & _REALLOC_BALLHANDLER_POS))}) "
                            f"— assist reallocation ×{realloc_mult:.2f}"
                        )
                if realloc_note:
                    parts.append(realloc_note)
                else:
                    parts.append(f"{own_team_abbr}: no reallocation-triggering absence")
            except Exception as exc:
                logger.debug(f"[wnba_opp_intel] Layer4 reallocation failed for {own_team_abbr}: {exc}")
        else:
            parts.append("reallocation skipped (own_team_abbr/role_label not provided)")

        # ── Layer 5: Positional Matchup → opponent frontcourt absence ──────
        if market == "player_rebounds" and opp_team_abbr:
            try:
                from core.intelligence.lineup_intel import get_lineup_intel
                opp_intel = get_lineup_intel(opp_team_abbr, "WNBA", bet_on_this_team=False)
                opp_out_positions = {
                    d["position"] for d in opp_intel.injuries_detail
                    if d.get("severity", 0.0) >= _REALLOC_SEVERITY_FLOOR
                }
                if opp_out_positions & _REALLOC_FRONTCOURT_POS:
                    matchup_mult = 1.05
                    matchup_note = (
                        f"{opp_team_abbr} missing frontcourt "
                        f"({', '.join(sorted(opp_out_positions & _REALLOC_FRONTCOURT_POS))}) "
                        f"— smaller opposing lineup, matchup×{matchup_mult:.2f}"
                    )
                    parts.append(matchup_note)
                else:
                    parts.append(f"{opp_team_abbr}: no favorable frontcourt matchup")
            except Exception as exc:
                logger.debug(f"[wnba_opp_intel] Layer5 matchup failed for {opp_team_abbr}: {exc}")
        elif market == "player_rebounds":
            parts.append("matchup skipped (opp_team_abbr not provided)")

    except Exception as exc:
        logger.debug(f"[wnba_opp_intel] error ({home_team} vs {away_team}): {exc}")

    return WNBAOppIntel(
        shooting_mult     = shoot_mult,
        rebound_mult      = reb_mult,
        blowout_mult      = blowout_mult,
        blowout_level     = blowout_level,
        reallocation_mult = realloc_mult,
        matchup_mult      = matchup_mult,
        reallocation_note = realloc_note,
        matchup_note      = matchup_note,
        diag              = " | ".join(parts),
    )
