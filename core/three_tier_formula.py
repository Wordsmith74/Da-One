"""
core/three_tier_formula.py

Three-tier pregame matchup classifier, built 2026-08-29 per explicit
request, on real data only -- see the module-level notes on each rule for
exactly what real source backs it and where a named metric from the
original spec wasn't available and had to be honestly substituted.

Output is SHADOW-ONLY: see core/market_gate.py and core/market_governance.py
-- moneyline is in ALLOWED_MARKETS (so candidates get generated, priced,
and logged) but NOT in PUBLICATION_MARKETS (so nothing here ever reaches
a published pick) until it clears the same walk-forward validation gate
every other market in this pipeline had to clear. MLB moneyline's last
graded real-money record was 35.3% win / -0.52 units over n=34 (see
core/market_governance.py's 2026-08-05 scope note) -- this formula
replaces the approach that produced that record, but "replaces" is not
"proven better than" until there's graded evidence for THIS formula
specifically.

--------------------------------------------------------------------------
Rule 1 -- Pitch Quality Supremacy (real data: Savant K%/BB%)
--------------------------------------------------------------------------
Spec named "Stuff+" and "K-BB%". Stuff+ (FanGraphs/PitchingBot proprietary
pitch-shape grading) has no free data source anywhere in this pipeline --
not built here; fabricating a same-named substitute would reintroduce the
exact "confident number from data that isn't what it claims to be"
problem this codebase's audit history is full of. K-BB% IS real and
buildable: data.fetch.get_savant_pitcher_advanced_stats() now pulls real
K% and BB% from Baseball Savant's public leaderboard (bb_percent column
added alongside the existing k_percent for this).

"Elite tier" / "standard weakness" thresholds below are fixed real-world
sabermetric benchmarks (approx. top/bottom quartile of full-season
starter K-BB%, per public MLB leaderboards), not a live-computed
percentile against today's actual starter pool -- computing a true daily
percentile would need every active starter's current K-BB% fetched and
ranked each run, which isn't wired in. Documented here rather than
silently treated as more precise than it is.

--------------------------------------------------------------------------
Rule 2 -- Bullpen Integrity (real data: MLB Stats API game logs)
--------------------------------------------------------------------------
core.intelligence.bullpen_intel.get_bullpen_freshness_tags() -- added
alongside this module -- returns real per-reliever appearance and pitch
counts over a rolling 72-hour window for a team's top 3 highest-usage
relief arms, and flags an arm "pulled" if it worked back-to-back-to-back
days or crossed a real-bullpen pitch-load caution threshold. Real data,
no fabrication; the only substitution from the original spec is that
"total pitch counts" is tracked per-arm from real game logs rather than
from a live pitch-count feed (MLB Stats API doesn't expose warm-up/
bullpen-session pitch counts, only in-game ones -- in-game pitches are
what's used here).

--------------------------------------------------------------------------
Rule 3 -- Platoon Suppression (real data: Statcast OBP/ISO, wRC+ proxy)
--------------------------------------------------------------------------
Spec named "WAA" (Wins Above Average) and "wOBA". Neither is published
anywhere this pipeline can reach -- WAA isn't available from the MLB
Stats API or Baseball Savant's public endpoints at all (it's a
Baseball-Reference-specific stat); true wOBA needs event-level linear
weights this pipeline doesn't compute. Not fabricated. Substituted with
data.statcast_lineup_platoon.get_top4_platoon_splits() -- real top-4-
batters-vs-pitcher-handedness OBP/ISO, already used for NRFI/YRFI in
core/game_markets.py, and that module's own docstring is explicit that
its wRC+ figure is "intentionally simple and clearly a proxy, not real
park/league-adjusted wRC+." Same honesty carried forward here rather
than re-labeled as WAA/wOBA to match the spec more closely than the data
actually supports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

# Rule 1 thresholds -- fixed benchmarks, not a live daily percentile (see
# module docstring). Roughly: elite = top-quartile full-season starter
# K-BB%; standard/weak = bottom-quartile.
_ELITE_K_BB = 0.20     # K-BB% >= 20% -> elite tier
_WEAK_K_BB = 0.11      # K-BB% <= 11% -> standard/below-average tier


@dataclass
class Rule1Result:
    elite_side: str | None = None       # "home", "away", or None
    home_k_bb: float | None = None
    away_k_bb: float | None = None
    cleared: bool = False               # True iff one side is elite AND the other is weak
    reason: str = ""


@dataclass
class Rule2Result:
    home_clean: bool | None = None
    away_clean: bool | None = None
    cleared: bool = False               # True iff the selected side's pen is clean
    reason: str = ""


@dataclass
class Rule3Result:
    suppressed_side: str | None = None  # which lineup is suppressed (the OPPONENT's)
    home_top4_wrc: float | None = None
    away_top4_wrc: float | None = None
    cleared: bool = False
    reason: str = ""


@dataclass
class MatchupClassification:
    status: Literal["advantage", "no_edge"]
    team: str | None = None             # set iff status == "advantage"
    rule1: Rule1Result = field(default_factory=Rule1Result)
    rule2: Rule2Result = field(default_factory=Rule2Result)
    rule3: Rule3Result = field(default_factory=Rule3Result)
    reason: str = ""


def evaluate_rule1_pitch_quality(
    home_pitcher_name: str, away_pitcher_name: str, season: int,
) -> Rule1Result:
    """Real Savant K%/BB% for both starters -> K-BB% -> elite vs. weak tiering."""
    from data.fetch import get_savant_pitcher_advanced_stats

    def _k_bb(name: str) -> float | None:
        try:
            row = get_savant_pitcher_advanced_stats(name, season)
        except Exception:
            return None
        if row is None or "K%" not in row.columns or "BB%" not in row.columns:
            return None
        k, bb = row["K%"].iloc[0], row["BB%"].iloc[0]
        if k is None or bb is None:
            return None
        return round(float(k) - float(bb), 4)

    home_k_bb = _k_bb(home_pitcher_name)
    away_k_bb = _k_bb(away_pitcher_name)

    if home_k_bb is None or away_k_bb is None:
        return Rule1Result(
            home_k_bb=home_k_bb, away_k_bb=away_k_bb,
            reason="Missing real K-BB% data for one or both starters -- cannot clear Rule 1.",
        )

    if home_k_bb >= _ELITE_K_BB and away_k_bb <= _WEAK_K_BB:
        return Rule1Result(
            elite_side="home", home_k_bb=home_k_bb, away_k_bb=away_k_bb, cleared=True,
            reason=f"Home K-BB% {home_k_bb:.1%} (elite) vs away {away_k_bb:.1%} (weak).",
        )
    if away_k_bb >= _ELITE_K_BB and home_k_bb <= _WEAK_K_BB:
        return Rule1Result(
            elite_side="away", home_k_bb=home_k_bb, away_k_bb=away_k_bb, cleared=True,
            reason=f"Away K-BB% {away_k_bb:.1%} (elite) vs home {home_k_bb:.1%} (weak).",
        )
    return Rule1Result(
        home_k_bb=home_k_bb, away_k_bb=away_k_bb,
        reason=f"No elite/weak split: home {home_k_bb:.1%}, away {away_k_bb:.1%}.",
    )


def evaluate_rule2_bullpen(
    home_abbr: str, away_abbr: str, game_date: date, candidate_side: str,
) -> Rule2Result:
    """
    Real per-reliever pitch/appearance data for the CANDIDATE side's
    bullpen specifically (Rule 2 only requires the selected team's pen be
    clean, per the spec -- it doesn't require the opponent's pen to be
    fatigued).
    """
    from core.intelligence.bullpen_intel import get_bullpen_freshness_tags

    try:
        home_tags = get_bullpen_freshness_tags(home_abbr, game_date)
        away_tags = get_bullpen_freshness_tags(away_abbr, game_date)
    except Exception as exc:
        return Rule2Result(reason=f"Bullpen freshness lookup failed: {exc}")

    home_clean, away_clean = home_tags.get("clean"), away_tags.get("clean")
    selected_clean = home_clean if candidate_side == "home" else away_clean

    if selected_clean is None:
        return Rule2Result(
            home_clean=home_clean, away_clean=away_clean,
            reason=f"No real bullpen data for {candidate_side} side -- cannot clear Rule 2.",
        )
    if selected_clean:
        return Rule2Result(
            home_clean=home_clean, away_clean=away_clean, cleared=True,
            reason=f"{candidate_side.title()} bullpen's top 3 arms are clean (no fatigue tags).",
        )
    return Rule2Result(
        home_clean=home_clean, away_clean=away_clean,
        reason=f"{candidate_side.title()} bullpen has a fatigued top-3 arm -- No Edge.",
    )


def evaluate_rule3_platoon(
    candidate_side: str,
    opposing_pitcher_throws: str | None,
    opposing_lineup_batter_ids: list[int],
    season: int,
) -> Rule3Result:
    """
    Real top-4-batters-vs-handedness OBP/ISO -> wRC+ proxy for the
    OPPONENT's lineup against the candidate side's own starter's throwing
    hand. Suppression = opponent's top-4 wRC+ proxy below league-average
    (100).
    """
    from data.statcast_lineup_platoon import get_top4_platoon_splits

    try:
        splits = get_top4_platoon_splits(
            opposing_lineup_batter_ids, opposing_pitcher_throws, season=season,
        )
    except Exception as exc:
        return Rule3Result(reason=f"Platoon split lookup failed: {exc}")

    wrc = splits.top4_wrc_plus_vs_hand
    if wrc is None or splits.batters_with_data == 0:
        return Rule3Result(reason="No real platoon data for opposing lineup -- cannot clear Rule 3.")

    other_side = "away" if candidate_side == "home" else "home"
    if candidate_side == "home":
        home_wrc, away_wrc = None, wrc
    else:
        home_wrc, away_wrc = wrc, None

    if wrc < 100.0:
        return Rule3Result(
            suppressed_side=other_side, home_top4_wrc=home_wrc, away_top4_wrc=away_wrc,
            cleared=True,
            reason=f"Opposing top-4 wRC+ proxy {wrc:.0f} (below league-average 100) vs. "
                   f"{candidate_side}'s starter.",
        )
    return Rule3Result(
        suppressed_side=None, home_top4_wrc=home_wrc, away_top4_wrc=away_wrc,
        reason=f"Opposing top-4 wRC+ proxy {wrc:.0f} -- not suppressed (>= league-average).",
    )


def classify_matchup(
    home_abbr: str, away_abbr: str,
    home_pitcher_name: str, away_pitcher_name: str,
    game_date: date, season: int,
    home_pitcher_throws: str | None, away_pitcher_throws: str | None,
    home_lineup_batter_ids: list[int], away_lineup_batter_ids: list[int],
) -> MatchupClassification:
    """
    Runs all three rules for whichever side Rule 1 flags as the
    pitch-quality-elite side (there's no "advantage" to evaluate for the
    other side once Rule 1 has already picked a direction -- Rule 1 is
    the entry gate, Rules 2/3 either confirm or deny it).

    Returns "no_edge" whenever any rule fails to clear, INCLUDING when a
    rule can't get real data -- consistent with this pipeline's "real
    data or no pick" policy elsewhere (see core/odds_client.py,
    core/player_props.py). Never guesses.
    """
    r1 = evaluate_rule1_pitch_quality(home_pitcher_name, away_pitcher_name, season)
    if not r1.cleared:
        return MatchupClassification(status="no_edge", rule1=r1, reason=r1.reason)

    side = r1.elite_side  # "home" or "away" -- the side with the elite starter
    r2 = evaluate_rule2_bullpen(home_abbr, away_abbr, game_date, side)
    if not r2.cleared:
        return MatchupClassification(status="no_edge", rule1=r1, rule2=r2, reason=r2.reason)

    opposing_throws = away_pitcher_throws if side == "home" else home_pitcher_throws
    opposing_lineup = away_lineup_batter_ids if side == "home" else home_lineup_batter_ids
    r3 = evaluate_rule3_platoon(side, opposing_throws, opposing_lineup, season)
    if not r3.cleared:
        return MatchupClassification(status="no_edge", rule1=r1, rule2=r2, rule3=r3, reason=r3.reason)

    team_abbr = home_abbr if side == "home" else away_abbr
    return MatchupClassification(
        status="advantage", team=team_abbr, rule1=r1, rule2=r2, rule3=r3,
        reason=f"Cleared all 3 rules: {r1.reason} {r2.reason} {r3.reason}",
    )
