"""
models/nrfi_handicapper.py

NRFI/YRFI (No/Yes Run First Inning) tiered handicapping framework --
translates NRFI_YRFI_F5_Elite_Handicapping_Reference.md into code that
plugs into the existing MLB pipeline the same way models/handicapper_rules.py
encodes general handicapper discipline and models/bayesian.py encodes
shrinkage. Consumed by core/game_markets.py (candidate generation) and
models/monte_carlo.py (simulate_nrfi_game / nrfi_edge_with_uncertainty).

Design principle (per the doc's own "Baseline Rates" section): first-inning
scoring is dominated by the league base rate -- roughly 72-75% of first
innings are scoreless. Every tier below nudges that base rate with a
DAMPENED, WEIGHTED multiplier rather than replacing it with an independent
estimate. A thin, noisy signal (small first-inning sample, one Statcast
number) should never be allowed to swamp the anchor -- that's the doc's
explicit warning under "Common Pitfalls": large deviations from the prior
need a real causal reason, not noise.

Four tiers, in the doc's own stated priority order:
  Tier 1 -- Starting pitcher first-inning quality      (heaviest weight)
  Tier 2 -- Opposing lineup top-of-order quality        (2nd heaviest)
  Tier 3 -- Park / weather / umpire environment         (moderate weight)
  Tier 4 -- Betting market intelligence (CLV, RLM)       (diagnostic ONLY --
            never modifies the projected lambda, exactly like
            handicapper_rules.public_fade_signal(), for the same reason:
            "public fade alone isn't a strategy, it's a slogan")

Reliability gatekeeper -- mirrors core/stability_filter.py's role in the
existing pipeline: a game doesn't reach a tradeable NRFI/YRFI edge just
because the tiers produced a number. Debut starts, injury-return starts,
and opener/bullpen games are excluded regardless of what the tiers say
(doc: "Reliability filter -- only trust first-inning splits for pitchers
with a meaningful sample (~50+ career starts)").

HONEST DATA-SOURCE NOTE (same contract-first pattern as
models/handicapper_rules.umpire_zone_factor / compare_to_consensus):
This repo has no first-inning-specific splits feed wired in anywhere --
the MLB Stats API linescore/boxscore endpoints used elsewhere in this repo
(core/mlb.py) give innings-level RUNS only, not batter-level first-inning
splits (FBF OBP, first-inning ERA/K%/BB%), and there's no Statcast/umpire
zone-history API contract in this codebase either (see
umpire_zone_factor's own note on why). Every tier function below therefore
accepts its real inputs as parameters and returns a NEUTRAL multiplier
(1.0) when they're not supplied, exactly like umpire_zone_factor returns
a neutral factor with an explicit "no_data" reason. Do NOT fabricate these
inputs from season-long stats to make a tier "work" -- season ERA as a
stand-in for first-inning ERA is precisely the pitfall the reference doc
warns against ("Season-long ERA/WHIP as your primary pitcher input").
Until a real first-inning splits source is wired in, candidate generation
(core/game_markets.py) runs on the league baseline lambda plus whatever
adjustment already-wired real data (team run-scoring history, pitcher
workload) can honestly support -- see get_pitcher_workload_first_inning_note()
below for why that data is NOT substituted in as a tier input either.
"""
from __future__ import annotations

from typing import Optional

from models.sport_config import MLB

NRFI_LEAGUE_SCORELESS_PCT = MLB["nrfi_league_scoreless_pct"]
NRFI_COMBINED_LAMBDA_BASELINE = MLB["nrfi_combined_lambda_baseline"]
MIN_CAREER_STARTS_FOR_RELIABILITY = MLB["min_career_starts_for_nrfi_reliability"]
TIER_WEIGHT_PITCHER = MLB["nrfi_tier_weight_pitcher"]
TIER_WEIGHT_LINEUP = MLB["nrfi_tier_weight_lineup"]
TIER_WEIGHT_ENVIRONMENT = MLB["nrfi_tier_weight_environment"]
NRFI_PARK_SCALE = MLB["nrfi_park_scale"]

# Multiplier band -- keeps any single tier (or their combination) from
# overriding the league prior on thin signal. A pitcher/lineup/environment
# read this extreme should show up as a real, sourced, cross-checked edge,
# not a single noisy input.
_MULT_FLOOR = 0.45
_MULT_CEIL = 2.20


# ---------------------------------------------------------------------------
# Reliability gatekeeper
# ---------------------------------------------------------------------------

def nrfi_reliability_gate(
    career_starts: Optional[int],
    is_debut: bool = False,
    is_injury_return_start: bool = False,
    is_opener_or_bullpen_game: bool = False,
    min_starts: int = MIN_CAREER_STARTS_FOR_RELIABILITY,
) -> tuple[bool, str]:
    """
    Returns (passed, reason). Mirrors core/stability_filter.py's role:
    gates a game OUT of a tradeable NRFI/YRFI edge regardless of what the
    tier scoring below produces, per the doc's reliability filter (50+
    career starts; exclude debuts/injury-returns/openers -- "splits are
    unstable on small samples").

    When career_starts is None (data not available), the gate FAILS closed
    (not passes silently) -- an unknown sample size is treated the same as
    an insufficient one, since the alternative (assuming reliable) is
    exactly the kind of silent confidence-inflation this gate exists to
    prevent.
    """
    if is_debut:
        return False, "season_debut_start"
    if is_injury_return_start:
        return False, "injury_return_start"
    if is_opener_or_bullpen_game:
        return False, "opener_or_bullpen_game"
    if career_starts is None:
        return False, "career_starts_unknown"
    if career_starts < min_starts:
        return False, f"career_starts_below_minimum({career_starts}<{min_starts})"
    return True, "reliable"


# ---------------------------------------------------------------------------
# Tier 1 -- Starting pitcher first-inning quality
# ---------------------------------------------------------------------------

def pitcher_first_inning_tier_multiplier(
    fbf_obp: Optional[float] = None,
    first_inning_era: Optional[float] = None,
    first_inning_k_pct: Optional[float] = None,
    first_inning_bb_pct: Optional[float] = None,
    league_fbf_obp: float = 0.320,
    league_first_inning_era: float = 4.10,
    league_first_inning_k_pct: float = 0.220,
    league_first_inning_bb_pct: float = 0.085,
) -> float:
    """
    Doc: "First-Batter-Faced (FBF) OBP -- the single highest-correlation
    event to a YRFI is the leadoff hitter reaching base" and "K%/BB% (first
    time through the order) -- use rate stats, not K/9."

    Returns a multiplier on ONE team's (the opponent batting against this
    pitcher's) expected first-inning lambda. >1.0 = pitcher projects worse
    than league norm for this tier (more YRFI lean); <1.0 = better (more
    NRFI lean). Each component is optional and skipped (not zeroed) when
    unavailable, so a partial read doesn't get diluted toward neutral by
    missing data -- the weighting renormalizes over whatever IS supplied.

    All four inputs are first-inning/first-time-through-order SPECIFIC
    stats. Do not pass season-long ERA/K%/BB% here -- see module docstring.
    """
    components: list[tuple[float, float]] = []  # (ratio, component_weight)

    if fbf_obp is not None and league_fbf_obp > 0:
        components.append((fbf_obp / league_fbf_obp, 0.40))
    if first_inning_era is not None and league_first_inning_era > 0:
        components.append((first_inning_era / league_first_inning_era, 0.30))
    if first_inning_k_pct is not None and first_inning_k_pct > 0:
        # Higher K% suppresses scoring -- invert so it moves the multiplier
        # the same direction (down) as the other "worse for offense" ratios.
        components.append((league_first_inning_k_pct / first_inning_k_pct, 0.15))
    if first_inning_bb_pct is not None and league_first_inning_bb_pct > 0:
        components.append((first_inning_bb_pct / league_first_inning_bb_pct, 0.15))

    if not components:
        return 1.0

    total_weight = sum(w for _, w in components)
    raw_mult = sum(r * w for r, w in components) / total_weight
    return max(_MULT_FLOOR, min(_MULT_CEIL, raw_mult))


# ---------------------------------------------------------------------------
# Tier 2 -- Opposing lineup top-of-order ("Top 4") quality
# ---------------------------------------------------------------------------

def lineup_top4_tier_multiplier(
    top4_wrc_plus_vs_hand: Optional[float] = None,
    top4_obp_vs_hand: Optional[float] = None,
    top4_iso_vs_hand: Optional[float] = None,
    missing_key_bat: bool = False,
    league_wrc_plus: float = 100.0,
    league_obp: float = 0.320,
    league_iso: float = 0.150,
) -> float:
    """
    Doc: "For NRFI/YRFI you effectively only care about hitters 1-4 in the
    order ... Use OBP, ISO, and wRC+ specifically vs. the starter's
    handedness (platoon splits, not season totals)" and "A missing 1- or
    2-hole hitter meaningfully lowers early-scoring probability."

    Returns a multiplier on the batting team's expected first-inning
    lambda. All three rate inputs must already be platoon-split (vs. the
    opposing starter's handedness) by the caller -- this function doesn't
    know handedness, it just combines whatever platoon-specific numbers
    it's given.
    """
    components: list[tuple[float, float]] = []

    if top4_wrc_plus_vs_hand is not None and league_wrc_plus > 0:
        components.append((top4_wrc_plus_vs_hand / league_wrc_plus, 0.45))
    if top4_obp_vs_hand is not None and league_obp > 0:
        components.append((top4_obp_vs_hand / league_obp, 0.35))
    if top4_iso_vs_hand is not None and league_iso > 0:
        components.append((top4_iso_vs_hand / league_iso, 0.20))

    if not components:
        raw_mult = 1.0
    else:
        total_weight = sum(w for _, w in components)
        raw_mult = sum(r * w for r, w in components) / total_weight

    if missing_key_bat:
        # Doc: fastest way to find value not yet priced by the market --
        # applied as a flat discount on top of the platoon-stat read, not a
        # replacement for it (a weak lineup missing its best bat is still
        # weaker than a strong lineup missing its best bat).
        raw_mult *= 0.85

    return max(_MULT_FLOOR, min(_MULT_CEIL, raw_mult))


# ---------------------------------------------------------------------------
# Tier 3 -- Environment (park / weather / umpire)
# ---------------------------------------------------------------------------

def environment_tier_multiplier(
    park_run_factor: float = 1.0,
    wind_out_mph: float = 0.0,
    umpire_zone_size: Optional[str] = None,   # "wide" | "narrow" | "average"
    umpire_volatility: Optional[str] = None,  # "high" | "average" | "low"
    park_scale: float = NRFI_PARK_SCALE,
) -> float:
    """
    Doc, section 10: park factor's early-game slice, wind at first pitch,
    and umpire zone size/volatility as "force multipliers" that often
    outweigh the pitching matchup itself.

    park_run_factor: full-game park run factor (1.0 = neutral, e.g. Coors
        ~1.25, a true pitcher's park ~0.90) -- dampened by park_scale since
        only a fraction of a full-game park effect plausibly applies to a
        single first inning (see sport_config.MLB['nrfi_park_scale']).
    wind_out_mph: positive = blowing out (raises scoring), negative = in.
        Doc: "at extreme parks, sustained wind blowing out can shift a
        total by 1.5-2 runs" -- scaled down heavily for a one-inning window.
    umpire_zone_size: "wide" upgrades pitchers (more called strikes) -> NRFI
        lean; "narrow" the opposite. "average"/None = no adjustment.
    umpire_volatility: "high" -> doc: inconsistent calls push deeper counts
        and more walks -> YRFI lean.
    """
    park_component = 1.0 + (park_run_factor - 1.0) * park_scale
    wind_component = 1.0 + (wind_out_mph / 10.0) * 0.05  # +10 mph sustained out -> +5%

    ump_component = 1.0
    if umpire_zone_size == "wide":
        ump_component *= 0.94
    elif umpire_zone_size == "narrow":
        ump_component *= 1.06

    if umpire_volatility == "high":
        ump_component *= 1.04
    elif umpire_volatility == "low":
        ump_component *= 0.98

    raw_mult = park_component * wind_component * ump_component
    return max(_MULT_FLOOR, min(_MULT_CEIL, raw_mult))


# ---------------------------------------------------------------------------
# Tier 4 -- Market intelligence (DIAGNOSTIC ONLY -- never touches lambda)
# ---------------------------------------------------------------------------

def market_signal_label(
    clv_pct: Optional[float] = None,
    reverse_line_movement_detected: bool = False,
) -> str:
    """
    DIAGNOSTIC ONLY, same philosophy and same contract as
    handicapper_rules.public_fade_signal(): never feeds into the lambda
    projection below. Doc: CLV is "the standard measure sharp bettors use
    to judge whether their process is sound independent of single-game
    outcomes"; reverse line movement is a signal to review, not a trigger.
    """
    if reverse_line_movement_detected:
        return "reverse_line_movement_flagged"
    if clv_pct is None:
        return "no_data"
    if clv_pct > 0:
        return "positive_clv_process_confirmed"
    if clv_pct < 0:
        return "negative_clv_review_process"
    return "flat_clv"


# ---------------------------------------------------------------------------
# Composite projection
# ---------------------------------------------------------------------------

def project_team_first_inning_lambda(
    league_prior_lambda: float,
    pitcher_mult: float = 1.0,
    lineup_mult: float = 1.0,
    environment_mult: float = 1.0,
) -> float:
    """
    Combine the three model-driven tiers (pitcher heaviest, then lineup,
    then environment -- weights from sport_config.MLB) into a single
    dampened multiplier applied to ONE team's league-prior first-inning
    lambda. Tier 4 (market) is intentionally excluded -- see
    market_signal_label()'s docstring.
    """
    combined_mult = (
        1.0
        + (pitcher_mult - 1.0) * TIER_WEIGHT_PITCHER
        + (lineup_mult - 1.0) * TIER_WEIGHT_LINEUP
        + (environment_mult - 1.0) * TIER_WEIGHT_ENVIRONMENT
    )
    combined_mult = max(_MULT_FLOOR, min(_MULT_CEIL, combined_mult))
    return max(0.02, league_prior_lambda * combined_mult)


def project_combined_first_inning_lambda(
    home_team_inputs: Optional[dict] = None,
    away_team_inputs: Optional[dict] = None,
    league_combined_lambda: float = NRFI_COMBINED_LAMBDA_BASELINE,
    home_reliability: Optional[tuple[bool, str]] = None,
    away_reliability: Optional[tuple[bool, str]] = None,
) -> dict:
    """
    Top-level entry point for core/game_markets.py.

    home_team_inputs / away_team_inputs: dicts of tier inputs governing that
    side's expected first-inning RUNS -- i.e. home_team_inputs["pitcher"]
    describes the AWAY starter (the pitcher the home lineup faces) and
    home_team_inputs["lineup"] describes the HOME top-4 hitters (vs. that
    away starter's handedness); away_team_inputs is the mirror image. Keys
    match the kwargs of pitcher_first_inning_tier_multiplier /
    lineup_top4_tier_multiplier / environment_tier_multiplier, grouped
    under "pitcher", "lineup", "environment" sub-dicts. Any missing/absent
    sub-dict defaults every component to neutral (1.0) via that tier
    function's own defaults.

    home_reliability / away_reliability: (passed, reason) tuples from
    nrfi_reliability_gate(), evaluated against the OPPOSING starter (the
    same pitcher referenced by that side's "pitcher" sub-dict above). When
    a side fails the gate, its pitcher-tier multiplier is dropped back to
    neutral (1.0) -- the reliability filter downgrades the READ, it doesn't
    zero out the team's baseline projection.

    Returns a dict with home_lambda, away_lambda, combined_lambda,
    per-side tier multipliers, and reliability pass/fail -- everything
    core/game_markets.py needs to hand to
    models.monte_carlo.simulate_nrfi_game / nrfi_edge_with_uncertainty.
    """
    home_in = home_team_inputs or {}
    away_in = away_team_inputs or {}
    league_prior_per_team = league_combined_lambda / 2.0

    def _side(inputs: dict, reliability: Optional[tuple[bool, str]]) -> dict:
        passed = True if reliability is None else reliability[0]
        reason = "not_evaluated" if reliability is None else reliability[1]

        p_mult = pitcher_first_inning_tier_multiplier(**inputs.get("pitcher", {}))
        l_mult = lineup_top4_tier_multiplier(**inputs.get("lineup", {}))
        e_mult = environment_tier_multiplier(**inputs.get("environment", {}))

        if not passed:
            # Reliability gate failed -- downgrade the pitcher tier (the
            # tier the gate is actually about) back to neutral rather than
            # trusting a small/unstable sample. Lineup and environment
            # reads aren't pitcher-sample-size-dependent, so they stand.
            p_mult = 1.0

        lam = project_team_first_inning_lambda(
            league_prior_per_team, pitcher_mult=p_mult, lineup_mult=l_mult, environment_mult=e_mult,
        )
        return {
            "lambda": lam, "pitcher_mult": p_mult, "lineup_mult": l_mult,
            "environment_mult": e_mult, "reliability_passed": passed, "reliability_reason": reason,
        }

    home_side = _side(home_in, home_reliability)
    away_side = _side(away_in, away_reliability)

    return {
        "home_lambda": home_side["lambda"],
        "away_lambda": away_side["lambda"],
        "combined_lambda": home_side["lambda"] + away_side["lambda"],
        "home_tiers": home_side,
        "away_tiers": away_side,
    }
