"""
MLB-ONLY advanced metric blending.

There is no WNBA equivalent of CSW%/SIERA/park-factor in this file on purpose:
basketball doesn't have a "park," and the analogous basketball advanced stats
(true shooting %, usage rate, defensive matchup difficulty) are different
enough mechanically that they live in models/wnba_advanced.py instead of being
forced into this file's shape. Do not import this module for WNBA processing.
"""
from models.sport_config import MLB


def project_k_pct_advanced(csw_pct, swstr_pct, raw_k_pct):
    """
    Blends CSW% (called+swinging strike %) and SwStr% with the shrunk raw K%
    to get a more stable strikeout-rate projection. CSW%/SwStr% are *process*
    stats (how often a pitcher generates whiffs/called strikes per pitch) and
    are more predictive going forward than recent K% outcomes alone, which can
    be inflated/deflated by sequencing luck.

    Falls back gracefully to raw_k_pct alone if advanced columns are missing
    (e.g. pybaseball lookup failed or name match was empty) -- this MUST NOT
    raise, since run_pipeline.py depends on this never crashing a live run.
    """
    if csw_pct is None and swstr_pct is None:
        return raw_k_pct

    weights = []
    values = []
    if csw_pct is not None:
        # CSW% correlates strongly with K% league-wide; treat it as a strong signal
        values.append(csw_pct * 1.05)
        weights.append(0.45)
    if swstr_pct is not None:
        # SwStr% alone underestimates total K% (doesn't count called third strikes)
        values.append(swstr_pct * 1.85)
        weights.append(0.25)

    values.append(raw_k_pct)
    weights.append(1.0 - sum(weights))  # remaining weight on the shrunk raw rate

    blended = sum(v * w for v, w in zip(values, weights))
    return max(0.0, min(1.0, blended))


def pitcher_quality_factor(siera):
    """
    Converts a SIERA into a multiplicative run-environment factor centered at 1.0
    (league-average SIERA ~= 4.00). Lower SIERA -> tougher pitcher -> factor < 1
    (suppresses runs scored against them); higher SIERA -> factor > 1.
    Missing SIERA falls back to neutral 1.0 rather than crashing.
    """
    if siera is None:
        return 1.0
    league_avg_siera = 4.00
    # Cap the swing so one extreme SIERA value (small sample) can't blow up the sim
    factor = 1.0 + (siera - league_avg_siera) * 0.06
    return max(0.7, min(1.3, factor))


def f5_park_factor(full_game_park_factor):
    """
    Scales a full-game park factor down for First-5-Innings use. Starters
    typically face the order ~2x in 5 innings vs ~3x in 9, and bullpen/late-game
    park effects (twilight, wind shifts) shouldn't bleed into an F5 number.
    Scale factor pulled from sport_config.MLB so it's not a silent magic number.
    """
    scale = MLB["f5_park_scale"]
    # Move the factor toward 1.0 (neutral) by (1 - scale), i.e. partially apply it
    return 1.0 + (full_game_park_factor - 1.0) * scale


# ---------------------------------------------------------------------------
# Static MLB park factors -- replaces the FanGraphs Guts-table scrape.
#
# WHY: data.fetch.get_park_factors() pulled FanGraphs' Guts page via
# pandas.read_html() and matched it by substring-searching the canonical
# park name (from data/name_registry.py) against column 0. Every park was
# missing a match -- not a handful of edge cases -- which means the join
# key itself was wrong (FanGraphs likely labels rows by team, not stadium
# name), not that a few spellings needed a tweak.
#
# FIX: same approach the other (rate-limit-free) engine's venue_intel.py
# uses for the identical problem -- a hardcoded, manually-maintained dict.
# No live dependency, no fuzzy join. pybaseball/lxml are no longer needed
# for the park-factor path at all (get_savant_pitcher_advanced_stats()'s
# CSW%/SIERA pull is unaffected and still uses pybaseball).
#
# Run factor, 100 = league neutral. Values ported from the working engine's
# venue_intel.py (its own hardcoded 2025 estimates), re-keyed here by the
# CANONICAL PARK NAME STRING data/name_registry.py's canonical_team()
# already returns in team["park"] -- so the join is an exact dict lookup,
# not a fuzzy match against an external, unverified source.
#
# VERIFIED entries (confirmed against this pipeline's actual run output --
# these are the exact strings that were failing to match before this fix):
#   Petco Park, Oracle Park, Angel Stadium, PNC Park,
#   Oriole Park at Camden Yards, Rogers Centre,
#   George M. Steinbrenner Field, Comerica Park, Citi Field,
#   Progressive Field, American Family Field, Target Field,
#   Rate Field, Busch Stadium, Fenway Park
#
# UNVERIFIED entries (the other 15 teams weren't in the slate that produced
# the run log above, so these are best-effort standard names -- confirm
# each against data/name_registry.py's canonical_team()[...]["park"] before
# trusting them. Two known landmines:
#   - Houston: officially renamed "Daikin Park" for the 2025 season
#     (was "Minute Maid Park") -- if name_registry.py wasn't updated to
#     match, this entry will silently miss exactly like the old bug did.
#   - Athletics: relocated to Sutter Health Park (West Sacramento) as a
#     multi-year interim home -- there is no real MLB park-factor history
#     for it yet, so it's left at neutral (100) rather than guessed.
# ---------------------------------------------------------------------------
MLB_PARK_FACTORS = {
    # -- verified against this pipeline's real run output --
    "Petco Park": 91,
    "Oracle Park": 92,
    "Angel Stadium": 97,
    "PNC Park": 98,
    "Oriole Park at Camden Yards": 98,
    "Rogers Centre": 101,
    "George M. Steinbrenner Field": 100,  # TB temp home -- no real run-factor history yet; left neutral
    "Comerica Park": 100,
    "Citi Field": 97,
    "Progressive Field": 100,
    "American Family Field": 100,
    "Target Field": 98,
    "Rate Field": 105,
    "Busch Stadium": 99,
    "Fenway Park": 106,

    # -- UNVERIFIED: confirm exact spelling against data/name_registry.py --
    "Chase Field": 96,                  # ARI
    "Truist Park": 102,                 # ATL
    "Wrigley Field": 97,                # CHC
    "Great American Ball Park": 109,    # CIN
    "Coors Field": 122,                 # COL
    "Daikin Park": 100,                 # HOU -- was "Minute Maid Park"; renamed 2025
    "Kauffman Stadium": 104,            # KC
    "Dodger Stadium": 95,                # LAD
    "loanDepot park": 93,                # MIA
    "Yankee Stadium": 103,               # NYY
    "Sutter Health Park": 100,           # OAK -- interim home; no real history, neutral
    "Citizens Bank Park": 107,           # PHI
    "T-Mobile Park": 95,                 # SEA
    "Globe Life Field": 104,             # TEX
    "Nationals Park": 97,                # WSH
}


def park_factor_by_name(park_name):
    """
    Static replacement for the old FanGraphs Guts-table lookup. Returns the
    park factor as a 0-1 ratio (100 = neutral -> 1.0), or 1.0 (neutral) with
    a warning if park_name isn't in MLB_PARK_FACTORS -- same safe-degrade
    behavior as before, just without needing pybaseball/lxml or a live
    FanGraphs fetch for this path anymore.
    """
    if not park_name:
        return 1.0
    factor = MLB_PARK_FACTORS.get(park_name)
    if factor is None:
        print(f"  [warn] no static park factor for '{park_name}' -- using neutral 1.0. "
              f"Add it to advanced_metrics.MLB_PARK_FACTORS (double check the exact "
              f"spelling data/name_registry.py's canonical_team() returns for this "
              f"team's 'park' field).")
        return 1.0
    return factor / 100.0
