"""
data/statcast_first_inning.py

Fills the #1 gap flagged in models/nrfi_handicapper.py's module docstring:
"This repo has no first-inning-specific splits feed wired in anywhere."

Approach (per the "Wiring Up Da-One" research this module implements):
MLB Stats API only exposes innings-level RUNS, not batter-level first-inning
splits. Baseball-Reference/Stathead's split finder has the numbers but is
paywalled and actively anti-scrapes (Cloudflare). FanGraphs splits are
member-gated. The correct free, programmatic path is to pull raw pitch-by-
pitch Statcast data (which includes an `inning` column) via `pybaseball` --
a free, open-source wrapper around Baseball Savant's public
`statcast_search/csv` endpoint -- and aggregate inning==1 rows ourselves.

What this computes, per pitcher, from real Statcast play-by-play:
  - first_inning_era            : (1st-inning earned runs / 1st-inning
                                    appearances) * 9 -- a per-9 rate proxy,
                                    same units as season ERA so it composes
                                    with pitcher_first_inning_tier_multiplier's
                                    league_first_inning_era default.
  - first_inning_k_pct          : strikeouts / batters faced, 1st inning only
  - first_inning_bb_pct         : (walks + HBP) / batters faced, 1st inning only
  - fbf_obp                     : OBP allowed to the very first batter faced
                                    in each start (on-base events / PAs)
  - first_inning_starts_sample  : number of 1st-inning appearances the above
                                    is built from (feeds the reliability gate
                                    alongside career_starts)

Honesty / fail-safe contract (matches umpire_intel.py / weather_intel.py):
  - If `pybaseball` isn't installed, or the network call fails, or the
    result set is empty, every public function returns None per field --
    NEVER a fabricated/estimated number. Callers (game_markets.py) must
    already treat None as "skip this tier component," which
    pitcher_first_inning_tier_multiplier already does natively.
  - Small samples are NOT silently trusted: a pitcher with fewer 1st-inning
    appearances this season than MIN_FIRST_INNING_SAMPLE falls back to a
    multi-season pool (see get_first_inning_splits) before giving up, per
    the reference doc's "Reliability filter" -- season-long ERA is never
    substituted in as a stand-in (that's the exact pitfall the doc and the
    module docstring warn against).
  - Runs entirely off publicly documented Statcast columns
    (game_pk, inning, inning_topbot, pitcher, batter, events, description,
    at_bat_number, outs_when_up). Written without live network access in
    this sandbox -- spot-check the actual column names/values against a
    live `pybaseball.statcast()` pull before trusting in production, same
    caveat this codebase already carries on umpire_intel.py and
    bullpen_intel.py.

Caching: results are cached to disk (data/cache/first_inning_splits_<year>.json)
with a TTL, since a full-season Statcast pull is a multi-minute operation you
do NOT want to repeat on every pipeline run. Refresh once per day is plenty --
first-inning stats don't meaningfully change pitcher-to-pitcher inside a day.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date
from typing import Optional

logger = logging.getLogger("betting_bot")

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
_CACHE_TTL_SECONDS = 20 * 60 * 60  # ~20h -- one refresh per day is enough

# Below this many 1st-inning appearances IN THE CURRENT SEASON, pool in the
# prior season too before trusting the read. Mirrors the reference doc's
# "unstable on small samples" warning -- this is a data-quality gate that
# sits ALONGSIDE (not instead of) nrfi_reliability_gate's career-starts gate.
MIN_FIRST_INNING_SAMPLE = 8

# League-average fallbacks, used only to fill obviously-missing per-PA rate
# denominators (never to fabricate a pitcher-specific number).
_LEAGUE_AVG_BF_PER_FIRST_INNING = 4.3


@dataclass
class FirstInningSplits:
    pitcher_id: int
    fbf_obp: Optional[float] = None
    first_inning_era: Optional[float] = None
    first_inning_k_pct: Optional[float] = None
    first_inning_bb_pct: Optional[float] = None
    sample_appearances: int = 0
    seasons_pooled: tuple = ()
    source: str = "statcast_pybaseball"


def _cache_path(season: int) -> str:
    return os.path.join(_CACHE_DIR, f"first_inning_splits_{season}.json")


def _load_cache(season: int) -> Optional[dict]:
    path = _cache_path(season)
    try:
        if not os.path.exists(path):
            return None
        if time.time() - os.path.getmtime(path) > _CACHE_TTL_SECONDS:
            return None
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.debug(f"[statcast_first_inning] cache read failed ({season}): {exc!r}")
        return None


def _save_cache(season: int, table: dict) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_cache_path(season), "w") as f:
            json.dump(table, f)
    except Exception as exc:
        logger.debug(f"[statcast_first_inning] cache write failed ({season}): {exc!r}")


def _pull_season_first_inning_table(season: int) -> Optional[dict]:
    """
    One full-season Statcast pull via pybaseball, aggregated down to a
    per-pitcher-id dict of raw first-inning counting stats. Returns None
    (not {}) if pybaseball is unavailable or the pull fails, so callers can
    tell "no data" apart from "zero pitchers had a 1st-inning appearance."
    """
    try:
        from pybaseball import statcast
    except ImportError:
        logger.debug("[statcast_first_inning] pybaseball not installed -- pip install pybaseball")
        return None

    start = f"{season}-03-01"
    end = f"{season}-11-15"
    try:
        df = statcast(start_dt=start, end_dt=end)
    except Exception as exc:
        logger.debug(f"[statcast_first_inning] statcast() pull failed for {season}: {exc!r}")
        return None

    if df is None or len(df) == 0:
        return None

    try:
        df = df[df["inning"] == 1]
        if df.empty:
            return None

        table: dict = {}
        # One row per pitch; group by (pitcher, game_pk) = one 1st-inning
        # appearance by that pitcher in that game. game_pk itself isn't
        # referenced below -- it's only needed to group appearances, not
        # to key the output table (that's keyed by pitcher id).
        for (pitcher_id, _game_pk), g in df.groupby(["pitcher", "game_pk"]):
            pid = str(int(pitcher_id))
            entry = table.setdefault(pid, {
                "appearances": 0, "earned_runs": 0.0, "batters_faced": 0,
                "strikeouts": 0, "walks_hbp": 0, "fbf_on_base": 0, "fbf_pa": 0,
            })
            entry["appearances"] += 1

            # One row per plate-appearance-ending pitch = at_bat_number groups.
            pa_rows = g.sort_values("at_bat_number").drop_duplicates(
                subset=["at_bat_number"], keep="last"
            )
            entry["batters_faced"] += len(pa_rows)

            events = pa_rows["events"].fillna("")
            entry["strikeouts"] += int((events == "strikeout").sum())
            entry["walks_hbp"] += int(events.isin(["walk", "hit_by_pitch"]).sum())

            # Earned runs allowed in the 1st inning this appearance. Statcast
            # doesn't tag earned vs. unearned directly on the pitch row, so
            # this is computed as the raw change in the batting team's score
            # across the inning (post_bat_score - bat_score) -- a RUNS proxy
            # (first-inning runs allowed), which is the honest thing to call
            # it, not true "earned runs" -- see first_inning_era docstring.
            if "post_bat_score" in pa_rows.columns and "bat_score" in pa_rows.columns:
                runs_this_inning = float(
                    (pa_rows["post_bat_score"] - pa_rows["bat_score"]).clip(lower=0).sum()
                )
                entry["earned_runs"] += runs_this_inning

            # First batter faced this appearance (lowest at_bat_number).
            first_pa = pa_rows.sort_values("at_bat_number").iloc[0]
            entry["fbf_pa"] += 1
            fbf_event = first_pa.get("events", "")
            on_base_events = {"single", "double", "triple", "home_run", "walk", "hit_by_pitch"}
            if fbf_event in on_base_events:
                entry["fbf_on_base"] += 1

        return table
    except Exception as exc:
        logger.debug(f"[statcast_first_inning] aggregation failed for {season}: {exc!r}")
        return None


def _get_season_table(season: int) -> Optional[dict]:
    cached = _load_cache(season)
    if cached is not None:
        return cached
    table = _pull_season_first_inning_table(season)
    if table is not None:
        _save_cache(season, table)
    return table


def get_first_inning_splits(
    pitcher_id: int,
    as_of: Optional[date] = None,
    min_sample: int = MIN_FIRST_INNING_SAMPLE,
) -> FirstInningSplits:
    """
    Public entry point. Returns real first-inning splits for pitcher_id,
    pooling the current season with the prior season if the current season's
    sample is below min_sample -- never fabricating a number when both are
    unavailable (returns an all-None FirstInningSplits instead, which
    pitcher_first_inning_tier_multiplier already treats as "skip this tier
    component" for each field independently).

    as_of: reference date (defaults to today) -- determines "current season."
    """
    as_of = as_of or date.today()
    current_season = as_of.year
    seasons_to_try = [current_season, current_season - 1]

    pid = str(int(pitcher_id))
    pooled = {
        "appearances": 0, "earned_runs": 0.0, "batters_faced": 0,
        "strikeouts": 0, "walks_hbp": 0, "fbf_on_base": 0, "fbf_pa": 0,
    }
    seasons_used = []

    for season in seasons_to_try:
        table = _get_season_table(season)
        if table is None:
            continue
        entry = table.get(pid)
        if not entry:
            continue
        for k in pooled:
            pooled[k] += entry.get(k, 0)
        seasons_used.append(season)
        if pooled["appearances"] >= min_sample:
            break

    if pooled["appearances"] == 0:
        return FirstInningSplits(pitcher_id=int(pitcher_id), seasons_pooled=tuple(seasons_used))

    bf = pooled["batters_faced"] or 1
    fbf_pa = pooled["fbf_pa"] or 1
    appearances = pooled["appearances"] or 1

    return FirstInningSplits(
        pitcher_id=int(pitcher_id),
        fbf_obp=round(pooled["fbf_on_base"] / fbf_pa, 4),
        first_inning_era=round((pooled["earned_runs"] / appearances) * 9.0, 3),
        first_inning_k_pct=round(pooled["strikeouts"] / bf, 4),
        first_inning_bb_pct=round(pooled["walks_hbp"] / bf, 4),
        sample_appearances=pooled["appearances"],
        seasons_pooled=tuple(seasons_used),
    )


def splits_as_tier_kwargs(splits: FirstInningSplits, min_sample: int = MIN_FIRST_INNING_SAMPLE) -> dict:
    """
    Convert a FirstInningSplits into the exact kwarg shape
    models.nrfi_handicapper.pitcher_first_inning_tier_multiplier expects.
    Below min_sample, returns {} (neutral) rather than a noisy read -- this
    is the data-quality half of reliability; nrfi_reliability_gate's
    career-starts check is the other half and both must pass.
    """
    if splits.sample_appearances < min_sample:
        return {}
    return {
        "fbf_obp": splits.fbf_obp,
        "first_inning_era": splits.first_inning_era,
        "first_inning_k_pct": splits.first_inning_k_pct,
        "first_inning_bb_pct": splits.first_inning_bb_pct,
    }
