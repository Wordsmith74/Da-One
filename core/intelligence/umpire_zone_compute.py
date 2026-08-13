"""
core/intelligence/umpire_zone_compute.py

Fills umpire_intel.py's deliberately-empty UMPIRE_ZONE_TENDENCY table from a
REAL source instead of leaving it permanently empty. Per the "Wiring Up
Da-One" research: there is no free, documented public API for umpire zone
tendency (UmpScorecards' richer data is Patreon-gated; their site-powering
API isn't public). The honest free path is to self-compute it from the same
two ingredients UmpScorecards itself uses:

  1. Statcast pitch-level called-ball/called-strike data
     (plate_x, plate_z, sz_top, sz_bot, description) -- free via pybaseball,
     same source data/statcast_first_inning.py already pulls.
  2. Umpire identity per pitch -- Statcast's own `umpire` column is blank/
     deprecated, so umpire identity must be joined in from Bill Petti's
     free, public umpire-ID file (game_pk -> umpire name), the standard
     community source for this join.

What this computes, per umpire:
  - zone_size: this umpire's called-strike rate on borderline/edge pitches
    (within the rule-book zone tolerance band) relative to the league-wide
    rate on the same pitch locations -- >1.0 = calls MORE edge pitches
    strikes than average (a "wide" zone, pitcher-friendly); <1.0 = "narrow"
    (hitter-friendly).
  - league_mean_adjustment: zone_size translated into the signed adjustment
    unit umpire_intel.UMPIRE_ZONE_TENDENCY already uses (positive = hitter-
    friendly/raises totals, negative = pitcher-friendly/lowers totals),
    via adj = (1.0 - zone_size) * ADJUSTMENT_SCALE -- the exact derivation
    umpire_intel.py's own table docstring already specifies as reasonable.
  - games: sample size backing the number, so callers can apply their own
    minimum-sample trust threshold (umpire_intel.MIN_TABLE_ENTRIES_TO_TRUST
    gates the whole TABLE's size; this per-umpire `games` count lets a
    caller additionally distrust a single umpire with too few starts, the
    same reliability-filter spirit the NRFI doc applies to pitchers).

Honesty / fail-safe contract: if pybaseball is unavailable, the Petti
umpire-ID file can't be fetched, or the join produces no matches, this
returns an EMPTY table -- never invented adjustments. Same rule
umpire_intel.py's own docstring states explicitly. This module does not
run automatically on every pipeline invocation (a full-season Statcast
pull is expensive); it's meant to be run periodically (e.g. weekly, via a
scheduled job) to refresh data/cache/umpire_zone_tendency.json, which
umpire_intel.py then loads lazily and cheaply on every real run.

Written without live network access in this sandbox: the Petti umpire-ID
file's exact column names/URL, and Statcast's called-strike zone-edge
convention, follow widely-documented public conventions but are unverified
live here -- spot-check before trusting in production, same caveat this
repo already carries elsewhere (umpire_intel.py, pitcher_intel.py).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date
from typing import Optional

logger = logging.getLogger("betting_bot")

_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "cache", "umpire_zone_tendency.json",
)
_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # weekly refresh is plenty

# Bill Petti's public, community-standard umpire-ID-to-game join file.
# URL per his own baseball_tools documentation
# (https://billpetti.github.io/baseball_tools/) -- verify this resolves to
# the current file location before relying on it; Petti has moved this
# file's host before.
_PETTI_UMPIRE_ID_URL = (
    "https://raw.githubusercontent.com/BillPetti/baseball_tools/master/"
    "harryPotter/mlb_umpire_ids.csv"
)

# Adjustment-unit scale -- same "runs for totals / fractional strikeouts for
# K props" unit umpire_intel.py's table docstring specifies, so this
# composes additively with pitcher_intel.py / bullpen_intel.py's existing
# adjustments. Kept modest: umpire zone effects are real but second-order
# next to pitcher/lineup quality (reference doc section 10 calls it a
# "force multiplier," not the primary driver).
ADJUSTMENT_SCALE = 1.5

# A called pitch within this many inches of the rule-book zone edge (in
# either plate_x or the sz_top/sz_bot band) counts as "borderline" -- the
# only pitches umpire skill/tendency actually differentiates; pitches
# thrown down the middle or a foot off the plate are called correctly by
# ~every umpire and would just dilute the signal toward 1.0 for everyone.
_EDGE_MARGIN_FT = 0.25

MIN_GAMES_PER_UMPIRE = 8


def _load_cache() -> Optional[dict]:
    try:
        if not os.path.exists(_CACHE_PATH):
            return None
        if time.time() - os.path.getmtime(_CACHE_PATH) > _CACHE_TTL_SECONDS:
            return None
        with open(_CACHE_PATH, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.debug(f"[umpire_zone_compute] cache read failed: {exc!r}")
        return None


def _save_cache(table: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(table, f)
    except Exception as exc:
        logger.debug(f"[umpire_zone_compute] cache write failed: {exc!r}")


def _load_umpire_id_map() -> Optional["object"]:
    """Returns a pandas DataFrame of game_pk -> umpire name, or None."""
    try:
        import pandas as pd
        return pd.read_csv(_PETTI_UMPIRE_ID_URL)
    except Exception as exc:
        logger.debug(f"[umpire_zone_compute] Petti umpire-ID file fetch failed: {exc!r}")
        return None


def build_umpire_zone_tendency_table(season: int) -> dict[str, dict]:
    """
    Full self-compute: pulls a season of Statcast pitch data, joins umpire
    identity from Petti's file on game_pk, computes each umpire's edge-call
    rate vs. league average, and returns
        {umpire_name: {"zone_size": float, "league_mean_adjustment": float,
                        "games": int}}
    Returns {} (not a fabricated table) on any failure at any step.
    """
    try:
        from pybaseball import statcast
    except ImportError:
        logger.debug("[umpire_zone_compute] pybaseball not installed -- pip install pybaseball")
        return {}

    ump_ids = _load_umpire_id_map()
    if ump_ids is None or ump_ids.empty:
        return {}

    # Expect columns close to game_pk / umpire name -- normalize common
    # variants defensively since this is an external, unversioned file.
    game_pk_col = next((c for c in ump_ids.columns if c.lower() in ("game_pk", "gamepk")), None)
    name_col = next(
        (c for c in ump_ids.columns if "ump" in c.lower() and "name" in c.lower()), None,
    )
    if game_pk_col is None or name_col is None:
        logger.debug(
            f"[umpire_zone_compute] unexpected Petti file columns: {list(ump_ids.columns)}"
        )
        return {}
    ump_ids = ump_ids[[game_pk_col, name_col]].rename(
        columns={game_pk_col: "game_pk", name_col: "hp_umpire"}
    ).dropna()

    try:
        df = statcast(start_dt=f"{season}-03-01", end_dt=f"{season}-11-15")
    except Exception as exc:
        logger.debug(f"[umpire_zone_compute] statcast() pull failed for {season}: {exc!r}")
        return {}
    if df is None or df.empty:
        return {}

    try:
        df = df[df["description"].isin(["called_strike", "ball"])].copy()
        df = df.merge(ump_ids, on="game_pk", how="inner")
        if df.empty:
            return {}

        # Borderline/edge classification: pitch is within _EDGE_MARGIN_FT of
        # either the horizontal plate edges (+/-0.83ft, the rulebook plate
        # half-width plus ball radius) or the vertical sz_top/sz_bot band.
        plate_half_width = 0.83
        df["edge_h"] = (df["plate_x"].abs() - plate_half_width).abs() <= _EDGE_MARGIN_FT
        df["edge_v"] = (
            ((df["plate_z"] - df["sz_top"]).abs() <= _EDGE_MARGIN_FT)
            | ((df["plate_z"] - df["sz_bot"]).abs() <= _EDGE_MARGIN_FT)
        )
        edge_df = df[df["edge_h"] | df["edge_v"]]
        if edge_df.empty:
            return {}

        edge_df = edge_df.copy()
        edge_df["is_strike"] = (edge_df["description"] == "called_strike").astype(int)

        league_edge_strike_rate = edge_df["is_strike"].mean()
        if not league_edge_strike_rate:
            return {}

        table: dict[str, dict] = {}
        grouped = edge_df.groupby("hp_umpire")
        for name, g in grouped:
            games = g["game_pk"].nunique()
            if games < MIN_GAMES_PER_UMPIRE:
                continue
            ump_rate = g["is_strike"].mean()
            zone_size = ump_rate / league_edge_strike_rate
            adj = round((1.0 - zone_size) * ADJUSTMENT_SCALE, 3)
            table[str(name)] = {
                "zone_size": round(float(zone_size), 4),
                "league_mean_adjustment": adj,
                "games": int(games),
            }
        return table
    except Exception as exc:
        logger.debug(f"[umpire_zone_compute] aggregation failed: {exc!r}")
        return {}


def load_or_build_umpire_zone_tendency_table(force_refresh: bool = False) -> dict[str, dict]:
    """
    Cheap entry point for umpire_intel.py: returns the cached table if
    fresh, otherwise rebuilds it (current season, falling back to the prior
    season if the current one has too little data early in the year) and
    caches the result. Safe to call on every pipeline run -- the expensive
    path only runs once per _CACHE_TTL_SECONDS.
    """
    if not force_refresh:
        cached = _load_cache()
        if cached is not None:
            return cached

    season = date.today().year
    table = build_umpire_zone_tendency_table(season)
    if len(table) < MIN_GAMES_PER_UMPIRE and season > 2015:
        # Early season / thin current-year sample -- fall back to last
        # season's full table rather than caching a near-empty one.
        prior = build_umpire_zone_tendency_table(season - 1)
        if len(prior) > len(table):
            table = prior

    if table:
        _save_cache(table)
    return table
