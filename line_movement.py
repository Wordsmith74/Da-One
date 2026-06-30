"""
line_movement.py

Real-time line movement detector — the engine's second confirmation step.

How it works
------------
1. The opening line was recorded in the candidate dict at slate-fetch time
   (candidate["sportsbook_line"] / candidate["opening_line"]).
2. Just before the broadcast step, this module makes one fresh Odds API
   call per sport (bypassing the slate cache) to fetch the CURRENT lines.
3. For each candidate the delta between opening and current line is computed
   and translated into a directional signal:

   OVER bet
     current_line > opening_line  → market moved our way (confirming)  +edge
     current_line < opening_line  → market moved against us (opposing)  -edge

   UNDER bet
     current_line < opening_line  → market moved our way (confirming)  +edge
     current_line > opening_line  → market moved against us (opposing)  -edge

Signal magnitudes
-----------------
  delta ≥ 1.0 pt  (large move / steam)
      confirming  +2.5 edge
      opposing    -3.0 edge  + "Steam against us" flag text

  0.5 ≤ delta < 1.0  (medium move)
      confirming  +1.5 edge
      opposing    -2.0 edge

  delta < 0.5         no signal applied (noise threshold)

API cost
--------
One fresh call per sport (not per game).  Results are cached in-process
with a 5-minute TTL so repeated calls within a session do not multiply
quota usage.

The slate cache is intentionally bypassed here — we specifically want to
know how the line has moved *since* we first fetched it.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any
from urllib import request as urllib_req
from urllib.error import HTTPError
from zoneinfo import ZoneInfo

logger = logging.getLogger("betting_bot")

_ET  = ZoneInfo("America/New_York")
_UTC = timezone.utc

BASE_URL = "https://api.the-odds-api.com/v4"

_SPORT_KEY: dict[str, str] = {
    "WNBA": "basketball_wnba",
    "NBA":  "basketball_nba",
    "MLB":  "baseball_mlb",
}

# In-process TTL cache  {sport: (result_dict, fetched_at)}
_LINE_CACHE: dict[str, tuple[dict[str, dict], datetime]] = {}
_CACHE_TTL_S = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Edge signal thresholds
# ---------------------------------------------------------------------------

_LARGE_MOVE  = 1.0   # points
_MEDIUM_MOVE = 0.5   # points

# Bucket names keyed by (size, direction) — used to look up calibrator values
_BUCKET_MAP: dict[tuple[str, str], tuple[str, str]] = {
    # (size, conf_or_opp) → (calibrator_bucket, display_label)
    ("large",  "confirming"): ("steam_confirming", "Steam confirming"),
    ("large",  "opposing"):   ("steam_opposing",   "Steam against us — suppressed"),
    ("medium", "confirming"): ("line_confirming",  "Line confirming"),
    ("medium", "opposing"):   ("line_opposing",    "Line opposing"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_api_keys() -> list[str]:
    keys: list[str] = []
    for name in ("THE_ODDS_API_KEY", "THE_ODDS_API_KEY_2", "THE_ODDS_API_KEY_3"):
        k = os.getenv(name, "").strip()
        if k:
            keys.append(k)
    return keys


def _fetch_fresh(sport_key: str) -> list[dict] | None:
    """
    Fetch current odds for *sport_key* directly (no slate cache).
    Returns the raw list of game dicts from the Odds API, or None on failure.
    """
    keys = _load_api_keys()
    if not keys:
        return None

    params = "regions=us&markets=totals&oddsFormat=american&dateFormat=iso"
    last_err: Exception | None = None

    for key in keys:
        url = f"{BASE_URL}/sports/{sport_key}/odds/?apiKey={key}&{params}"
        try:
            with urllib_req.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                remaining = resp.headers.get("x-requests-remaining", "?")
                logger.debug(
                    f"[line_movement] fresh fetch  sport={sport_key}  "
                    f"quota_remaining={remaining}  key=…{key[-6:]}"
                )
                return data if isinstance(data, list) else None
        except HTTPError as exc:
            if exc.code in (401, 429):
                last_err = exc
                continue
            logger.debug(f"[line_movement] HTTP {exc.code} fetching {sport_key}")
            return None
        except Exception as exc:
            logger.debug(f"[line_movement] fetch error for {sport_key}: {exc}")
            return None

    logger.debug(f"[line_movement] all keys failed for {sport_key}: {last_err}")
    return None


def _parse_lines(raw_games: list[dict]) -> dict[str, dict]:
    """
    Parse raw Odds API game list → dict keyed by (away_team, home_team) tuple
    string "{away_team}||{home_team}".

    Each value: {"over_line": float, "under_line": float, "book_count": int}
    """
    result: dict[str, dict] = {}
    for game in raw_games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        if not home or not away:
            continue

        over_lines: list[tuple[float, str]] = []
        under_lines: list[tuple[float, str]] = []

        for bk in game.get("bookmakers", []):
            bk_title = bk.get("title") or bk.get("key", "")
            for mkt in bk.get("markets", []):
                if mkt.get("key") != "totals":
                    continue
                for outcome in mkt.get("outcomes", []):
                    name = outcome.get("name", "").lower()
                    pt   = outcome.get("point")
                    if pt is None:
                        continue
                    if name == "over":
                        over_lines.append((float(pt), bk_title))
                    elif name == "under":
                        under_lines.append((float(pt), bk_title))

        if over_lines:
            median_over = sorted(l for l, _ in over_lines)[len(over_lines) // 2]
            result[f"{away}||{home}"] = {
                "over_line":  median_over,
                "under_line": sorted(l for l, _ in under_lines)[len(under_lines) // 2]
                              if under_lines else median_over,
                "book_count": len({b for _, b in over_lines}),
            }

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_current_lines(sport: str) -> dict[str, dict]:
    """
    Return current market lines for all today's games in *sport*.

    Result is cached for up to 5 minutes so calling this once per sport
    during a run is enough.

    Returns
    -------
    dict keyed by "{away_team}||{home_team}" (full names as returned by
    the Odds API, which match candidate["away_team"] / candidate["home_team"]).
    Empty dict on any failure.
    """
    sport_up  = sport.upper()
    sport_key = _SPORT_KEY.get(sport_up)
    if not sport_key:
        return {}

    now = datetime.now(_UTC)
    cached = _LINE_CACHE.get(sport_up)
    if cached:
        data, ts = cached
        if (now - ts).total_seconds() < _CACHE_TTL_S:
            return data

    raw = _fetch_fresh(sport_key)
    if raw is None:
        return {}

    parsed = _parse_lines(raw)
    _LINE_CACHE[sport_up] = (parsed, now)
    logger.debug(
        f"[line_movement] {sport_up}: {len(parsed)} current lines fetched."
    )
    return parsed


def get_line_movement_signals(
    sport: str,
    candidates: list[dict[str, Any]],
) -> dict[str, tuple[float, str]]:
    """
    Compute line movement signals for a list of candidates.

    Fetches the current market lines once for the sport and compares each
    candidate's opening_line (or sportsbook_line) to the current line.

    Parameters
    ----------
    sport      : 'WNBA' | 'NBA' | 'MLB'
    candidates : list of candidate dicts from the pipeline

    Returns
    -------
    dict  {bet_id: (edge_adjustment, signal_description)}

    Keys are present only when a meaningful signal exists (delta ≥ 0.5 pt).
    Missing key = no signal (neutral).
    """
    current = fetch_current_lines(sport)
    if not current:
        return {}

    signals: dict[str, tuple[float, str]] = {}

    for c in candidates:
        if c.get("player"):
            continue  # line movement is for game totals only

        bet_id    = c.get("bet_id", "")
        direction = c.get("direction", "over").lower()
        away_team = c.get("away_team", "")
        home_team = c.get("home_team", "")
        opening   = float(c.get("opening_line") or c.get("sportsbook_line") or 0)

        if not opening or not away_team or not home_team:
            continue

        game_key = f"{away_team}||{home_team}"
        cur_game = current.get(game_key)
        if cur_game is None:
            continue

        current_line = float(
            cur_game["over_line"] if direction == "over"
            else cur_game["under_line"]
        )
        delta = current_line - opening  # positive = line went up

        # Translate to directional signal
        # OVER: line up = confirming (market moved up toward our over bet)
        # OVER: line down = opposing (market moved against over)
        # UNDER: line down = confirming  (market moved down toward our under)
        # UNDER: line up = opposing
        if direction == "over":
            conf_or_opp = "confirming" if delta > 0 else "opposing"
        else:
            conf_or_opp = "confirming" if delta < 0 else "opposing"

        abs_delta = abs(delta)
        if abs_delta >= _LARGE_MOVE:
            size = "large"
        elif abs_delta >= _MEDIUM_MOVE:
            size = "medium"
        else:
            continue  # below noise threshold — no signal

        bucket, label = _BUCKET_MAP[(size, conf_or_opp)]

        # Pull calibrated adjustment (falls back to research prior when no data)
        try:
            from core.intelligence.signal_calibrator import get_adjustment
            edge_adj = get_adjustment(bucket).value
            cal_src  = get_adjustment(bucket).source
        except Exception:
            # Fallback constants if calibrator unavailable
            _FALLBACK = {
                "steam_confirming":  2.5,
                "steam_opposing":   -3.0,
                "line_confirming":   1.5,
                "line_opposing":    -2.0,
            }
            edge_adj = _FALLBACK.get(bucket, 0.0)
            cal_src  = "fallback"

        description = (
            f"{label}: opening={opening} → current={current_line} "
            f"(Δ{delta:+.1f}) for {direction.upper()} "
            f"[adj={edge_adj:+.1f}, {cal_src}]"
        )
        signals[bet_id] = (edge_adj, description, bucket)
        logger.debug(
            f"[line_movement] {bet_id}  {description}"
        )

    return signals
