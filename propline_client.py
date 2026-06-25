"""
core/propline_client.py — PropLine API client (api.prop-line.com)

Fetches supplementary prop odds to augment The Odds API bookmaker coverage.
Adds sharp/exchange books not available in The Odds API:
  Novig, Pinnacle (reliable), Matchbook, Smarkets, PrizePicks, Underdog.

API structure:
  GET https://api.prop-line.com/v1/sports/{sport_key}/odds
      ?apiKey=KEY&markets=market1,market2

Response format mirrors The Odds API (bookmakers → markets → outcomes).
This module normalises PropLine's two outcome formats to The Odds API format
so callers can merge bookmaker lists without any format awareness.

Public interface
---------------
fetch_propline_books(sport_key, markets)
  → dict[(home_team, away_team), list[bookmaker_dict]]
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_API_KEY  = os.environ.get("PROP_LINE_API_KEY", "")
_BASE_URL = "https://api.prop-line.com/v1"

# Strips trailing team abbreviation: "Brandon Marsh (PHI)" → "Brandon Marsh"
_TEAM_ABBREV_RE = re.compile(r"\s+\([A-Z]{2,4}\)$")

# Detects generic alt-line / group-bet descriptions that are NOT player names.
# Examples: "1+ Home Runs", "2+ Total Bases", "3+ Hits", "5+ Total Bases"
# These appear in DraftKings outcome lists alongside proper player-named lines.
_ALT_LINE_RE = re.compile(r"^\d+\+?\s+", re.IGNORECASE)

# Process-level response cache keyed by (sport_key, markets_str).
# One sport-level fetch covers all events — no per-event calls needed.
_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}


def _strip_team(name: str) -> str:
    """Remove trailing team abbreviation from a player name."""
    return _TEAM_ABBREV_RE.sub("", name).strip()


def _normalize_outcomes(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalise PropLine outcomes to The Odds API over/under format.

    PropLine serves two formats:

    O/U markets (e.g. batter_total_bases, pitcher_strikeouts):
        name="Over"/"Under", description="Player Name (TEAM)", point=1.5

    Binary markets (e.g. batter_hits, batter_home_runs):
        name="Player Name (TEAM)", description="...", point=null

    Both are normalised to:
        name="over"/"under", description="Player Name" (no team abbrev), point=float

    Binary markets map to direction="over", point=0.5 (did-it-happen).
    """
    out: list[dict[str, Any]] = []
    for o in outcomes:
        name  = (o.get("name") or "").strip()
        desc  = (o.get("description") or "").strip()
        point = o.get("point")
        price = o.get("price")
        if price is None:
            continue

        name_lower = name.lower()

        if name_lower in ("over", "under"):
            player = _strip_team(desc) if desc else ""
            # Skip alt-line group bets with non-player descriptions
            # e.g. "2+ Total Bases", "1+ Home Runs", "5+ Total Bases"
            if not player or _ALT_LINE_RE.match(player):
                continue
            out.append({
                "name":        name_lower,
                "description": player,
                "point":       float(point) if point is not None else 0.5,
                "price":       price,
            })
        else:
            player = _strip_team(name)
            # Skip generic alt-line names that aren't real player names
            if not player or _ALT_LINE_RE.match(player):
                continue
            out.append({
                "name":        "over",
                "description": player,
                "point":       0.5,
                "price":       price,
            })

    return out


def _match_event(
    target_home: str,
    target_away: str,
    lookup: dict[tuple[str, str], Any],
) -> Any | None:
    """
    Match a Odds-API event to a PropLine event by team names.

    Tries exact match first, then normalised (lowercase + stripped) match to
    handle minor formatting differences between the two APIs.
    """
    key = (target_home, target_away)
    if key in lookup:
        return lookup[key]
    # Normalised fallback
    th = target_home.lower().strip()
    ta = target_away.lower().strip()
    for (h, a), v in lookup.items():
        if h.lower().strip() == th and a.lower().strip() == ta:
            return v
    return None


def fetch_propline_books(
    sport_key: str,
    markets: list[str],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """
    Fetch prop odds from api.prop-line.com for an entire sport in one call.

    Parameters
    ----------
    sport_key : PropLine sport key, e.g. "baseball_mlb"
    markets   : market keys to request, e.g. ["pitcher_strikeouts", "batter_hits"]

    Returns
    -------
    dict mapping (home_team, away_team) → list of bookmaker dicts matching
    The Odds API shape:
        [{"key": "novig", "title": "Novig",
          "markets": [{"key": "pitcher_strikeouts", "outcomes": [...]}]}, ...]

    Returns {} on any error — PropLine is supplementary; callers must not fail
    if this dict is empty.
    """
    if not _API_KEY:
        logger.debug("[propline] PROP_LINE_API_KEY not set — skipping.")
        return {}
    if not markets:
        return {}

    markets_str = ",".join(markets)
    cache_key   = (sport_key, markets_str)

    if cache_key in _CACHE:
        raw_events = _CACHE[cache_key]
    else:
        url = (
            f"{_BASE_URL}/sports/{sport_key}/odds"
            f"?apiKey={_API_KEY}&markets={markets_str}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw_events = json.loads(resp.read())
            if not isinstance(raw_events, list):
                logger.warning(f"[propline] Unexpected response shape for {sport_key}")
                return {}
            _CACHE[cache_key] = raw_events
            logger.debug(
                f"[propline] {sport_key}: fetched {len(raw_events)} event(s) "
                f"for markets [{markets_str}]"
            )
        except Exception as exc:
            logger.warning(f"[propline] fetch error for {sport_key}: {exc}")
            return {}

    result: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for event in raw_events:
        home = (event.get("home_team") or "").strip()
        away = (event.get("away_team") or "").strip()
        if not home or not away:
            continue

        bookmakers: list[dict[str, Any]] = []
        for bk in event.get("bookmakers") or []:
            bk_key   = bk.get("key", "")
            bk_title = bk.get("title", bk_key)
            mkt_out: list[dict[str, Any]] = []
            for mkt in bk.get("markets") or []:
                mkt_key  = mkt.get("key", "")
                outcomes = _normalize_outcomes(mkt.get("outcomes") or [])
                if outcomes:
                    mkt_out.append({"key": mkt_key, "outcomes": outcomes})
            if mkt_out:
                bookmakers.append({
                    "key":     bk_key,
                    "title":   bk_title,
                    "markets": mkt_out,
                })

        if bookmakers:
            result[(home, away)] = bookmakers

    logger.debug(
        f"[propline] {sport_key}: {len(result)} event(s) have bookmaker data."
    )
    return result


def clear_cache() -> None:
    """Clear the process-level response cache (useful for testing)."""
    _CACHE.clear()
