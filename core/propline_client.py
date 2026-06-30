"""
propline_client.py

Thin adapter layer between the raw PropLine HTTP calls (data.fetch.
get_propline_odds) and the rest of the engine, which wants PropLine
bookmaker data shaped exactly like a single game's "bookmakers" list from
The Odds API: ``[{"title"/"key": ..., "markets": [{"key": ..., "outcomes":
[{"name": "over"/"under", "description": <player>, "point": <float>,
"price": <int>}]}]}]``.

PropLine's response is advertised as drop-in compatible with The Odds API
at the sport-level odds endpoint (event-keyed, each event carrying
home_team/away_team/bookmakers), so this is mostly bookkeeping:
    1. fetch_propline_books()  -- one call per sport per session, grouped
       by (home_team, away_team) so callers can do an O(1) lookup per game
       instead of re-fetching per event.
    2. _match_event()          -- look up a specific game's PropLine books
       by team names, tolerating minor naming differences between
       providers (case, whitespace, "St." vs "State", etc.).
    3. _normalize_outcomes()   -- defensive cleanup of each outcome so
       downstream code (core.player_props._merge_propline_books) can
       assume name is lowercase "over"/"under", description is stripped,
       and point/price are the right types.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_outcomes(bookmakers: list[dict]) -> list[dict]:
    """
    Defensively normalise a list of bookmaker dicts from PropLine so every
    outcome has a lowercase over/under name, a stripped player description,
    and a numeric point where possible.

    Malformed bookmakers/markets/outcomes are dropped rather than raising,
    since one bad record shouldn't void an entire event's data.
    """
    normalized: list[dict] = []
    for bk in bookmakers or []:
        if not isinstance(bk, dict):
            continue
        title = bk.get("title") or bk.get("key") or "PropLine"
        norm_markets = []
        for mkt in bk.get("markets", []) or []:
            if not isinstance(mkt, dict):
                continue
            mkt_key = mkt.get("key", "")
            norm_outcomes = []
            for oc in mkt.get("outcomes", []) or []:
                if not isinstance(oc, dict):
                    continue
                name = (oc.get("name") or "").strip().lower()
                desc = (oc.get("description") or "").strip()
                point = oc.get("point")
                price = oc.get("price")
                if name not in ("over", "under") or not desc:
                    continue
                try:
                    point = float(point) if point is not None else None
                except (TypeError, ValueError):
                    point = None
                try:
                    price = int(price) if price is not None else None
                except (TypeError, ValueError):
                    price = None
                if point is None or price is None:
                    continue
                norm_outcomes.append(
                    {"name": name, "description": desc, "point": point, "price": price}
                )
            if norm_outcomes:
                norm_markets.append({"key": mkt_key, "outcomes": norm_outcomes})
        if norm_markets:
            normalized.append({"title": title, "key": bk.get("key", title), "markets": norm_markets})
    return normalized


def _team_key(name: str) -> str:
    """Normalise a team name for fuzzy matching across providers."""
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def fetch_propline_books(
    sport_key: str, markets: list[str], regions: str = "us"
) -> dict[tuple[str, str], list[dict]]:
    """
    Fetch all of today's PropLine player-prop bookmaker data for *sport_key*
    in a single call, grouped by (home_team, away_team).

    Args:
        sport_key: Odds-API-style sport key, e.g. "basketball_wnba". Same
            keys PropLine uses since its API is advertised drop-in
            compatible with The Odds API.
        markets:   List of market keys to request, e.g.
            ["player_rebounds", "player_assists"].
        regions:   Bookmaker regions to request.

    Returns:
        dict mapping (home_team, away_team) -> list of normalized
        bookmaker dicts (see _normalize_outcomes). Empty dict on any
        fetch failure -- callers treat PropLine as a best-effort
        supplementary source and must keep working without it.
    """
    if not markets:
        return {}

    from data.fetch import get_propline_odds

    markets_str = ",".join(markets)
    try:
        events = get_propline_odds(sport_key, markets_str, regions=regions)
    except Exception as exc:
        logger.warning(f"[propline_client] fetch failed for {sport_key}: {exc}")
        return {}

    if not isinstance(events, list):
        return {}

    grouped: dict[tuple[str, str], list[dict]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        if not home_team or not away_team:
            continue
        books = _normalize_outcomes(event.get("bookmakers", []))
        if books:
            grouped[(home_team, away_team)] = books

    return grouped


def _match_event(
    home_team: str,
    away_team: str,
    propline_all: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    """
    Look up PropLine bookmaker data for a specific game by team names.

    Tries an exact (home, away) match first, then falls back to a
    normalised (lowercased, alnum-only) comparison to tolerate minor
    naming differences between The Odds API and PropLine (e.g. trailing
    whitespace, punctuation, "St." vs "State").

    Returns:
        List of normalized bookmaker dicts, or [] if no match found.
    """
    if not propline_all:
        return []

    direct = propline_all.get((home_team, away_team))
    if direct:
        return direct

    home_k = _team_key(home_team)
    away_k = _team_key(away_team)
    for (pl_home, pl_away), books in propline_all.items():
        if _team_key(pl_home) == home_k and _team_key(pl_away) == away_k:
            return books

    return []
