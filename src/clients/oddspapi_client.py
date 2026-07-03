"""
oddspapi_client.py

Client for OddsPapi (https://api.oddspapi.io/v4), covering MLB and WNBA.

IDs below were confirmed against a live discovery pull (sports.json,
tournaments_all.json, markets.json -- 32,815 markets total, 794 MLB /
4,902 WNBA) rather than guessed:

    Sport IDs:        MLB=13 (Baseball), WNBA=11 (Basketball)
    Tournament IDs:   MLB=109 (regular season), WNBA=486 (regular season)

Market catalog is season-agnostic (it's the menu of line types OddsPapi
supports, not tied to a year), so this client applies the 2026-season
restriction itself, at the fixture level, using the same
regular_season_start_month_day / regular_season_end_month_day bounds
models/sport_config.py already defines for the rest of the pipeline --
2026-03-20 to 2026-09-28 for MLB, 2026-05-15 to 2026-09-15 for WNBA.

Each specific line (handicap) is its own marketId in OddsPapi, so market
families are resolved by (marketType, period, playerProp) signature against
a live /markets response rather than hardcoded per-handicap IDs.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import requests

SEASON_YEAR = 2026

SPORT_IDS: dict[str, int] = {
    "MLB": 13,
    "WNBA": 11,
}

TOURNAMENT_IDS: dict[str, int] = {
    "MLB": 109,
    "WNBA": 486,
}

# Market families currently wired into the pipeline. Each is a
# (marketType, period, playerProp) signature resolved against a live
# /markets response -- see _load_market_index().
#
#   MLB:  game_total, run_line (== spread; MLB has no separate spread
#         market from the run line), pitcher_strikeouts
#   WNBA: game_total, spread, moneyline, points, rebounds, assists
MARKET_FAMILIES: dict[str, dict[str, dict[str, Any]]] = {
    "MLB": {
        "game_total": {"marketType": "totals", "period": "result", "playerProp": False},
        "run_line": {"marketType": "spreads", "period": "result", "playerProp": False},
        "pitcher_strikeouts": {
            "marketType": "playertotals-strikeouts",
            "period": "result",
            "playerProp": True,
        },
    },
    "WNBA": {
        "game_total": {"marketType": "totals", "period": "result", "playerProp": False},
        "spread": {"marketType": "spreads", "period": "result", "playerProp": False},
        "moneyline": {"marketType": "moneyline", "period": "result", "playerProp": False},
        "points": {"marketType": "playertotals-points", "period": "result", "playerProp": True},
        "rebounds": {"marketType": "playertotals-rebounds", "period": "result", "playerProp": True},
        "assists": {"marketType": "playertotals-assists", "period": "result", "playerProp": True},
    },
}

# Regular-season month/day bounds, mirrored from models/sport_config.py so
# the two stay in sync -- if that file's season dates ever change, update
# both.
_SEASON_BOUNDS_MD: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "MLB": ((3, 20), (9, 28)),
    "WNBA": ((5, 15), (9, 15)),
}


def season_date_bounds(league: str, year: int = SEASON_YEAR) -> tuple[str, str]:
    """
    Return (start_date, end_date) ISO strings for *league*'s regular season
    in *year*, e.g. ("2026-03-20", "2026-09-28") for MLB.
    """
    (sm, sd), (em, ed) = _SEASON_BOUNDS_MD[league]
    return date(year, sm, sd).isoformat(), date(year, em, ed).isoformat()


class OddsPapiClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("ODDSPAPI_API_KEY")
        self.base_url = "https://api.oddspapi.io/v4"
        self._market_index_cache: dict[str, dict[str, dict[float, dict]]] = {}

    # ------------------------------------------------------------------
    # Low-level request helper
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.api_key:
            raise RuntimeError(
                "No OddsPapi key configured. Set ODDSPAPI_API_KEY in Secrets."
            )
        params = dict(params or {})
        params["apiKey"] = self.api_key
        resp = requests.get(f"{self.base_url}/{path}", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"OddsPapi error on {path}: {data['error']}")
        return data

    # ------------------------------------------------------------------
    # Market catalog
    # ------------------------------------------------------------------

    def _load_market_index(self, league: str) -> dict[str, dict[float, dict]]:
        """
        Build {family_name: {handicap: {"marketId": int, "outcomes": [...]}}}
        for *league* from a live /markets pull. Cached per-instance so a
        single run only fetches the catalog once per league.
        """
        if league in self._market_index_cache:
            return self._market_index_cache[league]

        sport_id = SPORT_IDS[league]
        families = MARKET_FAMILIES[league]
        markets = self._get("markets")

        by_sport = [m for m in markets if m.get("sportId") == sport_id]
        index: dict[str, dict[float, dict]] = {name: {} for name in families}

        for name, sig in families.items():
            for m in by_sport:
                if (
                    m.get("marketType") == sig["marketType"]
                    and m.get("period") == sig["period"]
                    and m.get("playerProp") == sig["playerProp"]
                ):
                    index[name][m["handicap"]] = {
                        "marketId": m["marketId"],
                        "marketName": m["marketName"],
                        "outcomes": m["outcomes"],
                    }

        self._market_index_cache[league] = index
        return index

    def get_market_id(self, league: str, family: str, handicap: float) -> int | None:
        """Look up a single marketId, e.g. get_market_id('WNBA', 'points', 4.5)."""
        index = self._load_market_index(league)
        entry = index.get(family, {}).get(handicap)
        return entry["marketId"] if entry else None

    # ------------------------------------------------------------------
    # Fixtures -- restricted to the 2026 season
    # ------------------------------------------------------------------

    def get_fixtures(
        self, league: str, year: int = SEASON_YEAR
    ) -> list[dict[str, Any]]:
        """
        Return fixtures for *league*'s regular-season tournament, filtered
        to *year*'s season window (default 2026: 2026-03-20..2026-09-28 for
        MLB, 2026-05-15..2026-09-15 for WNBA).

        OddsPapi's /fixtures endpoint returns fixtures across all seasons
        for a tournamentId, so the date filter is applied client-side here
        rather than trusting the API to scope by year.
        """
        tournament_id = TOURNAMENT_IDS[league]
        start_date, end_date = season_date_bounds(league, year)

        raw = self._get("fixtures", {"tournamentId": tournament_id})
        if not isinstance(raw, list):
            return []

        fixtures = []
        for fx in raw:
            fx_date = fx.get("startTime", "")[:10]  # ISO date prefix
            if fx_date and start_date <= fx_date <= end_date:
                fixtures.append(fx)
        return fixtures

    # ------------------------------------------------------------------
    # Odds
    # ------------------------------------------------------------------

    def get_odds_for_fixture(
        self, fixture_id: int, league: str, families: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Fetch odds for one fixture, restricted to the given market
        *families* (default: all families defined for *league* in
        MARKET_FAMILIES).
        """
        index = self._load_market_index(league)
        wanted_families = families or list(index.keys())
        market_ids = [
            entry["marketId"]
            for fam in wanted_families
            for entry in index.get(fam, {}).values()
        ]
        if not market_ids:
            return {}

        return self._get(
            "odds",
            {"fixtureId": fixture_id, "marketIds": ",".join(str(i) for i in market_ids)},
        )

    def get_historical_lines(self, fixture_id: int, league: str | None = None) -> Any:
        """
        Fetch historical odds for a completed fixture.

        *league* is required to resolve market families -- without it there's
        no way to know which marketIds to ask for.
        """
        if league is None:
            raise ValueError("get_historical_lines requires league='MLB' or 'WNBA'")
        return self._get("odds/historical", {"fixtureId": fixture_id})
