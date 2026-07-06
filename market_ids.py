"""
⚠️ NOT WIRED IN — appears superseded, not a gap to fill.

This file is never imported anywhere in the codebase. The live pipeline's
odds provider is PropLine (see core/propline_client.py / data/fetch.py's
get_propline_odds), which does not need OddsPapi marketId resolution at all.

A separate, independently-maintained copy of this same MARKET_FAMILIES /
marketId-resolution concept already lives in src/clients/oddspapi_client.py,
which is the OddsPapi client actually in use -- but only by the standalone
discover_book_coverage.py discovery script (its own GitHub Actions workflow),
not by run_pipeline.py's live pick generation.

Before deleting or wiring this file in, confirm with the maintainer whether
OddsPapi is still meant to become a live odds source (in which case this and
src/clients/oddspapi_client.py's copy should be reconciled into one), or
whether this is leftover from a pre-PropLine migration and safe to remove.

oddspapi ID mappings for MLB and WNBA, derived from markets.json discovery
(32,815 total markets; 794 MLB / 4,902 WNBA).

Only wires up the market families currently needed by the pipeline:
  MLB:  game_total, run_line (== spread), pitcher_strikeouts
  WNBA: game_total, spread, moneyline, points, rebounds, assists

Each specific line (handicap) is its own marketId in oddspapi, so rather than
hardcoding thousands of IDs, this defines the (marketType, period, playerProp)
signature for each family and resolves handicap -> marketId at load time from
a markets.json payload (discovery dump or live /markets response).
"""

SPORT_IDS = {
    "MLB": 13,
    "WNBA": 11,
}

TOURNAMENT_IDS = {
    "MLB": 109,
    "WNBA": 486,
}

# (sportId, marketType, period, playerProp) signatures per family.
MARKET_FAMILIES = {
    "MLB": {
        "game_total": {"marketType": "totals", "period": "result", "playerProp": False},
        # MLB run line IS the spread market -- same family, no separate signature.
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

# spread and moneyline have no handicap axis worth indexing by "line" the same
# way totals/props do -- moneyline has none, spread's handicap IS the line.
# All families here resolve the same way: handicap -> marketId.


def load_market_index(markets_json, league):
    """
    Build {family_name: {handicap: {"marketId": int, "outcomes": [...]}}}
    for one league from a parsed markets.json list (list[dict], oddspapi schema).

    league: "MLB" or "WNBA"
    """
    sport_id = SPORT_IDS[league]
    families = MARKET_FAMILIES[league]
    index = {name: {} for name in families}

    by_sport = [m for m in markets_json if m.get("sportId") == sport_id]

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

    return index


def get_market_id(index, family, handicap):
    """Look up a single marketId for a resolved index, e.g. moneyline handicap=0."""
    entry = index.get(family, {}).get(handicap)
    return entry["marketId"] if entry else None


if __name__ == "__main__":
    import json
    import sys

    with open(sys.argv[1] if len(sys.argv) > 1 else "markets.json") as f:
        markets = json.load(f)

    for league in ("MLB", "WNBA"):
        idx = load_market_index(markets, league)
        print(f"\n{league}:")
        for family, handicaps in idx.items():
            print(f"  {family}: {len(handicaps)} lines")
            sample = next(iter(handicaps.items()), None)
            if sample:
                h, info = sample
                print(f"    e.g. handicap={h} -> marketId={info['marketId']} ({info['marketName']})")
