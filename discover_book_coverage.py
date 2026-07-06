"""
discover_book_coverage.py

One-off manual discovery script -- NOT part of the pipeline, safe to delete
once the book-weighting decision for WNBA is made.

Why this exists
----------------
core/devig.py's weight tables (PLAYER_PROP_WEIGHTS, GAME_MARKET_WEIGHTS) are
currently scoped to 3 named books (FanDuel/Pinnacle/BetOnline) based on
handicapping convention, not measured data. Before adding OddsPapi as a 4th
data source and deciding whether to fold its books into the existing 3-book
scheme or expand the weight table, we need to know which bookmakers OddsPapi
*actually* returns odds from on real WNBA fixtures, and how consistently --
not OddsPapi's own marketing number ("350+ bookmakers"), which is an
aggregate across all sports and says nothing about a niche-domestically
market like WNBA. A book that shows up on 1 fixture in 10 isn't worth a
weight-table slot no matter how "sharp" it is.

What this does
---------------
Pulls up to MAX_FIXTURES real WNBA fixtures from OddsPapi, requests odds for
every wired market family (game_total/spread/moneyline/points/rebounds/
assists), and tallies which bookmaker keys appear and how often -- split
into a game-market bucket and a player-prop bucket, since devig.py already
weights those two differently.

The exact JSON shape of OddsPapi's /odds response for a multi-market
request hasn't been confirmed against a live call from this environment
(no network access here), so the walk below is deliberately shape-agnostic:
it recurses through whatever comes back looking for the two documented key
names ("bookmakerOdds" on live odds, "bookmakers" on historical odds) rather
than assuming a fixed structure. Anything it can't attribute to a market
family lands in "unattributed" instead of being silently dropped, and the
full raw response for the first fixture is written to the report so the
shape can be inspected/confirmed by hand.

Run manually via the "Discover OddsPapi Book Coverage (WNBA)" GitHub Action
(workflow_dispatch). Requires the ODDSPAPI_API_KEY secret.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.clients.oddspapi_client import OddsPapiClient  # noqa: E402

LEAGUE = "WNBA"
MAX_FIXTURES = 25  # cap so a manual run doesn't burn the whole free-tier quota
GAME_FAMILIES = {"game_total", "spread", "moneyline"}
PROP_FAMILIES = {"points", "rebounds", "assists"}
_REQUEST_DELAY_SECONDS = 1.0  # polite pacing against the free-tier rate limit


def _walk_for_book_keys(
    node: object,
    market_id_to_family: dict[int, str],
    out_game: dict[str, int],
    out_prop: dict[str, int],
    out_unattributed: dict[str, int],
) -> None:
    """
    Recursively search an arbitrary JSON structure for the two documented
    OddsPapi shapes -- "bookmakerOdds": {book: {...}} on /odds and
    "bookmakers": {book: [...]} on /historical-odds -- and tally which book
    keys appear. If the enclosing dict also carries a "marketId" sibling
    key, use it to attribute the books to the game-market or player-prop
    bucket; otherwise they land in out_unattributed so nothing is silently
    dropped.
    """
    if isinstance(node, dict):
        market_id = node.get("marketId")
        for key in ("bookmakerOdds", "bookmakers"):
            books_node = node.get(key)
            if isinstance(books_node, dict):
                family = market_id_to_family.get(market_id)
                if family in GAME_FAMILIES:
                    target = out_game
                elif family in PROP_FAMILIES:
                    target = out_prop
                else:
                    target = out_unattributed
                for book_key in books_node:
                    target[book_key] += 1
        for value in node.values():
            _walk_for_book_keys(value, market_id_to_family, out_game, out_prop, out_unattributed)
    elif isinstance(node, list):
        for item in node:
            _walk_for_book_keys(item, market_id_to_family, out_game, out_prop, out_unattributed)


def main() -> None:
    client = OddsPapiClient()

    # Build marketId -> family_name once from the same index the real
    # pipeline uses, so counts can be split into game-market vs. player-prop
    # buckets (core/devig.py weights those two differently).
    index = client._load_market_index(LEAGUE)
    market_id_to_family: dict[int, str] = {}
    for family_name, by_handicap in index.items():
        for _handicap, entry in by_handicap.items():
            market_id_to_family[entry["marketId"]] = family_name

    fixtures = client.get_fixtures(LEAGUE)
    print(
        f"[discovery] {len(fixtures)} {LEAGUE} fixtures in the 2026 season window; "
        f"sampling up to {MAX_FIXTURES}."
    )
    sample = fixtures[:MAX_FIXTURES]

    game_counts: dict[str, int] = defaultdict(int)
    prop_counts: dict[str, int] = defaultdict(int)
    unattributed_counts: dict[str, int] = defaultdict(int)
    fixtures_checked = 0
    first_raw_sample = None

    for fx in sample:
        fixture_id = fx.get("fixtureId")
        if fixture_id is None:
            continue
        try:
            odds = client.get_odds_for_fixture(fixture_id, LEAGUE)
        except Exception as exc:
            print(f"[discovery] fixtureId={fixture_id}: fetch failed -- {exc}")
            continue

        if first_raw_sample is None:
            first_raw_sample = odds

        _walk_for_book_keys(odds, market_id_to_family, game_counts, prop_counts, unattributed_counts)
        fixtures_checked += 1
        time.sleep(_REQUEST_DELAY_SECONDS)

    def _report(label: str, counts: dict[str, int]) -> None:
        print(f"\n--- {label} (out of {fixtures_checked} fixtures checked) ---")
        if not counts:
            print("  (no books found)")
            return
        for book, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * n / fixtures_checked if fixtures_checked else 0.0
            print(f"  {book:<20} {n:>3}/{fixtures_checked}  ({pct:5.1f}%)")

    _report("GAME MARKETS (totals / spread / moneyline)", game_counts)
    _report("PLAYER PROPS (points / rebounds / assists)", prop_counts)
    if unattributed_counts:
        _report(
            "UNATTRIBUTED (couldn't map marketId -> family -- inspect raw sample in the report)",
            unattributed_counts,
        )

    report = {
        "league": LEAGUE,
        "fixtures_checked": fixtures_checked,
        "game_market_book_counts": dict(game_counts),
        "player_prop_book_counts": dict(prop_counts),
        "unattributed_book_counts": dict(unattributed_counts),
        "first_raw_odds_sample": first_raw_sample,
    }
    out_path = Path(__file__).resolve().parent / "book_coverage_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[discovery] wrote {out_path}")


if __name__ == "__main__":
    main()
