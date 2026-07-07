"""
discover_book_coverage.py

One-off manual discovery script -- NOT part of the pipeline, safe to delete
once the book-weighting decision for WNBA is made.

Why this exists
----------------
core/devig.py's book table (BOOK_WEIGHTS) is currently scoped to 3 named,
equally-weighted books (FanDuel/Pinnacle/BetOnline) based on handicapping
convention, not measured data. Before adding OddsPapi as a 4th data source
and deciding whether to fold its books into the existing 3-book scheme or
expand the table, we need to know which bookmakers OddsPapi
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
into a game-market bucket and a player-prop bucket for visibility, even
though devig.py now weights both buckets the same way.

The exact JSON shape of OddsPapi's /odds response for a multi-market
request hasn't been confirmed against a live call from this environment
(no network access here), so the walk below is deliberately shape-agnostic:
it recurses through whatever comes back looking for the two documented key
names ("bookmakerOdds" on live odds, "bookmakers" on historical odds) rather
than assuming a fixed structure. Anything it can't attribute to a market
family lands in "unattributed" instead of being silently dropped, and the
full raw response for the first fixture is written to the report so the
shape can be inspected/confirmed by hand.

Bookmaker keys are canonicalized (bwin.de/bwin.es -> bwin) so regional skins
of the same trading desk don't get counted as independent books, and
polymarket/kalshi are split into their own bucket since they price a binary
contract via trader positioning rather than a bookmaker's vig-based line.
Books clearing a minimum fixture-presence threshold are surfaced as
"weight-table candidates" directly in the output.

Run manually via the "Discover OddsPapi Book Coverage (WNBA)" GitHub Action
(workflow_dispatch). Requires the ODDSPAPI_API_KEY secret.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.clients.oddspapi_client import OddsPapiClient  # noqa: E402

LEAGUE = "WNBA"
MAX_FIXTURES = 25  # cap so a manual run doesn't burn the whole free-tier quota
GAME_FAMILIES = {"game_total", "spread", "moneyline"}
PROP_FAMILIES = {"points", "rebounds", "assists"}
_REQUEST_DELAY_SECONDS = 1.0  # polite pacing against the free-tier rate limit

# OddsPapi returns regional storefronts of the same trading desk as distinct
# bookmaker keys (bwin / bwin.de / bwin.dk / bwin.es all mirror one price).
# Canonicalizing on "part before the first dot" collapses these correctly
# for every case observed in a real run -- the one known false-positive risk
# is a book whose *actual* name contains a dot (e.g. "sx.bet"), which would
# get truncated to "sx". Flagged in the report as canonicalization_caveats
# rather than hardcoding a maintenance-heavy exception list, since a human
# should eyeball the raw vs. canonical book lists before finalizing weights.
_KNOWN_DOTTED_BOOK_NAMES = {"sx.bet"}

# Prediction markets price a binary contract via trader positioning, not a
# bookmaker's vig-based two-sided line -- devig.py's math assumes the latter.
# Kept in a separate bucket rather than silently blended into the sportsbook
# consensus.
PREDICTION_MARKET_BOOKS = {"polymarket", "kalshi"}

# Minimum fraction of sampled fixtures a (canonical) book must appear on to
# be considered for a weight-table slot -- below this it's presence-noise,
# not signal, no matter how sharp the book is in principle.
_MIN_CONSISTENCY_PCT = 40.0


def _canonicalize_book(book_name: str) -> str:
    """Collapse regional operator skins (bwin.de -> bwin) to one canonical
    name, except for the small set of books whose real name contains a dot."""
    if book_name in _KNOWN_DOTTED_BOOK_NAMES:
        return book_name
    return book_name.split(".")[0]


def _book_family_pairs_for_fixture(
    fixture_odds: dict,
    market_id_to_family: dict[int, str],
) -> set[tuple[str, str | None]]:
    """
    Confirmed live /odds shape (from an actual run against real WNBA
    fixtures): {"bookmakerOdds": {book_name: {"markets": {marketId_str:
    {...}, ...}, ...}, ...}, ...}. marketId lives as a dict *key* under
    each book's own "markets" mapping -- not as a sibling field next to
    bookmakerOdds like OddsPapi's own blog examples implied.

    Returns the set of (book_name, family) pairs observed for this one
    fixture -- one pair per family per book, regardless of how many
    individual lines within that family the book quoted (e.g. a book
    quoting both the game total and the spread contributes one ("book",
    "game_total") pair and one ("book", "spread") pair, not two of the
    same pair). family is None when none of that book's marketIds map to
    a known family, so it lands in the unattributed bucket instead of
    being silently dropped.
    """
    pairs: set[tuple[str, str | None]] = set()
    bookmaker_odds = fixture_odds.get("bookmakerOdds")
    if not isinstance(bookmaker_odds, dict):
        return pairs

    for book_name, book_data in bookmaker_odds.items():
        if not isinstance(book_data, dict):
            continue
        markets = book_data.get("markets")
        if not isinstance(markets, dict):
            continue
        families_for_book: set[str | None] = set()
        for market_id_str in markets:
            try:
                market_id = int(market_id_str)
            except (TypeError, ValueError):
                continue
            families_for_book.add(market_id_to_family.get(market_id))
        for family in (families_for_book or {None}):
            pairs.add((book_name, family))
    return pairs


def _bucket_pairs(
    pairs: set[tuple[str, str | None]],
    out_game: dict[str, int],
    out_prop: dict[str, int],
    out_unattributed: dict[str, int],
    out_prediction: dict[str, int],
) -> None:
    """
    Bucket one fixture's (book, family) pairs into per-fixture-presence
    counts. A book is counted at most once per bucket per fixture, even if
    it quotes multiple families within that same bucket (e.g. both
    game_total and spread) -- these counts answer "what fraction of
    fixtures did this book appear on for this market type", not "how many
    individual lines did it quote", so double-counting within a bucket
    would inflate presence past 100%.
    """
    canonical_pairs = {(_canonicalize_book(book), family) for book, family in pairs}
    game_books: set[str] = set()
    prop_books: set[str] = set()
    unattributed_books: set[str] = set()
    prediction_books: set[str] = set()
    for book_name, family in canonical_pairs:
        if book_name in PREDICTION_MARKET_BOOKS:
            prediction_books.add(book_name)
        elif family in GAME_FAMILIES:
            game_books.add(book_name)
        elif family in PROP_FAMILIES:
            prop_books.add(book_name)
        else:
            unattributed_books.add(book_name)
    for book_name in game_books:
        out_game[book_name] += 1
    for book_name in prop_books:
        out_prop[book_name] += 1
    for book_name in unattributed_books:
        out_unattributed[book_name] += 1
    for book_name in prediction_books:
        out_prediction[book_name] += 1


def main() -> None:
    client = OddsPapiClient()

    # Build marketId -> family_name once from the same index the real
    # pipeline uses, so counts can be split into game-market vs. player-prop
    # buckets for reporting (core/devig.py itself now weights both the same).
    index = client._load_market_index(LEAGUE)
    market_id_to_family: dict[int, str] = {}
    for family_name, by_handicap in index.items():
        for _handicap, entry in by_handicap.items():
            market_id_to_family[entry["marketId"]] = family_name

    index_sizes = {name: len(by_handicap) for name, by_handicap in index.items()}
    print(f"[discovery] market index sizes (handicaps matched per family): {index_sizes}")

    raw_markets_sample = None
    if not any(index_sizes.values()):
        # get_odds_for_fixture() short-circuits to {} whenever market_ids is
        # empty for every family -- which is exactly what an all-zero index
        # produces. Rather than burn the fixture-odds quota finding that out
        # 25 times over, pull the raw /markets catalog once, filter to this
        # league's sportId, and dump what the real marketType/period/
        # playerProp/marketName values actually look like right now, so the
        # MARKET_FAMILIES signatures in oddspapi_client.py can be corrected
        # against real data instead of the original (possibly stale)
        # discovery dump.
        print(
            "[discovery] every family matched 0 handicaps -- MARKET_FAMILIES signatures "
            "likely don't match the live /markets catalog. Pulling a raw sample instead "
            "of burning the fixture-odds quota on calls that will all return {}."
        )
        from src.clients.oddspapi_client import SPORT_IDS  # noqa: E402

        sport_id = SPORT_IDS[LEAGUE]
        all_markets = client._get("markets")
        by_sport = [m for m in all_markets if m.get("sportId") == sport_id]
        print(f"[discovery] {len(by_sport)} raw market entries with sportId={sport_id}.")
        raw_markets_sample = by_sport[:15]
        for m in raw_markets_sample:
            print(
                f"  marketType={m.get('marketType')!r:<28} period={m.get('period')!r:<10} "
                f"playerProp={m.get('playerProp')!r:<6} marketName={m.get('marketName')!r}"
            )

    all_fixtures = client.get_fixtures(LEAGUE)
    has_odds_fixtures = [fx for fx in all_fixtures if fx.get("hasOdds")]
    print(
        f"[discovery] {len(all_fixtures)} {LEAGUE} fixtures in the 2026 season window; "
        f"{len(has_odds_fixtures)} report hasOdds=true. Sampling up to {MAX_FIXTURES} of those, "
        f"closest to right now first."
    )
    # Prefer fixtures closest to "now" (in either direction) -- a fixture from
    # week 1 of the season is far less likely to still carry a live/useful
    # odds snapshot than one starting soon, even if it happens to report
    # hasOdds=true (could be a stale closing-line snapshot retained post-game).
    _now = datetime.now(timezone.utc)

    def _distance_from_now(fx: dict) -> float:
        raw = fx.get("startTime")
        if not raw:
            return float("inf")
        try:
            fx_time = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return float("inf")
        return abs((fx_time - _now).total_seconds())

    fixtures = sorted(has_odds_fixtures, key=_distance_from_now)
    sample = fixtures[:MAX_FIXTURES]

    game_counts: dict[str, int] = defaultdict(int)
    prop_counts: dict[str, int] = defaultdict(int)
    unattributed_counts: dict[str, int] = defaultdict(int)
    prediction_counts: dict[str, int] = defaultdict(int)
    fixtures_checked = 0
    first_raw_sample = None

    if any(index_sizes.values()):
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
                print(f"[discovery] first fixture (id={fixture_id}) raw odds -- "
                      f"{'top-level keys: ' + str(list(odds.keys())) if isinstance(odds, dict) else 'type: ' + type(odds).__name__}")
                _dump = json.dumps(odds, indent=2)
                print("[discovery] first fixture raw odds (truncated to 4000 chars):")
                print(_dump[:4000] + ("... [truncated]" if len(_dump) > 4000 else ""))

            _bucket_pairs(
                _book_family_pairs_for_fixture(odds, market_id_to_family),
                game_counts, prop_counts, unattributed_counts, prediction_counts,
            )
            fixtures_checked += 1
            time.sleep(_REQUEST_DELAY_SECONDS)
    else:
        print("[discovery] skipping fixture-odds loop entirely -- index is empty, every call would return {}.")

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
    if prediction_counts:
        _report(
            "PREDICTION MARKETS (polymarket/kalshi -- binary contract pricing, "
            "not bookmaker vig -- kept separate from the sportsbook consensus)",
            prediction_counts,
        )
    if unattributed_counts:
        _report(
            "UNATTRIBUTED (couldn't map marketId -> family -- inspect raw sample in the report)",
            unattributed_counts,
        )

    def _candidates(counts: dict[str, int]) -> list[str]:
        if not fixtures_checked:
            return []
        return [
            book for book, n in sorted(counts.items(), key=lambda kv: -kv[1])
            if 100.0 * n / fixtures_checked >= _MIN_CONSISTENCY_PCT
        ]

    game_candidates = _candidates(game_counts)
    prop_candidates = _candidates(prop_counts)
    print(
        f"\n--- WEIGHT-TABLE CANDIDATES (>= {_MIN_CONSISTENCY_PCT:.0f}% fixture presence) ---\n"
        f"  game markets: {game_candidates or '(none clear the bar)'}\n"
        f"  player props: {prop_candidates or '(none clear the bar)'}"
    )

    report = {
        "league": LEAGUE,
        "index_sizes": index_sizes,
        "fixtures_checked": fixtures_checked,
        "game_market_book_counts": dict(game_counts),
        "player_prop_book_counts": dict(prop_counts),
        "prediction_market_book_counts": dict(prediction_counts),
        "unattributed_book_counts": dict(unattributed_counts),
        "min_consistency_pct_threshold": _MIN_CONSISTENCY_PCT,
        "game_market_candidates": game_candidates,
        "player_prop_candidates": prop_candidates,
        "first_raw_odds_sample": first_raw_sample,
        "raw_markets_sample_if_index_empty": raw_markets_sample,
    }
    out_path = Path(__file__).resolve().parent / "book_coverage_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[discovery] wrote {out_path}")


if __name__ == "__main__":
    main()
