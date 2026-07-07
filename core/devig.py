"""
core/devig.py — Weighted multi-book devig for fair market probability.

Problem this replaces
----------------------
Every edge calculation in the pipeline (models/monte_carlo.py,
core/revalidation_engine.py, core/odds_client.py._implied_prob) currently
does this:

    implied_over = (100/(odds+100)) if odds > 0 else abs(odds)/(abs(odds)+100)
    edge_pct = (model_prob - implied_over) * 100

That is the RAW single-sided implied probability from ONE book's price —
vig included. A -110/-110 market implies ~52.4%/52.4% (105% total); the
extra ~5% is the vig, not real probability. Comparing model_prob against
that inflated number understates true edge, and — because the vig baked
into a -110 market isn't the same size as the vig in a -150/+130 market —
two picks with the "same" edge_pct aren't actually benchmarked against the
same thing.

What this module does instead
------------------------------
For each named book that quoted BOTH sides of a line, normalize that book's
own two prices so their implied probabilities sum to 1 (standard
multiplicative devig). Then blend those per-book fair probabilities with
EQUAL weight across whichever of FanDuel / Pinnacle / BetOnline actually
reported a two-sided price — no book is weighted more heavily than another,
and there's no split between player-prop and game-market weighting schemes.

If one of the three named books didn't post a two-sided price (missing
entirely, or only quoted one side), it's simply dropped and the average is
taken over whichever of the three actually reported — this module never
silently invents a number for a book that isn't there.

Data dependency
----------------
This requires "own-side odds" AND "opposing-side odds" for the SAME book,
which is why core/odds_client.py and core/player_props.py were changed
alongside this module to:
  1. request regions=us,eu,us2 (us2 = BetOnline on The Odds API) instead of
     regions=us / regions=us,eu, and
  2. store {"book", "line", "odds"} per bookmaker on BOTH the over and under
     side (previously "odds" was only kept as a single global "best_odds"
     winner-take-all across all books, and book_lines carried "line" only).

Each prop/game candidate now carries "book_lines" (this direction) and
"opposing_book_lines" (the other direction) — both lists of
{"book", "line", "odds"}. That's what get_weighted_fair_prob() consumes.

NOT wired in yet
-----------------
This module is self-contained and does not yet change any edge_pct that
gets computed today. Swapping models/monte_carlo.py's inline implied_over
for this module's output, and choosing which book's price is actually used
to grade/settle the bet (execution price), is the next step — intentionally
kept separate so the devig math can be reviewed/tested on its own first.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Book weighting scheme
# ---------------------------------------------------------------------------
# Keys are canonical book names (see _canonical_book below) — every book
# name coming out of odds_client.py / player_props.py gets normalized to one
# of these before being matched against this table, so "BetOnline.ag" and
# "betonline" both resolve to the same entry as "BetOnline".
#
# All three books get equal weight — there's no per-book market-share or
# sharpness weighting, and no distinction between player-prop and
# game-market schemes. A book only contributes if it quoted both sides of
# the line (see get_weighted_fair_prob below); this table just says which
# three books are in scope, not how much each one counts.

BOOK_WEIGHTS: dict[str, float] = {
    "FanDuel":   1.0 / 3.0,
    "Pinnacle":  1.0 / 3.0,
    "BetOnline": 1.0 / 3.0,
}

# Book that's actually used to price/execute the bet (i.e. what "american_odds"
# should reflect once this is wired in) — separate concern from which books
# feed the fair-probability consensus. Exposed here so callers don't have to
# hardcode "FanDuel" themselves.
EXECUTION_BOOK = "FanDuel"

# If the fraction of the three named books actually reporting falls below
# this, the result is flagged low_confidence=True rather than silently
# trusted as a real 3-book consensus. Set at "at least one out of three"
# (1/3) so a single-book devig doesn't pass as equivalent to a full blend.
_MIN_TRUSTED_WEIGHT_COVERAGE = 1.0 / 3.0

_BOOK_ALIASES: dict[str, str] = {
    "fanduel":        "FanDuel",
    "pinnacle":       "Pinnacle",
    "betonline":      "BetOnline",
    "betonline.ag":   "BetOnline",
    "bol":            "BetOnline",
}


def _canonical_book(name: str | None) -> str | None:
    """Normalize a bookmaker title/key to one of the three named books, or
    None if it isn't one of them (e.g. DraftKings, BetMGM — not in scope
    for either weighting scheme, simply ignored)."""
    if not name:
        return None
    return _BOOK_ALIASES.get(name.strip().lower())


# ---------------------------------------------------------------------------
# Core probability math
# ---------------------------------------------------------------------------

def american_to_prob(odds: int) -> float:
    """Raw (vig-included) implied probability from a single American price.

    This is exactly the formula currently duplicated in monte_carlo.py,
    revalidation_engine.py, and odds_client.py._implied_prob — kept here as
    the building block devig_two_way() normalizes, and as an explicit
    fallback for callers that want the old (non-devigged) behavior for
    comparison/logging.
    """
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def devig_two_way(over_odds: int, under_odds: int) -> tuple[float, float]:
    """
    Standard multiplicative devig: normalize one book's own two-sided price
    so the implied probabilities sum to 1.0 instead of ~1.02-1.06 (the vig).

    Returns (fair_over_prob, fair_under_prob). Both from the SAME book —
    mixing one book's over price with a different book's under price here
    would reintroduce exactly the inconsistency this module exists to fix.
    """
    p_over_raw = american_to_prob(over_odds)
    p_under_raw = american_to_prob(under_odds)
    total = p_over_raw + p_under_raw
    if total <= 0:
        return 0.5, 0.5
    return p_over_raw / total, p_under_raw / total


# ---------------------------------------------------------------------------
# Weighted multi-book consensus
# ---------------------------------------------------------------------------

def get_weights_for_market(market: str) -> dict[str, float]:
    """Return the book weight table. Kept as a function (rather than callers
    reading BOOK_WEIGHTS directly) so the market-type split can come back
    later without changing call sites; today every market uses the same
    equal weights regardless of *market*."""
    return BOOK_WEIGHTS


def get_weighted_fair_prob(
    direction: str,
    own_book_lines: list[dict[str, Any]],
    opposing_book_lines: list[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    """
    Blend a weighted, vig-free fair probability for *direction* ("over" or
    "under") from whichever of the named books in *weights* quoted BOTH
    sides of this line.

    own_book_lines / opposing_book_lines: lists of {"book","line","odds"} —
    exactly what odds_client.py / player_props.py now attach to each
    candidate as "book_lines" and "opposing_book_lines".

    Returns:
        {
          "fair_prob":         float | None,   # None if zero books matched
          "books_used":        [str, ...],      # canonical names actually blended
          "per_book_fair_prob": {book: prob},
          "weight_covered":    float,           # sum of weights actually used, 0-1
          "low_confidence":    bool,            # weight_covered below trust floor
          "method":            "weighted_devig" | "no_two_sided_match",
        }

    A book is only used if BOTH its own-side and opposing-side price are
    present — a book that only quotes one side can't be devigged on its own
    and is silently excluded (not defaulted to raw implied probability),
    since that would quietly reintroduce the vig for that one book's weight.
    """
    opp_odds_by_book: dict[str, int] = {}
    for entry in opposing_book_lines or []:
        book = _canonical_book(entry.get("book"))
        odds = entry.get("odds")
        if book and odds is not None:
            opp_odds_by_book[book] = int(odds)

    per_book_fair: dict[str, float] = {}
    for entry in own_book_lines or []:
        book = _canonical_book(entry.get("book"))
        own_odds = entry.get("odds")
        if not book or book not in weights or own_odds is None:
            continue
        opp_odds = opp_odds_by_book.get(book)
        if opp_odds is None:
            continue  # this book didn't quote the other side — can't devig it alone

        if direction == "over":
            fair_over, fair_under = devig_two_way(int(own_odds), opp_odds)
        else:
            fair_over, fair_under = devig_two_way(opp_odds, int(own_odds))

        per_book_fair[book] = fair_over if direction == "over" else fair_under

    if not per_book_fair:
        return {
            "fair_prob": None,
            "books_used": [],
            "per_book_fair_prob": {},
            "weight_covered": 0.0,
            "low_confidence": True,
            "method": "no_two_sided_match",
        }

    weight_covered = sum(weights[b] for b in per_book_fair)
    blended = sum(per_book_fair[b] * weights[b] for b in per_book_fair) / weight_covered

    return {
        "fair_prob": blended,
        "books_used": sorted(per_book_fair.keys()),
        "per_book_fair_prob": per_book_fair,
        "weight_covered": round(weight_covered, 4),
        "low_confidence": weight_covered < _MIN_TRUSTED_WEIGHT_COVERAGE,
        "method": "weighted_devig",
    }


def weighted_fair_prob_for_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """
    Convenience wrapper: given a candidate dict already shaped by
    odds_client.py / player_props.py (has "market", "direction",
    "book_lines", "opposing_book_lines"), run get_weighted_fair_prob() with
    the correct weighting scheme auto-selected by market type.
    """
    weights = get_weights_for_market(candidate.get("market", ""))
    return get_weighted_fair_prob(
        direction=candidate.get("direction", "over"),
        own_book_lines=candidate.get("book_lines", []),
        opposing_book_lines=candidate.get("opposing_book_lines", []),
        weights=weights,
    )


def get_execution_price(
    own_book_lines: list[dict[str, Any]],
    execution_book: str = EXECUTION_BOOK,
    fallback_odds: int | None = None,
) -> dict[str, Any]:
    """
    Return the price to actually bet at (as opposed to the books used to
    build the fair-probability benchmark). Defaults to FanDuel per the
    weighting brief — since that's presumably where the bet gets placed —
    falling back to *fallback_odds* (typically the existing best-odds
    winner-take-all) if FanDuel didn't quote this side.

    Not called anywhere yet — this is prep for wiring the edge calc to use
    a *devigged consensus* as the benchmark while still displaying/settling
    at a real, bettable price.
    """
    for entry in own_book_lines or []:
        if _canonical_book(entry.get("book")) == execution_book and entry.get("odds") is not None:
            return {"book": execution_book, "odds": int(entry["odds"]), "used_fallback": False}
    return {"book": execution_book, "odds": fallback_odds, "used_fallback": True}
