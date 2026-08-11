"""
core/book_exposure.py

Partial mitigation for book limiting (Fix #6).

Honest scope of this fix
-------------------------
Book limiting/restriction is fundamentally an operational problem, not a
code problem: a sportsbook restricts an account because a human reviews
its betting pattern and decides it's a winning player, and no amount of
code on your end changes their decision process. There is no code fix
that "solves" this the way the calibration/walk-forward fixes solve a
real bug -- multi-account rotation, bet-size variance, and timing
diversification are operational practices, not something this module can
automate responsibly (deliberately staggering bets to evade a book's risk
system is also the kind of thing that gets accounts closed faster, not
slower).

What this module DOES do: give you visibility you don't currently have.
Every published pick already carries a bookmaker_source (the book whose
line was actually used) and a stake_pct_bankroll. Nothing in this
pipeline aggregates that across a day/week to show you how concentrated
your action is on any one book -- which is exactly the pattern that gets
a book's attention fastest (one book seeing 80% of your total stake at
tiers priced better than average). This module computes that exposure
breakdown from pick_history.jsonl so you can see it and manually
diversify, rather than being surprised by a limit.

This does NOT: rotate accounts, throttle stake automatically, or predict
whether/when a book will act. Those require infrastructure (multiple
funded accounts, a policy for how you personally want to spread action)
that's a decision for you to make, not something safe to automate here.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from core.grading_utils import read_jsonl

DEFAULT_PICK_HISTORY_PATH = "output/pick_history.jsonl"

# A single book carrying more than this share of total stake over the
# lookback window is flagged -- not because 100% proof of anything, but
# because concentrated volume on one book, especially on your
# better-priced/higher-edge picks, is the single biggest lever you
# control over how fast that book notices you.
CONCENTRATION_WARN_THRESHOLD = 0.40


@dataclass
class BookExposure:
    book: str
    n_picks: int
    total_stake_pct: float          # sum of stake_pct_bankroll across picks at this book
    share_of_total_stake: float     # this book's stake as a fraction of all stake in the window
    avg_edge_pct: float             # average edge_pct of picks placed at this book
    flagged: bool                   # share_of_total_stake > CONCENTRATION_WARN_THRESHOLD


def compute_book_exposure(
    pick_history_path: str = DEFAULT_PICK_HISTORY_PATH,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """
    Aggregate published-pick stake by bookmaker_source over the trailing
    lookback_days, so concentration on a single book is visible before a
    limit happens rather than only explainable after.

    Uses ALL published picks in the window (not just graded ones) --
    exposure is about how much action a book has SEEN, which happens at
    pick time regardless of whether the game has been graded yet.
    """
    records = read_jsonl(pick_history_path)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    recent = [r for r in records if (r.get("generated_at") or "") >= cutoff]

    by_book: dict[str, list[dict]] = defaultdict(list)
    for r in recent:
        book = r.get("bookmaker_source") or r.get("book") or "unknown"
        by_book[book].append(r)

    total_stake = sum(
        (r.get("stake_pct_bankroll") or 0.0) for recs in by_book.values() for r in recs
    )

    exposures: list[BookExposure] = []
    for book, recs in sorted(by_book.items()):
        n = len(recs)
        stake_sum = sum((r.get("stake_pct_bankroll") or 0.0) for r in recs)
        edges = [r.get("edge_pct") for r in recs if r.get("edge_pct") is not None]
        avg_edge = round(sum(edges) / len(edges), 2) if edges else 0.0
        share = round(stake_sum / total_stake, 4) if total_stake else 0.0
        exposures.append(BookExposure(
            book=book, n_picks=n, total_stake_pct=round(stake_sum, 2),
            share_of_total_stake=share, avg_edge_pct=avg_edge,
            flagged=share > CONCENTRATION_WARN_THRESHOLD,
        ))

    exposures.sort(key=lambda e: e.share_of_total_stake, reverse=True)

    return {
        "lookback_days": lookback_days,
        "n_picks_in_window": len(recent),
        "total_stake_pct": round(total_stake, 2),
        "books": [
            {
                "book": e.book, "n_picks": e.n_picks,
                "total_stake_pct": e.total_stake_pct,
                "share_of_total_stake": e.share_of_total_stake,
                "avg_edge_pct": e.avg_edge_pct, "flagged": e.flagged,
            }
            for e in exposures
        ],
        "warnings": [
            f"{e.book}: {e.share_of_total_stake:.0%} of all stake in the last "
            f"{lookback_days}d ({e.n_picks} picks, avg edge {e.avg_edge_pct}%) -- "
            f"concentrated volume on one book, especially at above-average edge, "
            f"is what gets a book's risk system to look at an account. Consider "
            f"routing more action through other books that carry the same line."
            for e in exposures if e.flagged
        ],
    }


def print_exposure_report(payload: dict[str, Any]) -> None:
    print(f"=== Book Exposure (last {payload['lookback_days']}d, "
          f"{payload['n_picks_in_window']} picks) ===\n")
    for b in payload["books"]:
        flag = "  ⚠ CONCENTRATED" if b["flagged"] else ""
        print(f"  {b['book']:<16} n={b['n_picks']:3d}  "
              f"stake={b['total_stake_pct']:6.2f}%  "
              f"share={b['share_of_total_stake']:.0%}  "
              f"avg_edge={b['avg_edge_pct']}%{flag}")
    if payload["warnings"]:
        print()
        for w in payload["warnings"]:
            print(f"  WARNING: {w}")


if __name__ == "__main__":
    payload = compute_book_exposure()
    print_exposure_report(payload)
