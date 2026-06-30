"""
clv_tracker.py

Closing Line Value (CLV) Tracker.

CLV is the gold standard for verifying whether a betting model has genuine
edge.  If your picks consistently beat the closing line, the market is
confirming your assessment was sharp.  If you lose to the closing line, the
model is reacting to information the market already priced in.

How it works
------------
1. snapshot_odds()  — called at pick generation time; records the opening
   odds and line into the clv_snapshots table in results.db.
2. update_closing_line()  — called at bet close time; records the closing
   odds/line so CLV can be computed.
3. get_clv_summary()  — returns aggregate CLV stats for reporting.

CLV formula (American odds → implied probability)
-------------------------------------------------
  implied(odds) =  abs(odds) / (abs(odds) + 100) × 100   if odds < 0
                =  100 / (odds + 100) × 100               if odds ≥ 0

  CLV = implied(opening_odds) - implied(closing_odds)

  Positive CLV → you opened at a better price than market closed at → sharp.
  Negative CLV → market moved away from you → soft side.

Beat-rate target: CLV > 0 on ≥ 55 % of picks indicates a genuine edge.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger("betting_bot")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent.parent / "data" / "results.db"


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_table() -> None:
    """Create clv_snapshots table if it does not exist."""
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clv_snapshots (
                bet_id          TEXT PRIMARY KEY,
                snapshot_time   TEXT NOT NULL,
                opening_odds    INTEGER NOT NULL,
                opening_line    REAL NOT NULL,
                closing_odds    INTEGER,
                closing_line    REAL,
                close_time      TEXT,
                clv_pct         REAL,
                sport           TEXT,
                market          TEXT
            )
        """)


# ---------------------------------------------------------------------------
# Odds math
# ---------------------------------------------------------------------------

def _american_to_implied(odds: int | float) -> float:
    """Convert American odds to implied probability (0–100 %)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100) * 100.0
    return 100.0 / (odds + 100) * 100.0


def _compute_clv(opening_odds: int, closing_odds: int) -> float:
    """
    Positive = opening was a better price than closing (sharp).
    Negative = opening was worse (soft).
    """
    return round(_american_to_implied(opening_odds) - _american_to_implied(closing_odds), 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def snapshot_odds(
    bet_id:       str,
    opening_odds: int,
    opening_line: float,
    sport:        str = "",
    market:       str = "",
) -> None:
    """
    Record the opening odds and line at pick generation time.
    Safe to call multiple times — REPLACE INTO overwrites a stale snapshot.
    """
    try:
        _ensure_table()
        now = datetime.now(timezone.utc).isoformat()
        with _db() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO clv_snapshots
                    (bet_id, snapshot_time, opening_odds, opening_line, sport, market)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (bet_id, now, int(opening_odds), float(opening_line), sport, market),
            )
        logger.debug(
            f"[clv_tracker] Snapshot: {bet_id}  odds={opening_odds:+d}  line={opening_line}"
        )
    except Exception as exc:
        logger.warning(f"[clv_tracker] snapshot_odds failed for {bet_id}: {exc}")


def update_closing_line(
    bet_id:       str,
    closing_odds: int,
    closing_line: float,
) -> float | None:
    """
    Record the closing odds/line and compute CLV.
    Returns the CLV percentage, or None on failure.
    """
    try:
        _ensure_table()
        # Fetch opening odds
        with _db() as conn:
            row = conn.execute(
                "SELECT opening_odds FROM clv_snapshots WHERE bet_id = ?", (bet_id,)
            ).fetchone()

        if row is None:
            logger.debug(f"[clv_tracker] No snapshot found for {bet_id} — skipping CLV.")
            return None

        clv = _compute_clv(int(row["opening_odds"]), closing_odds)
        now = datetime.now(timezone.utc).isoformat()

        with _db() as conn:
            conn.execute(
                """
                UPDATE clv_snapshots
                SET closing_odds = ?,
                    closing_line = ?,
                    close_time   = ?,
                    clv_pct      = ?
                WHERE bet_id = ?
                """,
                (int(closing_odds), float(closing_line), now, clv, bet_id),
            )

        direction = "SHARP ✓" if clv > 0 else "soft ✗"
        logger.info(
            f"[clv_tracker] {bet_id}  CLV={clv:+.2f}%  ({direction})"
        )
        return clv

    except Exception as exc:
        logger.warning(f"[clv_tracker] update_closing_line failed for {bet_id}: {exc}")
        return None


def get_clv_summary(sport: str | None = None) -> dict[str, Any]:
    """
    Return aggregate CLV statistics across all graded bets.

    Returns
    -------
    dict with keys:
        total_bets     : int   — bets with a recorded CLV
        avg_clv        : float — mean CLV across all bets (positive = sharp overall)
        beat_rate      : float — % of bets where CLV > 0
        best_bet_id    : str   — bet with highest CLV
        worst_bet_id   : str   — bet with lowest CLV
        by_sport       : dict  — {sport: {avg_clv, beat_rate, count}}
    """
    try:
        _ensure_table()
        filters = "WHERE clv_pct IS NOT NULL"
        params: list[Any] = []
        if sport:
            filters += " AND sport = ?"
            params.append(sport.upper())

        with _db() as conn:
            rows = conn.execute(
                f"SELECT * FROM clv_snapshots {filters}", params
            ).fetchall()

        if not rows:
            return {
                "total_bets": 0, "avg_clv": 0.0, "beat_rate": 0.0,
                "best_bet_id": None, "worst_bet_id": None, "by_sport": {},
            }

        clvs       = [r["clv_pct"] for r in rows]
        avg_clv    = round(sum(clvs) / len(clvs), 3)
        beat_rate  = round(sum(1 for c in clvs if c > 0) / len(clvs) * 100, 1)
        best_row   = max(rows, key=lambda r: r["clv_pct"])
        worst_row  = min(rows, key=lambda r: r["clv_pct"])

        # By sport breakdown
        by_sport: dict[str, Any] = {}
        for row in rows:
            sp = row["sport"] or "UNKNOWN"
            if sp not in by_sport:
                by_sport[sp] = {"clvs": [], "count": 0}
            by_sport[sp]["clvs"].append(row["clv_pct"])
            by_sport[sp]["count"] += 1

        for sp, d in by_sport.items():
            sp_clvs = d.pop("clvs")
            d["avg_clv"]   = round(sum(sp_clvs) / len(sp_clvs), 3)
            d["beat_rate"] = round(sum(1 for c in sp_clvs if c > 0) / len(sp_clvs) * 100, 1)

        return {
            "total_bets":   len(rows),
            "avg_clv":      avg_clv,
            "beat_rate":    beat_rate,
            "best_bet_id":  best_row["bet_id"],
            "worst_bet_id": worst_row["bet_id"],
            "by_sport":     by_sport,
        }

    except Exception as exc:
        logger.warning(f"[clv_tracker] get_clv_summary failed: {exc}")
        return {"total_bets": 0, "avg_clv": 0.0, "beat_rate": 0.0,
                "best_bet_id": None, "worst_bet_id": None, "by_sport": {}}


def print_clv_report(sport: str | None = None) -> None:
    """Print a human-readable CLV report to stdout / log."""
    s = get_clv_summary(sport)
    if s["total_bets"] == 0:
        logger.info("[CLV Report] No graded bets with CLV data yet.")
        return

    verdict = "SHARP ✓" if s["avg_clv"] > 0 else "SOFT ✗"
    logger.info(
        f"\n{'━'*44}\n"
        f"  CLV REPORT{' — ' + sport.upper() if sport else ''}\n"
        f"{'━'*44}\n"
        f"  Total bets tracked : {s['total_bets']}\n"
        f"  Avg CLV            : {s['avg_clv']:+.2f}%  [{verdict}]\n"
        f"  Beat-rate          : {s['beat_rate']:.1f}%  (target ≥55%)\n"
        f"  Best pick (CLV)    : {s['best_bet_id']}\n"
        f"  Worst pick (CLV)   : {s['worst_bet_id']}\n"
        + (
            "\n  By sport:\n" +
            "\n".join(
                f"    {sp}: avg={d['avg_clv']:+.2f}%  beat={d['beat_rate']:.0f}%  n={d['count']}"
                for sp, d in s["by_sport"].items()
            ) if s["by_sport"] else ""
        ) +
        f"\n{'━'*44}"
    )
