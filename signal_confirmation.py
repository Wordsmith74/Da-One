"""
core/signal_confirmation.py

Pick Lifecycle: Candidate → Confirmed → Locked

A signal must persist across ≥3 scheduler refresh cycles AND be at least
min_minutes old before it is eligible for publication (Locked status).

  NBA / WNBA : min_cycles=3,  min_minutes=5
  MLB        : min_cycles=3,  min_minutes=5

The stable identity key is (game_id, market_norm, direction) — not bet_id,
which is regenerated on every run.

Rejection rules:
  - Edge weakens > 25 % from its first-sighting value → rejected
  - Direction flips can't happen (key includes direction)

Exposed API:
  gate_signals(bets, sport, *, dry_run=False)  → (ready_to_lock, held)
  purge_old_signals(days=2)                    → int (rows deleted)
  get_candidate_counts()                       → dict[str, int]  (status → count)
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from core.time_utils import now_est

if TYPE_CHECKING:
    from core.decision_gatekeeper import Bet

_DB_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "results.db")
)

# Minimum confirmation requirements per sport
_MIN_CYCLES: dict[str, int] = {
    "NBA":  3,
    "WNBA": 3,
    "MLB":  3,
}
_MIN_MINUTES: dict[str, int] = {
    "NBA":  5,
    "WNBA": 5,
    "MLB":  5,
}

# Reject a signal if edge drops by more than this fraction from first sighting
_EDGE_WEAKEN_RATIO = 0.25   # 25 % decline → candidate rejected


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _init_table() -> None:
    conn = _conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS signal_queue (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_key       TEXT    NOT NULL UNIQUE,
            sport            TEXT    NOT NULL,
            game_id          TEXT    NOT NULL,
            market           TEXT    NOT NULL,
            market_norm      TEXT    NOT NULL,
            direction        TEXT    NOT NULL,
            team             TEXT,
            first_seen_at    TEXT    NOT NULL,
            last_seen_at     TEXT    NOT NULL,
            first_edge       REAL    NOT NULL,
            first_conf       REAL    NOT NULL,
            current_edge     REAL,
            current_conf     REAL,
            cycle_count      INTEGER NOT NULL DEFAULT 1,
            status           TEXT    NOT NULL DEFAULT 'candidate',
            rejection_reason TEXT,
            locked_bet_id    TEXT,
            created_date     TEXT    NOT NULL DEFAULT (date('now'))
        );
        CREATE INDEX IF NOT EXISTS ix_sq_key  ON signal_queue(signal_key);
        CREATE INDEX IF NOT EXISTS ix_sq_date ON signal_queue(created_date);
    """)
    conn.commit()
    conn.close()


def _mkt_norm(market: str) -> str:
    return market.lower().strip().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gate_signals(
    bets: list,
    sport: str,
    *,
    dry_run: bool = False,
) -> tuple[list, list]:
    """
    Filter bets through the signal confirmation gate.

    dry_run=True  →  all bets pass immediately (bypass confirmation delay).

    Returns:
        ready_to_lock   — confirmed signals eligible for publication
        held            — candidates saved to queue but not yet ready
    """
    if dry_run:
        return list(bets), []

    _init_table()
    now        = now_est()
    today      = now.strftime("%Y-%m-%d")
    min_cycles  = _MIN_CYCLES.get(sport.upper(), 3)
    min_minutes = _MIN_MINUTES.get(sport.upper(), 5)

    ready: list = []
    held:  list = []
    conn = _conn()

    try:
        for bet in bets:
            gid  = getattr(bet, "game_id", "") or ""
            mkt  = getattr(bet, "market", "")  or ""
            mn   = _mkt_norm(mkt)
            drct = (getattr(bet, "direction", "") or "").lower()
            key  = f"{gid}|{mn}|{drct}"

            new_edge = float(getattr(bet, "edge_percentage", 0.0) or 0.0)
            new_conf = float(getattr(bet, "confidence_score", 0.0) or 0.0)

            row = conn.execute(
                "SELECT * FROM signal_queue WHERE signal_key = ?", (key,)
            ).fetchone()

            if row is None:
                # ── First sighting: register as Candidate ─────────────────
                conn.execute(
                    """
                    INSERT INTO signal_queue
                        (signal_key, sport, game_id, market, market_norm, direction,
                         team, first_seen_at, last_seen_at,
                         first_edge, first_conf, current_edge, current_conf,
                         cycle_count, status, created_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'candidate', ?)
                    """,
                    (
                        key, sport.upper(), gid, mkt, mn, drct,
                        getattr(bet, "team", "") or "",
                        now.isoformat(), now.isoformat(),
                        new_edge, new_conf,
                        new_edge, new_conf,
                        today,
                    ),
                )
                held.append(bet)
                continue

            # ── Signal seen before ────────────────────────────────────────
            first_edge = float(row["first_edge"] or 0.0)

            # Reject if edge weakened materially
            if first_edge > 0.0 and new_edge < first_edge * (1.0 - _EDGE_WEAKEN_RATIO):
                conn.execute(
                    """
                    UPDATE signal_queue
                    SET status='rejected',
                        rejection_reason=?,
                        last_seen_at=?,
                        current_edge=?
                    WHERE signal_key=?
                    """,
                    (
                        f"Edge weakened: {first_edge:.2f}% → {new_edge:.2f}%",
                        now.isoformat(), new_edge, key,
                    ),
                )
                held.append(bet)
                continue

            # Increment count and check confirmation thresholds
            new_count     = int(row["cycle_count"]) + 1
            first_seen    = datetime.fromisoformat(row["first_seen_at"])
            elapsed_min   = (now - first_seen).total_seconds() / 60.0
            criteria_met  = new_count >= min_cycles and elapsed_min >= min_minutes
            new_status    = "confirmed" if criteria_met else row["status"]

            conn.execute(
                """
                UPDATE signal_queue
                SET last_seen_at=?, current_edge=?, current_conf=?,
                    cycle_count=?, status=?
                WHERE signal_key=?
                """,
                (
                    now.isoformat(), new_edge, new_conf,
                    new_count, new_status, key,
                ),
            )

            if new_status == "confirmed":
                ready.append(bet)
            else:
                held.append(bet)

        conn.commit()
    finally:
        conn.close()

    return ready, held


def purge_old_signals(days: int = 2) -> int:
    """Remove signal_queue rows older than *days*. Returns number of rows deleted."""
    _init_table()
    cutoff = (now_est() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = _conn()
    try:
        cur = conn.execute(
            "DELETE FROM signal_queue WHERE created_date < ?", (cutoff,)
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def get_candidate_counts() -> dict[str, int]:
    """Return today's signal_queue counts grouped by status."""
    _init_table()
    today = now_est().strftime("%Y-%m-%d")
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM signal_queue "
            "WHERE created_date = ? GROUP BY status",
            (today,),
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}
    finally:
        conn.close()
