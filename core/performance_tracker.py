"""
performance_tracker.py

Computes and caches cumulative lifetime performance statistics from
the bets table.  Single source of truth for all dashboard metrics.

Call rebuild_stats() after any grading event — it is fully idempotent
and rewrites the single performance_stats row from scratch every time.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "data" / "results.db"

# Season start — all stats only count picks on or after this date.
# Updated via bot_config table; defaults to epoch (all history) if absent.
_DEFAULT_SEASON_START = "2000-01-01T00:00:00"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _get_season_start(c: sqlite3.Connection) -> str:
    """Return the season-start ISO timestamp from bot_config."""
    try:
        row = c.execute(
            "SELECT value FROM bot_config WHERE key='season_start'"
        ).fetchone()
        if row and row[0]:
            return row[0]
    except Exception:
        pass
    return _DEFAULT_SEASON_START


def init_performance_table() -> None:
    """Create the performance_stats table and daily_archive if they don't exist."""
    c = _conn()
    try:
        # Migrate existing DB: add by_market column if missing
        try:
            c.execute(
                "ALTER TABLE performance_stats ADD COLUMN by_market TEXT NOT NULL DEFAULT '{}'"
            )
            c.commit()
        except Exception:
            pass  # column already exists — safe to ignore

        c.executescript("""
            CREATE TABLE IF NOT EXISTS bot_config (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS performance_stats (
                id                   INTEGER PRIMARY KEY DEFAULT 1,
                total_wins           INTEGER NOT NULL DEFAULT 0,
                total_losses         INTEGER NOT NULL DEFAULT 0,
                total_pushes         INTEGER NOT NULL DEFAULT 0,
                total_open           INTEGER NOT NULL DEFAULT 0,
                total_bets           INTEGER NOT NULL DEFAULT 0,
                win_rate             REAL    NOT NULL DEFAULT 0,
                units_won            REAL    NOT NULL DEFAULT 0,
                units_lost           REAL    NOT NULL DEFAULT 0,
                net_units            REAL    NOT NULL DEFAULT 0,
                roi_pct              REAL    NOT NULL DEFAULT 0,
                current_streak       INTEGER NOT NULL DEFAULT 0,
                current_streak_type  TEXT    NOT NULL DEFAULT '',
                longest_win_streak   INTEGER NOT NULL DEFAULT 0,
                longest_loss_streak  INTEGER NOT NULL DEFAULT 0,
                tier_nuke            TEXT    NOT NULL DEFAULT '{}',
                tier_diamond         TEXT    NOT NULL DEFAULT '{}',
                tier_value           TEXT    NOT NULL DEFAULT '{}',
                by_sport             TEXT    NOT NULL DEFAULT '{}',
                by_market            TEXT    NOT NULL DEFAULT '{}',
                last_7d              TEXT    NOT NULL DEFAULT '{}',
                last_30d             TEXT    NOT NULL DEFAULT '{}',
                last_updated         TEXT    NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS daily_archive (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                date_et    TEXT    NOT NULL,
                sport      TEXT    NOT NULL DEFAULT 'ALL',
                wins       INTEGER NOT NULL DEFAULT 0,
                losses     INTEGER NOT NULL DEFAULT 0,
                pushes     INTEGER NOT NULL DEFAULT 0,
                net_units  REAL    NOT NULL DEFAULT 0,
                roi_pct    REAL    NOT NULL DEFAULT 0,
                win_rate   REAL    NOT NULL DEFAULT 0,
                bets       INTEGER NOT NULL DEFAULT 0,
                created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE(date_et, sport)
            );
        """)
        c.commit()
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tier_record(rows: list[dict], tier: str) -> dict:
    sub = [r for r in rows if r["tier"] == tier]
    w = sum(1 for r in sub if r["outcome"] == "win")
    l = sum(1 for r in sub if r["outcome"] == "loss")
    p = sum(1 for r in sub if r["outcome"] == "push")
    u = round(sum(r["pl"] for r in sub) / 100, 2)
    return {
        "wins": w, "losses": l, "pushes": p,
        "win_rate": round(w / (w + l) * 100, 1) if (w + l) else 0,
        "net_units": u,
    }


def _window_stats(rows: list[dict], since_iso: str) -> dict:
    sub = [r for r in rows if r["timestamp"] >= since_iso]
    w = sum(1 for r in sub if r["outcome"] == "win")
    l = sum(1 for r in sub if r["outcome"] == "loss")
    p = sum(1 for r in sub if r["outcome"] == "push")
    u = round(sum(r["pl"] for r in sub) / 100, 2)
    closed = w + l
    return {
        "wins": w, "losses": l, "pushes": p,
        "win_rate": round(w / closed * 100, 1) if closed else 0,
        "net_units": u,
        "bets": len(sub),
    }


# ---------------------------------------------------------------------------
# Main rebuild — recomputes everything from the bets table
# ---------------------------------------------------------------------------

def rebuild_stats() -> dict[str, Any]:
    """
    Recompute ALL performance statistics from the bets table and
    upsert the single performance_stats row.  Returns the computed dict.

    Always reads from bets (source of truth) — the stats table is just
    a cached projection that is replaced on every call.

    Markets excluded from all W/L/ROI accounting (model retired or removed):
    """
    init_performance_table()
    c = _conn()

    # Markets permanently excluded from W/L and ROI accounting.
    # Add a market key here to retroactively remove it from all stats.
    _EXCLUDED_MARKETS = ("outs_recorded",)
    _excluded_placeholder = ",".join("?" * len(_EXCLUDED_MARKETS))

    try:
        season_start = _get_season_start(c)

        # All closed bets on or after season start, oldest first (for streaks).
        # Bets whose wager_details market is in _EXCLUDED_MARKETS are omitted.
        raw = c.execute(f"""
            SELECT bet_id, sport, tier, actual_outcome, profit_loss, timestamp
            FROM bets
            WHERE status = 'closed'
              AND timestamp >= ?
              AND COALESCE(json_extract(wager_details, '$.market'), '') NOT IN ({_excluded_placeholder})
            ORDER BY timestamp ASC
        """, (season_start, *_EXCLUDED_MARKETS)).fetchall()

        open_count: int = c.execute(
            "SELECT COUNT(*) FROM bets WHERE status='open' AND timestamp >= ?",
            (season_start,),
        ).fetchone()[0]

        # Normalise rows — skip anything without a clean win/loss/push outcome
        rows: list[dict] = []
        for r in raw:
            outcome = (r["actual_outcome"] or "").lower()
            if outcome not in ("win", "loss", "push"):
                continue
            rows.append({
                "bet_id":    r["bet_id"],
                "sport":     (r["sport"] or "").upper(),
                "tier":      r["tier"] or "",
                "outcome":   outcome,
                "pl":        float(r["profit_loss"] or 0),
                "timestamp": r["timestamp"] or "",
            })

        # ── Lifetime totals ───────────────────────────────────────────
        total_wins   = sum(1 for r in rows if r["outcome"] == "win")
        total_losses = sum(1 for r in rows if r["outcome"] == "loss")
        total_pushes = sum(1 for r in rows if r["outcome"] == "push")
        total_bets   = len(rows)
        closed       = total_wins + total_losses
        win_rate     = round(total_wins / closed * 100, 1) if closed else 0.0

        total_pl     = sum(r["pl"] for r in rows)
        units_won    = round(sum(r["pl"] for r in rows if r["pl"] > 0) / 100, 2)
        units_lost   = round(abs(sum(r["pl"] for r in rows if r["pl"] < 0)) / 100, 2)
        net_units    = round(total_pl / 100, 2)
        total_staked = total_bets * 100.0
        roi_pct      = round(total_pl / total_staked * 100, 2) if total_staked else 0.0

        # ── Streaks ───────────────────────────────────────────────────
        current_streak      = 0
        current_streak_type = ""
        longest_win_streak  = 0
        longest_loss_streak = 0
        run_w = run_l = 0

        for r in rows:
            if r["outcome"] == "win":
                run_w += 1; run_l = 0
                longest_win_streak = max(longest_win_streak, run_w)
            elif r["outcome"] == "loss":
                run_l += 1; run_w = 0
                longest_loss_streak = max(longest_loss_streak, run_l)
            else:
                run_w = run_l = 0  # push resets streak

        if rows:
            last_outcome = rows[-1]["outcome"]
            if last_outcome == "win":
                current_streak, current_streak_type = run_w, "W"
            elif last_outcome == "loss":
                current_streak, current_streak_type = run_l, "L"
            else:
                current_streak, current_streak_type = 0, "P"

        # ── Tier records ──────────────────────────────────────────────
        tier_nuke    = _tier_record(rows, "Nuke")
        tier_diamond = _tier_record(rows, "Diamond")
        # tier_value column repurposed for Gold Standard (Edge/Value are retired)
        tier_value   = _tier_record(rows, "Gold Standard")

        # ── Per-market records (T009) ──────────────────────────────────
        # Pulls market labels from wager_details JSON so this works even
        # for bets whose rows dict doesn't include the market column.
        by_market: dict[str, dict] = {}
        try:
            mkt_raw = c.execute(f"""
                SELECT
                    json_extract(wager_details, '$.market') AS mkt,
                    actual_outcome,
                    COALESCE(profit_loss, 0)                AS pl
                FROM bets
                WHERE status = 'closed'
                  AND actual_outcome IN ('win','loss','push')
                  AND timestamp >= ?
                  AND COALESCE(json_extract(wager_details, '$.market'), '') NOT IN ({_excluded_placeholder})
            """, (season_start, *_EXCLUDED_MARKETS)).fetchall()
            from collections import defaultdict as _dd
            _mkt_rows: dict[str, list] = _dd(list)
            for mr in mkt_raw:
                _m = (mr["mkt"] or "").strip()
                if _m:
                    _mkt_rows[_m].append(mr)
            for _m, _mrs in sorted(_mkt_rows.items()):
                _w = sum(1 for r in _mrs if r["actual_outcome"] == "win")
                _l = sum(1 for r in _mrs if r["actual_outcome"] == "loss")
                _p = sum(1 for r in _mrs if r["actual_outcome"] == "push")
                _u = round(sum(float(r["pl"]) for r in _mrs) / 100, 2)
                _stk = len(_mrs) * 100.0
                by_market[_m] = {
                    "wins":     _w,
                    "losses":   _l,
                    "pushes":   _p,
                    "bets":     len(_mrs),
                    "win_rate": round(_w / (_w + _l) * 100, 1) if (_w + _l) else 0,
                    "net_units": _u,
                    "roi_pct":  round(sum(float(r["pl"]) for r in _mrs) / _stk * 100, 2)
                                if _stk else 0.0,
                }
        except Exception:
            by_market = {}

        # ── Sport records ─────────────────────────────────────────────
        by_sport: dict[str, dict] = {}
        for sp in sorted({r["sport"] for r in rows}):
            sub = [r for r in rows if r["sport"] == sp]
            w = sum(1 for r in sub if r["outcome"] == "win")
            l = sum(1 for r in sub if r["outcome"] == "loss")
            p = sum(1 for r in sub if r["outcome"] == "push")
            u = round(sum(r["pl"] for r in sub) / 100, 2)
            by_sport[sp] = {
                "wins": w, "losses": l, "pushes": p,
                "win_rate": round(w / (w + l) * 100, 1) if (w + l) else 0,
                "net_units": u,
            }

        # ── Rolling windows ───────────────────────────────────────────
        now_utc   = datetime.now(timezone.utc)
        since_7d  = (now_utc - timedelta(days=7)).isoformat()
        since_30d = (now_utc - timedelta(days=30)).isoformat()
        last_7d   = _window_stats(rows, since_7d)
        last_30d  = _window_stats(rows, since_30d)

        # ── Rebuild daily_archive ─────────────────────────────────────
        _rebuild_daily_archive(c, rows)

        # ── Upsert the single stats row ───────────────────────────────
        now_str = now_utc.isoformat()
        c.execute("""
            INSERT INTO performance_stats (
                id, total_wins, total_losses, total_pushes, total_open,
                total_bets, win_rate, units_won, units_lost, net_units, roi_pct,
                current_streak, current_streak_type,
                longest_win_streak, longest_loss_streak,
                tier_nuke, tier_diamond, tier_value,
                by_sport, by_market, last_7d, last_30d, last_updated
            ) VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                total_wins=excluded.total_wins,
                total_losses=excluded.total_losses,
                total_pushes=excluded.total_pushes,
                total_open=excluded.total_open,
                total_bets=excluded.total_bets,
                win_rate=excluded.win_rate,
                units_won=excluded.units_won,
                units_lost=excluded.units_lost,
                net_units=excluded.net_units,
                roi_pct=excluded.roi_pct,
                current_streak=excluded.current_streak,
                current_streak_type=excluded.current_streak_type,
                longest_win_streak=excluded.longest_win_streak,
                longest_loss_streak=excluded.longest_loss_streak,
                tier_nuke=excluded.tier_nuke,
                tier_diamond=excluded.tier_diamond,
                tier_value=excluded.tier_value,
                by_sport=excluded.by_sport,
                by_market=excluded.by_market,
                last_7d=excluded.last_7d,
                last_30d=excluded.last_30d,
                last_updated=excluded.last_updated
        """, (
            total_wins, total_losses, total_pushes, open_count,
            total_bets, win_rate, units_won, units_lost, net_units, roi_pct,
            current_streak, current_streak_type,
            longest_win_streak, longest_loss_streak,
            json.dumps(tier_nuke), json.dumps(tier_diamond), json.dumps(tier_value),
            json.dumps(by_sport), json.dumps(by_market),
            json.dumps(last_7d), json.dumps(last_30d),
            now_str,
        ))
        c.commit()

        return get_stats()

    finally:
        c.close()


def _rebuild_daily_archive(conn: sqlite3.Connection, rows: list[dict]) -> None:
    """
    Recompute the daily_archive table from the normalised rows.
    Uses INSERT OR REPLACE so each (date_et, sport) pair is idempotent.
    """
    from collections import defaultdict

    # Group by (date_str, sport)
    by_date_sport: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_date_all:   dict[str, list[dict]]              = defaultdict(list)

    for r in rows:
        # Extract ET date from UTC timestamp (approximate: subtract 5 hours)
        try:
            ts = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
            date_et = (ts - timedelta(hours=5)).strftime("%Y-%m-%d")
        except Exception:
            date_et = r["timestamp"][:10]
        by_date_sport[(date_et, r["sport"])].append(r)
        by_date_all[date_et].append(r)

    def _stats(subset: list[dict]) -> dict:
        w = sum(1 for x in subset if x["outcome"] == "win")
        l = sum(1 for x in subset if x["outcome"] == "loss")
        p = sum(1 for x in subset if x["outcome"] == "push")
        u = round(sum(x["pl"] for x in subset) / 100, 2)
        staked = len(subset) * 100
        roi = round(sum(x["pl"] for x in subset) / staked * 100, 2) if staked else 0
        wr  = round(w / (w + l) * 100, 1) if (w + l) else 0
        return {"wins": w, "losses": l, "pushes": p,
                "net_units": u, "roi_pct": roi, "win_rate": wr, "bets": len(subset)}

    # Clear and rebuild
    conn.execute("DELETE FROM daily_archive")

    for (date_et, sport), subset in by_date_sport.items():
        s = _stats(subset)
        conn.execute("""
            INSERT OR REPLACE INTO daily_archive
              (date_et, sport, wins, losses, pushes, net_units, roi_pct, win_rate, bets)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (date_et, sport, s["wins"], s["losses"], s["pushes"],
              s["net_units"], s["roi_pct"], s["win_rate"], s["bets"]))

    for date_et, subset in by_date_all.items():
        s = _stats(subset)
        conn.execute("""
            INSERT OR REPLACE INTO daily_archive
              (date_et, sport, wins, losses, pushes, net_units, roi_pct, win_rate, bets)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (date_et, "ALL", s["wins"], s["losses"], s["pushes"],
              s["net_units"], s["roi_pct"], s["win_rate"], s["bets"]))


# ---------------------------------------------------------------------------
# Read-only accessor
# ---------------------------------------------------------------------------

def get_stats() -> dict[str, Any]:
    """
    Return the cached performance stats row.
    Calls rebuild_stats() if the table is empty (first-run bootstrap).
    """
    init_performance_table()
    c = _conn()
    try:
        row = c.execute("SELECT * FROM performance_stats WHERE id=1").fetchone()
    finally:
        c.close()

    if not row:
        return rebuild_stats()

    def _j(v: Any) -> Any:
        try:
            return json.loads(v or "{}")
        except Exception:
            return {}

    return {
        "total_wins":           row["total_wins"],
        "total_losses":         row["total_losses"],
        "total_pushes":         row["total_pushes"],
        "total_open":           row["total_open"],
        "total_bets":           row["total_bets"],
        "win_rate":             row["win_rate"],
        "units_won":            row["units_won"],
        "units_lost":           row["units_lost"],
        "net_units":            row["net_units"],
        "roi_pct":              row["roi_pct"],
        "current_streak":       row["current_streak"],
        "current_streak_type":  row["current_streak_type"],
        "longest_win_streak":   row["longest_win_streak"],
        "longest_loss_streak":  row["longest_loss_streak"],
        "tier_nuke":            _j(row["tier_nuke"]),
        "tier_diamond":         _j(row["tier_diamond"]),
        "tier_value":           _j(row["tier_value"]),
        "by_sport":             _j(row["by_sport"]),
        "by_market":            _j(row["by_market"]) if "by_market" in row.keys() else {},
        "last_7d":              _j(row["last_7d"]),
        "last_30d":             _j(row["last_30d"]),
        "last_updated":         row["last_updated"],
    }


def get_daily_archive(limit: int = 90) -> list[dict]:
    """Return per-date ALL-sport rows from daily_archive, newest first."""
    init_performance_table()
    c = _conn()
    try:
        rows = c.execute("""
            SELECT date_et, wins, losses, pushes, net_units, roi_pct, win_rate, bets
            FROM daily_archive
            WHERE sport = 'ALL'
            ORDER BY date_et DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        c.close()
