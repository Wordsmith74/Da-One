"""
supabase_sync.py — Sync picks from local SQLite to Supabase.

Called automatically by scheduler.py after each picks/grade/reconcile cycle.
Can also be run directly: python core/supabase_sync.py [--date YYYY-MM-DD]

Required environment variables:
  SUPABASE_URL              — Project URL (e.g. https://xxxx.supabase.co)
  SUPABASE_SERVICE_ROLE_KEY — Service role key (bypasses RLS for writes)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT    = Path(__file__).resolve().parent.parent
_DB_PATH = _ROOT / "data" / "results.db"


# ── Internal helpers ────────────────────────────────────────────────────────

def _supabase_client():
    try:
        from supabase import create_client
    except ImportError:
        raise ImportError("supabase-py not installed — run: pip install supabase")
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    return create_client(url, key)


def _db():
    import sqlite3
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _parse_wager(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


_TIER_MAP = {
    "nuke": "nuke",
    "diamond": "diamond",
    "gold standard": "gold standard",
    "gold": "gold standard",
}

_MARKET_MAP = {
    "player_points":       "Points",
    "player_assists":      "Assists",
    "player_rebounds":     "Rebounds",
    "player_threes":       "3-Pointers",
    "player_blocks":       "Blocks",
    "player_steals":       "Steals",
    "pitcher_strikeouts":  "Strikeouts",
    "player_hits":         "Hits",
    "player_home_runs":    "Home Runs",
    "player_total_bases":  "Total Bases",
    "player_rbis":         "RBIs",
    "nrfi":                "NRFI",
    "yrfi":                "YRFI",
    "f5_innings":          "F5 Innings",
    "team_total":          "Team Total",
    "moneyline":           "Moneyline",
    "spread":              "Spread",
    "totals":              "Totals",
}


def _normalize_tier(t: str | None) -> str:
    return _TIER_MAP.get((t or "").lower().strip().replace("☢️ ", "").replace("💎 ", ""), "gold standard")


def _normalize_market(m: str | None) -> str:
    if not m:
        return ""
    return _MARKET_MAP.get(m.lower(), m.replace("_", " ").title())


# ── Sync picks ──────────────────────────────────────────────────────────────

def sync_picks(slate_date: str | None = None) -> int:
    """
    Upsert all published picks (sent_to_group=1) to Supabase picks table.
    Uses ON CONFLICT bet_id so re-runs are safe (idempotent).
    Returns number of rows upserted.
    """
    sb   = _supabase_client()
    conn = _db()
    try:
        sql    = "SELECT * FROM bets WHERE sent_to_group = 1 AND tier IS NOT NULL"
        params: list = []
        if slate_date:
            sql += " AND slate_date = ?"
            params.append(slate_date)

        rows = conn.execute(sql, params).fetchall()
        if not rows:
            logger.info("supabase_sync.picks: nothing to sync")
            return 0

        records = []
        for row in rows:
            wd = _parse_wager(row["wager_details"])

            # verified_at: portal expects ISO timestamp or null (not "9:02 AM ET")
            verified_raw = wd.get("verified_at")
            verified_at  = verified_raw if verified_raw and "T" in str(verified_raw) else None

            records.append({
                "bet_id":                 row["bet_id"],
                "tier":                   _normalize_tier(row["tier"]),
                "status":                 row["status"] or "open",
                "outcome":                row["actual_outcome"],
                "player":                 wd.get("player"),
                "market":                 _normalize_market(wd.get("market")),
                "direction":              wd.get("direction", "over"),
                "line":                   wd.get("sportsbook_line"),
                "team":                   wd.get("team"),
                "sport":                  row["sport"],
                "edge":                   row["edge_percentage"],
                "odds":                   row["sportsbook_odds"],
                "bookmaker_source":       row["bookmaker_source"] or "DraftKings",
                "model_probability":      row["model_probability"],
                "opening_line":           wd.get("opening_line"),
                "opening_edge":           row["opening_edge"],
                "is_locked":              bool(row["is_locked"]),
                "verified_at":            verified_at,
                "l5_avg":                 wd.get("l5_avg"),
                "l10_avg":                wd.get("l10_avg"),
                "weighted_projection":    wd.get("weighted_projection"),
                "mis_score":              wd.get("mis_score"),
                "market_agreement_score": wd.get("market_agreement_score"),
                "data_reliability_score": wd.get("data_reliability_score"),
                "effective_edge":         wd.get("effective_edge"),
                "sharp_signal":           wd.get("sharp_signal"),
                "rlm_detected":           bool(wd.get("rlm_detected", False)),
                "steam_detected":         bool(wd.get("steam_detected", False)),
                "reval_status":           row["revalidation_status"],
                "reval_reason":           row["revalidation_reason"],
                "reval_edge":             row["current_edge"],
                "reval_opening_edge":     row["opening_edge"],
                "reval_at":               row["revalidation_at"],
                "created_at":             row["published_at"] or row["timestamp"],
            })

        synced = 0
        for i in range(0, len(records), 50):
            sb.table("picks").upsert(records[i:i + 50], on_conflict="bet_id").execute()
            synced += len(records[i:i + 50])

        # Mark rows as synced so we only process deltas on future runs
        ids          = [r["bet_id"] for r in records]
        placeholders = ",".join("?" * len(ids))
        conn.execute(f"UPDATE bets SET sent_to_miniapp=1 WHERE bet_id IN ({placeholders})", ids)
        conn.commit()

        logger.info(f"supabase_sync.picks: upserted {synced} rows")
        return synced
    finally:
        conn.close()


# ── Sync revalidations ───────────────────────────────────────────────────────

def sync_revalidations() -> int:
    """
    Push new revalidation events from pick_regrade_history to Supabase.
    Uses alert_sent flag to process only new rows (idempotent).
    Returns number of rows inserted.
    """
    sb   = _supabase_client()
    conn = _db()
    try:
        rows = conn.execute("""
            SELECT id, bet_id, changed_at, change_type, reason, new_edge, prev_edge
            FROM   pick_regrade_history
            WHERE  alert_sent = 0
              AND  change_type IN ('voided','downgraded','upgraded','confirmed','revalidation')
        """).fetchall()

        if not rows:
            return 0

        records = [{
            "bet_id":               r["bet_id"],
            "revalidation_status":  r["change_type"] if r["change_type"] != "revalidation" else "confirmed",
            "revalidation_reason":  r["reason"],
            "current_edge":         r["new_edge"],
            "opening_edge":         r["prev_edge"],
            "revalidation_at":      r["changed_at"],
        } for r in rows]

        for i in range(0, len(records), 50):
            sb.table("revalidation_updates").insert(records[i:i + 50]).execute()

        ids          = [r["id"] for r in rows]
        placeholders = ",".join("?" * len(ids))
        conn.execute(
            f"UPDATE pick_regrade_history SET alert_sent=1 WHERE id IN ({placeholders})", ids
        )
        conn.commit()

        logger.info(f"supabase_sync.revals: inserted {len(records)} rows")
        return len(records)
    finally:
        conn.close()


# ── Master sync ──────────────────────────────────────────────────────────────

def sync_all(slate_date: str | None = None) -> dict:
    """
    Full sync: picks + revalidations. Safe to call multiple times (idempotent).
    Returns a summary dict: {picks, revalidations, ok, error?}
    """
    result: dict = {"picks": 0, "revalidations": 0, "ok": True}
    try:
        result["picks"]         = sync_picks(slate_date)
        result["revalidations"] = sync_revalidations()
    except Exception as exc:
        logger.error(f"supabase_sync error: {exc}", exc_info=True)
        result["ok"]    = False
        result["error"] = str(exc)
    return result


# ── Standalone CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    ap = argparse.ArgumentParser(description="Sync SQLite picks to Supabase")
    ap.add_argument("--date", metavar="YYYY-MM-DD", help="Restrict to a specific slate_date")
    args = ap.parse_args()

    out = sync_all(slate_date=args.date)
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["ok"] else 1)
