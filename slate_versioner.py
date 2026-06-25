"""
slate_versioner.py

Immutable pick-slate versioning system.

  – Every time picks are broadcast, this module snapshots the full slate.
  – The first snapshot of the day becomes Official Slate v1 (locked).
  – Subsequent broadcasts create v2, v3… and auto-detect what changed vs v1.
  – Change alerts are sent to Telegram when any material difference is found.
  – API endpoints expose version history and the change log.

Entry points:
    snapshot_slate(bets_by_sport, trigger_reason, sent_bet_ids, dry_run)
    get_today_versions()
    get_version_picks(version_id)
    get_today_changelog()
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "data" / "results.db"

TIER_LABEL = {"Nuke": "Nuke ☢️", "Diamond": "Diamond 💎", "Edge": "Edge ⚡",
              "S+": "Nuke ☢️", "S": "Diamond 💎", "Value": "Edge ⚡"}  # legacy compat


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_et() -> str:
    from core.time_utils import now_est
    return now_est().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Tier helpers
# ---------------------------------------------------------------------------

def _is_nuke(tier: str) -> bool:
    t = tier.strip().lower()
    return t in ("nuke", "s+")


def _is_diamond(tier: str) -> bool:
    t = tier.strip().lower()
    return t in ("diamond", "s")


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def snapshot_slate(
    bets_by_sport: dict,           # sport → list[BetDisplay]
    trigger_reason: str = "scheduled",
    sent_bet_ids: set[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Persist a full slate snapshot and return a result dict.

    Parameters
    ----------
    bets_by_sport   : Same dict passed to send_daily_picks().
    trigger_reason  : 'scheduled' | 'manual' | 'line_movement' | 'revalidation' | …
    sent_bet_ids    : Set of bet_ids that were actually sent (Nuke/Diamond/Edge).
                      When None, all bets in bets_by_sport are snapshotted.
    dry_run         : If True, do not write to DB or send alerts.

    Returns a dict: {version, version_id, changes, alert_sent}
    """
    if dry_run:
        return {"version": 0, "version_id": None, "changes": [], "alert_sent": False}

    date_et   = _today_et()
    now_str   = _now_iso()
    c         = _conn()

    try:
        # ── 1. What's the next version number for today? ──────────────────
        row = c.execute(
            "SELECT MAX(version) AS mv FROM slate_versions WHERE date_et=?",
            (date_et,),
        ).fetchone()
        prev_version = row["mv"] or 0
        new_version  = prev_version + 1
        is_locked    = 1 if new_version == 1 else 0

        # ── 2. Build pick rows from BetDisplay objects ────────────────────
        pick_rows: list[dict] = []
        for sport, bet_displays in bets_by_sport.items():
            for bd in bet_displays:
                bet  = bd.bet
                tier = (bet.tier.value if hasattr(bet.tier, "value") else str(bet.tier)) or "Value"
                snap = {
                    "bet_id":           bet.bet_id,
                    "sport":            sport,
                    "tier":             tier,
                    "is_nuke":          1 if _is_nuke(tier) else 0,
                    "is_diamond":       1 if _is_diamond(tier) else 0,
                    "team":             getattr(bet, "team", None),
                    "player":           getattr(bet, "player", None),
                    "market":           getattr(bet, "market", ""),
                    "direction":        getattr(bet, "direction", ""),
                    "sportsbook_line":  getattr(bet, "sportsbook_line", None),
                    "sportsbook_odds":  int(bd.american_odds) if hasattr(bd, "american_odds") else None,
                    "edge_percentage":  getattr(bet, "edge_percentage", 0),
                    "confidence_score": getattr(bet, "confidence_score", 0),
                    "model_probability": float(getattr(bd, "model_probability", 0)),
                    "bookmaker_source": getattr(bd, "bookmaker_source", ""),
                }
                snap["snapshot_json"] = json.dumps(snap)
                pick_rows.append(snap)

        # ── 3. Insert slate_versions row ──────────────────────────────────
        c.execute(
            """
            INSERT OR IGNORE INTO slate_versions
              (date_et, version, triggered_at, trigger_reason, picks_count, is_locked)
            VALUES (?,?,?,?,?,?)
            """,
            (date_et, new_version, now_str, trigger_reason, len(pick_rows), is_locked),
        )
        version_id = c.execute(
            "SELECT id FROM slate_versions WHERE date_et=? AND version=?",
            (date_et, new_version),
        ).fetchone()["id"]

        # ── 4. Insert picks ───────────────────────────────────────────────
        for p in pick_rows:
            c.execute(
                """
                INSERT INTO slate_version_picks
                  (version_id, bet_id, sport, tier, is_nuke, is_diamond,
                   team, player, market, direction, sportsbook_line,
                   sportsbook_odds, edge_percentage, confidence_score,
                   model_probability, bookmaker_source, snapshot_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    version_id,
                    p["bet_id"], p["sport"], p["tier"],
                    p["is_nuke"], p["is_diamond"],
                    p["team"], p["player"], p["market"], p["direction"],
                    p["sportsbook_line"], p["sportsbook_odds"],
                    p["edge_percentage"], p["confidence_score"],
                    p["model_probability"], p["bookmaker_source"],
                    p["snapshot_json"],
                ),
            )

        c.commit()

        # ── 5. Diff against v1 (only for v2+) ────────────────────────────
        changes: list[dict] = []
        alert_sent          = False

        if new_version > 1:
            v1_id = c.execute(
                "SELECT id FROM slate_versions WHERE date_et=? AND version=1",
                (date_et,),
            ).fetchone()
            if v1_id:
                v1_picks = _load_picks(c, v1_id["id"])
                v2_picks = pick_rows
                changes  = _diff_versions(v1_picks, v2_picks, date_et, new_version, trigger_reason, now_str)

                for ch in changes:
                    c.execute(
                        """
                        INSERT INTO slate_changes
                          (date_et, from_version, to_version, change_type, bet_id,
                           old_snapshot_json, new_snapshot_json, changed_fields_json,
                           trigger_reason, created_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            date_et, 1, new_version,
                            ch["change_type"], ch["bet_id"],
                            ch.get("old_snapshot_json"),
                            ch.get("new_snapshot_json"),
                            json.dumps(ch.get("changed_fields", {})),
                            trigger_reason, now_str,
                        ),
                    )
                c.commit()

                if changes:
                    alert_text = _format_change_alert(changes, new_version, trigger_reason, date_et)
                    _send_telegram(alert_text)
                    alert_sent = True

        return {
            "version":    new_version,
            "version_id": version_id,
            "is_v1":      is_locked == 1,
            "changes":    changes,
            "alert_sent": alert_sent,
        }

    finally:
        c.close()


# ---------------------------------------------------------------------------
# Pick loader (for diff)
# ---------------------------------------------------------------------------

def _load_picks(c: sqlite3.Connection, version_id: int) -> list[dict]:
    rows = c.execute(
        "SELECT * FROM slate_version_picks WHERE version_id=?",
        (version_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

_ODDS_CHANGE_THRESHOLD = 10      # points (American odds)
_CONF_CHANGE_THRESHOLD = 5.0     # percentage points
_EDGE_CHANGE_THRESHOLD = 3.0     # percentage points


def _diff_versions(
    v1: list[dict],
    v2: list[dict],
    date_et: str,
    to_version: int,
    trigger_reason: str,
    now_str: str,
) -> list[dict]:
    """Return a list of change dicts describing every material difference."""
    v1_map = {p["bet_id"]: p for p in v1}
    v2_map = {p["bet_id"]: p for p in v2}
    changes: list[dict] = []

    # Added picks
    for bid, p in v2_map.items():
        if bid not in v1_map:
            changes.append({
                "change_type":       "added",
                "bet_id":            bid,
                "old_snapshot_json": None,
                "new_snapshot_json": json.dumps(p),
                "changed_fields":    {},
            })

    # Removed picks
    for bid, p in v1_map.items():
        if bid not in v2_map:
            changes.append({
                "change_type":       "removed",
                "bet_id":            bid,
                "old_snapshot_json": json.dumps(p),
                "new_snapshot_json": None,
                "changed_fields":    {},
            })

    # Modified picks
    for bid in v1_map:
        if bid not in v2_map:
            continue
        old, new = v1_map[bid], v2_map[bid]
        fields: dict = {}

        if old.get("tier") != new.get("tier"):
            fields["tier"] = {"old": old.get("tier"), "new": new.get("tier")}

        if old.get("is_nuke") != new.get("is_nuke"):
            fields["nuke"] = {"old": bool(old.get("is_nuke")), "new": bool(new.get("is_nuke"))}

        if old.get("is_diamond") != new.get("is_diamond"):
            fields["diamond"] = {"old": bool(old.get("is_diamond")), "new": bool(new.get("is_diamond"))}

        old_odds = old.get("sportsbook_odds") or 0
        new_odds = new.get("sportsbook_odds") or 0
        if abs(old_odds - new_odds) >= _ODDS_CHANGE_THRESHOLD:
            fields["odds"] = {"old": old_odds, "new": new_odds}

        old_conf = float(old.get("confidence_score") or 0)
        new_conf = float(new.get("confidence_score") or 0)
        if abs(old_conf - new_conf) >= _CONF_CHANGE_THRESHOLD:
            fields["confidence"] = {
                "old": round(old_conf, 1),
                "new": round(new_conf, 1),
                "delta": round(new_conf - old_conf, 1),
            }

        old_edge = float(old.get("edge_percentage") or 0)
        new_edge = float(new.get("edge_percentage") or 0)
        if abs(old_edge - new_edge) >= _EDGE_CHANGE_THRESHOLD:
            fields["edge"] = {
                "old": round(old_edge, 1),
                "new": round(new_edge, 1),
                "delta": round(new_edge - old_edge, 1),
            }

        if fields:
            changes.append({
                "change_type":       "modified",
                "bet_id":            bid,
                "old_snapshot_json": json.dumps(old),
                "new_snapshot_json": json.dumps(new),
                "changed_fields":    fields,
            })

    return changes


# ---------------------------------------------------------------------------
# Telegram alert formatter
# ---------------------------------------------------------------------------

def _fmt_odds(o: int | None) -> str:
    if o is None:
        return "—"
    return f"+{o}" if o > 0 else str(o)


def _pick_label(snap: dict | None) -> str:
    if not snap:
        return "?"
    name  = snap.get("player") or snap.get("team") or snap.get("sport", "?")
    mkt   = (snap.get("market") or "").replace("_", " ").title()
    dirn  = (snap.get("direction") or "").upper()
    line  = snap.get("sportsbook_line")
    odds  = _fmt_odds(snap.get("sportsbook_odds"))
    parts = [name]
    if mkt:
        parts.append(f"{mkt} {dirn}" + (f" {line}" if line is not None else ""))
    if odds != "—":
        parts.append(f"({odds})")
    return "  ".join(parts)


def _format_change_alert(
    changes: list[dict],
    new_version: int,
    trigger_reason: str,
    date_et: str,
) -> str:
    from datetime import datetime as dt
    try:
        d = dt.strptime(date_et, "%Y-%m-%d")
        date_label = d.strftime("%b %-d, %Y")
    except Exception:
        date_label = date_et

    reason_labels = {
        "manual":       "Manual re-broadcast",
        "line_movement": "Line movement",
        "scheduled":    "Scheduled run",
        "revalidation": "Pregame revalidation",
        "injury":       "Injury update",
        "lineup":       "Lineup change",
    }
    reason_label = reason_labels.get(trigger_reason, trigger_reason.replace("_", " ").title())

    added    = [c for c in changes if c["change_type"] == "added"]
    removed  = [c for c in changes if c["change_type"] == "removed"]
    modified = [c for c in changes if c["change_type"] == "modified"]

    lines = [
        f"🚨 SLATE UPDATE — Official Slate v{new_version} 🚨",
        f"📅 {date_label}  |  Trigger: {reason_label}",
        "",
    ]

    if added:
        lines.append(f"➕ ADDED ({len(added)})")
        for ch in added:
            snap = json.loads(ch["new_snapshot_json"] or "{}")
            tier = snap.get("tier", "")
            prefix = "☢️" if _is_nuke(tier) else ("💎" if _is_diamond(tier) else "⚡")
            lines.append(f"  {prefix} {_pick_label(snap)}")
        lines.append("")

    if removed:
        lines.append(f"➖ REMOVED ({len(removed)})")
        for ch in removed:
            snap = json.loads(ch["old_snapshot_json"] or "{}")
            tier = snap.get("tier", "")
            prefix = "☢️" if _is_nuke(tier) else ("💎" if _is_diamond(tier) else "⚡")
            lines.append(f"  {prefix} {_pick_label(snap)}")
        lines.append("")

    if modified:
        lines.append(f"🔄 CHANGED ({len(modified)})")
        for ch in modified:
            snap  = json.loads(ch["new_snapshot_json"] or "{}")
            name  = snap.get("player") or snap.get("team") or "?"
            flds  = ch.get("changed_fields", {})
            parts = []
            if "tier" in flds:
                parts.append(f"tier {flds['tier']['old']} → {flds['tier']['new']}")
            if "nuke" in flds:
                st = "gained ☢️ Nuke" if flds["nuke"]["new"] else "lost ☢️ Nuke"
                parts.append(st)
            if "diamond" in flds:
                st = "gained 💎 Diamond" if flds["diamond"]["new"] else "lost 💎 Diamond"
                parts.append(st)
            if "odds" in flds:
                o = flds["odds"]
                parts.append(f"odds {_fmt_odds(o['old'])} → {_fmt_odds(o['new'])}")
            if "confidence" in flds:
                cf = flds["confidence"]
                sign = "+" if cf["delta"] >= 0 else ""
                parts.append(f"conf {cf['old']}% → {cf['new']}% ({sign}{cf['delta']}pp)")
            if "edge" in flds:
                ef = flds["edge"]
                sign = "+" if ef["delta"] >= 0 else ""
                parts.append(f"edge {ef['old']}% → {ef['new']}% ({sign}{ef['delta']}pp)")
            lines.append(f"  • {name}: {', '.join(parts)}")
        lines.append("")

    lines.append(
        "📋 View full change log in the MiniApp → Change Log tab."
    )
    return "\n".join(lines)


def _send_telegram(text: str) -> None:
    try:
        from output.telegram_formatter import send_to_telegram
        send_to_telegram(text)
    except Exception as exc:
        print(f"[slate_versioner] Telegram alert failed: {exc}")


# ---------------------------------------------------------------------------
# Read-only query helpers (used by API endpoints)
# ---------------------------------------------------------------------------

def get_today_versions(date_et: str | None = None) -> list[dict]:
    """Return all slate versions for a given date, newest first."""
    date_et = date_et or _today_et()
    c = _conn()
    try:
        rows = c.execute(
            """
            SELECT id, date_et, version, triggered_at, trigger_reason,
                   picks_count, is_locked
            FROM slate_versions
            WHERE date_et = ?
            ORDER BY version ASC
            """,
            (date_et,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        c.close()


def get_version_picks(version_id: int) -> list[dict]:
    """Return all picks snapshotted for a given version_id."""
    c = _conn()
    try:
        rows = c.execute(
            """
            SELECT bet_id, sport, tier, is_nuke, is_diamond,
                   team, player, market, direction,
                   sportsbook_line, sportsbook_odds,
                   edge_percentage, confidence_score,
                   model_probability, bookmaker_source
            FROM slate_version_picks
            WHERE version_id = ?
            ORDER BY is_nuke DESC, is_diamond DESC, edge_percentage DESC
            """,
            (version_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        c.close()


def get_today_changelog(date_et: str | None = None) -> list[dict]:
    """Return all slate_changes for today, newest first."""
    date_et = date_et or _today_et()
    c = _conn()
    try:
        rows = c.execute(
            """
            SELECT id, date_et, from_version, to_version, change_type,
                   bet_id, old_snapshot_json, new_snapshot_json,
                   changed_fields_json, trigger_reason, created_at
            FROM slate_changes
            WHERE date_et = ?
            ORDER BY created_at DESC
            """,
            (date_et,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["old_snapshot"]    = json.loads(d["old_snapshot_json"] or "null")
                d["new_snapshot"]    = json.loads(d["new_snapshot_json"] or "null")
                d["changed_fields"]  = json.loads(d["changed_fields_json"] or "{}")
            except Exception:
                pass
            result.append(d)
        return result
    finally:
        c.close()
