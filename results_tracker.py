"""
results_tracker.py

Logs every bet to a SQLite database, computes running ROI, and refines
the model's sport weights based on actual outcomes (the learning loop).

Database lives at: data/results.db  (created automatically)
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from core.decision_gatekeeper import Bet, Tier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DB_PATH     = Path(__file__).parent.parent / "data" / "results.db"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "sports_metrics.json"

# ---------------------------------------------------------------------------
# Market → weight-key mapping for model refinement
# ---------------------------------------------------------------------------
# Over-direction → the weight key to nudge DOWN when the model overestimates.
# Under-direction → the weight key to nudge DOWN when model overestimates UNDERs.

_MARKET_WEIGHT_MAP: dict[str, dict[str, dict[str, str]]] = {
    "WNBA": {
        "player_points":   {"over": "off_efficiency", "under": "def_efficiency", "group": "total_weights"},
        "player_assists":  {"over": "pace",            "under": "def_efficiency", "group": "total_weights"},
        "player_rebounds": {"over": "off_efficiency",  "under": "def_efficiency", "group": "total_weights"},
        "team_total":      {"over": "off_efficiency",  "under": "def_efficiency", "group": "total_weights"},
        "totals":          {"over": "off_efficiency",  "under": "def_efficiency", "group": "total_weights"},
        "team_spread":     {"over": "power_rating",    "under": "matchup",        "group": "spread_weights"},
    },
    "NBA": {
        "player_points":   {"over": "off_efficiency", "under": "def_efficiency", "group": "total_weights"},
        "player_assists":  {"over": "pace",            "under": "def_efficiency", "group": "total_weights"},
        "player_rebounds": {"over": "off_efficiency",  "under": "def_efficiency", "group": "total_weights"},
        "team_total":      {"over": "off_efficiency",  "under": "def_efficiency", "group": "total_weights"},
        "totals":          {"over": "off_efficiency",  "under": "def_efficiency", "group": "total_weights"},
        "team_spread":     {"over": "power_rating",    "under": "matchup",        "group": "spread_weights"},
    },
    "MLB": {
        "team_total":  {"over": "lineup_offense",       "under": "starting_pitcher_era", "group": "total_weights"},
        "totals":      {"over": "lineup_offense",       "under": "starting_pitcher_era", "group": "total_weights"},
        "team_spread": {"over": "pitcher_power",        "under": "bullpen",              "group": "spread_weights"},
    },
}

# Minimum resolved bets in a (sport, market, direction) bucket before we
# apply any weight adjustment. Prevents over-fitting on small samples.
MIN_SAMPLE_FOR_REFINEMENT = 10

# Maximum weight adjustment per run (keeps changes gradual)
MAX_NUDGE = 0.02


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def has_open_opposite_bet(
    sport:     str,
    game_id:   str,
    team:      str,
    market:    str,
    direction: str,
) -> bool:
    """
    Return True if the DB already contains an open bet on the **same**
    team / market / game in the **opposite** direction.

    Used as a cross-run contradiction guard: prevents the engine from
    broadcasting e.g. MIL team_total UNDER when MIL team_total OVER is
    already open for the same game, which guarantees one pick must lose.

    Requirements:
      - game_id must be non-empty (bets with blank game_id are skipped).
      - The existing bet must have status = 'open'.
      - direction comparison is case-insensitive.
    """
    if not game_id:
        return False

    opp = "under" if direction.lower() == "over" else "over"

    with _connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM bets
            WHERE sport   = ?
              AND game_id = ?
              AND status  = 'open'
              AND JSON_EXTRACT(wager_details, '$.team')      = ?
              AND JSON_EXTRACT(wager_details, '$.market')    = ?
              AND JSON_EXTRACT(wager_details, '$.direction') = ?
            """,
            (sport.upper(), game_id, team, market, opp),
        ).fetchone()
        return (row[0] or 0) > 0


def init_db() -> None:
    """Create the bets table if it doesn't exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                bet_id          TEXT    UNIQUE NOT NULL,
                sport           TEXT    NOT NULL,
                wager_details   TEXT    NOT NULL,
                model_probability REAL  NOT NULL,
                sportsbook_odds INTEGER NOT NULL,
                actual_outcome  TEXT,
                profit_loss     REAL,
                status          TEXT    NOT NULL DEFAULT 'open',
                stake           REAL    NOT NULL DEFAULT 100.0,
                tier            TEXT,
                edge_percentage REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_bets_sport_status
            ON bets (sport, status)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weight_adjustments (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                sport       TEXT NOT NULL,
                weight_group TEXT NOT NULL,
                weight_key  TEXT NOT NULL,
                old_value   REAL NOT NULL,
                new_value   REAL NOT NULL,
                reason      TEXT
            )
        """)
        _migrate_db(conn)


def _today_et() -> str:
    """Return today's date in America/New_York as YYYY-MM-DD."""
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _migrate_db(conn: sqlite3.Connection) -> None:
    """
    Safely add missing columns to existing tables on every startup.
    Uses PRAGMA table_info so each ALTER is executed exactly once.
    """
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(bets)")}

    new_cols: list[tuple[str, str]] = [
        ("sent_to_group",      "INTEGER NOT NULL DEFAULT 0"),
        ("bookmaker_source",   "TEXT    NOT NULL DEFAULT ''"),
        ("current_market_price","INTEGER"),
        ("sent_to_miniapp",    "INTEGER NOT NULL DEFAULT 0"),
        ("game_id",            "TEXT"),
        ("closing_price",      "INTEGER"),
        ("clv_pct",            "REAL"),
        ("mid_price",          "INTEGER"),
        ("line_move_dir",      "TEXT"),
        # ── Slate lock & revalidation columns ───────────────────────────────
        ("slate_date",         "TEXT"),
        ("is_locked",          "INTEGER NOT NULL DEFAULT 0"),
        ("published_at",       "TEXT"),
        ("opening_edge",       "REAL"),
        ("opening_confidence", "REAL"),
        ("opening_odds",       "INTEGER"),
        ("current_edge",       "REAL"),
        ("current_confidence", "REAL"),
        ("closing_edge",       "REAL"),
        ("revalidation_status","TEXT"),
        ("revalidation_reason","TEXT"),
        ("revalidation_at",    "TEXT"),
        ("revalidation_flags", "TEXT"),
    ]
    for col, typedef in new_cols:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE bets ADD COLUMN {col} {typedef}")

    # ── Audit log for every pick change during revalidation ─────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pick_audit_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            changed_at   TEXT    NOT NULL,
            bet_id       TEXT    NOT NULL,
            sport        TEXT,
            field_changed TEXT,
            old_value    TEXT,
            new_value    TEXT,
            reason       TEXT,
            alert_sent   INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_bet_id ON pick_audit_log (bet_id)"
    )

    # ── V3.0 pick regrade history ────────────────────────────────────────────
    # Captures every meaningful tier/confidence/edge change during revalidation.
    # One row per pick per revalidation cycle that results in a non-confirm action.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pick_regrade_history (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            bet_id           TEXT    NOT NULL,
            sport            TEXT,
            changed_at       TEXT    NOT NULL,
            version          INTEGER NOT NULL DEFAULT 1,
            prev_tier        TEXT,
            new_tier         TEXT,
            prev_confidence  REAL,
            new_confidence   REAL,
            prev_edge        REAL,
            new_edge         REAL,
            change_type      TEXT,
            reason           TEXT,
            alert_sent       INTEGER NOT NULL DEFAULT 0,
            snapshot_json    TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_regrade_bet_id "
        "ON pick_regrade_history (bet_id)"
    )


# ---------------------------------------------------------------------------
# Logging bets
# ---------------------------------------------------------------------------

def log_bet(
    bet: Bet,
    sportsbook_odds: int,
    model_probability: float,
    stake: float = 100.0,
    sent_to_group: bool = False,
) -> None:
    """
    Insert an 'open' bet entry. Called when the bot pushes a pick to Telegram.

    Args:
        bet:               Approved Bet from the Gatekeeper.
        sportsbook_odds:   American odds at time of pick (e.g. -110).
        model_probability: % probability the direction hits (from SimulationEngine).
        stake:             Unit stake in dollars. Defaults to $100.
        sent_to_group:     True when this pick was broadcast to the Telegram group.
    """
    init_db()
    details = {
        "team":             bet.team,
        "market":           bet.market,
        "direction":        bet.direction,
        "sportsbook_line":  bet.sportsbook_line,
        "edge_percentage":  bet.edge_percentage,
        "confidence_score": bet.confidence_score,
        "tier":             bet.tier.value if bet.tier else None,
        "player":           bet.player,
    }
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO bets
              (timestamp, bet_id, sport, wager_details, model_probability,
               sportsbook_odds, status, stake, tier, edge_percentage, sent_to_group)
            VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                bet.bet_id,
                bet.sport_type if hasattr(bet, "sport_type") else details.get("team", "UNK"),
                json.dumps(details),
                model_probability,
                sportsbook_odds,
                stake,
                details["tier"],
                bet.edge_percentage,
                int(sent_to_group),
            ),
        )


def log_bet_dict(
    bet_id: str,
    sport: str,
    wager_details: dict[str, Any],
    model_probability: float,
    sportsbook_odds: int,
    stake: float = 100.0,
    tier: str | None = None,
    edge_percentage: float = 0.0,
    sent_to_group: bool = False,
    bookmaker_source: str = "",
    line_move_dir: str | None = None,
) -> None:
    """
    Lower-level log function used by run.py / main.py when passing full bet dicts.

    Snaps opening values (edge, confidence, odds) at publication time so the
    pregame revalidation engine can compare against current conditions.

    Args:
        sent_to_group:    True when this pick was broadcast to the Telegram group.
        bookmaker_source: Bookmaker selected as the best line source.
        line_move_dir:    Signal calibrator bucket for the line movement signal.
    """
    init_db()
    now_iso    = datetime.now(timezone.utc).isoformat()
    slate_date = _today_et()

    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO bets
              (timestamp, bet_id, sport, wager_details, model_probability,
               sportsbook_odds, status, stake, tier, edge_percentage,
               sent_to_group, bookmaker_source, line_move_dir,
               slate_date, is_locked, published_at,
               opening_edge, opening_confidence, opening_odds,
               current_edge, current_confidence)
            VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?,
                    ?, 1, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso,
                bet_id,
                sport,
                json.dumps(wager_details),
                model_probability,
                sportsbook_odds,
                stake,
                tier,
                edge_percentage,
                int(sent_to_group),
                bookmaker_source or "",
                line_move_dir,
                # slate-lock snapshot
                slate_date,
                now_iso,
                edge_percentage,      # opening_edge
                model_probability,    # opening_confidence
                sportsbook_odds,      # opening_odds
                edge_percentage,      # current_edge  (same as opening at publish)
                model_probability,    # current_confidence
            ),
        )
        # If the row already existed (INSERT was ignored), update the tier in-place
        # so global tier re-ranking (Nuke/Diamond/Gold cap) is reflected.
        # Only applies to picks not yet broadcast (sent_to_group=0).
        if tier is not None:
            conn.execute(
                """
                UPDATE bets SET
                    tier             = ?,
                    edge_percentage  = ?,
                    wager_details    = json_set(
                        COALESCE(wager_details, '{}'),
                        '$.tier', ?
                    )
                WHERE bet_id        = ?
                  AND status        = 'open'
                  AND sent_to_group = 0
                """,
                (tier, edge_percentage, tier, bet_id),
            )


def close_bet(
    bet_id: str,
    actual_outcome: str,
    stake_override: float | None = None,
) -> dict[str, Any]:
    """
    Resolve an open bet entry with its actual outcome and compute profit/loss.

    Args:
        bet_id:         The unique bet ID.
        actual_outcome: 'win', 'loss', or 'push'.
        stake_override: Override the stored stake amount if needed.

    Returns:
        dict with profit_loss and a summary string.

    Raises:
        ValueError: If bet_id is not found or already closed.
    """
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM bets WHERE bet_id = ?", (bet_id,)
        ).fetchone()

        if not row:
            raise ValueError(f"Bet '{bet_id}' not found in database.")
        if row["status"] == "closed":
            raise ValueError(f"Bet '{bet_id}' is already closed.")

        stake = stake_override or row["stake"]
        odds  = row["sportsbook_odds"]
        pl    = _calculate_profit_loss(actual_outcome, odds, stake)

        conn.execute(
            """
            UPDATE bets
            SET actual_outcome = ?,
                profit_loss    = ?,
                status         = 'closed'
            WHERE bet_id = ?
            """,
            (actual_outcome, pl, bet_id),
        )

    return {
        "bet_id":        bet_id,
        "actual_outcome": actual_outcome,
        "profit_loss":   pl,
        "summary":       f"{bet_id} → {actual_outcome.upper()}  P&L: ${pl:+.2f}",
    }


def _calculate_profit_loss(outcome: str, american_odds: int, stake: float) -> float:
    outcome = outcome.lower()
    if outcome == "push":
        return 0.0
    if outcome == "loss":
        return -stake
    if outcome == "win":
        if american_odds > 0:
            return round(stake * american_odds / 100, 2)
        else:
            return round(stake * 100 / abs(american_odds), 2)
    raise ValueError(f"Unknown outcome '{outcome}'. Use 'win', 'loss', or 'push'.")


# ---------------------------------------------------------------------------
# ROI Calculator
# ---------------------------------------------------------------------------

def calculate_running_roi(sport: str | None = None) -> dict[str, Any]:
    """
    Compute ROI and performance stats from broadcast bets (sent_to_group=1).

    Only broadcast picks are included in P&L / ROI / win-rate figures so
    that non-broadcast candidates (which were filtered before publication)
    do not contaminate reported performance.  Total/open bet counts still
    reflect the full database.

    Args:
        sport: Filter to a specific sport (e.g. 'WNBA'). None = all sports.

    Returns:
        dict with:
            total_bets, closed_bets, open_bets,
            wins, losses, pushes, win_rate,
            total_staked, total_profit_loss, roi_pct,
            by_tier (breakdown per tier),
            by_sport (breakdown per sport, if sport=None)
    """
    init_db()

    where = "WHERE status = 'closed' AND sent_to_group = 1"
    params: list[Any] = []
    if sport:
        where += " AND sport = ?"
        params.append(sport.upper())

    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM bets {where}", params
        ).fetchall()

        total_bets  = conn.execute("SELECT COUNT(*) FROM bets" + (f" WHERE sport=?" if sport else ""), params).fetchone()[0]
        open_bets   = conn.execute("SELECT COUNT(*) FROM bets WHERE status='open'" + (f" AND sport=?" if sport else ""), params).fetchone()[0]

    closed_bets = len(rows)
    wins = pushes = losses = 0
    total_staked = total_pl = 0.0
    by_tier:  dict[str, dict[str, Any]] = {}
    by_sport: dict[str, dict[str, Any]] = {}

    for row in rows:
        outcome = (row["actual_outcome"] or "").lower()
        pl      = row["profit_loss"] or 0.0
        stake   = row["stake"] or 100.0
        tier    = row["tier"] or "Unknown"
        sp      = row["sport"]

        if outcome == "win":    wins   += 1
        elif outcome == "loss": losses += 1
        elif outcome == "push": pushes += 1

        total_staked += stake
        total_pl     += pl

        # by_tier
        if tier not in by_tier:
            by_tier[tier] = {"bets": 0, "wins": 0, "losses": 0, "pl": 0.0}
        by_tier[tier]["bets"]   += 1
        by_tier[tier]["wins"]   += (1 if outcome == "win"  else 0)
        by_tier[tier]["losses"] += (1 if outcome == "loss" else 0)
        by_tier[tier]["pl"]     += pl

        # by_sport
        if sp not in by_sport:
            by_sport[sp] = {"bets": 0, "wins": 0, "pl": 0.0}
        by_sport[sp]["bets"] += 1
        by_sport[sp]["wins"] += (1 if outcome == "win" else 0)
        by_sport[sp]["pl"]   += pl

    win_rate = round(wins / closed_bets * 100, 1) if closed_bets else 0.0
    roi_pct  = round(total_pl / total_staked * 100, 2) if total_staked else 0.0

    for t in by_tier.values():
        t["win_rate"] = round(t["wins"] / t["bets"] * 100, 1) if t["bets"] else 0.0
        t["pl"]       = round(t["pl"], 2)
    for s in by_sport.values():
        s["win_rate"] = round(s["wins"] / s["bets"] * 100, 1) if s["bets"] else 0.0
        s["pl"]       = round(s["pl"], 2)

    return {
        "total_bets":        total_bets,
        "closed_bets":       closed_bets,
        "open_bets":         open_bets,
        "wins":              wins,
        "losses":            losses,
        "pushes":            pushes,
        "win_rate":          win_rate,
        "total_staked":      round(total_staked, 2),
        "total_profit_loss": round(total_pl, 2),
        "roi_pct":           roi_pct,
        "by_tier":           by_tier,
        "by_sport":          by_sport,
    }


def format_roi_report(sport: str | None = None) -> str:
    """Return a Telegram-ready ROI summary string."""
    r   = calculate_running_roi(sport)
    from core.time_utils import format_est
    now = format_est(datetime.now(timezone.utc), "%A, %B %d %Y  %I:%M %p ET")
    sport_label = sport or "ALL SPORTS"

    tier_lines = ""
    for tier, data in sorted(r["by_tier"].items()):
        pl_sign = "+" if data["pl"] >= 0 else ""
        tier_lines += (
            f"  {tier:<8} {data['bets']} bets  "
            f"WR {data['win_rate']}%  "
            f"P&L ${pl_sign}{data['pl']:.2f}\n"
        )

    roi_emoji = "📈" if r["roi_pct"] >= 0 else "📉"
    pl_sign   = "+" if r["total_profit_loss"] >= 0 else ""

    return (
        f"╔══════════════════════════════════════╗\n"
        f"║  {roi_emoji}  DAILY ROI REPORT  ·  {sport_label}\n"
        f"║  📅  {now}\n"
        f"╚══════════════════════════════════════╝\n"
        f"\n"
        f"  📊 Record:      {r['wins']}W – {r['losses']}L – {r['pushes']}P\n"
        f"  🎯 Win Rate:    {r['win_rate']}%\n"
        f"  💵 Total Staked: ${r['total_staked']:,.2f}\n"
        f"  💰 Net P&L:     ${pl_sign}{r['total_profit_loss']:,.2f}\n"
        f"  {roi_emoji} ROI:          {'+' if r['roi_pct'] >= 0 else ''}{r['roi_pct']}%\n"
        f"  📂 Open Bets:   {r['open_bets']}\n"
        f"\n"
        f"  ── By Tier ──────────────────────────\n"
        f"{tier_lines}"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  ⚡ Powered by Multi-Sport Prediction Engine\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


# ---------------------------------------------------------------------------
# Model Refinement — the learning loop
# ---------------------------------------------------------------------------

def update_model_priors(sport: str, min_sample: int = MIN_SAMPLE_FOR_REFINEMENT) -> list[dict[str, Any]]:
    """
    Compare model_probability vs actual_outcome for closed bets and nudge
    the sports_metrics.json weights when the model is consistently miscalibrated.

    Logic
    -----
    For each (sport, market, direction) bucket with ≥ min_sample resolved bets:
      1. actual_hit_rate = wins / (wins + losses)
      2. avg_model_prob  = mean(model_probability) for that bucket
      3. calibration_err = avg_model_prob - actual_hit_rate   (positive = overconfident)
      4. If |calibration_err| > 0.08 (8%), look up the responsible weight key
         and nudge it by ±MAX_NUDGE, then renormalize the weight group to 1.0.
      5. Write the updated config back to sports_metrics.json.
      6. Record every change in the weight_adjustments table.

    Args:
        sport:      Sport to analyse (e.g. 'WNBA').
        min_sample: Minimum resolved bets required before adjusting a weight.

    Returns:
        List of adjustment dicts (empty if no adjustments were made).
    """
    init_db()
    sport = sport.upper()

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT wager_details, model_probability, actual_outcome
            FROM bets
            WHERE sport = ? AND status = 'closed'
              AND actual_outcome IN ('win', 'loss')
            """,
            (sport,),
        ).fetchall()

    if not rows:
        return []

    # ── Bucket rows by (market, direction) ──────────────────────────────────
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        details   = json.loads(row["wager_details"])
        market    = details.get("market", "")
        direction = details.get("direction", "").lower()
        key       = (market, direction)
        buckets.setdefault(key, []).append({
            "model_prob": row["model_probability"] / 100.0,
            "hit":        1 if row["actual_outcome"] == "win" else 0,
        })

    adjustments: list[dict[str, Any]] = []

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    if sport not in config:
        return []

    sport_config = config[sport]
    market_map   = _MARKET_WEIGHT_MAP.get(sport, {})

    for (market, direction), records in buckets.items():
        if len(records) < min_sample:
            continue

        actual_hit_rate  = sum(r["hit"] for r in records) / len(records)
        avg_model_prob   = sum(r["model_prob"] for r in records) / len(records)
        calibration_err  = avg_model_prob - actual_hit_rate   # + = overconfident

        if abs(calibration_err) <= 0.08:
            continue    # within acceptable range, no change needed

        mapping = market_map.get(market)
        if not mapping:
            continue

        weight_group = mapping["group"]
        # Overconfident on OVER → over_key is too influential → reduce it
        # Overconfident on UNDER → under_key is too influential → reduce it
        nudge_key = mapping["over"] if direction == "over" else mapping["under"]
        nudge_dir = -1 if calibration_err > 0 else +1   # reduce if overconfident

        weights = sport_config.get(weight_group, {})
        if nudge_key not in weights:
            continue

        old_val = weights[nudge_key]
        delta   = min(MAX_NUDGE, abs(calibration_err) * 0.1) * nudge_dir
        new_val = max(0.01, round(old_val + delta, 4))

        if new_val == old_val:
            continue

        # Adjust and renormalize so group sums to 1.0
        weights[nudge_key] = new_val
        total = sum(weights.values())
        weights = {k: round(v / total, 4) for k, v in weights.items()}
        # Fix rounding residual on the first key
        residual = round(1.0 - sum(weights.values()), 4)
        first_key = next(iter(weights))
        weights[first_key] = round(weights[first_key] + residual, 4)

        sport_config[weight_group] = weights
        config[sport] = sport_config

        reason = (
            f"{sport} {market} {direction.upper()}: "
            f"model avg {avg_model_prob:.1%} vs actual {actual_hit_rate:.1%} "
            f"(err {calibration_err:+.1%})"
        )

        adj = {
            "sport":        sport,
            "weight_group": weight_group,
            "weight_key":   nudge_key,
            "old_value":    old_val,
            "new_value":    new_val,
            "reason":       reason,
        }
        adjustments.append(adj)

        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO weight_adjustments
                  (timestamp, sport, weight_group, weight_key, old_value, new_value, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    sport,
                    weight_group,
                    nudge_key,
                    old_val,
                    new_val,
                    reason,
                ),
            )

    if adjustments:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)

    return adjustments


def get_open_bets(sport: str | None = None) -> list[dict[str, Any]]:
    """Return all open bets as dicts."""
    init_db()
    where  = "WHERE status = 'open'"
    params: list[Any] = []
    if sport:
        where += " AND sport = ?"
        params.append(sport.upper())
    with _connect() as conn:
        rows = conn.execute(f"SELECT * FROM bets {where} ORDER BY timestamp DESC", params).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Group-broadcast tracking
# ---------------------------------------------------------------------------

def mark_sent_to_group(bet_id: str) -> None:
    """
    Flip sent_to_group = 1 on a bet that was successfully broadcast to the
    Telegram group.  Called by the broadcast orchestrator after each send.
    """
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE bets SET sent_to_group = 1 WHERE bet_id = ?",
            (bet_id,),
        )


def get_yesterday_group_bets(sport: str | None = None) -> list[dict[str, Any]]:
    """
    Return all bets broadcast to the group (sent_to_group=1) during yesterday
    in Eastern Time, ordered newest first.

    The date range is computed in UTC from yesterday's ET midnight boundaries
    so late-evening games (e.g. 11 PM ET = 3 AM UTC next day) still resolve
    correctly.
    """
    _EST = ZoneInfo("America/New_York")
    from core.time_utils import now_utc, convert_to_est

    yesterday_et = (convert_to_est(now_utc()) - timedelta(days=1)).date()
    day_start_et = datetime(yesterday_et.year, yesterday_et.month, yesterday_et.day, tzinfo=_EST)
    day_end_et   = day_start_et + timedelta(days=1)
    start_utc    = day_start_et.astimezone(timezone.utc).isoformat()
    end_utc      = day_end_et.astimezone(timezone.utc).isoformat()

    init_db()
    where  = "WHERE sent_to_group = 1 AND timestamp >= ? AND timestamp < ?"
    params: list[Any] = [start_utc, end_utc]
    if sport:
        where += " AND sport = ?"
        params.append(sport.upper())

    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM bets {where} ORDER BY timestamp DESC", params
        ).fetchall()
    return [dict(r) for r in rows]


def calculate_group_roi(
    date_et: str | None = None,
    sport: str | None = None,
) -> dict[str, Any]:
    """
    Compute ROI for bets sent to the Telegram group (sent_to_group=1) from
    a specific calendar date in Eastern Time.

    Args:
        date_et: Target date as 'YYYY-MM-DD' in ET (defaults to yesterday ET).
        sport:   Filter to one sport; None = all sports.

    Returns:
        dict with record stats, P&L, ROI, and a by_tier breakdown.
        Also includes 'open_bets' count for picks not yet resolved.
    """
    from core.time_utils import now_utc, convert_to_est
    from datetime import date as _date

    _EST = ZoneInfo("America/New_York")
    est_now = convert_to_est(now_utc())

    if date_et is None:
        target_date = (est_now - timedelta(days=1)).date()
    else:
        target_date = _date.fromisoformat(date_et)

    day_start_et = datetime(target_date.year, target_date.month, target_date.day, tzinfo=_EST)
    day_end_et   = day_start_et + timedelta(days=1)
    start_utc    = day_start_et.astimezone(timezone.utc).isoformat()
    end_utc      = day_end_et.astimezone(timezone.utc).isoformat()

    init_db()
    where  = "WHERE sent_to_group = 1 AND timestamp >= ? AND timestamp < ?"
    params: list[Any] = [start_utc, end_utc]
    if sport:
        where += " AND sport = ?"
        params.append(sport.upper())

    with _connect() as conn:
        all_rows = conn.execute(f"SELECT * FROM bets {where}", params).fetchall()

    closed    = [r for r in all_rows if r["status"] == "closed"]
    open_rows = [r for r in all_rows if r["status"] == "open"]

    wins = losses = pushes = 0
    total_staked = total_pl = 0.0
    by_tier: dict[str, dict[str, Any]] = {}

    for row in closed:
        outcome = (row["actual_outcome"] or "").lower()
        pl      = row["profit_loss"] or 0.0
        stake   = row["stake"] or 100.0
        tier    = row["tier"] or "Unknown"

        if outcome == "win":    wins   += 1
        elif outcome == "loss": losses += 1
        elif outcome == "push": pushes += 1

        total_staked += stake
        total_pl     += pl

        if tier not in by_tier:
            by_tier[tier] = {"bets": 0, "wins": 0, "losses": 0, "pl": 0.0}
        by_tier[tier]["bets"]   += 1
        by_tier[tier]["wins"]   += (1 if outcome == "win" else 0)
        by_tier[tier]["losses"] += (1 if outcome == "loss" else 0)
        by_tier[tier]["pl"]     += pl

    closed_count = len(closed)
    win_rate = round(wins / closed_count * 100, 1) if closed_count else 0.0
    roi_pct  = round(total_pl / total_staked * 100, 2) if total_staked else 0.0

    for t in by_tier.values():
        t["win_rate"] = round(t["wins"] / t["bets"] * 100, 1) if t["bets"] else 0.0
        t["pl"]       = round(t["pl"], 2)

    return {
        "target_date":       target_date.isoformat(),
        "total_group":       len(all_rows),
        "closed_bets":       closed_count,
        "open_bets":         len(open_rows),
        "wins":              wins,
        "losses":            losses,
        "pushes":            pushes,
        "win_rate":          win_rate,
        "total_staked":      round(total_staked, 2),
        "total_profit_loss": round(total_pl, 2),
        "roi_pct":           roi_pct,
        "by_tier":           by_tier,
    }


def get_tier_record_30d(tier_value: str) -> str:
    """
    Return a W–L record string for a given tier, respecting the season_start
    gate in bot_config so the card always shows the current-season record.

    Args:
        tier_value: Tier enum value string — "Nuke", "Diamond", "Gold Standard".

    Returns:
        e.g. "4W – 2L" or "0W – 0L"
    """
    init_db()
    with _connect() as conn:
        # Read season start; fall back to 30-day window if not set
        cfg = conn.execute(
            "SELECT value FROM bot_config WHERE key='season_start'"
        ).fetchone()
        season_start = cfg["value"] if cfg and cfg["value"] else None
        cutoff = season_start if season_start else "datetime('now', '-30 days')"

        if season_start:
            rows = conn.execute(
                """
                SELECT actual_outcome FROM bets
                WHERE tier = ?
                  AND status = 'closed'
                  AND timestamp >= ?
                """,
                (tier_value, season_start),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT actual_outcome FROM bets
                WHERE tier = ?
                  AND status = 'closed'
                  AND timestamp >= datetime('now', '-30 days')
                """,
                (tier_value,),
            ).fetchall()

    if not rows:
        return "0W – 0L"

    wins   = sum(1 for r in rows if (r["actual_outcome"] or "").lower() == "win")
    losses = sum(1 for r in rows if (r["actual_outcome"] or "").lower() == "loss")
    return f"{wins}W – {losses}L"


def format_morning_recap(sport: str | None = None) -> str:
    """
    Return a Telegram-ready morning recap string covering only the bets sent
    to the group (sent_to_group=1) during yesterday in ET.

    Closed bets show full W/L/P&L stats; open bets are counted as pending.
    """
    from core.time_utils import now_utc, convert_to_est

    r           = calculate_group_roi(sport=sport)
    est_now     = convert_to_est(now_utc())
    yesterday   = (est_now - timedelta(days=1)).strftime("%A, %B %d %Y")
    sport_label = sport or "ALL SPORTS"
    roi_emoji   = "📈" if r["roi_pct"] >= 0 else "📉"
    pl_sign     = "+" if r["total_profit_loss"] >= 0 else ""

    tier_lines = ""
    for tier_name, data in sorted(r["by_tier"].items()):
        pl_s = "+" if data["pl"] >= 0 else ""
        tier_lines += (
            f"  {tier_name:<8} {data['bets']} bet(s)  "
            f"WR {data['win_rate']}%  "
            f"P&L ${pl_s}{data['pl']:.2f}\n"
        )

    no_data_line  = "  ℹ️  No group bets logged for yesterday.\n" if not r["total_group"] else ""
    pending_line  = f"  ⏳ Pending Resolution: {r['open_bets']} bet(s)\n" if r["open_bets"] else ""
    tier_section  = tier_lines if tier_lines else "  No resolved bets to report.\n"

    return (
        f"☀️ MORNING RECAP  ·  {sport_label}\n"
        f"🗓️ {yesterday}\n"
        f"{'━' * 42}\n"
        f"\n"
        f"  Yesterday's Group Feed Results:\n"
        f"\n"
        f"{no_data_line}"
        f"  📊 Record:       {r['wins']}W – {r['losses']}L – {r['pushes']}P\n"
        f"  🎯 Win Rate:     {r['win_rate']}%\n"
        f"  💵 Total Staked: ${r['total_staked']:,.2f}\n"
        f"  💰 Net P&L:      ${pl_sign}{r['total_profit_loss']:,.2f}\n"
        f"  {roi_emoji} ROI:           {'+' if r['roi_pct'] >= 0 else ''}{r['roi_pct']}%\n"
        f"{pending_line}"
        f"\n"
        f"  ── By Tier ──────────────────────────\n"
        f"{tier_section}"
        f"{'━' * 42}\n"
        f"  ⚡ Powered by Multi-Sport Prediction Engine\n"
        f"{'━' * 42}"
    )
