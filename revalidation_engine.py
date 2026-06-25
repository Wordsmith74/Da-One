"""
revalidation_engine.py

Pregame Revalidation System.

Runs 30-60 minutes before game time. Evaluates every open, locked pick
against current conditions. The only permitted actions are:

  CONFIRM   — edge intact, no material change
  UPGRADE   — edge improved (line moved in model's favour, or edge +20%+)
  DOWNGRADE — edge weakened (adverse line move, or edge dropped >45%)
  VOID      — original model assumptions no longer valid (SP scratch,
               major injury, extreme line move)

No new picks are ever created. The official published slate is immutable.
All changes are written to bets.revalidation_* and logged to pick_audit_log.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "results.db"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Adverse line movement (pts) required for each action, keyed by sport.
# "Adverse" = line moved against the bet direction (up for OVER, down for UNDER).
_VOID_LINE:      dict[str, float] = {"MLB": 2.5,  "WNBA": 5.0,  "NBA": 5.0}
_DOWNGRADE_LINE: dict[str, float] = {"MLB": 1.0,  "WNBA": 2.5,  "NBA": 2.5}
_UPGRADE_LINE:   dict[str, float] = {"MLB": 0.5,  "WNBA": 1.0,  "NBA": 1.0}  # favourable move

# Player prop stale-line threshold — direction-agnostic absolute move.
# The original bet conditions no longer exist at any sportsbook if the line
# shifted this much in EITHER direction. Apply regardless of adverse/favorable.
_PROP_STALE_LINE_THRESHOLD: float = 1.0

# Injury impact thresholds (LineupIntelFactor.edge_adjustment magnitude)
_INJURY_VOID_IMPACT       = 0.30
_INJURY_DOWNGRADE_IMPACT  = 0.12

# Edge ratio thresholds
_EDGE_VOID_RATIO      = 0.50   # current_edge < opening_edge * 0.50 → void (spec §INVALIDATION #2)
_EDGE_DOWNGRADE_RATIO = 0.55   # current_edge < opening_edge * 0.55 → downgrade
_EDGE_UPGRADE_RATIO   = 1.20   # current_edge > opening_edge * 1.20 → upgrade


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _implied_prob(american_odds: int) -> float:
    if american_odds >= 0:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)


def _recompute_edge(current_odds: int, model_prob_pct: float) -> float:
    """Edge % given current market odds and the original model probability."""
    implied = _implied_prob(current_odds)
    model   = model_prob_pct / 100.0
    return round((model - implied) * 100.0, 2)


def _adverse_move(current_line: float, original_line: float, direction: str) -> float:
    """
    Positive value = line moved *against* the bet.
    OVER: line went up   → positive adverse
    UNDER: line went down → positive adverse
    """
    delta = current_line - original_line
    return delta if direction.lower() == "over" else -delta


def _today_et() -> str:
    from zoneinfo import ZoneInfo
    from datetime import date
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Fetch current odds for a sport (one Odds API call per sport, then cached)
# ---------------------------------------------------------------------------
def _get_fresh_candidates(sport: str) -> list[dict[str, Any]]:
    """Re-fetch today's odds candidates for *sport*. Returns [] on failure."""
    try:
        from core.odds_client import fetch_todays_candidates
        return fetch_todays_candidates(sport)
    except Exception as exc:
        logger.warning(f"[revalidation] Odds fetch failed for {sport}: {exc}")
        return []


def _get_fresh_prop_candidates(sport: str) -> list[dict[str, Any]]:
    """
    Re-fetch live player prop candidates for *sport* via the Odds API.

    Uses raw_mode=True so Rule 2 (cross-book consensus gate) is bypassed and
    candidates with significant book spread are still returned — using the
    consensus median as the line. This is essential for detecting stale lines
    on already-published picks whose consensus has shifted significantly since
    publication (causing books to spread out around the new centre).
    Returns [] on failure.
    """
    try:
        from core.player_props import get_player_prop_candidates
        return get_player_prop_candidates(sport, raw_mode=True)
    except Exception as exc:
        logger.warning(f"[revalidation] Prop fetch failed for {sport}: {exc}")
        return []


def _find_candidate(
    candidates: list[dict[str, Any]],
    team: str,
    market: str,
    direction: str,
) -> dict[str, Any] | None:
    """Find the matching candidate by team + market + direction."""
    t = team.upper()
    m = market.lower()
    d = direction.lower()
    for c in candidates:
        if (
            str(c.get("team", "")).upper() == t
            and str(c.get("market", "")).lower() == m
            and str(c.get("direction", "")).lower() == d
        ):
            return c
    return None


def _find_prop_candidate(
    candidates: list[dict[str, Any]],
    player: str,
    market: str,
    direction: str,
) -> dict[str, Any] | None:
    """
    Find the matching player prop candidate by player name + market + direction.
    Market comparison is normalised (spaces↔underscores, case-insensitive).
    """
    p = player.lower()
    m = market.lower().replace(" ", "_")
    d = direction.lower()
    for c in candidates:
        c_player    = str(c.get("player", "")).lower()
        c_market    = str(c.get("market", "")).lower().replace(" ", "_")
        c_direction = str(c.get("direction", "")).lower()
        if c_player == p and c_direction == d and (m in c_market or c_market in m):
            return c
    return None


# ---------------------------------------------------------------------------
# Pitcher intel — detect SP change vs original model context
# ---------------------------------------------------------------------------
def _check_pitcher_change(
    home_abbr: str,
    away_abbr: str,
) -> tuple[bool, str]:
    """
    Returns (changed: bool, reason: str).
    Compares today's probable starters against a second call. If no prior
    snapshot is available we cannot detect a change, so we return (False, "").
    """
    try:
        from core.intelligence.pitcher_intel import get_pitcher_intel
        result = get_pitcher_intel(home_abbr, away_abbr)
        if result is None:
            return False, ""
        # If pitcher data unavailable or FIP at league baseline, flag as uncertain
        factor_text = getattr(result, "factor_text", "") or ""
        if "league" in factor_text.lower():
            return False, "SP data unavailable — using league baseline"
        return False, ""
    except Exception as exc:
        logger.debug(f"[revalidation] pitcher_intel error: {exc}")
        return False, ""


# ---------------------------------------------------------------------------
# Lineup intel — detect late injuries
# ---------------------------------------------------------------------------
def _check_lineup(
    team: str,
    sport: str,
) -> tuple[float, str]:
    """
    Returns (injury_impact: float, description: str).
    impact > 0 means injuries detected; magnitude = edge_adjustment abs value.
    """
    try:
        from core.intelligence.lineup_intel import get_lineup_intel
        result = get_lineup_intel(team, sport, bet_on_this_team=True)
        if result is None:
            return 0.0, ""
        impact = abs(getattr(result, "edge_adjustment", 0.0))
        desc   = getattr(result, "factor_text", "") or ""
        return impact, desc
    except Exception as exc:
        logger.debug(f"[revalidation] lineup_intel error: {exc}")
        return 0.0, ""


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------
def _decide(
    sport:          str,
    direction:      str,
    adverse:        float,
    opening_edge:   float,
    current_edge:   float,
    injury_impact:  float,
    sp_changed:     bool,
    pitcher_reason: str   = "",
    lineup_desc:    str   = "",
    is_prop:        bool  = False,
    abs_line_move:  float = 0.0,
    orig_line:      float = 0.0,
    current_line:   float = 0.0,
) -> tuple[str, str]:
    """
    Returns (status, reason):
      status ∈ {'confirmed', 'upgraded', 'downgraded', 'voided'}
    """
    void_line     = _VOID_LINE.get(sport, 2.5)
    downgrade_line= _DOWNGRADE_LINE.get(sport, 1.0)
    upgrade_line  = _UPGRADE_LINE.get(sport, 0.5)

    reasons: list[str] = []

    # ── PROP: stale-line void (direction-agnostic) ───────────────────────────
    # For player props the existing adverse/favorable logic is insufficient.
    # If the sportsbook line has moved ≥ threshold in EITHER direction, the
    # original bet conditions no longer exist at any book — void unconditionally.
    if is_prop and abs_line_move >= _PROP_STALE_LINE_THRESHOLD:
        delta = current_line - orig_line
        sign  = "+" if delta >= 0 else ""
        return "voided", (
            f"Sportsbook line moved: {orig_line} → {current_line} "
            f"({sign}{delta:.1f} pts). "
            f"Original {direction.upper()} {orig_line} no longer available at sportsbooks."
        )

    # ── VOID conditions ─────────────────────────────────────────────────────
    if sp_changed and sport == "MLB":
        return "voided", f"Starting pitcher change: {pitcher_reason}"
    if injury_impact >= _INJURY_VOID_IMPACT:
        return "voided", f"Major injury impact ({injury_impact:.2f}): {lineup_desc}"
    # ── Edge collapse void (spec §INVALIDATION CONDITIONS #2) ───────────────
    # Current edge < 50 % of opening edge → original thesis is invalidated.
    if opening_edge > 0 and current_edge < opening_edge * _EDGE_VOID_RATIO:
        return "voided", (
            f"Edge collapsed: {opening_edge:.1f}% → {current_edge:.1f}% "
            f"(below {_EDGE_VOID_RATIO:.0%} of original — original thesis invalidated)"
        )

    if adverse >= void_line:
        dir_label = "over" if direction.lower() == "over" else "under"
        return "voided", (
            f"Extreme adverse line move ({adverse:+.1f} pts against {dir_label}). "
            f"Original edge invalidated."
        )

    # ── DOWNGRADE conditions ─────────────────────────────────────────────────
    if adverse >= downgrade_line:
        reasons.append(f"Line moved {adverse:.1f} pts against model")
    if injury_impact >= _INJURY_DOWNGRADE_IMPACT:
        reasons.append(f"Injury concern ({injury_impact:.2f}): {lineup_desc}")
    if opening_edge > 0 and current_edge < opening_edge * _EDGE_DOWNGRADE_RATIO:
        reasons.append(
            f"Edge weakened: {opening_edge:.1f}% → {current_edge:.1f}%"
        )
    if reasons:
        return "downgraded", "; ".join(reasons)

    # ── UPGRADE conditions ───────────────────────────────────────────────────
    upgrade_reasons: list[str] = []
    if adverse <= -upgrade_line:
        upgrade_reasons.append(
            f"Line moved {abs(adverse):.1f} pts in model's favour"
        )
    if opening_edge > 0 and current_edge > opening_edge * _EDGE_UPGRADE_RATIO:
        upgrade_reasons.append(
            f"Edge strengthened: {opening_edge:.1f}% → {current_edge:.1f}%"
        )
    if upgrade_reasons:
        return "upgraded", "; ".join(upgrade_reasons)

    # ── CONFIRM ──────────────────────────────────────────────────────────────
    return "confirmed", "Edge intact. No material change detected."


# ---------------------------------------------------------------------------
# Apply change to DB
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Telegram alert governance — spec §TELEGRAM NOTIFICATION GOVERNANCE
# ---------------------------------------------------------------------------
# Telegram is an EXCEPTION-ONLY system.  Only two alert types are permitted,
# regardless of internal revalidation_status:
#   1. Edge Loss Alert  — current_edge < 50 % of opening edge
#   2. Edge Reversal    — positive EV flipped to negative EV (current_edge < 0)
#
# All other events (upgrades, downgrades, line moves, internal simulation
# updates, tier comparisons, re-ranking) are MiniApp-only and must NEVER
# trigger a Telegram message.
#
# Alerts are restricted to picks originally broadcast to the Telegram group
# (sent_to_group = 1).  Once a transition fires an alert, subsequent
# revalidation cycles skip re-alerting for the same pick (prev_status guard).

_EDGE_WARNING_RATIO = 0.50   # current_edge < opening_edge × this → Edge Loss Alert


def _send_edge_warning_alert(change: dict[str, Any]) -> bool:
    """
    Edge Loss Alert — fires when current_edge < 50 % of the original edge.
    Message type: EDGE WARNING / SIGNIFICANT EDGE DECAY.
    """
    import os
    import json as _j
    import urllib.request

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id   = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        return False

    opening_edge = float(change.get("opening_edge") or 0)
    current_edge = float(change.get("current_edge") or 0)
    player    = change.get("player")
    team      = change.get("team", "")
    market    = str(change.get("market", "totals") or "totals")
    direction = str(change.get("direction", "over") or "over").upper()
    o_line    = change.get("opening_line", "")

    subject   = player if player else team
    mkt_label = market.replace("_", " ").title()
    pick_desc = f"{subject} {mkt_label} {direction}"
    if o_line:
        pick_desc += f" {o_line}"

    decay_pct = round((opening_edge - current_edge) / opening_edge * 100) if opening_edge else 0

    text = (
        f"⚠️ *EDGE WARNING — SIGNIFICANT EDGE DECAY*\n"
        f"\n"
        f"*Pick:* {pick_desc}\n"
        f"\n"
        f"*Opening Edge:* +{opening_edge:.1f}%\n"
        f"*Current Edge:* +{current_edge:.1f}%\n"
        f"*Edge Decay:* −{decay_pct:.0f}% of original value\n"
        f"\n"
        f"Edge has fallen below 50% of the published value.\n"
        f"Reduce or avoid this position."
    )

    try:
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        data = _j.dumps(payload).encode()
        req  = urllib.request.Request(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data=data, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            sent = resp.status == 200
            if sent:
                logger.info(
                    f"[revalidation] EDGE WARNING sent for "
                    f"{change.get('bet_id','?')} "
                    f"({opening_edge:.1f}% → {current_edge:.1f}%)"
                )
            return sent
    except Exception as exc:
        logger.warning(f"[revalidation] Edge warning alert failed: {exc}")
        return False


def _send_edge_reversal_alert(change: dict[str, Any]) -> bool:
    """
    Edge Reversal Alert — fires when EV flipped from positive to negative.
    Message type: EDGE REVERSAL / MARKET INVALIDATION DETECTED.
    """
    import os
    import json as _j
    import urllib.request

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id   = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        return False

    opening_edge = float(change.get("opening_edge") or 0)
    current_edge = float(change.get("current_edge") or 0)
    player    = change.get("player")
    team      = change.get("team", "")
    market    = str(change.get("market", "totals") or "totals")
    direction = str(change.get("direction", "over") or "over").upper()
    o_line    = change.get("opening_line", "")
    reason    = str(change.get("reason", "") or "").strip()

    subject   = player if player else team
    mkt_label = market.replace("_", " ").title()
    pick_desc = f"{subject} {mkt_label} {direction}"
    if o_line:
        pick_desc += f" {o_line}"

    reason_line = f"\n*Reason:*\n{reason}\n" if reason else "\n"

    text = (
        f"🚨 *EDGE REVERSAL — MARKET INVALIDATION DETECTED*\n"
        f"\n"
        f"*Pick:* {pick_desc}\n"
        f"{reason_line}"
        f"*Opening Edge:* +{opening_edge:.1f}%\n"
        f"*Current Edge:* {current_edge:.1f}% *(NEGATIVE)*\n"
        f"\n"
        f"Model has crossed from positive to negative expected value.\n"
        f"Original thesis is invalidated. Do not place this bet."
    )

    try:
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        data = _j.dumps(payload).encode()
        req  = urllib.request.Request(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data=data, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            sent = resp.status == 200
            if sent:
                logger.info(
                    f"[revalidation] EDGE REVERSAL sent for "
                    f"{change.get('bet_id','?')} "
                    f"({opening_edge:.1f}% → {current_edge:.1f}%)"
                )
            return sent
    except Exception as exc:
        logger.warning(f"[revalidation] Edge reversal alert failed: {exc}")
        return False


def _check_and_send_edge_alerts(change: dict[str, Any]) -> bool:
    """
    Router: evaluate both edge alert conditions and send the appropriate alert.

    Telegram governance (spec §TELEGRAM NOTIFICATION GOVERNANCE):
    - ONLY fires for picks originally broadcast to the group (sent_to_group=1).
    - Edge Reversal (positive → negative EV) has priority over Edge Warning.
    - Edge Warning fires only when current_edge < opening_edge × 0.50.
    - No alert for upgrades, downgrades, confirms, tier changes, or any other
      condition not listed above.

    Returns True if any alert was sent.
    """
    # Spec: "Telegram notifications apply ONLY to picks originally published
    # to the Telegram group."
    if not change.get("sent_to_group"):
        logger.debug(
            f"[revalidation] {change.get('bet_id','?')}: "
            f"not sent_to_group — no Telegram alert."
        )
        return False

    opening_edge = float(change.get("opening_edge") or 0)
    current_edge = float(change.get("current_edge") or 0)

    if opening_edge <= 0:
        return False   # can't compute meaningful decay

    # Prevent re-alerting: if this pick was already voided/alerted, skip.
    prev_status = change.get("prev_status", "none") or "none"
    if prev_status == "voided":
        logger.debug(
            f"[revalidation] {change.get('bet_id','?')}: "
            f"already voided — skipping duplicate Telegram alert."
        )
        return False

    # Trigger 2 — Edge Reversal (positive EV → negative EV): higher priority
    if current_edge < 0:
        return _send_edge_reversal_alert(change)

    # Trigger 1 — Edge Loss Alert (current < 50 % of original)
    if current_edge < opening_edge * _EDGE_WARNING_RATIO:
        return _send_edge_warning_alert(change)

    logger.debug(
        f"[revalidation] {change.get('bet_id','?')}: "
        f"edge {opening_edge:.1f}% → {current_edge:.1f}% — "
        f"thresholds not met, no Telegram alert."
    )
    return False


def _write_change(change: dict[str, Any]) -> None:
    """Write revalidation result to bets table, pick_audit_log, and pick_regrade_history."""
    now = datetime.now(timezone.utc).isoformat()
    bet_id = change["bet_id"]
    status = change["revalidation_status"]

    with _connect() as conn:
        # 1. Update bets — scalars + targeted wager_details patch via json_set().
        # json_set() is a no-op on keys that already exist and are unchanged,
        # so it is safe to run on every revalidation cycle.
        conn.execute(
            """
            UPDATE bets SET
                current_edge         = ?,
                current_confidence   = ?,
                revalidation_status  = ?,
                revalidation_reason  = ?,
                revalidation_at      = ?,
                revalidation_flags   = ?,
                wager_details        = json_set(
                    COALESCE(wager_details, '{}'),
                    '$.current_effective_edge',  ?,
                    '$.edge_decay_revalidation', ?,
                    '$.sharp_signal',            ?,
                    '$.rlm_detected',            json(?),
                    '$.steam_detected',          json(?),
                    '$.mis_score',               ?
                )
            WHERE bet_id = ?
            """,
            (
                change.get("current_edge"),
                change.get("current_confidence"),
                status,
                change.get("reason", ""),
                now,
                json.dumps(change.get("flags", {})),
                # wager_details json_set args
                change.get("current_effective_edge"),
                change.get("edge_decay_revalidation"),
                change.get("fresh_sharp_signal"),
                json.dumps(change.get("fresh_rlm_detected", False)),
                json.dumps(change.get("fresh_steam_detected", False)),
                change.get("fresh_mis_score", 0),
                bet_id,
            ),
        )

        # 2. Audit log entry (only for non-confirm changes)
        if status != "confirmed":
            conn.execute(
                """
                INSERT INTO pick_audit_log
                  (changed_at, bet_id, sport, field_changed,
                   old_value, new_value, reason, alert_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    now,
                    bet_id,
                    change.get("sport"),
                    "revalidation_status",
                    change.get("prev_status", "none"),
                    status,
                    change.get("reason", ""),
                ),
            )

        # 3. V3.0 pick_regrade_history (non-confirm only)
        if status != "confirmed":
            _alert_sent = 0
            try:
                _alert_sent = int(_check_and_send_edge_alerts(change))
            except Exception:
                pass
            import json as _json_rh
            conn.execute(
                """
                INSERT INTO pick_regrade_history
                  (bet_id, sport, changed_at, version,
                   prev_tier, new_tier,
                   prev_confidence, new_confidence,
                   prev_edge, new_edge,
                   change_type, reason, alert_sent, snapshot_json)
                VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bet_id,
                    change.get("sport"),
                    now,
                    change.get("tier"),               # prev_tier (published tier)
                    change.get("tier"),               # new_tier (unchanged — revalidation adjusts edge, not tier)
                    change.get("opening_confidence"), # prev_confidence (at publication)
                    change.get("current_confidence"), # new_confidence (current revalidation)
                    change.get("opening_edge"),
                    change.get("current_edge"),
                    status,
                    change.get("reason", ""),
                    _alert_sent,
                    _json_rh.dumps({
                        k: v for k, v in change.items()
                        if k not in ("reason",) and not isinstance(v, dict)
                    }),
                ),
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_revalidation(
    sports:  list[str] | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """
    Revalidate all open, locked picks for today.

    Returns a list of change dicts — one per pick that is NOT 'confirmed'
    (i.e. picks whose status upgraded, downgraded, or voided).
    """
    today = _today_et()

    # ── Fetch today's open locked picks ─────────────────────────────────────
    sport_filter = (
        f"AND sport IN ({','.join('?' for _ in sports)})" if sports else ""
    )
    params: list[Any] = [today]
    if sports:
        params.extend([s.upper() for s in sports])

    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT bet_id, sport, tier, wager_details, sportsbook_odds,
                   model_probability, edge_percentage,
                   opening_edge, opening_confidence, opening_odds,
                   current_edge, current_confidence,
                   revalidation_status, sent_to_group
            FROM bets
            WHERE slate_date = ?
              AND status = 'open'
              AND is_locked = 1
              {sport_filter}
            """,
            params,
        ).fetchall()

    if not rows:
        logger.info("[revalidation] No open locked picks for today — nothing to do.")
        return []

    logger.info(f"[revalidation] Evaluating {len(rows)} open pick(s) for {today}.")

    # ── Fetch fresh odds per sport (one batch call each) ────────────────────
    sports_needed = {row["sport"] for row in rows}
    fresh: dict[str, list[dict]] = {}
    for sport in sports_needed:
        logger.info(f"[revalidation] Fetching current odds for {sport}…")
        fresh[sport] = _get_fresh_candidates(sport)
        logger.info(
            f"[revalidation] {sport}: {len(fresh[sport])} current candidates."
        )

    # ── Fetch fresh prop lines if any open picks are player props ────────────
    has_props = any(
        bool(json.loads(row["wager_details"]).get("player"))
        for row in rows
        if row["wager_details"]
    )
    fresh_props: dict[str, list[dict]] = {}
    if has_props:
        for sport in sports_needed:
            logger.info(f"[revalidation] Fetching current prop lines for {sport}…")
            fresh_props[sport] = _get_fresh_prop_candidates(sport)
            logger.info(
                f"[revalidation] {sport}: {len(fresh_props[sport])} fresh prop candidates."
            )

    # ── Evaluate each pick ───────────────────────────────────────────────────
    all_changes: list[dict[str, Any]] = []

    for row in rows:
        bet_id = row["bet_id"]
        sport  = row["sport"]

        try:
            wd = json.loads(row["wager_details"])
        except Exception:
            logger.warning(f"[revalidation] Bad wager_details for {bet_id} — skipping.")
            continue

        team      = wd.get("team",           "")
        market    = wd.get("market",         "totals")
        direction = wd.get("direction",      "over")
        orig_line = float(wd.get("sportsbook_line", 0.0) or 0.0)
        home_abbr = wd.get("home_team",      team)
        away_abbr = wd.get("away_team",      team)
        player    = wd.get("player")
        is_prop   = bool(player)

        # Opening snapshot (use stored value if available, else fall back)
        opening_edge  = float(row["opening_edge"]  or row["edge_percentage"] or 0.0)
        opening_conf  = float(row["opening_confidence"] or row["model_probability"] or 0.0)
        opening_odds  = int(  row["opening_odds"]  or row["sportsbook_odds"] or -110)

        # ── Find matching candidate in fresh odds ────────────────────────────
        candidate = _find_candidate(fresh.get(sport, []), team, market, direction)
        if candidate:
            current_line  = float(candidate.get("sportsbook_line", orig_line))
            current_odds  = int(  candidate.get("american_odds",   opening_odds))
            current_edge  = _recompute_edge(current_odds, opening_conf)
        else:
            current_line  = orig_line
            current_odds  = opening_odds
            current_edge  = opening_edge
            logger.debug(
                f"[revalidation] {bet_id}: no current market found — "
                f"keeping original line."
            )

        # ── Task C: recompute effective edge with fresh market signals ────────
        # Pull signals from fresh candidate if available, else fall back to
        # what was stored at publication time in wager_details.
        _fresh_sharp  = (candidate.get("sharp_signal")   if candidate else None) or wd.get("sharp_signal",  "no_sharp")
        _fresh_rlm    = (candidate.get("rlm_detected")   if candidate else None)
        if _fresh_rlm is None:
            _fresh_rlm = wd.get("rlm_detected", False)
        _fresh_steam  = (candidate.get("steam_detected") if candidate else None)
        if _fresh_steam is None:
            _fresh_steam = wd.get("steam_detected", False)
        _fresh_mis    = int((candidate.get("mis_score") if candidate else None) or wd.get("mis_score", 0) or 0)

        try:
            from core.market_intelligence import compute_effective_edge as _cee_rv
            current_effective_edge = _cee_rv(
                raw_edge       = current_edge,
                sharp_signal   = str(_fresh_sharp),
                rlm_detected   = bool(_fresh_rlm),
                steam_detected = bool(_fresh_steam),
                mis_score      = _fresh_mis,
            )
        except Exception as _cee_exc:
            logger.debug(f"[revalidation] {bet_id}: effective_edge recompute failed — {_cee_exc}")
            current_effective_edge = current_edge

        edge_decay_reval = round(current_edge - current_effective_edge, 2)
        logger.debug(
            f"[revalidation] {bet_id}: raw_edge={current_edge:.2f}% "
            f"eff_edge={current_effective_edge:.2f}% decay={edge_decay_reval:+.2f}% "
            f"sharp={_fresh_sharp} rlm={_fresh_rlm} steam={_fresh_steam} mis={_fresh_mis}"
        )

        # ── For player props: get the current sportsbook line directly ───────
        # Game-level candidates don't carry prop lines, so we need a separate
        # lookup. The stale-line void fires if the prop line moved ≥ threshold.
        if is_prop and player:
            prop_c = _find_prop_candidate(
                fresh_props.get(sport, []), player, market, direction
            )
            if prop_c:
                current_line  = float(prop_c.get("sportsbook_line", orig_line))
                current_odds  = int(  prop_c.get("american_odds",   opening_odds))
                current_edge  = _recompute_edge(current_odds, opening_conf)
                logger.debug(
                    f"[revalidation] {bet_id}: prop line {orig_line} → {current_line} "
                    f"(Δ{current_line - orig_line:+.1f})"
                )
            else:
                logger.debug(
                    f"[revalidation] {bet_id}: prop not found in fresh data — "
                    "keeping original line (market may be unavailable)."
                )

        abs_line_move = abs(current_line - orig_line)

        # ── Adverse line move ────────────────────────────────────────────────
        adverse = _adverse_move(current_line, orig_line, direction)

        # ── Injury check ─────────────────────────────────────────────────────
        injury_impact, lineup_desc = _check_lineup(team, sport)

        # ── Pitcher check (MLB only) ─────────────────────────────────────────
        sp_changed    = False
        pitcher_reason = ""
        if sport == "MLB" and market in ("totals", "team_total", "runline"):
            sp_changed, pitcher_reason = _check_pitcher_change(home_abbr, away_abbr)

        # ── Decision ─────────────────────────────────────────────────────────
        status, reason = _decide(
            sport          = sport,
            direction      = direction,
            adverse        = adverse,
            opening_edge   = opening_edge,
            current_edge   = current_edge,
            injury_impact  = injury_impact,
            sp_changed     = sp_changed,
            pitcher_reason = pitcher_reason,
            lineup_desc    = lineup_desc,
            is_prop        = is_prop,
            abs_line_move  = abs_line_move,
            orig_line      = orig_line,
            current_line   = current_line,
        )

        # Current bookmaker implied probability for the ORIGINAL direction.
        # Used by the reversal alert to check the ≥50 ppt shift threshold.
        _current_implied_pct = round(_implied_prob(current_odds) * 100.0, 1)

        change: dict[str, Any] = {
            "bet_id":                   bet_id,
            "sport":                    sport,
            "tier":                     row["tier"],
            "team":                     team,
            "market":                   market,
            "direction":                direction,
            "player":                   wd.get("player"),
            "opening_edge":             opening_edge,
            "opening_confidence":       opening_conf,
            "opening_odds":             opening_odds,
            "current_edge":             current_edge,
            "current_confidence":       opening_conf,   # model prob doesn't change
            "current_odds":             current_odds,
            "opening_line":             orig_line,
            "current_line":             current_line,
            # Alert governance fields
            "sent_to_group":            int(row["sent_to_group"] or 0),
            "model_probability":        float(row["model_probability"] or opening_conf),
            "current_implied_prob":     _current_implied_pct,
            "adverse_move":             adverse,
            "injury_impact":            injury_impact,
            "revalidation_status":      status,
            "prev_status":              row["revalidation_status"] or "none",
            "reason":                   reason,
            # Task C — fresh market signals and recomputed effective edge
            "current_effective_edge":   current_effective_edge,
            "edge_decay_revalidation":  edge_decay_reval,
            "fresh_sharp_signal":       _fresh_sharp,
            "fresh_rlm_detected":       bool(_fresh_rlm),
            "fresh_steam_detected":     bool(_fresh_steam),
            "fresh_mis_score":          _fresh_mis,
            "flags": {
                "adverse_line_move":    adverse,
                "injury_impact":        injury_impact,
                "sp_changed":           sp_changed,
                "candidate_found":      candidate is not None,
            },
        }

        all_changes.append(change)

        if dry_run:
            logger.info(
                f"[revalidation][DRY] {bet_id}: {status.upper()} — {reason}"
            )
        else:
            _write_change(change)
            logger.info(f"[revalidation] {bet_id}: {status.upper()} — {reason}")

    # Return only non-confirm changes for alerting
    notable = [c for c in all_changes if c["revalidation_status"] != "confirmed"]
    logger.info(
        f"[revalidation] Complete. "
        f"{len(all_changes)} evaluated, "
        f"{len(notable)} notable change(s)."
    )
    return notable


def get_todays_revalidation_summary() -> list[dict[str, Any]]:
    """
    Return all picks for today that have a revalidation_status set.
    Used by the API endpoint and MiniApp.
    """
    today = _today_et()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT bet_id, sport, tier, wager_details, sportsbook_odds,
                   edge_percentage, model_probability,
                   opening_edge, opening_confidence, opening_odds,
                   current_edge, current_confidence,
                   revalidation_status, revalidation_reason, revalidation_at,
                   revalidation_flags
            FROM bets
            WHERE slate_date = ?
              AND revalidation_status IS NOT NULL
            ORDER BY revalidation_at DESC
            """,
            (today,),
        ).fetchall()
    result = []
    for row in rows:
        try:
            wd = json.loads(row["wager_details"])
        except Exception:
            wd = {}
        result.append({
            "bet_id":             row["bet_id"],
            "sport":              row["sport"],
            "tier":               row["tier"],
            "team":               wd.get("team"),
            "player":             wd.get("player"),
            "market":             wd.get("market"),
            "direction":          wd.get("direction"),
            "opening_line":       wd.get("sportsbook_line"),
            "opening_odds":       row["opening_odds"] or row["sportsbook_odds"],
            "opening_edge":       row["opening_edge"] or row["edge_percentage"],
            "opening_confidence": row["opening_confidence"] or row["model_probability"],
            "current_edge":       row["current_edge"],
            "current_confidence": row["current_confidence"],
            "revalidation_status":  row["revalidation_status"],
            "revalidation_reason":  row["revalidation_reason"],
            "revalidation_at":      row["revalidation_at"],
            "flags":                json.loads(row["revalidation_flags"] or "{}"),
        })
    return result


def get_audit_log(bet_id: str | None = None, limit: int = 50) -> list[dict]:
    """Return pick_audit_log rows, optionally filtered by bet_id."""
    with _connect() as conn:
        if bet_id:
            rows = conn.execute(
                "SELECT * FROM pick_audit_log WHERE bet_id=? ORDER BY id DESC LIMIT ?",
                (bet_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM pick_audit_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]
