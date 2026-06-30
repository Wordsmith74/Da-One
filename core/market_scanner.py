"""
core/market_scanner.py

Approved Market Coverage & Scan Priority Framework.

Exports
-------
SCAN_PRIORITY         Per-sport ordered market scan list (informational / audit).
composite_score()     Edge×40% + Confidence×30% + Liquidity×15% + ROI×15%.
apply_per_game_caps() Enforce max-1-per-market-type and max-3-per-game rules.
log_market_audit()    Write per-game market coverage audit to a JSON-L file.
get_market_historical_roi()  Pull per-market ROI from performance_stats.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Scan priority — the engine MUST evaluate markets in this order before
# declaring NO PLAY for any game.  Informational only; ranking is always by
# composite_score(), not by position in this list.
# ─────────────────────────────────────────────────────────────────────────────

SCAN_PRIORITY: dict[str, list[str]] = {
    "MLB": [
        "Strikeouts",           # pitcher props  (player_props.py)
        "Outs Recorded",
        "Earned Runs",
        "F5 Total",             # first-5 innings total  (game_markets.py)
        "F5 Moneyline",
        "F5 Run Line",
        "Team Total",           # team totals
        "Totals",               # full-game total  (odds_client.py)
        "Moneyline",            # full-game ML
        "Run Line",             # full-game run line
    ],
    "NBA": [
        "Points",               # player props
        "Pts+Reb+Ast",
        "Assists",
        "Rebounds",
        "Team Total",           # team totals
        "Q1 Total",             # quarter totals
        "Q1 Moneyline",
        "Q1 Spread",
        "H1 Total",             # half totals
        "H1 Moneyline",
        "H1 Spread",
        "Totals",               # full-game
        "Spread",
        "Moneyline",
    ],
    "WNBA": [
        "Points",
        "Assists",
        "Rebounds",
        "Team Total",
        "Totals",
        "Spread",
        "Moneyline",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Composite score
# ─────────────────────────────────────────────────────────────────────────────

def composite_score(
    edge_pct: float,
    confidence: float,
    book_count: int,
    historical_roi: float | None = None,
) -> float:
    """
    Rank candidates by expected value using four components:

        Final Score = Edge×40% + Confidence×30% + Liquidity×15% + ROI×15%

    All components are normalised to [0, 100] before weighting.

    Parameters
    ----------
    edge_pct        Calibrated edge percentage (0–15 % typical range).
    confidence      Confidence score (0–100).
    book_count      Number of distinct books offering the line (proxy for liquidity).
    historical_roi  Per-market ROI percentage (None → neutral 50 points).
    """
    # Normalise edge: 0% → 0, 15% → 100 (linear; capped)
    edge_score = min(100.0, max(0.0, edge_pct * (100.0 / 15.0)))

    # Confidence already 0–100
    conf_score = min(100.0, max(0.0, float(confidence)))

    # Liquidity: each distinct book ≈ 14 points; 7 books = 100
    liq_score = min(100.0, book_count * 14.3)

    # Historical ROI: maps [-20%, +20%] → [0, 100]; no data → 50
    if historical_roi is None:
        roi_score = 50.0
    else:
        roi_score = min(100.0, max(0.0, (historical_roi + 20.0) * 2.5))

    return round(
        edge_score  * 0.40
        + conf_score  * 0.30
        + liq_score   * 0.15
        + roi_score   * 0.15,
        2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-game market caps
# ─────────────────────────────────────────────────────────────────────────────

def apply_per_game_caps(
    bets: list[Any],           # list[Bet] — avoid circular import
    max_per_market: int = 1,
    max_per_game:   int = 3,
    roi_lookup: dict[str, float] | None = None,
) -> tuple[list[Any], list[Any]]:
    """
    Enforce per-game market caps on the approved bet list.

    Rules (applied in order)
    ------------------------
    1. Within each (game_id, market_type) pair keep the highest composite-score
       bet only.  Excess bets are demoted.
    2. Across each game_id keep at most *max_per_game* bets ranked by composite
       score.  Excess bets are demoted.

    Parameters
    ----------
    bets            Approved Bet objects from run_gatekeeper().
    max_per_market  Maximum recommendations per market type per game (default 1).
    max_per_game    Maximum recommendations per game total (default 3).
    roi_lookup      Optional {market_label: roi_pct} for composite scoring.

    Returns
    -------
    (kept, demoted)
        kept    — bets that survive the caps (still in priority order)
        demoted — bets removed by caps (caller may log/flag them)
    """
    roi_lookup = roi_lookup or {}

    def _score(bet: Any) -> float:
        roi = roi_lookup.get(str(getattr(bet, "market", "")))
        return composite_score(
            edge_pct      = float(getattr(bet, "edge_percentage",  0)),
            confidence    = float(getattr(bet, "confidence_score", 0)),
            book_count    = int(getattr(bet, "mis_score", 0) // 10) or 3,
            historical_roi= roi,
        )

    from collections import defaultdict

    # Group by game_id
    by_game: dict[str, list[Any]] = defaultdict(list)
    for bet in bets:
        by_game[getattr(bet, "game_id", "") or "__no_game__"].append(bet)

    kept:    list[Any] = []
    demoted: list[Any] = []

    for game_id, game_bets in by_game.items():
        # Sort descending by composite score
        ranked = sorted(game_bets, key=_score, reverse=True)

        # Rule 1: max 1 per market type per game
        seen_markets: set[str] = set()
        after_market_cap: list[Any] = []
        for bet in ranked:
            mkt = str(getattr(bet, "market", "")).lower().strip()
            if mkt not in seen_markets:
                seen_markets.add(mkt)
                after_market_cap.append(bet)
            else:
                demoted.append(bet)

        # Rule 2: max max_per_game total per game
        game_kept   = after_market_cap[:max_per_game]
        game_excess = after_market_cap[max_per_game:]

        kept.extend(game_kept)
        demoted.extend(game_excess)

    return kept, demoted


# ─────────────────────────────────────────────────────────────────────────────
# Market audit logging
# ─────────────────────────────────────────────────────────────────────────────

_AUDIT_DIR = Path(__file__).parent.parent / "data" / "market_audit"


def log_market_audit(
    sport: str,
    date_str: str,
    candidates_by_market: dict[str, list[dict[str, Any]]],
    approved: list[Any],
) -> None:
    """
    Append a per-game market coverage audit entry to
    data/market_audit/{SPORT}_{DATE}.jsonl.

    Logged fields for each game
    ---------------------------
    sport, date, game_id, markets_scanned, markets_with_candidates,
    markets_rejected, markets_qualified (≥ edge threshold),
    published (in approved list), timestamp
    """
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = _AUDIT_DIR / f"{sport.upper()}_{date_str}.jsonl"

    # Build set of published bet_ids for fast lookup
    published_ids = {getattr(b, "bet_id", "") for b in approved}

    # Aggregate by game_id across all market buckets
    from collections import defaultdict
    game_data: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "markets_scanned":    set(),
        "markets_with_lines": set(),
        "markets_qualified":  [],
        "published":          [],
    })

    scan_order = SCAN_PRIORITY.get(sport.upper(), [])
    for market_label, cands in candidates_by_market.items():
        for c in cands:
            gid = c.get("game_id", "unknown")
            game_data[gid]["markets_scanned"].add(market_label)
            if c.get("sportsbook_line") is not None:
                game_data[gid]["markets_with_lines"].add(market_label)

    for bet in approved:
        gid  = getattr(bet, "game_id", "unknown")
        mkt  = getattr(bet, "market", "")
        bid  = getattr(bet, "bet_id", "")
        game_data[gid]["markets_qualified"].append(mkt)
        if bid in published_ids:
            game_data[gid]["published"].append(bid)

    now_iso = datetime.now(timezone.utc).isoformat()
    with open(audit_path, "a", encoding="utf-8") as fh:
        for game_id, data in game_data.items():
            scanned   = sorted(data["markets_scanned"])
            with_lines = sorted(data["markets_with_lines"])
            rejected  = sorted(data["markets_scanned"] - data["markets_with_lines"])
            entry = {
                "sport":                   sport.upper(),
                "date":                    date_str,
                "game_id":                 game_id,
                "scan_priority":           scan_order,
                "markets_scanned":         scanned,
                "markets_with_lines":      with_lines,
                "markets_rejected":        rejected,
                "markets_qualified":       data["markets_qualified"],
                "published_bet_ids":       data["published"],
                "no_play":                 len(data["markets_qualified"]) == 0,
                "timestamp":               now_iso,
            }
            fh.write(json.dumps(entry) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Historical ROI lookup (for composite scoring)
# ─────────────────────────────────────────────────────────────────────────────

def get_market_historical_roi() -> dict[str, float]:
    """
    Return {market_label: roi_pct} computed from graded bets.
    Returns an empty dict on any error (falls back to neutral 50 in composite_score).
    """
    try:
        import sqlite3
        from pathlib import Path as _P
        db = _P(__file__).parent.parent / "data" / "results.db"
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT
                json_extract(wager_details, '$.market') AS mkt,
                actual_outcome,
                COALESCE(profit_loss, 0)                AS pl
            FROM bets
            WHERE status = 'closed'
              AND actual_outcome IN ('win','loss','push')
        """).fetchall()
        conn.close()

        from collections import defaultdict
        by_mkt: dict[str, dict[str, float]] = defaultdict(lambda: {"pl": 0.0, "n": 0})
        for r in rows:
            mkt = (r["mkt"] or "").strip()
            if not mkt:
                continue
            by_mkt[mkt]["pl"] += float(r["pl"])
            by_mkt[mkt]["n"]  += 1

        result: dict[str, float] = {}
        for mkt, d in by_mkt.items():
            if d["n"] >= 5:   # minimum sample for meaningful ROI
                staked = d["n"] * 100.0
                result[mkt] = round(d["pl"] / staked * 100, 2)
        return result
    except Exception:
        return {}
