"""
core/game_truth.py — Game Truth Protocol

Pre-gatekeeper filter that enforces a single "Value Vector" per game and
applies sport-specific volatility thresholds to line movements across runs.

Four-rule execution flow (per the protocol spec)
-------------------------------------------------
1. Ingest all simulated candidates for a game simultaneously.
2. Calculate the Value Vector — the market with the highest composite signal
   (edge_percentage × confidence_score / 100) for that game.
3. Volatility check against persisted state from the previous run:
      MLB  — re-evaluate only if line moved ≥ 0.20 pts; else treat as noise.
      NBA  — re-evaluate only if line moved ≥ 1.50 pts.
      WNBA — re-evaluate only if line moved ≥ 2.00 pts.
   If the movement is below threshold, suppress all candidates for that game
   (the previous evaluation stands).
4. Line shopping — only the Value Vector candidates survive to the gatekeeper,
   which then selects the best-odds book among them.

State persistence
-----------------
State is written to `data/game_truth_state.json` after every run and loaded
at the start of the next run. Entries older than 30 minutes are expired
automatically, giving each scheduler cycle a clean slate when needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("betting_bot")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOLATILITY_THRESHOLDS: dict[str, float] = {
    "MLB":  0.20,
    "NBA":  1.50,
    "WNBA": 1.00,
}

_DEFAULT_THRESHOLD = 0.50   # fallback for unknown sports
_STATE_TTL_MINUTES = 30     # state older than this is expired on next load
_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "game_truth_state.json"


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class GameTruthState:
    """Persisted evaluation snapshot for one game."""
    game_id:          str
    sport:            str
    value_vector:     str    # winning market type, e.g. "team_total"
    locked_direction: str    # "over" or "under"
    locked_line:      float  # line when Value Vector was last evaluated
    edge_score:       float  # edge × conf / 100 at lock time
    timestamp:        str    # ISO-8601 UTC
    picks_published:  bool   = False  # True once a pick for this game was sent to Telegram


# ---------------------------------------------------------------------------
# State persistence helpers
# ---------------------------------------------------------------------------

def _load_state() -> dict[str, GameTruthState]:
    """
    Load persisted game-truth state from disk.
    Entries older than _STATE_TTL_MINUTES are silently dropped.
    """
    if not _STATE_FILE.exists():
        return {}

    try:
        raw: dict[str, Any] = json.loads(_STATE_FILE.read_text())
    except Exception as exc:
        logger.warning(f"[GameTruth] Could not read state file: {exc}")
        return {}

    cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=_STATE_TTL_MINUTES)
    result: dict[str, GameTruthState] = {}

    for game_id, entry in raw.items():
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                logger.debug(
                    f"[GameTruth] Expiring stale state for {game_id} "
                    f"(age > {_STATE_TTL_MINUTES} min)"
                )
                continue
            result[game_id] = GameTruthState(**entry)
        except Exception as exc:
            logger.debug(f"[GameTruth] Skipping malformed state entry {game_id}: {exc}")

    return result


def _save_state(state: dict[str, GameTruthState]) -> None:
    """Persist current game-truth state to disk."""
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {gid: asdict(s) for gid, s in state.items()}
        _STATE_FILE.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        logger.warning(f"[GameTruth] Could not save state file: {exc}")


def mark_picks_published(game_ids: list[str]) -> None:
    """
    Mark the given game_ids as having their picks published (sent to Telegram).
    Call this immediately after the broadcast succeeds so subsequent intraday
    re-runs suppress already-published picks correctly.
    """
    if not game_ids:
        return
    state = _load_state()
    for gid in game_ids:
        if gid in state:
            state[gid] = GameTruthState(
                game_id          = state[gid].game_id,
                sport            = state[gid].sport,
                value_vector     = state[gid].value_vector,
                locked_direction = state[gid].locked_direction,
                locked_line      = state[gid].locked_line,
                edge_score       = state[gid].edge_score,
                timestamp        = state[gid].timestamp,
                picks_published  = True,
            )
            logger.debug(f"[GameTruth] {gid} — marked picks_published=True")
    _save_state(state)


# ---------------------------------------------------------------------------
# Core protocol
# ---------------------------------------------------------------------------

def apply_game_truth_protocol(
    processed: list[tuple[dict[str, Any], float, float, float]],
    sport: str,
) -> list[tuple[dict[str, Any], float, float, float]]:
    """
    Apply the Game Truth Protocol to a list of simulated candidates.

    Parameters
    ----------
    processed : list of (candidate_dict, edge_pct, confidence, model_prob)
        Output of the simulation step in _run_sport_pipeline().
    sport : str
        Sport identifier ("MLB", "NBA", "WNBA").

    Returns
    -------
    Filtered list — only the Value Vector candidates survive, subject to
    the volatility threshold check.  Candidates without a game_id pass
    through unchanged (no game grouping possible).
    """
    if not processed:
        return processed

    threshold = VOLATILITY_THRESHOLDS.get(sport.upper(), _DEFAULT_THRESHOLD)
    state     = _load_state()
    now_utc   = datetime.now(tz=timezone.utc).isoformat()

    # Split candidates: those with a game_id vs. ungrouped
    grouped:   dict[str, list[tuple[dict[str, Any], float, float, float]]] = {}
    ungrouped: list[tuple[dict[str, Any], float, float, float]] = []

    for entry in processed:
        c = entry[0]
        gid = c.get("game_id", "").strip()
        if gid:
            grouped.setdefault(gid, []).append(entry)
        else:
            ungrouped.append(entry)

    surviving: list[tuple[dict[str, Any], float, float, float]] = list(ungrouped)

    for game_id, entries in grouped.items():

        # ── Step 2: Dual Value Vector ────────────────────────────────────────
        # Score each candidate's market by edge × confidence / 100.
        # Markets are split into two lanes: game markets (totals/spreads) and
        # prop markets (player/pitcher props).  The top market from EACH lane
        # independently survives to the gatekeeper so a game can contribute
        # both a prop pick and a game-total pick simultaneously.

        _GAME_MKT_TOKENS = {
            "totals", "total", "first_5", "f5", "spread",
            "moneyline", "run_line", "team_total",
        }

        def _is_game_market(mkt_norm: str) -> bool:
            return any(tok in mkt_norm for tok in _GAME_MKT_TOKENS)

        market_scores: dict[str, float] = {}
        for (c, edge, conf, _) in entries:
            mkt   = c.get("market", "").strip().lower().replace(" ", "_")
            score = edge * conf / 100.0
            market_scores[mkt] = market_scores.get(mkt, 0.0) + score

        if not market_scores:
            surviving.extend(entries)
            continue

        game_mkt_scores = {m: s for m, s in market_scores.items() if _is_game_market(m)}
        prop_mkt_scores = {m: s for m, s in market_scores.items() if not _is_game_market(m)}

        top_game_mkt = max(game_mkt_scores, key=lambda m: game_mkt_scores[m]) if game_mkt_scores else None
        top_prop_mkt = max(prop_mkt_scores, key=lambda m: prop_mkt_scores[m]) if prop_mkt_scores else None

        # Primary value vector for state tracking (game market preferred)
        value_vector = top_game_mkt or top_prop_mkt
        winning_vectors = {v for v in (top_game_mkt, top_prop_mkt) if v}

        vv_entries = [
            entry for entry in entries
            if entry[0].get("market", "").strip().lower().replace(" ", "_") in winning_vectors
        ]

        if not vv_entries:
            surviving.extend(entries)
            continue

        # Best candidate within the primary value vector (for state tracking)
        primary_vv_entries = [
            e for e in vv_entries
            if e[0].get("market", "").strip().lower().replace(" ", "_") == value_vector
        ] or vv_entries
        best_entry = max(primary_vv_entries, key=lambda e: e[1] * e[2] / 100.0)
        best_c, best_edge, best_conf, _ = best_entry
        best_line      = float(best_c.get("sportsbook_line", 0.0))
        best_direction = str(best_c.get("direction", "over")).lower()
        best_score     = best_edge * best_conf / 100.0

        # ── Step 3: Volatility check (primary game vector only) ──────────────
        prev = state.get(game_id)

        # Only suppress when picks have already been published for this game.
        # Before first publication the 3-cycle confirmation flow must be able
        # to converge; rapid same-session cycles must never block the first send.
        if prev is not None and prev.sport.upper() == sport.upper() and prev.picks_published:
            if prev.value_vector == value_vector:
                line_delta = abs(best_line - prev.locked_line)

                if line_delta < threshold:
                    logger.info(
                        f"[GameTruth] {game_id} — line movement {line_delta:.2f} "
                        f"< threshold {threshold:.2f} ({sport}) → suppressed "
                        f"(picks already published; Value Vector: {value_vector}, "
                        f"prev_line={prev.locked_line}, current={best_line})"
                    )
                    continue  # nothing from this game survives

                logger.info(
                    f"[GameTruth] {game_id} — line movement {line_delta:.2f} "
                    f">= threshold {threshold:.2f} ({sport}) → re-evaluating "
                    f"(prev_line={prev.locked_line} → {best_line})"
                )

            else:
                logger.info(
                    f"[GameTruth] {game_id} — Value Vector changed "
                    f"{prev.value_vector} → {value_vector} → re-evaluating"
                )

        # Update / create state entry for this game
        already_published = prev.picks_published if (prev is not None) else False
        state[game_id] = GameTruthState(
            game_id          = game_id,
            sport            = sport.upper(),
            value_vector     = value_vector,
            locked_direction = best_direction,
            locked_line      = best_line,
            edge_score       = best_score,
            timestamp        = now_utc,
            picks_published  = already_published,
        )

        # ── Step 4: Pass both game + prop value vector candidates ────────────
        surviving.extend(vv_entries)

        n_filtered = len(entries) - len(vv_entries)
        if n_filtered:
            filtered_mkts = set(
                e[0].get("market", "?") for e in entries
                if e[0].get("market", "").strip().lower().replace(" ", "_") not in winning_vectors
            )
            logger.info(
                f"[GameTruth] {game_id} — Vectors: {', '.join(sorted(winning_vectors))} "
                f"| filtered out {n_filtered} candidate(s) from: "
                f"{', '.join(sorted(filtered_mkts))}"
            )
        else:
            logger.debug(
                f"[GameTruth] {game_id} — Vectors: {', '.join(sorted(winning_vectors))} "
                f"(no candidates filtered)"
            )

    _save_state(state)

    logger.info(
        f"[GameTruth] {sport}: {len(processed)} candidates in → "
        f"{len(surviving)} out "
        f"({len(processed) - len(surviving)} suppressed by protocol)"
    )

    return surviving
