"""
core/intelligence/signal_calibrator.py — Dynamic Signal Calibration

Replaces fixed adjustment scalars with data-driven values that learn from
historical bet outcomes.

How it works
------------
1. At startup (and every RECALIBRATE_EVERY_H hours) the calibrator reads all
   graded bets from the DB and groups them by signal bucket.

2. For each bucket it computes the empirical ROI lift vs the baseline
   (all graded bets with no signal):
       empirical_adj = roi_with_signal − roi_baseline

3. The final adjustment is a Bayesian blend of the research prior and the
   empirical value.  With <MIN_SAMPLES graded bets in a bucket the prior
   dominates; as data accumulates the empirical value takes over:

       confidence  = min(1.0, n_graded / MIN_SAMPLES)
       blended_adj = prior_adj × (1 − confidence) + empirical_adj × confidence

4. Callers receive a SignalAdjustment namedtuple with the blended value plus
   metadata (confidence, n_graded, source).

Signal buckets
--------------
  steam_confirming   — large line move (≥1.0 pt) in our direction
  steam_opposing     — large line move against our direction
  line_confirming    — medium line move (0.5–1.0 pt) in our direction
  line_opposing      — medium line move against our direction
  consensus_high     — multi-book dispersion ≥ 1.0 pt, our line favourable
  consensus_medium   — multi-book dispersion ≥ 0.5 pt, our line favourable
  sharp_confirmed    — Pinnacle / LowVig is the bookmaker source
  reverse_line       — public % strongly on one side but line moved other way

Research priors
---------------
Conservative estimates from published sports betting research.  Intentionally
understated so the system errs on the side of caution until real data arrives.

  steam_confirming : +2.5   (Levitt 2004; Gray & Gray 1997 — steam = sharp action)
  steam_opposing   : −3.0
  line_confirming  : +1.5
  line_opposing    : −2.0
  consensus_high   : +1.5   (Pinnacle dispersion study — Franck et al. 2010)
  consensus_medium : +0.8
  sharp_confirmed  : +1.2   (Closing-line efficiency at sharp books)
  reverse_line     : +1.8   (Contrarian value when public % diverges from line)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("betting_bot")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "results.db"

# Minimum graded bets in a bucket before empirical data starts to outweigh prior
MIN_SAMPLES: int = 50

# How many hours between full DB re-reads
RECALIBRATE_EVERY_H: float = 4.0

# ---------------------------------------------------------------------------
# Research priors  {bucket_name: prior_adjustment}
# ---------------------------------------------------------------------------

PRIORS: dict[str, float] = {
    "steam_confirming": 2.5,
    "steam_opposing":  -3.0,
    "line_confirming":  1.5,
    "line_opposing":   -2.0,
    "consensus_high":   1.5,
    "consensus_medium": 0.8,
    "sharp_confirmed":  1.2,
    "reverse_line":     1.8,
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalAdjustment:
    """Calibrated edge adjustment for one signal bucket."""
    signal:     str
    value:      float   # blended adjustment (edge points)
    prior:      float   # research prior
    empirical:  float   # empirical value (or prior if insufficient data)
    confidence: float   # 0.0 → all prior, 1.0 → all empirical
    n_graded:   int     # graded bets in this bucket
    source:     str     # "prior" | "blended" | "empirical"


# ---------------------------------------------------------------------------
# Process-level cache
# ---------------------------------------------------------------------------

_CACHE: dict[str, SignalAdjustment] = {}
_CACHE_BUILT_AT: datetime | None = None


def _cache_stale() -> bool:
    if _CACHE_BUILT_AT is None:
        return True
    age_h = (datetime.now(timezone.utc) - _CACHE_BUILT_AT).total_seconds() / 3600
    return age_h >= RECALIBRATE_EVERY_H


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_graded_bets() -> list[dict[str, Any]]:
    """
    Return all graded bets (win / loss / push) from the DB with the columns
    needed for calibration.
    """
    if not _DB_PATH.exists():
        return []
    try:
        con = sqlite3.connect(str(_DB_PATH))
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            """
            SELECT actual_outcome, profit_loss, stake,
                   line_move_dir, bookmaker_source,
                   edge_percentage, sport, wager_details
            FROM bets
            WHERE actual_outcome IN ('win', 'loss', 'push')
              AND stake > 0
            """
        )
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return rows
    except Exception as exc:
        logger.debug(f"[signal_calibrator] DB read failed: {exc}")
        return []


def _roi(bets: list[dict[str, Any]]) -> float:
    """Mean profit/loss as a fraction of mean stake, or 0 if empty."""
    if not bets:
        return 0.0
    total_pnl   = sum(b.get("profit_loss") or 0.0 for b in bets)
    total_stake = sum(b.get("stake")       or 0.0 for b in bets)
    return total_pnl / total_stake if total_stake > 0 else 0.0


# ---------------------------------------------------------------------------
# Calibration core
# ---------------------------------------------------------------------------

def _build_cache() -> None:
    """Read DB, compute empirical ROI lifts, blend with priors, populate cache."""
    global _CACHE, _CACHE_BUILT_AT

    bets = _load_graded_bets()
    baseline_roi = _roi(bets)
    n_total = len(bets)

    new_cache: dict[str, SignalAdjustment] = {}

    for signal, prior_adj in PRIORS.items():

        # ── Identify bets that belong to this signal bucket ────────────────
        if signal in ("steam_confirming", "steam_opposing",
                      "line_confirming",  "line_opposing"):
            bucket = [b for b in bets if b.get("line_move_dir") == signal]

        elif signal == "sharp_confirmed":
            _SHARP_KEYS = {
                "Pinnacle", "pinnacle", "LowVig", "lowvig",
                "BetAnySports", "betanysports", "Matchbook", "matchbook",
            }
            bucket = [b for b in bets if b.get("bookmaker_source") in _SHARP_KEYS]

        elif signal in ("consensus_high", "consensus_medium"):
            import json as _json
            bucket = []
            for b in bets:
                try:
                    wd = _json.loads(b.get("wager_details") or "{}")
                    cs = wd.get("consensus_signal")
                    if signal == "consensus_high"   and cs == "high":
                        bucket.append(b)
                    elif signal == "consensus_medium" and cs in ("high", "medium"):
                        bucket.append(b)
                except Exception:
                    pass

        elif signal == "reverse_line":
            # Reverse line = line moved against heavy public betting side.
            # We store this as line_move_dir="steam_confirming" with
            # wager_details["public_pct_against"] > 0.6 (future extension).
            # For now use the same bucket as steam_confirming as a proxy.
            bucket = [b for b in bets if b.get("line_move_dir") == "steam_confirming"]

        else:
            bucket = []

        n_graded = len(bucket)

        # ── Empirical ROI lift ──────────────────────────────────────────────
        if n_graded > 0:
            bucket_roi = _roi(bucket)
            # Convert ROI lift to edge-adjustment units.
            # Empirically: 1% ROI lift ≈ 1.0 edge-point in this system's scale.
            # We cap the raw lift to ±8 edge points to avoid outlier domination.
            raw_lift = (bucket_roi - baseline_roi) * 100  # convert to pct points
            empirical_adj = max(-8.0, min(8.0, raw_lift))
        else:
            empirical_adj = prior_adj  # no data → fall back to prior as empirical

        # ── Bayesian blend ──────────────────────────────────────────────────
        confidence  = min(1.0, n_graded / MIN_SAMPLES)
        blended     = prior_adj * (1 - confidence) + empirical_adj * confidence
        blended     = round(blended, 2)

        if confidence == 0.0:
            source = "prior"
        elif confidence < 0.8:
            source = "blended"
        else:
            source = "empirical"

        new_cache[signal] = SignalAdjustment(
            signal     = signal,
            value      = blended,
            prior      = prior_adj,
            empirical  = round(empirical_adj, 2),
            confidence = round(confidence, 3),
            n_graded   = n_graded,
            source     = source,
        )

    _CACHE = new_cache
    _CACHE_BUILT_AT = datetime.now(timezone.utc)

    if bets:
        logger.info(
            f"[signal_calibrator] Calibrated {len(PRIORS)} signals from "
            f"{n_total} graded bets  baseline_roi={baseline_roi*100:.1f}%"
        )
        for sig, adj in _CACHE.items():
            logger.debug(
                f"  {sig:22s}  adj={adj.value:+.2f}  "
                f"prior={adj.prior:+.1f}  empirical={adj.empirical:+.2f}  "
                f"conf={adj.confidence:.2f}  n={adj.n_graded}  [{adj.source}]"
            )
    else:
        logger.debug("[signal_calibrator] No graded bets — using research priors.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_adjustment(signal: str) -> SignalAdjustment:
    """
    Return the calibrated adjustment for *signal*.

    Rebuilds the cache if stale.  Falls back gracefully to the research prior
    when no historical data exists for the bucket.

    Parameters
    ----------
    signal : one of the PRIORS keys (e.g. "steam_confirming")

    Returns
    -------
    SignalAdjustment with .value = blended edge adjustment (edge points).
    """
    if _cache_stale():
        try:
            _build_cache()
        except Exception as exc:
            logger.warning(f"[signal_calibrator] calibration failed: {exc}")
            # Return pure prior on failure
            prior = PRIORS.get(signal, 0.0)
            return SignalAdjustment(
                signal=signal, value=prior, prior=prior, empirical=prior,
                confidence=0.0, n_graded=0, source="prior"
            )

    if signal in _CACHE:
        return _CACHE[signal]

    # Unknown signal — return zero (no adjustment)
    return SignalAdjustment(
        signal=signal, value=0.0, prior=0.0, empirical=0.0,
        confidence=0.0, n_graded=0, source="prior"
    )


def get_all_adjustments() -> dict[str, SignalAdjustment]:
    """Return a snapshot of all calibrated adjustments."""
    if _cache_stale():
        try:
            _build_cache()
        except Exception:
            pass
    return dict(_CACHE)


def invalidate_cache() -> None:
    """Force a full recalibration on the next call to get_adjustment()."""
    global _CACHE_BUILT_AT
    _CACHE_BUILT_AT = None
