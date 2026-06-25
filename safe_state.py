"""
safe_state.py

Safe State Manager — Rule 4 of the data resilience protocol.

  "The system must never operate on stale data from a previous day.
   If no current data can be fetched from any provider, the engine
   must enter a Safe State, which stops all parlay generation
   processes until the connection is restored."

How it works
------------
1. check_connectivity() in data_fetcher runs a lightweight probe at
   the start of every daily run.

2. The result is passed to report_and_evaluate(connectivity_report),
   which logs the status of every source (Rule 4: "explicitly report
   the status of the data pull at the start of the daily process").

3. If no source responds, enter_safe_state() is called.  This:
     a. Sets the module-level SAFE_STATE flag.
     b. Writes a machine-readable status file so the MiniApp can
        display a "data unavailable" banner.
     c. Sends a Telegram alert via the provided alert function.

4. is_active() is the gate callers check before running the engine.

5. clear() is called when a subsequent probe succeeds so the engine
   can resume on the next run.

Thread-safety
-------------
Safe State is set/read only from the main thread of a single cron
invocation.  No locking is needed for this use-case; the module-level
dict is reset on each new process launch.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("betting_bot")

# ---------------------------------------------------------------------------
# State store — reset on every process launch
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "active":      False,
    "reason":      "",
    "entered_at":  None,
    "source_report": {},
}

# Path for machine-readable status (read by MiniApp /api/status endpoint)
_STATUS_FILE = Path(__file__).resolve().parent.parent / "data" / "system_status.json"


# ---------------------------------------------------------------------------
# Internal writer
# ---------------------------------------------------------------------------

def _write_status_file(status: str, detail: str) -> None:
    """Persist current system status so the MiniApp can display it."""
    try:
        _STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status":     status,           # "ok" | "safe_state" | "degraded"
            "detail":     detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "sources":    _state["source_report"],
        }
        _STATUS_FILE.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        logger.debug(f"[safe_state] Could not write status file: {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def report_and_evaluate(
    connectivity_report: Any,          # data_fetcher.ConnectivityReport
    alert_fn: Callable[[str], None] | None = None,
) -> bool:
    """
    Log the data-pull status report (Rule 4) and determine whether to
    enter Safe State.

    Parameters
    ----------
    connectivity_report : ConnectivityReport from data_fetcher.check_connectivity()
    alert_fn            : callable(message) that sends a Telegram alert

    Returns
    -------
    True  → at least one source is responding; engine may proceed.
    False → all sources down; Safe State entered; engine must be halted.
    """
    _state["source_report"] = connectivity_report.results

    # Rule 4: always log the status at the start of the daily process
    logger.info(connectivity_report.summary())

    if connectivity_report.any_source_ok:
        # Clear any previous Safe State if sources recovered
        if _state["active"]:
            _clear(previous_reason=_state["reason"])
        _write_status_file("ok", "Data sources responding normally.")
        return True

    # All sources down — enter Safe State
    reason = (
        "All data sources unavailable: "
        + ", ".join(
            f"{src} ({'OK' if ok else 'FAIL'})"
            for src, ok in connectivity_report.results.items()
        )
    )
    enter_safe_state(reason, alert_fn=alert_fn)
    return False


def enter_safe_state(
    reason: str,
    alert_fn: Callable[[str], None] | None = None,
) -> None:
    """
    Activate Safe State.

    Logs an ERROR, persists the status file, and fires a Telegram alert
    if *alert_fn* is provided.  Subsequent calls to is_active() return
    True until clear() is called.
    """
    _state["active"]     = True
    _state["reason"]     = reason
    _state["entered_at"] = datetime.now(timezone.utc).isoformat()

    logger.error(
        f"\n{'⚠' * 50}\n"
        f"  SAFE STATE ACTIVATED\n"
        f"  {reason}\n"
        f"  All parlay generation is HALTED.\n"
        f"  Restore data connectivity before the next run.\n"
        f"{'⚠' * 50}"
    )

    _write_status_file("safe_state", reason)

    if alert_fn:
        try:
            alert_fn(
                "🚨 *SAFE STATE ACTIVATED*\n\n"
                f"All data sources are unreachable.\n\n"
                f"*Reason:* {reason}\n\n"
                "_Parlay generation has been HALTED. "
                "No picks will be sent until data connectivity is restored._"
            )
        except Exception as exc:
            logger.warning(f"[safe_state] Could not send Telegram alert: {exc}")


def is_active() -> bool:
    """
    Return True if the engine is in Safe State and should NOT run.
    Call this gate before every pipeline invocation.
    """
    return _state["active"]


def _clear(previous_reason: str = "") -> None:
    """Internal: reset Safe State and log recovery."""
    _state["active"]     = False
    _state["reason"]     = ""
    _state["entered_at"] = None
    logger.info(
        "[safe_state] Safe State CLEARED — data sources responding. "
        f"(Was: {previous_reason})"
    )


def get_status() -> dict[str, Any]:
    """Return a copy of the current state dict for reporting."""
    return {**_state}
