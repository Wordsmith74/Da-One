"""
core/alerts.py

Neutral alert sink. Replaces direct Telegram API calls that previously lived
inline in core/slate_versioner.py and core/revalidation_engine.py.

This intentionally does NOT talk to Telegram, a MiniApp, or any chat
platform. It logs the alert and appends it to a local JSONL file so nothing
is silently dropped -- if/when a real notification channel is wired back
in, point send_alert() at it instead of editing every call site again.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger("betting_bot")

_ALERTS_LOG_PATH = os.environ.get(
    "ALERTS_LOG_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "alerts.jsonl"),
)


def send_alert(text: str, *, source: str = "unknown", level: str = "info") -> bool:
    """
    Record an alert that would previously have gone to Telegram.

    text   : the alert message body.
    source : which subsystem raised it (e.g. "slate_versioner",
             "revalidation_engine") -- kept for triage, not required.
    level  : "info" | "warning" | "critical" -- routed to the matching
             logger level; critical alerts are still logged loudly even
             though there's no external channel to push them to right now.

    Returns True on successful local persistence (mirrors the old
    send-result boolean shape so existing call sites that check truthiness
    keep working).
    """
    log_fn = {"info": logger.info, "warning": logger.warning}.get(level, logger.critical)
    log_fn(f"[alert:{source}] {text}")

    try:
        os.makedirs(os.path.dirname(_ALERTS_LOG_PATH), exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "level": level,
            "text": text,
        }
        with open(_ALERTS_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        return True
    except Exception as exc:
        logger.warning(f"[alerts] failed to persist alert locally: {exc}")
        return False
