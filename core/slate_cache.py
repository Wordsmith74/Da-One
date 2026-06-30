"""
slate_cache.py

Producer-consumer JSON cache for a single day's fetched candidates,
keyed by sport (or a derived cache key like "WNBA_EXPANDED").

This exists so that multiple pipeline stages/runs within the same session
(e.g. game totals, expanded game markets, player props -- all pulling from
the same underlying Odds API "today's slate") don't each spend a separate
API credit re-fetching the same data. The first caller within a ~30 minute
window pays the network cost; everyone after reads the cached JSON.

File layout: data/slate_cache/{KEY}_{date}.json
    {
        "cached_at": "<ISO-8601 UTC timestamp>",
        "candidates": [ ... ]
    }
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

_CACHE_DIR = os.path.join("data", "slate_cache")
_MAX_AGE_SECONDS = 30 * 60  # 30 minutes


def _cache_path(key: str, date_str: str) -> str:
    safe_key = str(key).upper().replace("/", "_").replace(" ", "_")
    return os.path.join(_CACHE_DIR, f"{safe_key}_{date_str}.json")


def read_slate(key: str, date_str: str) -> list[dict[str, Any]] | None:
    """
    Return cached candidates for *key*/*date_str* if the cache file exists
    and is no older than 30 minutes, else None (cache miss/stale).

    Never raises -- any I/O or parse error is treated as a cache miss so
    callers always fall through to a live fetch.
    """
    path = _cache_path(key, date_str)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    cached_at_raw = payload.get("cached_at")
    if not cached_at_raw:
        return None

    try:
        cached_at = datetime.fromisoformat(cached_at_raw)
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)
    except Exception:
        return None

    age = (datetime.now(timezone.utc) - cached_at).total_seconds()
    if age > _MAX_AGE_SECONDS:
        return None

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return None

    return candidates


def write_slate(key: str, date_str: str, candidates: list[dict[str, Any]]) -> None:
    """
    Persist *candidates* to the slate cache for *key*/*date_str*.

    Best-effort: failures to write the cache (e.g. read-only filesystem)
    are swallowed -- caching is a performance optimisation, not a
    correctness requirement, so a write failure must never break the
    pipeline.
    """
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = _cache_path(key, date_str)
        payload = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "candidates": candidates,
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)
    except Exception:
        pass
