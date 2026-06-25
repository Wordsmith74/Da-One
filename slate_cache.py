"""
slate_cache.py — Producer-consumer JSON cache for daily bet-candidate slates.

Architecture
------------
Producer (odds_client.fetch_todays_candidates):
    After fetching and validating from The Odds API, write the candidate list
    to a per-sport per-date JSON file.  Subsequent calls in the same session
    read from the cache rather than re-hitting the API.

Consumer (_run_sport_pipeline in main.py):
    Candidates are always read from the slate returned by fetch_todays_candidates,
    which transparently serves the cache when it is fresh.  The engine never
    calls the network directly — it always reads from this local state file.

Benefits
--------
• Decouples data ingestion from analysis: a network surge / API timeout never
  stalls the Bayesian engine mid-pipeline.
• Eliminates redundant API calls within a 30-minute session window.
• Preserves the full validated candidate structure so the engine sees exactly
  the same data whether it's reading live or cached.
• Quota savings: with a 30-min cache, multiple intra-day dry-runs share one
  API hit instead of consuming a credit per run.

File layout
-----------
    data/slate_cache/{SPORT}_{YYYY-MM-DD}.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(__file__).parent.parent / "data" / "slate_cache"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_path(sport: str, date_str: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{sport.upper()}_{date_str}.json"


def _serialize_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert non-JSON-serialisable fields to plain types for storage."""
    out = []
    for c in candidates:
        row = dict(c)
        gt = row.get("game_time_utc")
        if isinstance(gt, datetime):
            row["game_time_utc"] = gt.isoformat()
        out.append(row)
    return out


def _deserialize_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore fields that were serialised to strings back to their native types."""
    for c in candidates:
        raw = c.get("game_time_utc")
        if isinstance(raw, str) and raw:
            try:
                c["game_time_utc"] = datetime.fromisoformat(
                    raw.replace("Z", "+00:00")
                )
            except ValueError:
                pass  # leave as string — downstream code guards with .date()
    return candidates


def _filter_started_games(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove candidates whose game has already started in ET.

    Applied when reading from cache so that a slate written early in the day
    never serves in-progress games to later engine cycles.  The conversion
    is safe for both UTC-aware and ET-aware datetimes stored under
    'game_time_utc' because astimezone() normalises either correctly.
    """
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
    now_et = datetime.now(_ET)
    out: list[dict[str, Any]] = []
    for c in candidates:
        gt = c.get("game_time_utc")
        if not isinstance(gt, datetime):
            out.append(c)
            continue
        try:
            game_et = gt.astimezone(_ET)
            if game_et > now_et:
                out.append(c)
        except Exception:
            out.append(c)  # keep on conversion error
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_slate(
    sport: str,
    candidates: list[dict[str, Any]],
    date_str: str,
) -> None:
    """
    Persist a validated candidate list for *sport* on *date_str* (YYYY-MM-DD).

    No-op when candidates is empty so we never cache an empty result —
    the next call will re-fetch rather than silently serve nothing.
    """
    if not candidates:
        return
    path = _cache_path(sport, date_str)
    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "sport":      sport.upper(),
        "date":       date_str,
        "count":      len(candidates),
        "candidates": _serialize_candidates(candidates),
    }
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")


def read_slate(
    sport: str,
    date_str: str,
    max_age_min: int = 30,
) -> list[dict[str, Any]] | None:
    """
    Return cached candidates if the cache file exists and is ≤ max_age_min old.

    Returns None on cache miss, stale entry, or any read/parse error so the
    caller falls through to a live API fetch.
    """
    path = _cache_path(sport, date_str)
    if not path.exists():
        return None
    try:
        payload   = json.loads(path.read_text(encoding="utf-8"))
        written   = datetime.fromisoformat(payload["written_at"].replace("Z", "+00:00"))
        age_s     = (datetime.now(timezone.utc) - written).total_seconds()
        if age_s > max_age_min * 60:
            return None  # stale — caller will re-fetch
        candidates = payload.get("candidates") or []
        if not candidates:
            return None
        candidates = _deserialize_candidates(candidates)
        candidates = _filter_started_games(candidates)
        if not candidates:
            return None  # all games started — fall through to live fetch
        return candidates
    except Exception:
        return None  # corrupt cache — caller will re-fetch


def clear_stale_caches(keep_days: int = 2) -> None:
    """Remove JSON files older than *keep_days* days (called by scheduler cleanup)."""
    if not _CACHE_DIR.exists():
        return
    cutoff = datetime.now(timezone.utc).timestamp() - keep_days * 86400
    for f in _CACHE_DIR.glob("*.json"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass
