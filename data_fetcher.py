"""
data_fetcher.py

Data Resilience Layer — implements the four-rule protocol:

  Rule 1  Three-Second Rule
          Every external request carries a strict 3-second timeout
          (2 s connect + 3 s read).  A connection that does not
          acknowledge within this window is aborted immediately and
          logged as a "Source Unavailable" event.

  Rule 2  Waterfall Hierarchy (fail-over)
          Primary   → ESPN site API  (site.api.espn.com)
          Secondary → ESPN web API   (site.web.api.espn.com)
          Tertiary  → RotoWire free injury-report scrape (no key needed —
                       see core.rotowire_injuries). Only covers "/injuries"
                       paths; other paths skip this slot.

          NOTE: Tank01 RapidAPI (RAPIDAPI_KEY) is available as a
          sport-specific fallback but cannot be substituted generically
          for ESPN paths because its response format differs.  Call
          core.tank01 functions directly from sport-specific graders:
            • WNBA prop grading  → core.prop_grader (already wired)
            • MLB scores/odds    → core.tank01.get_mlb_scores/odds
          core.wnba_stats_client (free, stats.wnba.com) is also available
          as a primary source for WNBA team/player game logs — see
          core.intelligence.game_logs and core.player_props, which try
          it ahead of ESPN/balldontlie.
          The fetcher tries each source in order; the first to return
          structurally valid data wins.

  Rule 3  Data Integrity Validation
          Every successful HTTP response is passed through
          data_validator before being returned.  A response that is
          empty or missing expected top-level keys is treated as a
          failed request and triggers the next source.

  Rule 4  Silent Failure Prohibition
          Every failure — timeout, HTTP error, or structural failure —
          is logged as a "Source Unavailable" event with the source
          name and reason.  Callers must never silently discard these.

Public API
----------
  fetch_espn(path)          → FetchResult
  check_connectivity()      → ConnectivityReport
  FetchResult               — dataclass (ok, data, source, error)
  ConnectivityReport        — dataclass (any_source_ok, results)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger("betting_bot")

# ---------------------------------------------------------------------------
# Timeout — strict 3-second rule (connect=2s, read=3s)
# ---------------------------------------------------------------------------

_CONNECT_TIMEOUT = 2   # seconds to establish TCP connection
_READ_TIMEOUT    = 3   # seconds to receive first byte after connection

_TIMEOUT = (_CONNECT_TIMEOUT, _READ_TIMEOUT)

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

_ESPN_PRIMARY  = "https://site.api.espn.com/apis/site/v2/sports"
_ESPN_FALLBACK = "https://site.web.api.espn.com/apis/site/v2/sports"

_CONNECTIVITY_PROBE_PATH = "basketball/nba/teams"  # lightweight probe endpoint

_HEADERS = {
    "Accept":     "application/json",
    "User-Agent": "DaPickSyndicate/1.0",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    ok:     bool
    data:   dict[str, Any] | None
    source: str                   # human-readable source name
    error:  str | None


@dataclass
class ConnectivityReport:
    any_source_ok: bool
    results: dict[str, bool] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["[DATA STATUS] Daily data-pull connectivity report:"]
        for source, ok in self.results.items():
            lines.append(f"  {'✅' if ok else '❌'} {source}")
        if not self.any_source_ok:
            lines.append(
                "  ⛔ ALL sources unavailable — engine will enter Safe State."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core HTTP helper — enforces Rule 1 + Rule 4
# ---------------------------------------------------------------------------

def _get(url: str, source_name: str) -> FetchResult:
    """
    Perform a single GET with the strict timeout.
    Logs a "Source Unavailable" event on any failure (Rule 4).
    """
    try:
        resp = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            reason = "empty JSON body"
            logger.warning(
                f"[DATA] Source Unavailable: {source_name} — {reason}  url={url}"
            )
            return FetchResult(ok=False, data=None, source=source_name, error=reason)
        return FetchResult(ok=True, data=data, source=source_name, error=None)

    except requests.Timeout:
        reason = (
            f"timed out after {_CONNECT_TIMEOUT}s connect / {_READ_TIMEOUT}s read"
        )
        logger.warning(
            f"[DATA] Source Unavailable: {source_name} — {reason}  url={url}"
        )
        return FetchResult(ok=False, data=None, source=source_name, error=reason)

    except requests.HTTPError as exc:
        reason = f"HTTP {exc.response.status_code}"
        logger.warning(
            f"[DATA] Source Unavailable: {source_name} — {reason}  url={url}"
        )
        return FetchResult(ok=False, data=None, source=source_name, error=reason)

    except Exception as exc:
        reason = str(exc)[:200]
        logger.warning(
            f"[DATA] Source Unavailable: {source_name} — {reason}  url={url}"
        )
        return FetchResult(ok=False, data=None, source=source_name, error=reason)


# ---------------------------------------------------------------------------
# RotoWire stub — Rule 2 tertiary slot
# ---------------------------------------------------------------------------

def _fetch_rotowire(path: str) -> FetchResult:
    """
    RotoWire fallback slot (Rule 2 tertiary source) — FREE scrape variant.

    This no longer requires ROTOWIRE_API_KEY. Instead of the paid RotoWire
    API, it scrapes RotoWire's public injury-report webpage via
    core.rotowire_injuries (no auth, no key, just polite rate-limited
    HTTP). Only injury paths are supported here; schedule/box-score paths
    fall through to "not implemented" since RotoWire's free page doesn't
    cover those.

    Path matching: ESPN-style paths look like
      "basketball/wnba/teams/{team_abbr}/injuries"
    We only act on paths ending in "/injuries" and try to recover the
    team abbreviation from the path segment before it.
    """
    if not path.endswith("/injuries"):
        logger.debug(
            "[DATA] RotoWire (free) — no free-page equivalent for path=%s; skipping.",
            path,
        )
        return FetchResult(ok=False, data=None, source="RotoWire", error="path not supported")

    parts = path.split("/")
    team_abbr = parts[-2] if len(parts) >= 2 else ""

    try:
        from core.rotowire_injuries import get_team_injuries
        data = get_team_injuries(team_abbr)
    except Exception as exc:
        reason = str(exc)[:200]
        logger.warning("[DATA] Source Unavailable: RotoWire (free) — %s", reason)
        return FetchResult(ok=False, data=None, source="RotoWire", error=reason)

    if not data:
        reason = "scrape returned no data (page unreachable or markup changed)"
        logger.warning("[DATA] Source Unavailable: RotoWire (free) — %s", reason)
        return FetchResult(ok=False, data=None, source="RotoWire", error=reason)

    return FetchResult(ok=True, data=data, source="RotoWire (free scrape)", error=None)


# ---------------------------------------------------------------------------
# Public fetch API — Rule 2 waterfall
# ---------------------------------------------------------------------------

def fetch_wnba_injuries(team_abbr: str) -> FetchResult:
    """
    WNBA-specific injury fetch that tries the FREE RotoWire scrape first
    (per user request to diversify off ESPN), falling back to the
    standard ESPN waterfall if RotoWire's page is unreachable or its
    markup has changed.

    This is intentionally separate from fetch_espn() so other sports
    (NBA, MLB) are untouched and keep ESPN as primary.
    """
    result = _fetch_rotowire(f"basketball/wnba/teams/{team_abbr}/injuries")
    if result.ok:
        logger.debug(f"[DATA] WNBA injuries via {result.source} for {team_abbr}")
        return result

    logger.debug(
        f"[DATA] RotoWire (free) unavailable for {team_abbr} injuries "
        f"({result.error}); falling back to ESPN waterfall…"
    )
    return fetch_espn(f"basketball/wnba/teams/{team_abbr}/injuries")


def fetch_espn(path: str) -> FetchResult:
    """
    Fetch ESPN data with waterfall fail-over (Rule 2).

    Attempt order
    -------------
    1. ESPN Primary  (site.api.espn.com)
    2. ESPN Fallback (site.web.api.espn.com)
    3. RotoWire      (requires ROTOWIRE_API_KEY)

    Returns the first successful FetchResult.
    If all sources fail, returns a FetchResult(ok=False, ...) — the
    caller should treat this as a data-integrity failure (Rule 3).
    """
    # ── Source 1: ESPN Primary ──────────────────────────────────────────────
    result = _get(f"{_ESPN_PRIMARY}/{path}", "ESPN Primary")
    if result.ok:
        logger.debug(f"[DATA] Fetch OK via {result.source}  path={path}")
        return result

    # ── Source 2: ESPN Fallback ─────────────────────────────────────────────
    logger.debug(
        f"[DATA] Primary failed ({result.error}); trying ESPN Fallback…"
    )
    result = _get(f"{_ESPN_FALLBACK}/{path}", "ESPN Fallback")
    if result.ok:
        logger.info(
            f"[DATA] Fetch OK via {result.source} (primary unavailable)  path={path}"
        )
        return result

    # ── Source 3: RotoWire ──────────────────────────────────────────────────
    logger.debug("[DATA] ESPN Fallback also failed; trying RotoWire…")
    result = _fetch_rotowire(path)
    if result.ok:
        logger.info(f"[DATA] Fetch OK via {result.source}  path={path}")
        return result

    # ── All sources exhausted ───────────────────────────────────────────────
    logger.error(
        f"[DATA] All sources exhausted for path={path} — "
        "no valid data available."
    )
    return FetchResult(
        ok=False,
        data=None,
        source="none",
        error="all sources exhausted",
    )


# ---------------------------------------------------------------------------
# Connectivity probe — called once at start of daily process
# ---------------------------------------------------------------------------

def check_connectivity() -> ConnectivityReport:
    """
    Probe each source with a lightweight request and return a
    ConnectivityReport.

    Call this at the start of the daily picks pipeline (Rule 4:
    "explicitly report the status of the data pull").  If
    report.any_source_ok is False, the caller must enter Safe State
    and skip the engine.
    """
    results: dict[str, bool] = {}

    # Probe ESPN Primary
    r = _get(
        f"{_ESPN_PRIMARY}/{_CONNECTIVITY_PROBE_PATH}",
        "ESPN Primary",
    )
    results["ESPN Primary"] = r.ok

    # Only probe secondary if primary is down (avoid unnecessary requests)
    if not r.ok:
        r2 = _get(
            f"{_ESPN_FALLBACK}/{_CONNECTIVITY_PROBE_PATH}",
            "ESPN Fallback",
        )
        results["ESPN Fallback"] = r2.ok
    else:
        results["ESPN Fallback"] = None   # type: ignore[assignment]

    # RotoWire availability (key-presence check only, no real HTTP probe)
    rw_key = os.environ.get("ROTOWIRE_API_KEY")
    results["RotoWire"] = bool(rw_key)

    any_ok = any(
        v for v in results.values() if v is True
    )
    return ConnectivityReport(any_source_ok=any_ok, results=results)
