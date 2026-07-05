"""
data_fetcher.py

Thin re-export of data.fetch for callers under core/ -- previously this was
a byte-for-byte hand copy of data/fetch.py, which meant every fix had to be
applied twice and silently drifted (see git history). data.fetch is the one
real implementation; this module just re-exports it and adds the helpers
core.intelligence.* actually needs that data.fetch never defined:
fetch_espn(), fetch_wnba_injuries(), and fetch_mlb_injuries().

NOTE: fetch_espn()/fetch_wnba_injuries() used to return a bare dict/None,
but every caller in core.intelligence checks `result.ok` / `result.data` /
`result.error`. That mismatch meant those checks always raised
AttributeError, silently caught by each caller's own try/except -- i.e.
lineup_intel/rest_travel/stat_model/wnba_opp_intel were failing closed on
every call, independent of whether ESPN actually responded. Fixed by
introducing FetchResult below and having all fetch_*() functions return it.

Rule 1 -- strict timeout: fetch_espn() uses a hard 3-second timeout per
    attempt so a slow ESPN response never stalls the pipeline.
Rule 2 -- waterfall: ESPN primary -> ESPN retry (covers transient
    blips/rate limiting) -> for injury paths only, RotoWire via
    data.fetch.get_wnba_team_injuries() (which itself tries RotoWire first,
    ESPN second -- see that function's docstring).
Rule 3 -- structure validation: NOT done here. core.data_validator is the
    Rule-3 layer and runs on whatever this returns; fetch_espn() only
    fetches and returns raw JSON (or None).
Rule 4 -- every source failure is logged as a "Source Unavailable" event.

Fail-safe: fetch_espn() and fetch_wnba_injuries() return None on total
failure rather than raising -- callers (lineup_intel, rest_travel,
stat_model, wnba_opp_intel) are all written to treat None as "skip this
factor, contribute zero adjustment."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

from data.fetch import *  # noqa: F401,F403 -- re-export the real implementation
from data.fetch import get_wnba_team_injuries  # explicit, used below
from data.fetch import get_mlb_team_injuries    # explicit, used below

logger = logging.getLogger("betting_bot")

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
_ESPN_TIMEOUT = 3.0  # seconds -- Rule 1


@dataclass
class FetchResult:
    """
    Uniform return type for every fetch_*() helper in this module.

    core.intelligence.lineup_intel / rest_travel / stat_model /
    wnba_opp_intel all check `result.ok` and read `result.data` /
    `result.error` -- previously fetch_espn() and fetch_wnba_injuries()
    returned a bare dict/None instead, so every one of those `.ok` checks
    raised AttributeError (silently swallowed by each caller's own
    try/except), meaning EVERY injury/schedule/stat lookup was failing
    closed regardless of whether ESPN actually responded. This dataclass
    is the missing contract those callers were already written against.
    """
    ok:    bool
    data:  dict[str, Any] | None = None
    error: str | None = None


def fetch_espn(path: str, timeout: float = _ESPN_TIMEOUT) -> FetchResult:
    """
    Fetch a path under ESPN's public "site" API, e.g.
    "basketball/wnba/teams/LV/injuries" or
    "baseball/mlb/teams/LAD/schedule".

    Tries the request once, and on any failure (timeout, connection error,
    non-2xx) retries it once more before giving up -- this absorbs the kind
    of transient blip/rate-limit hiccup ESPN's public API is known to throw,
    without ever blocking the pipeline for more than ~2x the timeout.

    Returns FetchResult(ok=True, data=<parsed JSON>) on success, or
    FetchResult(ok=False, error=<str>) if both attempts fail (logged as a
    Source Unavailable event -- Rule 4).
    """
    url = f"{_ESPN_BASE}/{path.lstrip('/')}"

    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return FetchResult(ok=True, data=r.json())
        except Exception as exc:
            last_exc = exc
            if attempt == 1:
                logger.debug(f"[data_fetcher] ESPN attempt 1 failed for {path}: {exc} -- retrying once")
                continue

    logger.warning(f"[data_fetcher] Source Unavailable: ESPN ({path}) -- {last_exc}")
    return FetchResult(ok=False, error=str(last_exc))


def fetch_wnba_injuries(team_abbr: str, roster_names: list[str] | None = None) -> FetchResult:
    """
    WNBA injury fetch with the RotoWire -> ESPN waterfall already
    implemented in data.fetch.get_wnba_team_injuries(): RotoWire (free,
    no API key) first, ESPN second.
    """
    try:
        data = get_wnba_team_injuries(team_abbr, roster_names=roster_names)
    except Exception as exc:
        logger.warning(f"[data_fetcher] Source Unavailable: WNBA injuries ({team_abbr}) -- {exc}")
        return FetchResult(ok=False, error=str(exc))

    if data is None:
        err = "no source returned data"
        logger.warning(f"[data_fetcher] Source Unavailable: WNBA injuries ({team_abbr}) -- {err}")
        return FetchResult(ok=False, error=err)
    return FetchResult(ok=True, data=data)


def fetch_mlb_injuries(team_abbr: str) -> FetchResult:
    """
    MLB injury fetch with the RotoWire -> ESPN waterfall implemented in
    data.fetch.get_mlb_team_injuries(): RotoWire's free injury-news scrape
    first (data/rotowire_injuries_mlb.py -- no API key), ESPN second.

    ESPN's own MLB injuries endpoint has been intermittently 500ing (see
    run logs: "Source Unavailable: ESPN (baseball/mlb/teams/*/injuries)
    -- 500 Server Error"), which is the immediate reason this exists --
    but RotoWire is tried first regardless of whether ESPN is currently
    healthy, since it's the free source you asked to prefer.
    """
    try:
        data = get_mlb_team_injuries(team_abbr)
    except Exception as exc:
        logger.warning(f"[data_fetcher] Source Unavailable: MLB injuries ({team_abbr}) -- {exc}")
        return FetchResult(ok=False, error=str(exc))

    if data is None:
        err = "no source returned data"
        logger.warning(f"[data_fetcher] Source Unavailable: MLB injuries ({team_abbr}) -- {err}")
        return FetchResult(ok=False, error=err)
    return FetchResult(ok=True, data=data)
