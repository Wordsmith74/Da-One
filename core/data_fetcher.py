"""
data_fetcher.py

Thin re-export of data.fetch for callers under core/ -- previously this was
a byte-for-byte hand copy of data/fetch.py, which meant every fix had to be
applied twice and silently drifted (see git history). data.fetch is the one
real implementation; this module just re-exports it and adds the two
helpers core.intelligence.* actually needs that data.fetch never defined:
fetch_espn() and fetch_wnba_injuries().

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

import requests

from data.fetch import *  # noqa: F401,F403 -- re-export the real implementation
from data.fetch import get_wnba_team_injuries  # explicit, used below

logger = logging.getLogger("betting_bot")

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
_ESPN_TIMEOUT = 3.0  # seconds -- Rule 1


def fetch_espn(path: str, timeout: float = _ESPN_TIMEOUT) -> dict | None:
    """
    Fetch a path under ESPN's public "site" API, e.g.
    "basketball/wnba/teams/LV/injuries" or
    "baseball/mlb/teams/LAD/schedule".

    Tries the request once, and on any failure (timeout, connection error,
    non-2xx) retries it once more before giving up -- this absorbs the kind
    of transient blip/rate-limit hiccup ESPN's public API is known to throw,
    without ever blocking the pipeline for more than ~2x the timeout.

    Returns the parsed JSON dict on success, or None if both attempts fail
    (logged as a Source Unavailable event -- Rule 4).
    """
    url = f"{_ESPN_BASE}/{path.lstrip('/')}"

    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_exc = exc
            if attempt == 1:
                logger.debug(f"[data_fetcher] ESPN attempt 1 failed for {path}: {exc} -- retrying once")
                continue

    logger.warning(f"[data_fetcher] Source Unavailable: ESPN ({path}) -- {last_exc}")
    return None


def fetch_wnba_injuries(team_abbr: str, roster_names: list[str] | None = None) -> dict | None:
    """
    WNBA injury fetch with the RotoWire -> ESPN waterfall already
    implemented in data.fetch.get_wnba_team_injuries(): RotoWire (free,
    no API key) first, ESPN second.

    This wrapper just adds the same Rule 4 "Source Unavailable" logging
    convention as fetch_espn() so callers in core.intelligence get
    consistent log lines regardless of which helper they used.
    """
    try:
        data = get_wnba_team_injuries(team_abbr, roster_names=roster_names)
    except Exception as exc:
        data = None
        logger.warning(f"[data_fetcher] Source Unavailable: WNBA injuries ({team_abbr}) -- {exc}")
        return None

    if data is None:
        logger.warning(f"[data_fetcher] Source Unavailable: WNBA injuries ({team_abbr}) -- no source returned data")
    return data
