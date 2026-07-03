"""
rotowire_injuries.py

Free, keyless WNBA injury source. Scrapes RotoWire's public injury report
page (no login, no API key) — distinct from the paid RotoWire API that
core.data_fetcher's old stub pointed at.

Why scraping instead of "the RotoWire API"
-------------------------------------------
RotoWire's official API requires a paid ROTOWIRE_API_KEY. Their injury
*report page*, however, is publicly served HTML meant for human readers
and carries no auth wall. This module fetches that page and parses it.

No official injury-status API exists for WNBA (the league doesn't publish
one) — every source (ESPN, RotoWire, CBS, etc.) is an editorial product
built by reporters. This is simply a free one instead of ESPN's.

Output shape matches what core.intelligence.lineup_intel expects from
data_fetcher.fetch_espn(): a dict with an "injuries" list of
  {"athlete": {"shortName": ..., "position": {"abbreviation": ...}},
   "status": "Out" | "Doubtful" | "Questionable" | "Day-To-Day"}
so lineup_intel.py needs ZERO changes — it just receives data from a
different upstream source through the same FetchResult contract.

IMPORTANT — could not be live-tested
-------------------------------------
This sandbox has no network egress. The CSS selectors below are based on
RotoWire's known injury-report page structure as of 2026-06 but WILL
need a quick sanity check against the live HTML (e.g. via `curl` or
viewing the page source) before trusting this in production. If RotoWire
changes their markup, _parse_injury_table() is the only function that
needs updating.

Rate limiting
-------------
This is someone's public webpage, not a paid API — be polite. One fetch
per team per process run is cached; a blanket _MIN_GAP throttle is
enforced across all calls.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.request

logger = logging.getLogger("betting_bot")

_INJURY_URL = "https://www.rotowire.com/basketball/wnba-injury-report.php"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_TIMEOUT  = 8
_MIN_GAP  = 1.0
_LAST_CALL = 0.0

# Process-level cache — whole page is fetched once per run, then filtered
# per-team in memory (one HTTP call serves every team).
_PAGE_CACHE: str | None = None

# Team full-name fragments RotoWire uses in their injury table, mapped to
# the abbreviations the rest of the engine uses.
_TEAM_NAME_TO_ABBR: dict[str, str] = {
    "atlanta dream": "ATL", "chicago sky": "CHI", "connecticut sun": "CON",
    "dallas wings": "DAL", "indiana fever": "IND", "las vegas aces": "LVA",
    "los angeles sparks": "LAS", "minnesota lynx": "MIN",
    "new york liberty": "NYL", "phoenix mercury": "PHX",
    "seattle storm": "SEA", "washington mystics": "WAS",
    "golden state valkyries": "GSV", "portland fire": "POR",
    "toronto tempo": "TOR",
}

# Normalize whatever status string RotoWire uses to the vocabulary
# lineup_intel._STATUS_SEVERITY already understands.
_STATUS_NORMALIZE: dict[str, str] = {
    "out": "out", "ir": "injured reserve", "injured reserve": "injured reserve",
    "suspended": "suspended", "doubtful": "doubtful",
    "questionable": "questionable", "gtd": "questionable",
    "game-time decision": "questionable",
    "day-to-day": "day-to-day", "day to day": "day-to-day",
    "probable": "probable",
}

_ROW_RE = re.compile(
    r'<tr[^>]*class="[^"]*injury-report[^"]*"[^>]*>(.*?)</tr>',
    re.IGNORECASE | re.DOTALL,
)
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
_TAG_RE  = re.compile(r"<[^>]+>")


def _clean(html_fragment: str) -> str:
    return _TAG_RE.sub("", html_fragment).strip()


def _fetch_page() -> str | None:
    global _PAGE_CACHE, _LAST_CALL
    if _PAGE_CACHE is not None:
        return _PAGE_CACHE

    gap = time.monotonic() - _LAST_CALL
    if gap < _MIN_GAP:
        time.sleep(_MIN_GAP - gap)

    req = urllib.request.Request(_INJURY_URL, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            _LAST_CALL = time.monotonic()
            html = resp.read().decode("utf-8", errors="replace")
            _PAGE_CACHE = html
            return html
    except urllib.error.HTTPError as exc:
        _LAST_CALL = time.monotonic()
        logger.warning("[rotowire] HTTP %s fetching injury page", exc.code)
        return None
    except Exception as exc:
        _LAST_CALL = time.monotonic()
        logger.debug("[rotowire] fetch failed: %s", exc)
        return None


def _parse_injury_table(html: str) -> list[dict]:
    """
    Parse RotoWire's injury-report rows into a flat list of:
      {"team_abbr": ..., "name": ..., "position": ..., "status": ...}

    NOTE: the regex-based row/cell extraction here is a best-effort parse
    of RotoWire's known table structure. If their markup changes, or if
    this comes back empty in production, switch to BeautifulSoup against
    a saved copy of the live page and adjust _ROW_RE / _CELL_RE / the
    column order assumed in this function.
    """
    rows: list[dict] = []
    current_team: str | None = None

    # RotoWire groups rows under team headers; do a simple linear scan
    # rather than assuming a fixed DOM depth.
    for team_name, abbr in _TEAM_NAME_TO_ABBR.items():
        if team_name in html.lower():
            current_team = abbr  # last-seen header before a row block; refined below

    for match in _ROW_RE.finditer(html):
        row_html = match.group(1)
        cells = [_clean(c) for c in _CELL_RE.findall(row_html)]
        if len(cells) < 3:
            continue
        # Expected column order: Player | Position | Status | (Injury | Est. Return)
        name, position, status = cells[0], cells[1], cells[2]
        norm_status = _STATUS_NORMALIZE.get(status.strip().lower(), status.strip().lower())
        rows.append({
            "name": name,
            "position": position,
            "status": norm_status,
        })

    return rows


def get_team_injuries(team_abbr: str) -> dict | None:
    """
    Return injury data for *team_abbr* in the same shape
    core.data_fetcher.FetchResult.data normally carries from ESPN, i.e.:
      {"injuries": [{"athlete": {"shortName": ..., "position": {...}},
                     "status": ...}, ...]}

    Returns None on any fetch/parse failure — caller treats this as a
    failed source and moves to the next waterfall step.
    """
    html = _fetch_page()
    if not html:
        return None

    all_rows = _parse_injury_table(html)
    if not all_rows:
        logger.debug("[rotowire] parsed 0 injury rows — markup may have changed")
        return None

    # Filter the league-wide row list down to this team using the free
    # stats.wnba.com roster (core.wnba_stats_client) as a name lookup,
    # since the scraped rows don't reliably carry a team tag (see NOTE
    # in _parse_injury_table).
    team_filtered = True
    roster_names: set[str] = set()
    try:
        from core.wnba_stats_client import get_team_roster
        roster = get_team_roster(team_abbr)
        roster_names = {
            (r.get("PLAYER") or "").strip().lower()
            for r in roster if r.get("PLAYER")
        }
    except Exception as exc:
        logger.debug("[rotowire] roster lookup for team filter failed: %s", exc)

    if roster_names:
        filtered_rows = [r for r in all_rows if r["name"].strip().lower() in roster_names]
        if filtered_rows:
            all_rows = filtered_rows
        else:
            # Roster lookup worked but no overlap found — likely a name-
            # format mismatch (e.g. "A. Wilson" vs "Aliyah Wilson") rather
            # than "this team really has zero injuries". Don't silently
            # return the WHOLE LEAGUE as this team's injuries — that would
            # corrupt the edge calc. Be honest that filtering failed.
            logger.debug(
                "[rotowire] roster fetched but no name overlap for %s — "
                "returning unfiltered as last resort.", team_abbr,
            )
            team_filtered = False
    else:
        team_filtered = False

    injuries = [
        {
            "athlete": {
                "shortName": r["name"],
                "position": {"abbreviation": r["position"]},
            },
            "status": r["status"],
        }
        for r in all_rows
    ]
    if not team_filtered:
        # We cannot trust an unfiltered league-wide list as this team's
        # injury report — fail this source rather than feed bad data
        # into the edge calculation (Rule 3 spirit: validate before use).
        return None

    return {"injuries": injuries, "_source": "RotoWire (free scrape)", "_team_filtered": True}
