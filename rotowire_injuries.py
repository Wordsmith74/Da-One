"""
rotowire_injuries.py

Free, keyless WNBA injury source for sports-engine -- ported from
Wordsmith74's core/rotowire_injuries.py. Scrapes RotoWire's public
injury-report page (no login, no API key) instead of relying on ESPN.

Used by data/fetch.py's get_wnba_team_injuries() as the primary source,
with ESPN (get_espn_team_injuries) as fallback if this fails.

IMPORTANT -- not live-tested
-----------------------------
No network access in this sandbox. The row/cell regex below is best-effort
against RotoWire's known table structure as of 2026-06. If it returns 0
rows in production, that's the first thing to check -- _parse_injury_table()
is the only function that needs updating if their markup changed.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.request

logger = logging.getLogger("sports_engine")

_INJURY_URL = "https://www.rotowire.com/basketball/wnba-injury-report.php"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_TIMEOUT = 8
_MIN_GAP = 1.0
_LAST_CALL = 0.0

_PAGE_CACHE: str | None = None

_TEAM_NAME_TO_ABBR: dict[str, str] = {
    "atlanta dream": "ATL", "chicago sky": "CHI", "connecticut sun": "CON",
    "dallas wings": "DAL", "indiana fever": "IND", "las vegas aces": "LVA",
    "los angeles sparks": "LAS", "minnesota lynx": "MIN",
    "new york liberty": "NYL", "phoenix mercury": "PHX",
    "seattle storm": "SEA", "washington mystics": "WAS",
    "golden state valkyries": "GSV", "portland fire": "POR",
    "toronto tempo": "TOR",
}

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
_TAG_RE = re.compile(r"<[^>]+>")


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
    rows: list[dict] = []
    for match in _ROW_RE.finditer(html):
        row_html = match.group(1)
        cells = [_clean(c) for c in _CELL_RE.findall(row_html)]
        if len(cells) < 3:
            continue
        name, position, status = cells[0], cells[1], cells[2]
        norm_status = _STATUS_NORMALIZE.get(status.strip().lower(), status.strip().lower())
        rows.append({"name": name, "position": position, "status": norm_status})
    return rows


def get_team_injuries(team_abbr: str, roster_names: set[str] | None = None) -> dict | None:
    """
    Returns {"injuries": [{"athlete": {...}, "status": ...}], "_source": ...}
    in the same shape ESPN's injuries endpoint returns, so callers (and
    models/injury_intel.py) need no special-casing.

    roster_names: optional set of lowercased player names for *team_abbr*
    (e.g. from get_wnba_player_game_logs / a roster call) used to filter
    the league-wide scraped list down to this team. Without it, this
    returns None rather than risk mixing in another team's injuries.
    """
    html = _fetch_page()
    if not html:
        return None

    all_rows = _parse_injury_table(html)
    if not all_rows:
        logger.debug("[rotowire] parsed 0 injury rows -- markup may have changed")
        return None

    if roster_names:
        filtered = [r for r in all_rows if r["name"].strip().lower() in roster_names]
        if not filtered:
            logger.debug("[rotowire] no name overlap for %s -- treating as unavailable", team_abbr)
            return None
        all_rows = filtered
    else:
        logger.debug("[rotowire] no roster_names provided -- cannot safely filter by team")
        return None

    injuries = [
        {
            "athlete": {"shortName": r["name"], "position": {"abbreviation": r["position"]}},
            "status": r["status"],
        }
        for r in all_rows
    ]
    return {"injuries": injuries, "_source": "RotoWire (free scrape)"}
