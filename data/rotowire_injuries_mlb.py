"""
rotowire_injuries_mlb.py

Free, keyless MLB injury source. Scrapes RotoWire's public injury *news*
feed (no login, no API key) as a replacement for ESPN's MLB injuries
endpoint, which is currently returning 500s for several teams
(baseball/mlb/teams/{TEAM}/injuries).

Why this URL and not "the RotoWire API" or the injury-report.php page
-----------------------------------------------------------------------
- RotoWire's official API (api.rotowire.com) requires a paid key. Not free.
- RotoWire's tabular "MLB Injury Report" page
  (rotowire.com/baseball/injury-report.php) renders its table client-side
  via JS after the page loads and gates it behind "Subscribe Now" --
  fetching the raw HTML returns a "Loading MLB Injury Report" placeholder,
  no usable rows. Confirmed by direct fetch of that URL.
- RotoWire's "MLB Injury News" feed
  (rotowire.com/baseball/news.php?injuries=all) IS free and keyless: the
  headline, player, team, position, and injury body-part tag are all
  present in the server-rendered HTML (only the deeper "Analysis" text is
  paywalled, which we don't need). Confirmed by direct fetch -- this is
  the URL this module uses.

Per-team URL params (e.g. "?team=ATL") on that endpoint route to a
different "team news" view (transactions included, injuries not
filtered), so instead of trying to build a per-team URL we fetch the
league-wide injuries=all feed once per process run and filter the parsed
rows by team abbreviation in memory -- same pattern as
data/rotowire_injuries.py (the WNBA version).

Status inference caveat
------------------------
Unlike a tabular injury report, this is a news feed: each entry is a
headline about an already-injured player (rehab update, IL move, day-to-day
call, etc.), not a standardized "Out/Doubtful/Questionable" tag. This
module infers a status from headline/body keywords (see
_infer_status()). When no clear signal is found, it defaults to the
lowest-severity bucket ("day-to-day") rather than guessing "out", so a
parsing miss under-states impact instead of over-stating it.

IMPORTANT -- selectors not live-tested against raw HTML
----------------------------------------------------------
This sandbox has no network egress, so the regexes below were written
against a readability-extracted rendering of the live page (fetched
externally), not the raw HTML source. The page structure (player link,
bolded position, team name, injury tag, date, blurb) is accurate as of
2026-07, but if get_recent_mlb_injuries() logs "parsed 0 rows", pull the
live HTML (curl/view-source) and adjust _ENTRY_RE / _clean_entry().

Output shape matches what core.data_fetcher.fetch_espn() normally returns
for "baseball/mlb/teams/{TEAM}/injuries", so lineup_intel.py needs ZERO
changes beyond picking this source for MLB:
  {"injuries": [{"athlete": {"shortName": ..., "position": {"abbreviation": ...}},
                 "status": "out" | "doubtful" | "questionable" | "day-to-day" | "probable"}],
   "_source": "RotoWire (free scrape)"}

Rate limiting
-------------
Public webpage, not a paid API -- be polite. Whole feed is cached
per-process; a blanket _MIN_GAP throttle applies across calls.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.request

logger = logging.getLogger("betting_bot")

_INJURY_NEWS_URL = "https://www.rotowire.com/baseball/news.php?injuries=all"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_TIMEOUT   = 8
_MIN_GAP   = 1.0
_LAST_CALL = 0.0

# Process-level cache -- one HTTP call serves every team for this run.
_PAGE_CACHE: str | None = None

# All 30 MLB team abbreviations as RotoWire's news feed displays them.
# Kept in sync with core.mlb._MLB_TEAM_IDS's key set (that module is the
# canonical abbr list elsewhere in this codebase).
_KNOWN_ABBRS = {
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "ATH",  # RotoWire uses "ATH" for the Athletics; alias handled below
    "PHI", "PIT", "SD", "SF", "SEA", "STL", "TB", "TEX", "TOR", "WSH",
}

# Map RotoWire's abbreviation quirks to the abbreviations the rest of the
# engine uses (core.mlb._MLB_TEAM_IDS has no "ATH" -- it's "OAK").
_ABBR_ALIAS = {"ATH": "OAK"}

# Headline/body keyword -> normalized status, checked in this priority
# order (most specific/severe first). Matched against lowercased text.
_STATUS_RULES: list[tuple[str, str]] = [
    ("season-ending", "out"),
    ("season ending", "out"),
    ("done for the year", "out"),
    ("undergo surgery", "out"),
    ("underwent surgery", "out"),
    ("ruled out", "out"),
    ("placed on the 10-day injured list", "out"),
    ("placed on the 15-day injured list", "out"),
    ("placed on the 60-day injured list", "out"),
    ("moved to the 60-day il", "out"),
    ("moved to 60-day il", "out"),
    ("injured list", "out"),
    (" il ", "out"),
    ("shut down", "out"),
    ("out with", "out"),
    ("out monday", "out"),
    ("out tuesday", "out"),
    ("out wednesday", "out"),
    ("out thursday", "out"),
    ("out friday", "out"),
    ("out saturday", "out"),
    ("out sunday", "out"),
    ("doubtful", "doubtful"),
    ("questionable", "questionable"),
    ("game-time decision", "questionable"),
    ("gtd", "questionable"),
    ("day-to-day", "day-to-day"),
    ("day to day", "day-to-day"),
    ("scratched", "day-to-day"),
    ("removed from", "day-to-day"),
]

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(fragment: str) -> str:
    return _TAG_RE.sub("", fragment).strip()


def _infer_status(headline: str, blurb: str) -> str:
    text = f"{headline} {blurb}".lower()
    for needle, status in _STATUS_RULES:
        if needle in text:
            return status
    # No clear signal -- conservative low-severity default (see module
    # docstring): every entry on this feed IS injury-related, so we don't
    # return "probable"/healthy, just the least severe active tier.
    return "day-to-day"


def _fetch_page() -> str | None:
    global _PAGE_CACHE, _LAST_CALL
    if _PAGE_CACHE is not None:
        return _PAGE_CACHE

    gap = time.monotonic() - _LAST_CALL
    if gap < _MIN_GAP:
        time.sleep(_MIN_GAP - gap)

    req = urllib.request.Request(_INJURY_NEWS_URL, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            _LAST_CALL = time.monotonic()
            html = resp.read().decode("utf-8", errors="replace")
            _PAGE_CACHE = html
            return html
    except urllib.error.HTTPError as exc:
        _LAST_CALL = time.monotonic()
        logger.warning("[rotowire-mlb] HTTP %s fetching injury news feed", exc.code)
        return None
    except Exception as exc:
        _LAST_CALL = time.monotonic()
        logger.debug("[rotowire-mlb] fetch failed: %s", exc)
        return None


# Each news entry (per the fetched page structure) is, in order:
#   player link -> headline link -> bolded position -> team name ->
#   (optional injury-location tag) -> date -> blurb paragraph
# This regex captures one entry block up to the next player link (or the
# "Get More News" footer) as a best-effort span, then sub-extracts fields.
_ENTRY_START_RE = re.compile(
    r'/baseball/player/[a-z0-9\-]+["\'][^>]*>([^<]+)</a>', re.IGNORECASE
)
_POSITION_RE = re.compile(r"<b>\s*([A-Z0-9]{1,3})\s*</b>", re.IGNORECASE)
_TEAM_RE = re.compile(
    r"(Arizona Diamondbacks|Atlanta Braves|Baltimore Orioles|Boston Red Sox|"
    r"Chicago Cubs|Chicago White Sox|Cincinnati Reds|Cleveland Guardians|"
    r"Colorado Rockies|Detroit Tigers|Houston Astros|Kansas City Royals|"
    r"Los Angeles Angels|Los Angeles Dodgers|Miami Marlins|Milwaukee Brewers|"
    r"Minnesota Twins|New York Mets|New York Yankees|Oakland Athletics|"
    r"Athletics|Philadelphia Phillies|Pittsburgh Pirates|San Diego Padres|"
    r"San Francisco Giants|Seattle Mariners|St\.?\s*Louis Cardinals|"
    r"Tampa Bay Rays|Texas Rangers|Toronto Blue Jays|Washington Nationals)"
)

_TEAM_NAME_TO_ABBR = {
    "arizona diamondbacks": "ARI", "atlanta braves": "ATL", "baltimore orioles": "BAL",
    "boston red sox": "BOS", "chicago cubs": "CHC", "chicago white sox": "CWS",
    "cincinnati reds": "CIN", "cleveland guardians": "CLE", "colorado rockies": "COL",
    "detroit tigers": "DET", "houston astros": "HOU", "kansas city royals": "KC",
    "los angeles angels": "LAA", "los angeles dodgers": "LAD", "miami marlins": "MIA",
    "milwaukee brewers": "MIL", "minnesota twins": "MIN", "new york mets": "NYM",
    "new york yankees": "NYY", "oakland athletics": "OAK", "athletics": "OAK",
    "philadelphia phillies": "PHI", "pittsburgh pirates": "PIT", "san diego padres": "SD",
    "san francisco giants": "SF", "seattle mariners": "SEA", "st. louis cardinals": "STL",
    "st louis cardinals": "STL", "tampa bay rays": "TB", "texas rangers": "TEX",
    "toronto blue jays": "TOR", "washington nationals": "WSH",
}


def _parse_entries(html: str) -> list[dict]:
    """
    Parse the injuries=all feed into flat rows:
      {"name": ..., "position": ..., "team_abbr": ..., "status": ...}

    Best-effort: splits the page on player-profile links, then looks
    within each resulting chunk for the position, team name, and a status
    inferred from the surrounding headline/blurb text. Chunks where no
    team can be matched are dropped (can't attribute them safely).
    """
    names = list(_ENTRY_START_RE.finditer(html))
    rows: list[dict] = []

    for i, m in enumerate(names):
        start = m.end()
        end = names[i + 1].start() if i + 1 < len(names) else min(len(html), start + 4000)
        chunk = html[start:end]
        name = _clean(m.group(1))
        if not name:
            continue

        pos_m = _POSITION_RE.search(chunk)
        position = pos_m.group(1).upper() if pos_m else "?"

        team_m = _TEAM_RE.search(chunk)
        if not team_m:
            continue  # can't safely attribute this row to a team
        team_abbr = _TEAM_NAME_TO_ABBR.get(team_m.group(1).lower().strip(), None)
        if not team_abbr:
            continue

        headline = ""
        headline_m = re.search(r'headlines/[a-z0-9\-]+["\'][^>]*>([^<]+)</a>', chunk, re.IGNORECASE)
        if headline_m:
            headline = _clean(headline_m.group(1))

        blurb = _clean(chunk)[:400]  # rough text blob for keyword scanning

        rows.append({
            "name": name,
            "position": position,
            "team_abbr": team_abbr,
            "status": _infer_status(headline, blurb),
        })

    return rows


def get_recent_mlb_injuries(team_abbr: str) -> dict | None:
    """
    Return recent injury-news-derived status for *team_abbr* in the same
    shape core.data_fetcher.FetchResult.data normally carries from ESPN.

    Returns None on any fetch/parse failure, or if zero rows matched this
    team -- callers treat this as a failed source and fall back to ESPN.
    """
    team_abbr = _ABBR_ALIAS.get(team_abbr.upper(), team_abbr.upper())

    html = _fetch_page()
    if not html:
        return None

    all_rows = _parse_entries(html)
    if not all_rows:
        logger.debug("[rotowire-mlb] parsed 0 injury-news rows -- markup may have changed")
        return None

    # One news item per player per event; keep only the most recent entry
    # per player (first occurrence, since the feed is newest-first).
    seen: set[str] = set()
    team_rows = []
    for r in all_rows:
        if r["team_abbr"] != team_abbr:
            continue
        key = r["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        team_rows.append(r)

    if not team_rows:
        # Genuinely could mean "no recent injury news for this team" --
        # but we can't distinguish that from "parsing missed them", so
        # fail this source and let the ESPN fallback have a shot too.
        return None

    injuries = [
        {
            "athlete": {
                "shortName": r["name"],
                "position": {"abbreviation": r["position"]},
            },
            "status": r["status"],
        }
        for r in team_rows
    ]
    return {"injuries": injuries, "_source": "RotoWire (free scrape)"}
