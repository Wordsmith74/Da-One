"""
rotowire_injuries_wnba.py

Free, keyless WNBA injury source -- replaces data/rotowire_injuries.py.

Why this file exists / what was wrong with rotowire_injuries.py
-----------------------------------------------------------------
The old module scraped RotoWire's tabular "WNBA Injury Report" page
(rotowire.com/wnba/injury-report.php) and filtered the league-wide table
down to one team by cross-referencing a team roster via
`core.wnba_stats_client.get_team_roster()`. That module does not exist
anywhere in this codebase -- the import fails, is swallowed by a broad
except, and get_team_injuries() then refuses to return an unfiltered
league-wide list (correctly, to avoid corrupting the edge calc), so it
always returned None. Net effect: WNBA injury data has been silently
empty regardless of ESPN's status.

On top of that, injury-report.php itself renders its table client-side
behind a "Subscribe Now" gate -- fetching the raw HTML just returns a
"Loading WNBA Injury Report" placeholder, no usable rows even with a
working roster filter.

This module instead scrapes RotoWire's free injury *news* feed:
    https://www.rotowire.com/wnba/news.php?view=injuries
Confirmed by direct fetch: server-rendered, free, no login wall, and each
entry already carries its own team name (logo alt text + full team name
in the body), so -- unlike the old module -- no roster lookup is needed
at all to attribute a row to a team. This is the same pattern used for
MLB in data/rotowire_injuries_mlb.py; see that module's docstring for the
general rationale (paid API vs. gated table vs. free news feed).

Status inference
-----------------
WNBA's news feed uses clearer status language than MLB's did ("has been
ruled out for Monday's game", "is probable for", "is listed as
questionable") since these are almost always tied to the league's
official pregame injury report. _infer_status() keyword-matches on that
language. Unclear entries default to "day-to-day" (lowest severity)
rather than guessing "out" -- an under-statement is safer than an
over-statement for the edge calculation.

IMPORTANT -- selectors not live-tested against raw HTML
----------------------------------------------------------
Same caveat as rotowire_injuries_mlb.py: written against a readability-
extracted rendering of the live page, not raw HTML source, since this
sandbox has no network egress. If get_recent_wnba_injuries() logs "parsed
0 rows", pull the live HTML and adjust _ENTRY_START_RE / _parse_entries().

Output shape matches what core.data_fetcher.fetch_espn() normally returns
for "basketball/wnba/teams/{TEAM}/injuries":
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

_INJURY_NEWS_URL = "https://www.rotowire.com/wnba/news.php?view=injuries"

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

_TEAM_NAME_TO_ABBR: dict[str, str] = {
    "atlanta dream": "ATL", "chicago sky": "CHI", "connecticut sun": "CON",
    "dallas wings": "DAL", "indiana fever": "IND", "las vegas aces": "LVA",
    "los angeles sparks": "LAS", "minnesota lynx": "MIN",
    "new york liberty": "NYL", "phoenix mercury": "PHX",
    "seattle storm": "SEA", "washington mystics": "WAS",
    "golden state valkyries": "GSV", "portland fire": "POR",
    "toronto tempo": "TOR",
}

_TEAM_RE = re.compile(
    r"(Atlanta Dream|Chicago Sky|Connecticut Sun|Dallas Wings|Indiana Fever|"
    r"Las Vegas Aces|Los Angeles Sparks|Minnesota Lynx|New York Liberty|"
    r"Phoenix Mercury|Seattle Storm|Washington Mystics|Golden State Valkyries|"
    r"Portland Fire|Toronto Tempo)"
)

# Headline/body keyword -> normalized status, checked in this priority
# order (most specific/severe first). Matched against lowercased text.
# WNBA's feed leans on official pregame-report language, so these are
# more literal than the MLB module's rules.
_STATUS_RULES: list[tuple[str, str]] = [
    ("season-ending", "out"),
    ("out for the season", "out"),
    ("ruled out", "out"),
    ("is out for", "out"),
    ("is out", "out"),
    ("won't play", "out"),
    ("won't return", "out"),
    ("won't suit up", "out"),
    ("will not return", "out"),
    ("not playing", "out"),
    ("doubtful", "doubtful"),
    ("questionable", "questionable"),
    ("iffy for", "questionable"),
    ("game-time decision", "questionable"),
    ("probable", "probable"),
    ("likely to play", "probable"),
    ("day-to-day", "day-to-day"),
    ("day to day", "day-to-day"),
]

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(fragment: str) -> str:
    return _TAG_RE.sub("", fragment).strip()


def _infer_status(text: str) -> str:
    lowered = text.lower()
    for needle, status in _STATUS_RULES:
        if needle in lowered:
            return status
    # No clear signal -- conservative low-severity default (see module
    # docstring): every entry on this feed IS injury-related, so we don't
    # return "healthy", just the least severe active tier.
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
        logger.warning("[rotowire-wnba] HTTP %s fetching injury news feed", exc.code)
        return None
    except Exception as exc:
        _LAST_CALL = time.monotonic()
        logger.debug("[rotowire-wnba] fetch failed: %s", exc)
        return None


_ENTRY_START_RE = re.compile(
    r'/wnba/player/[a-z0-9\-]+["\'][^>]*>([^<]+)</a>', re.IGNORECASE
)
_POSITION_RE = re.compile(r"<b>\s*([A-Z]{1,2})\s*</b>", re.IGNORECASE)


def _parse_entries(html: str) -> list[dict]:
    """
    Parse the view=injuries feed into flat rows:
      {"name": ..., "position": ..., "team_abbr": ..., "status": ...}

    Best-effort: splits the page on player-profile links, then looks
    within each resulting chunk for position, team, and inferred status.
    Chunks where no team can be matched are dropped (can't attribute
    them safely).
    """
    names = list(_ENTRY_START_RE.finditer(html))
    rows: list[dict] = []

    for i, m in enumerate(names):
        start = m.end()
        end = names[i + 1].start() if i + 1 < len(names) else min(len(html), start + 3000)
        chunk = html[start:end]
        name = _clean(m.group(1))
        if not name:
            continue

        pos_m = _POSITION_RE.search(chunk)
        position = pos_m.group(1).upper() if pos_m else "?"

        team_m = _TEAM_RE.search(chunk)
        if not team_m:
            continue  # can't safely attribute this row to a team
        team_abbr = _TEAM_NAME_TO_ABBR.get(team_m.group(1).lower().strip())
        if not team_abbr:
            continue

        rows.append({
            "name": name,
            "position": position,
            "team_abbr": team_abbr,
            "status": _infer_status(_clean(chunk)[:500]),
        })

    return rows


def get_recent_wnba_injuries(team_abbr: str) -> dict | None:
    """
    Return recent injury-news-derived status for *team_abbr* in the same
    shape core.data_fetcher.FetchResult.data normally carries from ESPN.

    Returns None on any fetch/parse failure, or if zero rows matched this
    team -- callers treat this as a failed source and fall back to ESPN.
    """
    team_abbr = team_abbr.upper()

    html = _fetch_page()
    if not html:
        return None

    all_rows = _parse_entries(html)
    if not all_rows:
        logger.debug("[rotowire-wnba] parsed 0 injury-news rows -- markup may have changed")
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
