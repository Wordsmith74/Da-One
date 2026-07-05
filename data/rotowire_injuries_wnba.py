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

Parsing approach -- DOM traversal, not tag-specific regex
----------------------------------------------------------
See rotowire_injuries_mlb.py's docstring for the full rationale: this
module used to regex raw HTML assuming a specific `<b>` tag around the
position, which silently broke (0 rows, every team, every run) once
RotoWire changed that markup. It now parses the real DOM with
BeautifulSoup: find each player-profile link, climb ancestors to the
smallest one containing a full team name and not already containing a
second player link, and read position/headline/blurb from within that.

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

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

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


_PLAYER_HREF_RE = re.compile(r"/wnba/player/[a-z0-9\-]+", re.IGNORECASE)


def _parse_entries(html: str) -> list[dict]:
    """
    Parse the view=injuries feed into flat rows:
      {"name": ..., "position": ..., "team_abbr": ..., "status": ...}

    Walks the real DOM (via BeautifulSoup): for each player-profile link,
    climb ancestors to the smallest one that contains a full team name
    and doesn't already contain a second player link (i.e. hasn't
    absorbed a neighboring entry). Position comes from a bold-ish tag
    inside that container if present; its absence doesn't drop the row.
    Entries where no team can be matched are dropped (can't attribute
    them safely).
    """
    if BeautifulSoup is None:
        logger.warning("[rotowire-wnba] beautifulsoup4 not installed -- cannot parse injury feed")
        return []

    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []

    player_links = soup.find_all("a", href=_PLAYER_HREF_RE)

    for a in player_links:
        name = _clean(a.get_text(strip=True))
        if not name:
            continue

        container = None
        team_abbr = None
        node = a
        for _ in range(8):
            parent = node.parent
            if parent is None:
                break
            if len(parent.find_all("a", href=_PLAYER_HREF_RE)) > 1:
                break
            node = parent
            text = node.get_text(" ", strip=True)
            team_m = _TEAM_RE.search(text)
            if team_m:
                candidate = _TEAM_NAME_TO_ABBR.get(team_m.group(1).lower().strip())
                if candidate:
                    team_abbr = candidate
                    container = node
                    break

        if not team_abbr or container is None:
            continue

        position = "?"
        bold = container.find(["b", "strong"])
        if bold:
            bt = _clean(bold.get_text(strip=True)).upper()
            if re.fullmatch(r"[A-Z]{1,2}", bt):
                position = bt

        blurb = _clean(container.get_text(" ", strip=True))[:500]

        rows.append({
            "name": name,
            "position": position,
            "team_abbr": team_abbr,
            "status": _infer_status(blurb),
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
        # See rotowire_injuries_mlb.py's equivalent branch: this means
        # the feed parsed fine but had no rows tagged for this team --
        # routine, not a failure, so debug-level only.
        logger.debug(
            "[rotowire-wnba] feed parsed %d total row(s) but none tagged for %s -- "
            "no recent injury news for this team, or a team-name match miss",
            len(all_rows), team_abbr,
        )
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
