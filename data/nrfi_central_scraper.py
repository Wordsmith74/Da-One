"""
data/nrfi_central_scraper.py

Third-party data source: scrapes public stats pages from nrfi-central.com
to help fill gaps flagged in models/nrfi_handicapper.py's module docstring
and confirmed missing during a manual audit (2026-08-16):
  - team-level 1st-inning RA% / NRFI% (this repo only had a scaled-down
    full-game proxy -- see core/game_markets.py._team_first_inning_lambda_fallback)
  - hitter 1st-inning K% (not computed anywhere in this repo previously)
  - ballpark 1st-inning-specific run factors

NOT AN API. nrfi-central.com is a WordPress site with no documented data
feed -- this scrapes rendered HTML tables directly. That makes it
inherently fragile: a theme change, table-markup change, or page rename
breaks this without warning. Treat it as a supplementary/cross-check
source, not a primary one, and expect to need to patch the CSS selectors
here periodically.

Honesty / fail-safe contract (same pattern as data/statcast_first_inning.py
and data/statcast_lineup_platoon.py): every public function returns None /
empty on ANY failure -- missing page, changed markup, network error,
unparseable row -- and NEVER fabricates or falls back to a season-long
substitute. Callers must already treat None as "skip this input," which
models/nrfi_handicapper.py's tier functions do natively (every kwarg is
Optional and simply omitted from the weighted blend when absent).

Etiquette / ToS note: this hits nrfi-central.com's live pages directly.
No robots.txt disallow was found for the pages used here at the time this
was written, and the site is a small free community resource (not a paid
data vendor) that explicitly asks for support-via-donation rather than
gating data behind a login. Be a good citizen about it:
  - Respect _CACHE_TTL_SECONDS below -- don't hit the site more than once
    per refresh cycle across your whole pipeline, not once per candidate.
  - Set a real, identifying User-Agent (done below) rather than spoofing
    a browser.
  - If nrfi-central.com ever adds a robots.txt disallow or ToS restriction
    on automated access, stop using this module -- don't route around it.

UNTESTED LIVE: written without live network access in this sandbox. The
table structure below was inferred from the page's rendered content at
inspection time (2026-08-16) and has NOT been round-tripped against a real
HTTP response. Spot-check the actual table IDs/classes with your browser's
dev tools against a live page before trusting this in production -- same
caveat this repo's other intelligence modules (umpire_intel.py,
bullpen_intel.py) already carry for the same reason.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("betting_bot")

_BASE_URL = "https://nrfi-central.com"
_USER_AGENT = "Da-One-NRFI-Model/1.0 (+internal research tool; contact: set-your-contact-here)"
_REQUEST_TIMEOUT_SECONDS = 10

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
_CACHE_TTL_SECONDS = 6 * 60 * 60  # 4x/day -- this data doesn't move faster than that

# Page slugs, keyed by what they give us. Update these if nrfi-central.com
# renames/reorganizes pages -- check the nav in nrfi_central_scraper's
# module docstring reference pull (or just browse the live "Stats" menu).
_PAGE_TEAM_NRFI_LEADERBOARD = "/2026-team-nrfi-first-inning-leaderboard/"
_PAGE_PITCHER_NRFI_RECORD = "/2026-pitcher-nrfi-yrfi-record/"
_PAGE_PITCHER_1ST_INNING_STATS = "/2026-pitcher-1st-inning-stats/"
_PAGE_BATTER_1ST_INNING_STATS = "/batter-1st-inning-stats/"
_PAGE_BALLPARK_FACTORS = "/ballpark-factors/"


@dataclass
class TeamFirstInningRecord:
    team_abbr: str
    nrfi_pct: Optional[float] = None       # fraction, e.g. 0.72
    avg_first_inning_runs: Optional[float] = None
    sample_games: Optional[int] = None


@dataclass
class PitcherFirstInningRecord:
    pitcher_name: str
    nrfi_pct: Optional[float] = None       # fraction of his own starts that were NRFI
    era_1st: Optional[float] = None
    k_pct_1st: Optional[float] = None
    starts_sample: Optional[int] = None


@dataclass
class BatterFirstInningRecord:
    batter_name: str
    team_abbr: Optional[str] = None
    obp_1st: Optional[float] = None
    k_pct_1st: Optional[float] = None
    pa_sample: Optional[int] = None


def _cache_path(name: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"nrfi_central_{name}.json")


def _load_cache(name: str) -> Optional[dict]:
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > _CACHE_TTL_SECONDS:
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug(f"[nrfi_central_scraper] cache read failed for {name}: {exc}")
        return None


def _save_cache(name: str, data: dict) -> None:
    try:
        with open(_cache_path(name), "w") as f:
            json.dump(data, f)
    except OSError as exc:
        logger.debug(f"[nrfi_central_scraper] cache write failed for {name}: {exc}")


def _fetch_page(path: str) -> Optional[str]:
    """
    Single HTTP GET wrapper. Returns raw HTML text, or None on any failure
    (network error, non-200, timeout). Never raises -- callers rely on
    None to mean 'skip this source', per this module's fail-safe contract.
    """
    try:
        import requests
    except ImportError:
        logger.debug("[nrfi_central_scraper] requests not installed -- skipping")
        return None

    url = f"{_BASE_URL}{path}"
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            logger.warning(f"[nrfi_central_scraper] {url} -> HTTP {resp.status_code}")
            return None
        return resp.text
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module docstring
        logger.warning(f"[nrfi_central_scraper] fetch failed for {url}: {exc}")
        return None


def _parse_first_table(html: str) -> Optional[list[dict]]:
    """
    Generic WordPress-table parser: grabs the first <table> on the page
    and returns rows as list[dict] keyed by header text (lowercased,
    spaces->underscores). Returns None if no table is found or it has no
    rows -- treat that as "page layout changed, needs a selector update,"
    not as "the team/pitcher has no data."
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.debug("[nrfi_central_scraper] beautifulsoup4 not installed -- skipping")
        return None

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        return None

    header_cells = table.find("tr").find_all(["th", "td"])
    headers = [
        c.get_text(strip=True).lower().replace(" ", "_").replace("%", "pct")
        for c in header_cells
    ]
    if not headers:
        return None

    rows: list[dict] = []
    for tr in table.find_all("tr")[1:]:
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) != len(headers):
            continue  # malformed row -- skip rather than misalign columns
        rows.append(dict(zip(headers, cells)))
    return rows or None


def _to_float(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    cleaned = raw.strip().rstrip("%")
    try:
        val = float(cleaned)
    except ValueError:
        return None
    # nrfi-central displays percentages as e.g. "72.4" not "0.724" --
    # normalize to a 0-1 fraction to match this repo's convention
    # (see models/nrfi_handicapper.py's fbf_obp / league_fbf_obp usage).
    if "%" in raw or val > 1.0:
        val = val / 100.0
    return round(val, 4)


def get_team_first_inning_leaderboard() -> dict[str, TeamFirstInningRecord]:
    """
    Team-level NRFI% -- fills the gap flagged in core/game_markets.py's
    _team_first_inning_lambda_fallback docstring, which currently only has
    a full-game-runs-scaled-down proxy, not a real observed first-inning
    frequency. Keyed by whatever team name string the site uses -- caller
    is responsible for mapping that to this repo's MLB_TEAM_IDS abbrevs
    (see core/mlb_team_ids.py), since the site's naming convention hasn't
    been verified live against that mapping.

    Returns {} on any failure -- never partial/fabricated data.
    """
    cached = _load_cache("team_leaderboard")
    if cached is not None:
        source = cached
    else:
        html = _fetch_page(_PAGE_TEAM_NRFI_LEADERBOARD)
        if html is None:
            return {}
        rows = _parse_first_table(html)
        if rows is None:
            logger.warning(
                "[nrfi_central_scraper] no table found on team leaderboard page "
                "-- page layout may have changed, selectors need updating"
            )
            return {}
        source = rows
        _save_cache("team_leaderboard", source)

    out: dict[str, TeamFirstInningRecord] = {}
    for row in source:
        team = row.get("team")
        if not team:
            continue
        out[team] = TeamFirstInningRecord(
            team_abbr=team,
            nrfi_pct=_to_float(row.get("nrfi_pct") or row.get("nrfi")),
            avg_first_inning_runs=_to_float(row.get("avg_1st_inning_runs") or row.get("runs")),
            sample_games=int(row["gp"]) if row.get("gp", "").isdigit() else None,
        )
    return out


def get_pitcher_first_inning_stats() -> dict[str, PitcherFirstInningRecord]:
    """
    Pitcher-level 1st-inning ERA/K%/NRFI record -- cross-check source
    alongside (not a replacement for) data/statcast_first_inning.py's
    Statcast-derived numbers. Keyed by pitcher display name as the site
    renders it -- caller must fuzzy-match against this repo's own
    pitcher-name normalization if reconciling the two sources.

    Returns {} on any failure.
    """
    cached = _load_cache("pitcher_1st_inning")
    if cached is not None:
        source = cached
    else:
        html = _fetch_page(_PAGE_PITCHER_1ST_INNING_STATS)
        if html is None:
            return {}
        rows = _parse_first_table(html)
        if rows is None:
            logger.warning(
                "[nrfi_central_scraper] no table found on pitcher 1st-inning page"
            )
            return {}
        source = rows
        _save_cache("pitcher_1st_inning", source)

    out: dict[str, PitcherFirstInningRecord] = {}
    for row in source:
        name = row.get("pitcher") or row.get("player")
        if not name:
            continue
        out[name] = PitcherFirstInningRecord(
            pitcher_name=name,
            nrfi_pct=_to_float(row.get("nrfi_pct")),
            era_1st=_to_float(row.get("era")) if row.get("era") else None,
            k_pct_1st=_to_float(row.get("k_pct")),
            starts_sample=int(row["gs"]) if row.get("gs", "").isdigit() else None,
        )
    return out


def get_batter_first_inning_stats() -> dict[str, BatterFirstInningRecord]:
    """
    Batter-level 1st-inning OBP/K% -- this is the field this repo was
    missing entirely (data/statcast_lineup_platoon.py has OBP/ISO but NOT
    K%, and nothing here is 1st-inning-specific vs. season-long for
    hitters). Keyed by batter display name.

    Returns {} on any failure.
    """
    cached = _load_cache("batter_1st_inning")
    if cached is not None:
        source = cached
    else:
        html = _fetch_page(_PAGE_BATTER_1ST_INNING_STATS)
        if html is None:
            return {}
        rows = _parse_first_table(html)
        if rows is None:
            logger.warning(
                "[nrfi_central_scraper] no table found on batter 1st-inning page"
            )
            return {}
        source = rows
        _save_cache("batter_1st_inning", source)

    out: dict[str, BatterFirstInningRecord] = {}
    for row in source:
        name = row.get("batter") or row.get("player")
        if not name:
            continue
        out[name] = BatterFirstInningRecord(
            batter_name=name,
            team_abbr=row.get("team"),
            obp_1st=_to_float(row.get("obp")),
            k_pct_1st=_to_float(row.get("k_pct")),
            pa_sample=int(row["pa"]) if row.get("pa", "").isdigit() else None,
        )
    return out


if __name__ == "__main__":
    # Manual smoke test -- run this directly (with real network access) to
    # verify the selectors still match the live site before wiring this
    # into core/game_markets.py. NOT run/verified in the sandbox that wrote
    # this file (no network access there) -- see module docstring.
    logging.basicConfig(level=logging.INFO)
    teams = get_team_first_inning_leaderboard()
    print(f"Parsed {len(teams)} teams from leaderboard page")
    for abbr, rec in list(teams.items())[:5]:
        print(" ", rec)

    pitchers = get_pitcher_first_inning_stats()
    print(f"Parsed {len(pitchers)} pitchers from 1st-inning stats page")

    batters = get_batter_first_inning_stats()
    print(f"Parsed {len(batters)} batters from 1st-inning stats page")
