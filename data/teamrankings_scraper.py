"""
data/teamrankings_scraper.py

Third-party data source: scrapes public MLB team-stat pages from
teamrankings.com. TeamRankings publishes exactly the team-level 1st-inning
data this repo's fallback lacked -- real observed "No Run First Inning
(NRFI) %" and "Yes Run First Inning (YRFI) %" by team, plus 1st-inning
runs/game and OPPONENT 1st-inning runs/game (i.e. how often a team's own
pitching staff allows a 1st-inning run) -- rather than the scaled-down
full-game-runs proxy core/game_markets.py currently falls back to
(_team_first_inning_lambda_fallback).

============================ IMPORTANT ====================================
UNVERIFIED / LIKELY BLOCKED. A direct fetch attempt against
teamrankings.com from the sandbox that wrote this file was REJECTED by the
site's own bot-detection layer on the very first request (not a fluke --
this is a real defense the site runs). That means:
  - A bare `requests.get()` like the one below may simply fail every time
    from a data-center IP / non-browser client, regardless of what headers
    are set.
  - Getting past it reliably typically requires a real browser session
    (e.g. Playwright/Selenium) rather than a plain HTTP client, and even
    then may violate the site's Terms of Service -- READ
    https://www.teamrankings.com's ToS/robots.txt yourself before relying
    on this for anything beyond occasional manual/personal use. This repo
    does not make that legal call for you.
  - This module is provided as a best-effort starting point, NOT a tested,
    working integration. Treat every function's empty-dict return as the
    expected/likely outcome until you've confirmed otherwise from your own
    network.
=============================================================================

Honesty / fail-safe contract, same as every other data/ module in this
repo: any failure (blocked request, network error, changed markup) returns
{} / None, never a fabricated number.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("betting_bot")

_BASE_URL = "https://www.teamrankings.com"
_USER_AGENT = "Da-One-NRFI-Model/1.0 (+internal research tool; contact: set-your-contact-here)"
_REQUEST_TIMEOUT_SECONDS = 10

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
_CACHE_TTL_SECONDS = 6 * 60 * 60

_PAGE_TEAM_NRFI_PCT = "/mlb/stat/no-run-first-inning-pct"
_PAGE_TEAM_YRFI_PCT = "/mlb/stat/yes-run-first-inning-pct"
_PAGE_TEAM_OPP_NRFI_PCT = "/mlb/stat/opponent-no-run-first-inning-pct"
_PAGE_TEAM_1ST_INNING_RUNS = "/mlb/stat/1st-inning-runs-per-game"
_PAGE_TEAM_OPP_1ST_INNING_RUNS = "/mlb/stat/opponent-1st-inning-runs-per-game"


@dataclass
class TeamRankingsFirstInningRecord:
    team_name: str                          # TeamRankings' own naming, NOT this repo's abbrevs
    nrfi_pct: Optional[float] = None        # this team's OWN offense: % of their games w/ no 1st-inn run
    opp_nrfi_pct: Optional[float] = None    # this team's PITCHING: % of games opponent doesn't score in 1st
    first_inning_runs_per_game: Optional[float] = None       # offense
    opp_first_inning_runs_per_game: Optional[float] = None   # pitching (what THEY allow)


def _cache_path(name: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"teamrankings_{name}.json")


def _load_cache(name: str) -> Optional[list]:
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > _CACHE_TTL_SECONDS:
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug(f"[teamrankings_scraper] cache read failed for {name}: {exc}")
        return None


def _save_cache(name: str, data: list) -> None:
    try:
        with open(_cache_path(name), "w") as f:
            json.dump(data, f)
    except OSError as exc:
        logger.debug(f"[teamrankings_scraper] cache write failed for {name}: {exc}")


def _fetch_and_parse_stat_table(path: str, cache_key: str) -> Optional[list[dict]]:
    """
    TeamRankings' stat pages render a single sortable table: rank, team,
    current-season value, last-3, last-1, home, away, prior-season. We
    only care about "team" and the current-season "value" column here.

    Returns None on ANY failure (including the bot-detection block this
    module's docstring warns about) -- never fabricates.
    """
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        logger.debug(f"[teamrankings_scraper] missing dependency: {exc}")
        return None

    url = f"{_BASE_URL}{path}"
    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[teamrankings_scraper] request failed for {url}: {exc}")
        return None

    if resp.status_code != 200:
        # This is the expected failure mode per this module's docstring --
        # bot detection typically returns 403 (or a CAPTCHA-serving 200
        # that won't contain the real table). Either way, bail cleanly.
        logger.warning(
            f"[teamrankings_scraper] {url} -> HTTP {resp.status_code} "
            f"(likely bot-detection block -- see module docstring)"
        )
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", class_="tr-table")
    if table is None:
        table = soup.find("table")  # fallback: first table on page
    if table is None:
        logger.warning(f"[teamrankings_scraper] no table found at {url}")
        return None

    rows: list[dict] = []
    body_rows = table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")[1:]
    for tr in body_rows:
        cells = [c.get_text(strip=True) for c in tr.find_all("td")]
        if len(cells) < 3:
            continue
        # Column layout per TeamRankings' standard stat-page template:
        # [rank, team, current_value, last3, last1, home, away, prior_season]
        rows.append({"team": cells[1], "value": cells[2]})

    if not rows:
        return None
    _save_cache(cache_key, rows)
    return rows


def _to_float(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    cleaned = raw.strip().rstrip("%")
    try:
        val = float(cleaned)
    except ValueError:
        return None
    if "%" in raw or val > 1.0:
        val = val / 100.0
    return round(val, 4)


def get_team_first_inning_records() -> dict[str, TeamRankingsFirstInningRecord]:
    """
    Fetches and merges all 4 TeamRankings 1st-inning pages into one
    per-team record. Any individual page that fails just leaves that
    field None on the record rather than dropping the team entirely --
    partial real data beats an all-or-nothing failure here, since each
    page is an independent HTTP request that can fail independently.

    Returns {} only if EVERY page fails (e.g. fully bot-blocked).
    """
    nrfi_rows = _fetch_and_parse_stat_table(_PAGE_TEAM_NRFI_PCT, "team_nrfi_pct")
    opp_nrfi_rows = _fetch_and_parse_stat_table(_PAGE_TEAM_OPP_NRFI_PCT, "team_opp_nrfi_pct")
    runs_rows = _fetch_and_parse_stat_table(_PAGE_TEAM_1ST_INNING_RUNS, "team_1st_inning_runs")
    opp_runs_rows = _fetch_and_parse_stat_table(_PAGE_TEAM_OPP_1ST_INNING_RUNS, "team_opp_1st_inning_runs")

    if not any([nrfi_rows, opp_nrfi_rows, runs_rows, opp_runs_rows]):
        logger.warning(
            "[teamrankings_scraper] all 4 source pages failed -- likely "
            "bot-detection block (see module docstring). Returning {}."
        )
        return {}

    out: dict[str, TeamRankingsFirstInningRecord] = {}

    def _ensure(team: str) -> TeamRankingsFirstInningRecord:
        if team not in out:
            out[team] = TeamRankingsFirstInningRecord(team_name=team)
        return out[team]

    for rows, attr in [
        (nrfi_rows, "nrfi_pct"),
        (opp_nrfi_rows, "opp_nrfi_pct"),
        (runs_rows, "first_inning_runs_per_game"),
        (opp_runs_rows, "opp_first_inning_runs_per_game"),
    ]:
        if not rows:
            continue
        for row in rows:
            rec = _ensure(row["team"])
            val = _to_float(row["value"])
            setattr(rec, attr, val)

    return out


if __name__ == "__main__":
    # Manual smoke test -- run with real network access to see whether
    # this actually gets past TeamRankings' bot detection. Expect this to
    # print "0 teams" / all-empty until proven otherwise; see module
    # docstring for why this could not be verified in the sandbox that
    # wrote this file.
    logging.basicConfig(level=logging.INFO)
    records = get_team_first_inning_records()
    print(f"Parsed {len(records)} teams")
    for name, rec in list(records.items())[:5]:
        print(" ", rec)
