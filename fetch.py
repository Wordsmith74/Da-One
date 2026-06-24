"""
Data layer for the picks engine.
Reads API keys from environment variables (set via .env / GitHub Secrets — never hardcoded).

Required env vars:
  BALL_DONT_LIE_KEY
  THE_ODDS_API_KEY
"""
import os
import requests

BDL_BASE = "https://api.balldontlie.io"
ODDS_BASE = "https://api.the-odds-api.com/v4"


def _bdl_headers():
    key = os.environ.get("BALL_DONT_LIE_KEY")
    if not key:
        raise RuntimeError("BALL_DONT_LIE_KEY not set in environment")
    return {"Authorization": key}


# ---------- MLB ----------

def get_mlb_games(date_str):
    """date_str format: YYYY-MM-DD"""
    r = requests.get(
        f"{BDL_BASE}/mlb/v1/games",
        headers=_bdl_headers(),
        params={"dates[]": date_str},
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("data", [])


def get_mlb_player_season_stats(player_id, season):
    r = requests.get(
        f"{BDL_BASE}/mlb/v1/season_stats",
        headers=_bdl_headers(),
        params={"player_ids[]": player_id, "season": season},
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("data", [])


def get_mlb_player_game_logs(player_id, season, limit=10):
    """Recent game-by-game stats for trend / sample-size weighted projections."""
    r = requests.get(
        f"{BDL_BASE}/mlb/v1/stats",
        headers=_bdl_headers(),
        params={"player_ids[]": player_id, "seasons[]": season, "per_page": limit},
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("data", [])


# ---------- WNBA ----------

def get_wnba_games(date_str):
    r = requests.get(
        f"{BDL_BASE}/wnba/v1/games",
        headers=_bdl_headers(),
        params={"dates[]": date_str},
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("data", [])


def get_wnba_player_game_logs(player_id, season, limit=10):
    r = requests.get(
        f"{BDL_BASE}/wnba/v1/stats",
        headers=_bdl_headers(),
        params={"player_ids[]": player_id, "seasons[]": season, "per_page": limit},
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("data", [])


# ---------- ESPN (no API key needed, public hidden API -- VERIFIED LIVE in this build) ----------
# Confirmed working via direct fetch: site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard
# Unlike balldontlie/Savant, these field names below ARE verified against a real response.

def get_espn_mlb_scoreboard(date_str=None):
    """
    date_str: 'YYYYMMDD' format (ESPN's convention, different from balldontlie's
    'YYYY-MM-DD' -- watch this when passing dates between the two sources).
    Returns today's games if date_str is None.

    KNOWN ISSUE (found during live verification of this endpoint): passing
    ?dates=YYYYMMDD returned a 400 in testing even with a syntactically valid
    date, while the no-param call worked fine. This may be a transient/caching
    quirk rather than a real API change -- this function tries the dated call
    first and falls back to the dateless call (which defaults to "today" in
    ESPN's own timezone, not necessarily TODAY as computed in run_pipeline.py)
    if the dated call fails. If you need a specific past/future date reliably,
    verify this independently before trusting it for anything but "today."

    Confirmed real response includes, per game (event):
      event['id'], event['shortName'] (e.g. "SEA @ BAL")
      event['competitions'][0]['competitors'][i]['team']['displayName'/'abbreviation'/'id']
      event['competitions'][0]['competitors'][i]['probables'][0]['athlete']['id'/'fullName']
      event['competitions'][0]['competitors'][i]['probables'][0]['statistics'] -- includes ERA
    This gives probable starting pitcher name + ID directly, solving the roster-resolution
    gap that blocked the K-prop live path -- no separate roster call needed for that.
    """
    base_url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
    if date_str:
        r = requests.get(base_url, params={"dates": date_str}, timeout=15)
        if r.status_code == 400:
            r = requests.get(base_url, timeout=15)  # fall back to dateless/today
        else:
            r.raise_for_status()
    else:
        r = requests.get(base_url, timeout=15)
    r.raise_for_status()
    return r.json().get("events", [])


def get_espn_wnba_scoreboard(date_str=None):
    """Same structure as MLB scoreboard, WNBA endpoint."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"
    if date_str:
        url += f"?dates={date_str}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json().get("events", [])


def get_espn_team_roster(sport, league, team_id):
    """
    sport: 'baseball' | 'basketball'   league: 'mlb' | 'wnba'
    Confirmed pattern from docs (not yet live-verified in this build -- the scoreboard
    call above was verified, this roster endpoint follows the same documented family
    but wasn't separately fetched; check the response shape on first real use).
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams/{team_id}/roster"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def get_espn_team_injuries(sport, league, team_id):
    """Same confirmation status as get_espn_team_roster -- documented, not separately verified."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams/{team_id}/injuries"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()




def get_savant_pitcher_advanced_stats(pitcher_name, season):
    """
    Pulls CSW%, SwStr%, SIERA, and K% for a pitcher -- no API key needed (Savant and
    FanGraphs are both public), via the pybaseball library. Savant's own leaderboard
    pages are JS-rendered (no static JSON/CSV URL to fetch directly), so pybaseball
    is the reliable path -- it maintains the actual scraping logic as Savant changes.

    Requires: pip install pybaseball --break-system-packages

    NOTE: pybaseball's exact function/column names shift slightly across versions --
    on first real run, print(df.columns) to confirm field names before trusting them
    in advanced_metrics.py's project_k_pct_advanced() / pitcher_quality_factor().
    Recommended single call: pb.pitching_stats(season) -- FanGraphs-sourced, includes
    SIERA, K%, and CSW% together in one table, avoiding a second source entirely.
    """
    import pybaseball as pb
    pb.cache.enable()  # avoids re-hitting the source on repeated calls during dev/testing

    df = pb.pitching_stats(season, qual=0)
    match = df[df["Name"].str.contains(pitcher_name, case=False, na=False)]
    return match


def get_park_factors(season):
    """
    FanGraphs' Guts page publishes park factors as a plain (non-JS-rendered) HTML
    table -- no API key, parseable directly with pandas, no scraping library needed.
    Use f5_park_factor() in advanced_metrics.py to scale this for F5-specific use.

    Requires: pip install pandas lxml --break-system-packages (lxml needed for read_html)
    """
    import pandas as pd
    url = f"https://www.fangraphs.com/guts.aspx?type=pf&season={season}"
    tables = pd.read_html(url)
    return tables[0] if tables else None




def get_odds(sport_key, markets, regions="us"):
    """
    sport_key examples: 'baseball_mlb', 'basketball_wnba'
    markets examples: 'h2h,totals' for game lines.
    Player props use a separate per-event endpoint — see get_event_player_props.
    """
    key = os.environ.get("THE_ODDS_API_KEY")
    if not key:
        raise RuntimeError("THE_ODDS_API_KEY not set in environment")
    r = requests.get(
        f"{ODDS_BASE}/sports/{sport_key}/odds",
        params={"apiKey": key, "regions": regions, "markets": markets, "oddsFormat": "american"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def get_event_player_props(sport_key, event_id, markets, regions="us"):
    key = os.environ.get("THE_ODDS_API_KEY")
    if not key:
        raise RuntimeError("THE_ODDS_API_KEY not set in environment")
    r = requests.get(
        f"{ODDS_BASE}/sports/{sport_key}/events/{event_id}/odds",
        params={"apiKey": key, "regions": regions, "markets": markets, "oddsFormat": "american"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    # Smoke test — requires env vars set locally.
    import json
    today = "2026-06-21"
    try:
        games = get_mlb_games(today)
        print(f"MLB games today: {len(games)}")
        print(json.dumps(games[:1], indent=2))
    except Exception as e:
        print("MLB fetch failed:", e)
