"""
Data layer for the picks engine.
Reads API keys from environment variables (set via .env / GitHub Secrets — never hardcoded).

Required env vars:
  BALL_DONT_LIE_KEY
  THE_ODDS_API_KEY
"""
import os
import time
import requests

BDL_BASE = "https://api.balldontlie.io"
ODDS_BASE = "https://api.the-odds-api.com/v4"

# Gap between balldontlie calls -- the earlier 0.5s/2s gaps weren't enough;
# your tier's quota appears tight enough that back-to-back calls during
# player/roster discovery (dozens in a row) still get rate limited even
# with a few seconds between them and a short backoff. This is more
# conservative on purpose.
BDL_REQUEST_DELAY_SECONDS = 4.0
BDL_MAX_RETRIES = 4


def _bdl_headers():
    # Accept either env var name -- sports-engine's own convention
    # (BALL_DONT_LIE_KEY) or Wordsmith74's (BALLDONTLIE_API_KEY), since the
    # same key is the same provider and setting one shouldn't silently be
    # ignored because the other file used a different name.
    key = os.environ.get("BALL_DONT_LIE_KEY") or os.environ.get("BALLDONTLIE_API_KEY")
    if not key:
        raise RuntimeError("BALL_DONT_LIE_KEY (or BALLDONTLIE_API_KEY) not set in environment")
    return {"Authorization": key}


def _bdl_get(path, params=None, timeout=15):
    """
    Shared GET wrapper for all balldontlie endpoints.

    - Waits BDL_REQUEST_DELAY_SECONDS before each call so a run with many
      lookups doesn't hammer the API back-to-back.
    - On a 401, fails immediately -- retrying will never fix an auth error,
      and the old behaviour (retrying anyway) caused 40+ minute runs when
      the key lacked access to an endpoint (e.g. MLB stats on a free tier).
    - On a 429, respects the API's own Retry-After header if present,
      otherwise backs off on an increasing schedule (15s, 30s, 45s, 60s)
      and retries up to BDL_MAX_RETRIES times. If it still fails after
      that, it raises -- this rides out a real per-minute quota window
      rather than looping indefinitely.
    """
    last_exc = None
    for attempt in range(BDL_MAX_RETRIES + 1):
        time.sleep(BDL_REQUEST_DELAY_SECONDS)
        r = requests.get(f"{BDL_BASE}{path}", headers=_bdl_headers(), params=params, timeout=timeout)

        if r.status_code == 401:
            # Auth failure -- retrying will never help, fail immediately.
            # Most likely cause: your balldontlie plan doesn't cover this
            # endpoint (e.g. MLB stats requires a paid tier).
            raise requests.exceptions.HTTPError(
                f"401 Unauthorized on {path} -- check BALL_DONT_LIE_KEY has access to this endpoint"
            )

        if r.status_code == 429:
            wait_s = float(r.headers.get("Retry-After", 15 * (attempt + 1)))
            print(f"  [warn] balldontlie rate limit hit on {path} (attempt {attempt + 1}/"
                  f"{BDL_MAX_RETRIES + 1}) -- waiting {wait_s}s")
            time.sleep(wait_s)
            last_exc = requests.exceptions.HTTPError(f"429 from {path} after {attempt + 1} attempt(s)")
            continue

        r.raise_for_status()
        return r.json()

    raise last_exc


# ---------- MLB ----------

def get_mlb_games(date_str):
    """date_str format: YYYY-MM-DD"""
    return _bdl_get("/mlb/v1/games", params={"dates[]": date_str}).get("data", [])


def get_mlb_player_season_stats(player_id, season):
    return _bdl_get(
        "/mlb/v1/season_stats",
        params={"player_ids[]": player_id, "season": season},
    ).get("data", [])


def _mlb_statsapi_id_from_name(pitcher_name):
    """
    Resolve a pitcher name to an MLB Stats API person ID via the
    statsapi.mlb.com people search endpoint (free, no key).
    Returns the integer person ID or None if not found.
    In-memory cache scoped to one run to avoid repeat lookups.
    """
    key = pitcher_name.strip().lower()
    if key in _mlb_statsapi_id_cache:
        return _mlb_statsapi_id_cache[key]

    r = requests.get(
        "https://statsapi.mlb.com/api/v1/people/search",
        params={"names": pitcher_name, "sportId": 1},
        timeout=15,
    )
    r.raise_for_status()
    people = r.json().get("people", [])
    pid = people[0]["id"] if people else None
    _mlb_statsapi_id_cache[key] = pid
    return pid

_mlb_statsapi_id_cache = {}


def get_mlb_player_game_logs(player_id, season, limit=10, pitcher_name=None):
    """
    Recent game-by-game pitching stats for trend / sample-size weighted projections.

    Uses the official MLB Stats API (statsapi.mlb.com) -- free, no API key,
    no rate-limit issues. player_id passed in is the ESPN athlete ID; since
    ESPN and MLB Stats API use different ID namespaces we resolve the correct
    MLB Stats API person ID via a name search first (cached in-memory per run).
    pitcher_name is used for that resolution -- pass it from the caller.

    Returns a list of dicts with keys matching what run_pipeline.py expects:
      strikeouts, batters_faced, innings_pitched, player (dict with full_name)
    """
    # Resolve ESPN athlete ID -> MLB Stats API person ID via name search
    mlb_id = None
    if pitcher_name:
        mlb_id = _mlb_statsapi_id_from_name(pitcher_name)
    if mlb_id is None:
        # Fall back to using the passed ID directly in case it happens to be correct
        mlb_id = player_id

    url = (
        f"https://statsapi.mlb.com/api/v1/people/{mlb_id}/stats"
        f"?stats=gameLog&group=pitching&season={season}&gameType=R"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    splits = (
        r.json()
        .get("stats", [{}])[0]
        .get("splits", [])
    )
    # Most recent games first
    splits = list(reversed(splits))[:limit]

    out = []
    for s in splits:
        stat = s.get("stat", {}) or {}
        player_info = s.get("player", {}) or {}
        # MLB Stats API returns innings_pitched as a string e.g. "5.2",
        # and any field can be None if the game was incomplete -- guard all of them.
        def _int(val, default=0):
            try:
                return int(val) if val is not None else default
            except (TypeError, ValueError):
                return default

        def _float(val, default=0.0):
            try:
                return float(val) if val is not None else default
            except (TypeError, ValueError):
                return default

        out.append({
            "strikeouts": _int(stat.get("strikeOuts")),
            "batters_faced": _int(stat.get("battersFaced")),
            "innings_pitched": _float(stat.get("inningsPitched")),
            "player": {
                "full_name": player_info.get("fullName") or "",
            },
        })
    return out



# ---------- WNBA ----------

def get_wnba_games(date_str):
    return _bdl_get("/wnba/v1/games", params={"dates[]": date_str}).get("data", [])


def get_wnba_player_game_logs(player_id, season, limit=10):
    return _bdl_get(
        "/wnba/v1/stats",
        params={"player_ids[]": player_id, "seasons[]": season, "per_page": limit},
    ).get("data", [])


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


def get_wnba_team_injuries(team_abbr, roster_names=None):
    """
    WNBA injury fetch, ESPN-diversified per Wordsmith74 port: tries the
    FREE RotoWire public injury-report scrape first (no API key --
    see data/rotowire_injuries.py), and only falls back to ESPN's
    get_espn_team_injuries() if RotoWire is unreachable or its markup
    changed. roster_names (lowercased player-name set) is required for
    RotoWire to safely filter the league-wide page down to one team;
    without it, this skips straight to ESPN.

    Returns the same {"injuries": [...]} shape either way -- callers
    (models/injury_intel.py) don't need to know which source answered.
    """
    from data.rotowire_injuries import get_team_injuries as _rotowire_get

    try:
        data = _rotowire_get(team_abbr, roster_names=roster_names)
        if data:
            return data
    except Exception as exc:
        print(f"  [warn] RotoWire (free) injury fetch failed for {team_abbr}: {exc} -- falling back to ESPN")

    # ESPN fallback -- requires team_id, not team_abbr; caller of this
    # convenience function should generally prefer passing roster_names
    # so RotoWire succeeds and this fallback is rarely hit. If you need
    # the ESPN path directly, call get_espn_team_injuries(sport, league,
    # team_id) yourself -- this wrapper is WNBA/team_abbr specific.
    print(f"  [warn] No free RotoWire data for {team_abbr}; ESPN fallback requires team_id "
          f"-- call get_espn_team_injuries('basketball', 'wnba', team_id) directly if needed.")
    return None




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

    SUPERSEDED for run_pipeline.py's F5 path: every park failed to match against
    this table's column 0 (join key mismatch, not a spelling typo -- see
    models/advanced_metrics.py's MLB_PARK_FACTORS and park_factor_by_name() for
    the static-dict replacement now used there). Left here unused-but-intact in
    case another caller still depends on the raw FanGraphs table.
    """
    import pandas as pd
    url = f"https://www.fangraphs.com/guts.aspx?type=pf&season={season}"
    tables = pd.read_html(url)
    return tables[0] if tables else None




# In-memory cache for player name -> search results, scoped to one run.
# Several pitchers/teams can repeat across a run (e.g. same starter shows up
# in K-prop discovery more than once), so this avoids redundant lookups
# against the same name within a single pipeline run.
_player_search_cache = {}


def get_mlb_player_search(name):
    """
    balldontlie player search by name -- standard pattern across their sports
    APIs (confirmed for /mlb/v1/players in their docs; NOT live-verified in
    this sandbox). Used to resolve an ESPN probable-pitcher name to a
    balldontlie player_id so live_fetch_mlb_pitcher_k_prop can pull game logs.
    Returns the raw list of matches -- caller picks the best match (usually
    the first, but check len() > 1 for common-surname collisions).
    """
    cache_key = name.strip().lower()
    if cache_key in _player_search_cache:
        return _player_search_cache[cache_key]

    results = _bdl_get("/mlb/v1/players", params={"search": name}).get("data", [])
    _player_search_cache[cache_key] = results
    return results


def get_wnba_team_players(team_id):
    """
    balldontlie players list filtered by team -- same documented (not
    live-verified) pattern as get_mlb_player_search. Used for WNBA
    roster resolution: "which players on this team should we generate
    point props for" since there's no separate probable-starters concept
    in basketball the way there is for MLB starting pitchers.
    """
    return _bdl_get("/wnba/v1/players", params={"team_ids[]": team_id}).get("data", [])


def get_mlb_team_recent_runs(team_id, before_date, n=5, lookback_days=20):
    """
    Returns up to *n* most recent completed games' runs-scored for
    *team_id*, looking backward from *before_date* (YYYY-MM-DD, exclusive)
    up to *lookback_days* days. Used to fill the F5-total model's
    home_recent_runs/away_recent_runs inputs (previously hardcoded to []).

    NOT live-verified: assumes each balldontlie MLB game dict has
    "status" (checked for a completed-game marker), "home_team": {"id":...},
    "away_team": {"id":...}, "home_team_score", "away_team_score". If any
    of these field names are wrong, this returns [] (same safe degrade as
    before -- process_mlb_f5 already protects against an empty list via
    its caller) and prints which date/field failed so it's a one-line fix.
    """
    from datetime import datetime, timedelta

    runs = []
    cursor = datetime.strptime(before_date, "%Y-%m-%d")
    days_checked = 0

    while len(runs) < n and days_checked < lookback_days:
        cursor -= timedelta(days=1)
        days_checked += 1
        date_str = cursor.strftime("%Y-%m-%d")
        try:
            games = get_mlb_games(date_str)
        except Exception:
            continue

        for g in games:
            status = (g.get("status") or "").lower()
            if status and "final" not in status and "complete" not in status:
                continue  # skip in-progress/scheduled games for this lookback

            home = g.get("home_team", {}) or {}
            away = g.get("away_team", {}) or {}
            if home.get("id") == team_id:
                score = g.get("home_team_score")
            elif away.get("id") == team_id:
                score = g.get("away_team_score")
            else:
                continue

            if score is not None:
                runs.append(score)

    if not runs:
        print(f"  [warn] get_mlb_team_recent_runs: no completed games found for team "
              f"{team_id} in the {lookback_days} days before {before_date} -- check "
              f"'status'/'home_team_score' field names against a live response")

    return runs[:n]


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


# ---------- PropLine (alternative odds/props source, being trialed alongside the-odds-api) ----------
# Response format is advertised as drop-in compatible with the-odds-api, so these mirror
# get_odds() / get_event_player_props() above. Set PROPLINE_API_KEY in env -- never hardcode it.
PROPLINE_BASE = "https://api.prop-line.com/v1"


def _propline_key():
    key = os.environ.get("PROPLINE_API_KEY")
    if not key:
        raise RuntimeError("PROPLINE_API_KEY not set in environment")
    return key


def get_propline_sports():
    """Lists available sport keys -- useful for a one-off check that your key/account covers
    the sports you need (confirmed live: baseball_mlb, basketball_wnba, among many others)."""
    r = requests.get(f"{PROPLINE_BASE}/sports", params={"apiKey": _propline_key()}, timeout=15)
    r.raise_for_status()
    return r.json()


def get_propline_odds(sport_key, markets, regions="us"):
    """PropLine equivalent of get_odds() -- same call shape, different provider."""
    r = requests.get(
        f"{PROPLINE_BASE}/sports/{sport_key}/odds",
        params={"apiKey": _propline_key(), "regions": regions, "markets": markets, "oddsFormat": "american"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def get_propline_event_player_props(sport_key, event_id, markets, regions="us"):
    """PropLine equivalent of get_event_player_props() -- same call shape, different provider."""
    r = requests.get(
        f"{PROPLINE_BASE}/sports/{sport_key}/events/{event_id}/odds",
        params={"apiKey": _propline_key(), "regions": regions, "markets": markets, "oddsFormat": "american"},
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
