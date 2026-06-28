"""
Data layer for the picks engine.
Reads API keys from environment variables (set via .env / GitHub Secrets — never hardcoded).

Required env vars:
  BALL_DONT_LIE_KEY
  THE_ODDS_API_KEY
"""
import os
import time
import threading
import requests

from player_id_cache import load_cache as _load_player_id_cache, \
    save_cache as _save_player_id_cache, lookup as _lookup_player_id, \
    remember as _remember_player_id

BDL_BASE = "https://api.balldontlie.io"
ODDS_BASE = "https://api.the-odds-api.com/v4"

# Gap between balldontlie calls -- the earlier 0.5s/2s gaps weren't enough;
# your tier's quota appears tight enough that back-to-back calls during
# player/roster discovery (dozens in a row) still get rate limited even
# with a few seconds between them and a short backoff. This is more
# conservative on purpose.
BDL_REQUEST_DELAY_SECONDS = 4.0
BDL_MAX_RETRIES = 4

# run_pipeline.py fetches K-props and WNBA props from thread pools (6-8
# workers), and ALL of those threads eventually call _bdl_get for the same
# balldontlie quota. BDL_REQUEST_DELAY_SECONDS used to only pace each thread
# against itself -- with multiple threads it did nothing, since they'd all
# sleep 4s and then hit the API at roughly the same instant, guaranteeing
# 429s that each thread then backs off from independently (15/30/45/60s,
# compounding across threads instead of sharing one wait). This lock makes
# the delay+request a single global bottleneck so threads queue politely
# against the real shared quota instead of colliding.
_BDL_LOCK = threading.Lock()

# Persistent WNBA name->balldontlie-id cache (separate from balldontlie's own
# in-process _player_search_cache below, which only lives for one run).
# Loaded lazily on first use rather than at import time, since the WNBA props
# path may run multiple threads concurrently (ThreadPoolExecutor in
# run_pipeline_final.py) and they all share one load -- guarded by the same
# lock used for every read/write/flush of it.
_wnba_id_cache = None
_wnba_id_cache_dirty = False
_wnba_id_cache_lock = threading.Lock()


def _get_wnba_id_cache():
    global _wnba_id_cache
    with _wnba_id_cache_lock:
        if _wnba_id_cache is None:
            _wnba_id_cache = _load_player_id_cache()
        return _wnba_id_cache


def flush_wnba_id_cache():
    """Persist any WNBA name->balldontlie-id mappings learned this run to
    output/player_id_cache.json. Call this ONCE, after all WNBA players for
    the run have been processed -- not per-player -- same batch-friendly
    pattern player_id_cache.py's own docstring describes for the MLB path.
    A no-op if nothing new was resolved this run (e.g. every player was
    already cached, or USE_LIVE_DATA is off)."""
    global _wnba_id_cache_dirty
    with _wnba_id_cache_lock:
        if _wnba_id_cache is not None and _wnba_id_cache_dirty:
            _save_player_id_cache(_wnba_id_cache)
            _wnba_id_cache_dirty = False


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
        # Hold the lock across the pacing sleep AND the request itself, so
        # only one thread is ever "in flight" to balldontlie at a time --
        # other threads block here instead of firing simultaneously and
        # tripping the rate limit together.
        with _BDL_LOCK:
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
#
# stats.wnba.com runs on the same backend as stats.nba.com, scoped to the
# WNBA via LeagueID="10" -- free, no API key, same pattern as the MLB Stats
# API trick above. Two real caveats, unlike the MLB path:
#   1. It 403s without browser-like headers (Referer/User-Agent/x-nba-stats-token).
#   2. stats.nba.com is known to block traffic from datacenter/cloud IPs
#      (AWS, DigitalOcean, Heroku, and likely GitHub Actions runners too) --
#      it can work fine locally and still get rejected in CI.
# So every call here is ONE short-timeout attempt, no retries -- on ANY
# failure (403, timeout, unexpected schema) it falls straight back to the
# existing balldontlie path rather than burning time retrying a source
# that's plausibly blocked for the whole run.

WNBA_STATS_BASE = "https://stats.wnba.com/stats"

_wnba_player_index_cache = None  # {name_lower: {"id":, "team_name":, "team_abbr":}}, built once per run


def _wnba_stats_headers():
    return {
        "Host": "stats.wnba.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
        "Referer": "https://www.wnba.com/",
        "Origin": "https://stats.wnba.com",
        "Connection": "keep-alive",
    }


def _wnba_stats_get(endpoint, params, timeout=8):
    """One-shot GET against stats.wnba.com -- no retries, no backoff.
    Raises on any failure; callers are expected to catch and fall back."""
    r = requests.get(
        f"{WNBA_STATS_BASE}/{endpoint}",
        params=params,
        headers=_wnba_stats_headers(),
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def _wnba_resultset_to_dicts(data, result_set_name=None):
    """stats.nba.com/stats.wnba.com responses are columnar: {"resultSets": [
    {"name":, "headers": [...], "rowSet": [[...], ...]}, ...]}. Flatten the
    first (or named) resultSet into a list of {header: value} dicts."""
    result_sets = data.get("resultSets") or data.get("resultSet") or []
    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    rs = None
    if result_set_name:
        rs = next((r for r in result_sets if r.get("name") == result_set_name), None)
    if rs is None and result_sets:
        rs = result_sets[0]
    if not rs:
        return []
    headers = rs.get("headers", [])
    return [dict(zip(headers, row)) for row in rs.get("rowSet", [])]


def _wnba_player_index(season):
    """Builds (once per run) a name-lookup of every current WNBA player ->
    {id, team_name, team_abbreviation} via commonallplayers (LeagueID=10).
    This sidesteps needing a balldontlie-team-id -> NBA-team-id mapping --
    we match on team name string instead, which both sources expose."""
    global _wnba_player_index_cache
    if _wnba_player_index_cache is not None:
        return _wnba_player_index_cache

    data = _wnba_stats_get("commonallplayers", params={
        "LeagueID": "10", "Season": season, "IsOnlyCurrentSeason": "1",
    })
    rows = _wnba_resultset_to_dicts(data, "CommonAllPlayers")
    index = {}
    for row in rows:
        name = (row.get("DISPLAY_FIRST_LAST") or "").strip()
        if not name:
            continue
        index[name.lower()] = {
            "id": row.get("PERSON_ID"),
            "team_id": row.get("TEAM_ID"),
            "team_name": (row.get("TEAM_NAME") or "").strip(),
            "team_abbr": (row.get("TEAM_ABBREVIATION") or "").strip(),
        }
    _wnba_player_index_cache = index
    return index


def get_wnba_games(date_str):
    return _bdl_get("/wnba/v1/games", params={"dates[]": date_str}).get("data", [])


def get_wnba_team_players(team_id, team_name=None):
    """Roster for a team. Tries stats.wnba.com first (matched by team_name,
    since balldontlie team IDs don't map to NBA-stats team IDs), falls back
    to balldontlie on any failure or no match.

    Returns balldontlie-shaped dicts: {"id", "first_name", "last_name", "full_name"}
    so existing call sites don't need to change shape, only pass team_name.
    """
    if team_name:
        try:
            season = str(__import__("datetime").datetime.now().year)
            index = _wnba_player_index(season)
            tname = team_name.strip().lower()
            matches = [
                (name, info) for name, info in index.items()
                if info["team_name"].lower() == tname or tname in info["team_name"].lower()
            ]
            if matches:
                out = []
                for name, info in matches:
                    parts = name.split(" ", 1)
                    out.append({
                        "id": info["id"],
                        "first_name": parts[0] if parts else name,
                        "last_name": parts[1] if len(parts) > 1 else "",
                        "full_name": name.title(),
                    })
                return out
            print(f"  [warn] stats.wnba.com had no roster match for team_name={team_name!r} "
                  f"-- falling back to balldontlie")
        except Exception as e:
            print(f"  [warn] stats.wnba.com roster fetch failed ({type(e).__name__}: {e}) "
                  f"-- falling back to balldontlie")

    return _bdl_get("/wnba/v1/players", params={"team_ids[]": team_id}).get("data", [])


def get_wnba_player_game_logs(player_id, season, limit=10, player_name=None, team_abbr=None):
    """Recent game logs for a player. Tries stats.wnba.com first (resolved
    by player_name, since balldontlie player IDs don't map to NBA-stats
    player IDs), falls back to balldontlie on any failure or no match.

    Returns balldontlie-shaped log dicts: {"pts", "min", "player": {"full_name"},
    "team": {"abbreviation"}} -- the shape live_fetch_wnba_player_prop() and
    discover_wnba_player_props() already expect, so no downstream changes
    needed beyond passing player_name through.
    """
    if player_name:
        try:
            index = _wnba_player_index(season)
            info = index.get(player_name.strip().lower())
            if info and info.get("id"):
                data = _wnba_stats_get("playergamelog", params={
                    "LeagueID": "10", "PlayerID": info["id"], "Season": season,
                    "SeasonType": "Regular Season",
                })
                rows = _wnba_resultset_to_dicts(data, "PlayerGameLog")[:limit]
                if rows:
                    abbr = team_abbr or info.get("team_abbr") or ""

                    def _parse_min(val):
                        # stats.wnba.com returns MIN as "32:15" (mm:ss) or a number.
                        # Convert to float minutes so downstream > 0 comparisons work.
                        if val is None:
                            return 0.0
                        try:
                            s = str(val)
                            if ":" in s:
                                parts = s.split(":")
                                return float(parts[0]) + float(parts[1]) / 60.0
                            return float(s)
                        except (ValueError, TypeError):
                            return 0.0

                    return [
                        {
                            "pts": row.get("PTS", 0),
                            "min": _parse_min(row.get("MIN")),
                            "player": {"full_name": player_name.title()},
                            "team": {"abbreviation": abbr},
                        }
                        for row in rows
                    ]
            print(f"  [warn] stats.wnba.com had no game log match for player_name={player_name!r} "
                  f"-- falling back to balldontlie")
        except Exception as e:
            print(f"  [warn] stats.wnba.com game log fetch failed ({type(e).__name__}: {e}) "
                  f"-- falling back to balldontlie")

    # player_id is the player's NAME at this point (discover_wnba_player_props
    # has no numeric ID to give us), so balldontlie's player_ids[] -- which
    # requires a numeric ID -- 404s if we pass it straight through. Resolve
    # the name to balldontlie's numeric ID first, same pattern as
    # get_mlb_player_search. Only skip resolution if player_id already looks
    # numeric (caller passed a real ID directly).
    bdl_player_id = player_id
    if player_name and not str(player_id).strip().isdigit():
        try:
            matches = get_wnba_player_search(player_name)
        except Exception as e:
            print(f"  [warn] balldontlie player search failed for {player_name!r} "
                  f"({type(e).__name__}: {e}) -- cannot fall back, skipping player")
            return []
        if not matches:
            print(f"  [warn] balldontlie has no player match for {player_name!r} -- skipping player")
            return []
        bdl_player_id = matches[0]["id"]

    return _bdl_get(
        "/wnba/v1/player_stats",
        params={"player_ids[]": bdl_player_id, "seasons[]": season, "per_page": limit},
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




# Module-level cache so we only fetch the Savant leaderboard once per run,
# not once per pitcher -- it's a ~1MB CSV and hitting it 30 times is wasteful.
_savant_stats_cache = {}


def get_savant_pitcher_advanced_stats(pitcher_name, season):
    """
    Pulls CSW%, SwStr%, and K% for a pitcher from Baseball Savant's public
    CSV export -- no API key, no pybaseball, no FanGraphs dependency.

    Savant publishes a stable CSV leaderboard URL that returns directly
    without JS rendering:
      https://baseballsavant.mlb.com/leaderboard/custom?...&type=pitcher&csv=true

    The full leaderboard is fetched once per season per run (cached in
    _savant_stats_cache) then filtered by name for each pitcher, so 30
    pitchers cost one HTTP call, not 30.

    Column names returned by Savant CSV (verified 2026):
      player_name, k_percent, whiff_percent, csw_percent
    These are mapped to the keys advanced_metrics.py expects:
      K%, SwStr%, CSW%
    SIERA is not available from Savant directly -- falls back to None,
    which project_k_pct_advanced() already handles gracefully.
    """
    import pandas as pd
    import io

    cache_key = str(season)
    if cache_key not in _savant_stats_cache:
        url = (
            "https://baseballsavant.mlb.com/leaderboard/custom"
            f"?year={season}&type=pitcher&filter=&min=1&selections=k_percent,"
            "whiff_percent,csw_percent&csv=true"
        )
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        # Normalise column names: lowercase, strip spaces
        df.columns = [c.strip().lower() for c in df.columns]
        _savant_stats_cache[cache_key] = df

    df = _savant_stats_cache[cache_key]

    # Match on player_name column (Savant uses "Last, First" format in some
    # exports and "First Last" in others -- handle both)
    name_col = next((c for c in df.columns if "name" in c), None)
    if name_col is None:
        return None

    last = pitcher_name.strip().split()[-1]
    match = df[df[name_col].str.contains(last, case=False, na=False)]
    if match.empty:
        return None

    # Map Savant column names to what advanced_metrics.py expects
    row = match.iloc[[0]].copy()
    rename = {
        "k_percent": "K%",
        "whiff_percent": "SwStr%",
        "csw_percent": "CSW%",
    }
    for savant_col, pipeline_col in rename.items():
        if savant_col in row.columns:
            row[pipeline_col] = row[savant_col]

    # Savant publishes these as 0-100 percentages (e.g. 22.5 meaning 22.5%).
    # project_k_pct_advanced() and shrink_mlb_k_pct() both expect 0-1 fractions.
    # Without this division, k_pct enters monte_carlo.py as ~22 instead of ~0.22,
    # pegging rng.random() < k_pct as always True and producing over_prob = 1.0
    # on every pick -- the all-overs / exaggerated-edge bug.
    for col in ["K%", "SwStr%", "CSW%"]:
        if col in row.columns:
            row[col] = row[col] / 100.0

    return row


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


def get_wnba_player_search(name):
    """
    balldontlie player search by name, same pattern as get_mlb_player_search,
    against /wnba/v1/players. Used by get_wnba_player_game_logs to resolve a
    player NAME (the only identifier discover_wnba_player_props has) to
    balldontlie's numeric player_id before calling /wnba/v1/player_stats.

    Checks the persistent name->id cache (output/player_id_cache.json, same
    file the MLB path uses) before ever hitting balldontlie -- WNBA rosters
    barely change night to night, so a player resolved once stays resolved
    across runs, cutting the search-call volume (and rate-limit risk) down
    a lot over a season. Keys are prefixed "wnba:" so these entries can
    never collide with MLB's entries (ESPN athlete IDs, a different ID
    system entirely) in that same shared cache file.

    On a cache miss, tries the combined `search` param first. If that comes
    back with zero matches -- observed even for players who are certainly in
    any real WNBA database (A'ja Wilson, Breanna Stewart, etc.), so this
    isn't a "player doesn't exist" case -- falls back to separate
    first_name/last_name params, which the WNBA endpoint documents as
    alternatives to `search`. A successful resolution either way gets
    written back to the persistent cache (flushed to disk once per run via
    flush_wnba_id_cache(), not on every call).
    """
    cache_key = f"wnba:{name.strip().lower()}"
    if cache_key in _player_search_cache:
        return _player_search_cache[cache_key]

    id_cache = _get_wnba_id_cache()
    with _wnba_id_cache_lock:
        cached_id = _lookup_player_id(id_cache, cache_key)
    if cached_id is not None:
        results = [{"id": cached_id}]
        _player_search_cache[cache_key] = results
        return results

    results = _bdl_get("/wnba/v1/players", params={"search": name}).get("data", [])

    if not results:
        parts = name.strip().split()
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            results = _bdl_get(
                "/wnba/v1/players",
                params={"first_name": first, "last_name": last},
            ).get("data", [])

    if results:
        global _wnba_id_cache_dirty
        with _wnba_id_cache_lock:
            _remember_player_id(id_cache, cache_key, results[0]["id"])
            _wnba_id_cache_dirty = True

    _player_search_cache[cache_key] = results
    return results


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
        timeout=10,  # tight timeout -- a slow response here stalls the whole pitcher loop
    )
    r.raise_for_status()
    return r.json()


# ---------- PropLine (alternative odds/props source, being trialed alongside the-odds-api) ----------
# Response format is advertised as drop-in compatible with the-odds-api, so these mirror
# get_odds() / get_event_player_props() above. Set PROPLINE_API_KEY in env -- never hardcode it.
PROPLINE_BASE = "https://api.prop-line.com/v1"


def _propline_key():
    key = os.environ.get("PROP_LINE_API_KEY") or os.environ.get("PROPLINE_API_KEY")
    if not key:
        raise RuntimeError("PROP_LINE_API_KEY (or PROPLINE_API_KEY) not set in environment")
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
        timeout=10,  # tight timeout -- a slow response here stalls the whole pitcher loop
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
