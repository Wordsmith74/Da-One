"""
Glue script: chains every layer into one pipeline and writes output/picks.json.

Pipeline order (matters):
  1. Raw stats in (from data/fetch.py in production; mock data here for demo)
  2. Bayesian shrinkage on rate stats (bayesian.py)
  3. Advanced metric blending -- CSW%, SIERA, etc. (advanced_metrics.py)
  4. Ramp-up detection for IL returns (ramp_detection.py)
  5. Season context adjustment, regular vs postseason (season_context.py)
  6. Monte Carlo simulation -> real probabilities (monte_carlo.py)
  7. Edge threshold filter (compare sim probability to market implied probability)
  8. Contradiction check -- drop conflicting/tense picks (contradiction_check.py)
  9. Line movement check -- drop picks where the market moved hard since generation (line_movement.py)
  10. Write output/picks.json for the web page

This file uses MOCK raw inputs (clearly labeled) since this sandbox has no network
access to call real APIs. Swap mock_fetch_*() functions for real data/fetch.py calls
to go live -- everything downstream is unchanged.
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

from fetch import (
    get_mlb_games, get_mlb_player_game_logs, get_wnba_games, get_wnba_player_game_logs,
    get_odds, get_event_player_props, get_savant_pitcher_advanced_stats, get_park_factors,
    get_espn_mlb_scoreboard, get_mlb_player_search, get_wnba_team_players,
    get_mlb_team_recent_runs,
)

from models.bayesian import shrink_mlb_k_pct, shrink_wnba_stat
from models.advanced_metrics import project_k_pct_advanced, pitcher_quality_factor, f5_park_factor
from models.ramp_detection import auto_adjust_workload_input
from models.season_context import detect_phase, adjust_for_postseason
from models.monte_carlo import (
    simulate_f5_game, summarize_f5_total, summarize_f5_moneyline,
    simulate_pitcher_ks, summarize_over_under, simulate_wnba_stat,
    f5_edge_with_uncertainty, k_prop_edge_with_uncertainty, wnba_edge_with_uncertainty,
)
from models.contradiction_check import filter_contradictions
from models.line_movement import apply_line_movement_filter
from models.injury_intel import compute_injury_adjustment
from models.sport_config import MLB, WNBA
from models.handicapper_rules import kelly_stake
from data.cache_history import append_picks

import sys

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

USE_LIVE_DATA = True

# ---------- Structured run log ----------
# Every print() in this file ALSO writes here, so the full run can be dumped
# to output/run_log.json -- when you run this on GitHub Actions and something
# goes wrong, paste that file's content back rather than just the console
# tail; it captures stage-by-stage detail the console scroll may have lost.
RUN_LOG = []


def log(level, stage, message):
    """level: 'info' | 'warn' | 'error'. Always prints AND records structured."""
    entry = {"level": level, "stage": stage, "message": str(message)}
    RUN_LOG.append(entry)
    prefix = {"info": "  ", "warn": "[warn] ", "error": "[ERROR] "}[level]
    print(f"{prefix}[{stage}] {message}")


def run_preflight_checks():
    """
    Run BEFORE touching any live data. Checks the things that fail silently
    or with a confusing traceback otherwise: missing env vars, missing
    packages. Printing this block first means a failed GitHub Actions run
    tells you WHY in the first 10 lines instead of buried in a stack trace
    from deep inside fetch.py.
    """
    print("=== Preflight checks ===")
    ok = True

    for var in ("BALL_DONT_LIE_KEY", "THE_ODDS_API_KEY"):
        if os.environ.get(var):
            log("info", "preflight", f"{var} is set")
        else:
            log("warn", "preflight", f"{var} is NOT set -- any live call needing it will raise RuntimeError")
            if USE_LIVE_DATA:
                ok = False

    for pkg in ("requests", "pandas"):
        try:
            __import__(pkg)
            log("info", "preflight", f"package '{pkg}' importable")
        except ImportError as e:
            log("error" if USE_LIVE_DATA else "warn", "preflight", f"package '{pkg}' missing: {e}")
            if USE_LIVE_DATA:
                ok = False

    # pybaseball and lxml are only needed for the Savant/park-factor paths --
    # warn, don't hard-fail preflight, since F5/K-prop totals can still run
    # without advanced metrics (they fall back to neutral values).
    for pkg in ("pybaseball", "lxml"):
        try:
            __import__(pkg)
            log("info", "preflight", f"package '{pkg}' importable")
        except ImportError as e:
            log("warn", "preflight", f"package '{pkg}' missing: {e} -- "
                f"advanced metrics (CSW%/SIERA/park factors) will fall back to neutral values")

    print(f"=== Preflight {'PASSED' if ok else 'FAILED -- see warnings above'} ===\n")
    return ok


# ---------- Discovery: find today's actual games/matchups (no hardcoded IDs) ----------

def discover_mlb_f5_matchups():
    """Returns today's MLB games as a list, so we don't hardcode team IDs."""
    games = get_mlb_games(TODAY)
    if not games:
        print("  [info] No MLB games found for today.")
    return games


def discover_wnba_matchups():
    games = get_wnba_games(TODAY)
    if not games:
        print("  [info] No WNBA games found for today.")
    return games


def discover_mlb_probable_pitchers():
    """
    Roster resolution for MLB K-props: uses get_espn_mlb_scoreboard (verified
    live in this build) to pull today's probable starters directly, then
    resolves each pitcher's NAME to a balldontlie player_id via
    get_mlb_player_search (needed because live_fetch_mlb_pitcher_k_prop's
    game-log lookup uses balldontlie's IDs, not ESPN's).

    Returns a list of dicts: {"pitcher_id", "pitcher_name", "home_name",
    "away_name"} -- one entry per probable starter found (typically 2 per
    game, home + away). Skips a pitcher gracefully (with a log line, not a
    crash) if ESPN has no probable listed yet or the name search returns
    zero/ambiguous balldontlie matches.
    """
    out = []
    try:
        events = get_espn_mlb_scoreboard()
    except Exception as e:
        print(f"  [warn] ESPN MLB scoreboard fetch failed: {e} -- no probable pitchers resolved")
        return out

    for event in events:
        try:
            competitors = event["competitions"][0]["competitors"]
            home_c = next(c for c in competitors if c.get("homeAway") == "home")
            away_c = next(c for c in competitors if c.get("homeAway") == "away")
            home_name = home_c["team"]["displayName"]
            away_name = away_c["team"]["displayName"]
        except (KeyError, IndexError, StopIteration) as e:
            print(f"  [warn] could not parse competitors for event {event.get('id')}: {e} -- skipping")
            continue

        for side_c, side_label in ((home_c, "home"), (away_c, "away")):
            probables = side_c.get("probables") or []
            if not probables:
                print(f"  [info] no probable pitcher listed yet for {side_label} side of "
                      f"{away_name} @ {home_name} -- skipping that side")
                continue
            athlete = probables[0].get("athlete", {})
            pitcher_name = athlete.get("fullName")
            if not pitcher_name:
                continue
            try:
                matches = get_mlb_player_search(pitcher_name)
            except Exception as e:
                print(f"  [warn] balldontlie player search failed for '{pitcher_name}': {e} -- skipping")
                continue
            if not matches:
                print(f"  [warn] no balldontlie player_id found for probable pitcher "
                      f"'{pitcher_name}' -- skipping (check name spelling/format mismatch)")
                continue
            if len(matches) > 1:
                print(f"  [warn] {len(matches)} balldontlie matches for '{pitcher_name}' -- "
                      f"using the first; verify this is correct for common surnames")
            out.append({
                "pitcher_id": matches[0].get("id"),
                "pitcher_name": pitcher_name,
                "home_name": home_name,
                "away_name": away_name,
            })
    return out


def discover_wnba_player_props(games, max_players_per_team=3):
    """
    Roster resolution for WNBA point props: basketball has no "probable
    starter" concept like MLB pitching, so instead this pulls each team's
    roster (get_wnba_team_players) and, for each player, checks their most
    recent game log for actual minutes played -- skipping anyone who hasn't
    logged minutes recently (inactive/deep bench/injured-and-not-playing),
    then takes up to max_players_per_team by recent minutes per team.

    This makes an extra game-log call per roster player to check recent
    minutes, which is slower than ideal for large rosters -- acceptable for
    a once-a-day pipeline run, but flagged here in case team roster sizes
    make this slow enough to need trimming (e.g. cap at top 8 by listed
    position before checking minutes).
    """
    out = []
    season = TODAY[:4]
    for game in games:
        home_team = game.get("home_team", {}) or {}
        away_team = game.get("away_team", {}) or {}
        for team in (home_team, away_team):
            team_id = team.get("id")
            if not team_id:
                continue
            try:
                players = get_wnba_team_players(team_id)
            except Exception as e:
                print(f"  [warn] WNBA roster fetch failed for team {team_id}: {e} -- skipping team")
                continue

            candidates = []
            for p in players:
                pid = p.get("id")
                if not pid:
                    continue
                try:
                    logs = get_wnba_player_game_logs(pid, season, limit=1)
                except Exception:
                    continue
                if logs and logs[0].get("min", 0) and logs[0]["min"] > 0:
                    candidates.append((pid, logs[0]["min"]))

            candidates.sort(key=lambda t: t[1], reverse=True)
            for pid, _ in candidates[:max_players_per_team]:
                out.append({"player_id": pid, "season": season})

            if not candidates:
                print(f"  [info] no players with recent minutes found for WNBA team {team_id} "
                      f"-- roster data may be stale or this team hasn't played yet this season")
    return out


# ---------- LIVE fetch functions ----------
# NOTE: field names below follow balldontlie's documented schema as of this build,
# but I have NOT been able to verify them against a live response in this sandbox
# (no network access here). First live run: check the printed raw JSON below and
# adjust field names if anything KeyErrors -- they're isolated to these functions.

def find_odds_event(odds_list, home_name, away_name=None):
    """
    Match a discovered game (by canonical team name) to an Odds-API event
    dict. Shared by F5, MLB-K-prop, and WNBA-prop wiring so the matching
    logic (and its caveats) live in exactly one place instead of three
    slightly-different inline lambdas.

    Matches on substring containment in either direction (Odds API team
    names and balldontlie/canonical names don't always match exactly --
    e.g. "Athletics" vs "Oakland Athletics") and, if away_name is given,
    requires the away side to also line up so a date with two same-home
    doubleheaders (rare, but MLB) doesn't grab the wrong game.
    """
    for o in odds_list:
        o_home = o.get("home_team", "")
        o_away = o.get("away_team", "")
        home_match = home_name in o_home or o_home in home_name
        if not home_match:
            continue
        if away_name and not (away_name in o_away or o_away in away_name):
            continue
        return o
    return None


def live_fetch_mlb_f5_matchup(game, season):
    """
    game: one game dict from discover_mlb_f5_matchups() -- already today's real
    game, no separate team-ID lookup needed.
    """
    from data.name_registry import canonical_team, get_unresolved_log

    raw_home = game.get("home_team", {}).get("display_name", game.get("home_team", {}).get("abbreviation", "HOME"))
    raw_away = game.get("away_team", {}).get("display_name", game.get("away_team", {}).get("abbreviation", "AWAY"))

    home_team = canonical_team(raw_home, "mlb")
    away_team = canonical_team(raw_away, "mlb")
    if home_team is None or away_team is None:
        unresolved = [u for u in get_unresolved_log() if u["input"] in (raw_home, raw_away)]
        print(f"  [warn] could not canonicalize team name(s): {unresolved} -- "
              f"add the missing alias to data/name_registry.py rather than guessing downstream")
        # Fall back to raw strings for display only -- but park-factor lookup below
        # requires a canonical match, so it'll correctly stay neutral (1.0) for this game
        home_name, away_name = raw_home, raw_away
        home_park = None
    else:
        home_name, away_name = home_team["full"], away_team["full"]
        home_park = home_team["park"]

    park_factor = 1.0
    try:
        pf_table = get_park_factors(season)
        if pf_table is not None and home_park is not None:
            # Match against the canonical park name, not a raw team-name substring --
            # this is the exact kind of cross-source mismatch the name_registry exists
            # to eliminate. FanGraphs' Guts table column names still need first-run
            # verification (see note below), but the JOIN key is now stable.
            row = pf_table[pf_table.iloc[:, 0].astype(str).str.contains(home_park, case=False, na=False)]
            if len(row) > 0:
                park_factor = float(row.iloc[0].get("Basic", 100)) / 100  # FanGraphs scales as 100=neutral
            else:
                print(f"  [warn] park '{home_park}' not found in Guts table -- using neutral 1.0. "
                      f"NOTE: confirm the exact park-name spelling FanGraphs uses by printing "
                      f"pf_table.iloc[:,0].tolist() on first real run.")
    except Exception as e:
        print(f"  [warn] Park factor lookup failed: {e} -- using neutral 1.0")

    odds_data = []
    try:
        odds_data = get_odds("baseball_mlb", markets="totals")
        # NOTE: The Odds API's default 'totals' market is FULL-GAME, not F5. F5 totals
        # are typically a separate market key on the books that offer them (varies by
        # book -- check your odds provider's market list for the exact F5 market key).
        # This WILL need adjusting once you see real market keys -- print(odds_data)
        # on first run to find the right key for your books.
    except Exception as e:
        print(f"  [warn] Odds fetch failed: {e}")

    # F5 totals market key -- The Odds API doesn't expose a single confirmed
    # universal key for "first 5 innings" across all books; the candidates
    # below are the commonly-documented ones as of 2026-06. Tries each in
    # turn and uses whichever actually returns a totals market for this
    # game. NOT live-verified -- if none of these work for your book
    # selection, print(odds_data) once and add the right key here.
    f5_keys_tried = ["totals_1st_5_innings", "totals_h1", "alternate_totals_1st_5_innings"]
    game_odds = find_odds_event(odds_data, home_name, away_name)
    market_f5_total_line, market_f5_total_odds = None, -110

    if game_odds:
        for book in game_odds.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") in f5_keys_tried:
                    outcomes = market.get("outcomes", [])
                    over = next((o for o in outcomes if o.get("name") == "Over"), None)
                    if over:
                        market_f5_total_line = over.get("point")
                        market_f5_total_odds = over.get("price", -110)
                        break
            if market_f5_total_line is not None:
                break

    if not game_odds or market_f5_total_line is None:
        print(f"  [info] No F5 totals line resolved for {away_name} @ {home_name} "
              f"(tried market keys {f5_keys_tried}) -- skipping this matchup")
        return None

    # Recent team-level runs -- last 5 completed games each, via the new
    # get_mlb_team_recent_runs helper (see data/fetch.py). Falls back to []
    # (same as before) if team IDs aren't resolvable or balldontlie's score
    # field names don't match what this assumes -- process_mlb_f5 already
    # needs a non-empty list to compute an average, so an empty result here
    # safely produces "no pick" rather than a crash or a div-by-zero.
    home_id = game.get("home_team", {}).get("id")
    away_id = game.get("away_team", {}).get("id")
    home_recent_runs, away_recent_runs = [], []
    if home_id is not None:
        home_recent_runs = get_mlb_team_recent_runs(home_id, TODAY, n=5)
    if away_id is not None:
        away_recent_runs = get_mlb_team_recent_runs(away_id, TODAY, n=5)

    # Probable-starter SIERA -- matches this game to today's ESPN scoreboard
    # (already verified live elsewhere in this build) by team name, then
    # looks up each probable's SIERA via pybaseball/Savant. home_opp_pitcher_siera
    # is the AWAY team's starter (the pitcher HOME's hitters face), and vice
    # versa -- matches the naming convention already used by process_mlb_f5's
    # pitcher_quality_factor() calls.
    home_opp_pitcher_siera, away_opp_pitcher_siera = None, None
    try:
        from data.name_registry import canonical_player
        events = get_espn_mlb_scoreboard()
        espn_event = next(
            (e for e in events
             if any(home_name in c.get("team", {}).get("displayName", "") or
                    c.get("team", {}).get("displayName", "") in home_name
                    for c in e.get("competitions", [{}])[0].get("competitors", []))),
            None,
        )
        if espn_event:
            competitors = espn_event["competitions"][0]["competitors"]
            home_c = next(c for c in competitors if c.get("homeAway") == "home")
            away_c = next(c for c in competitors if c.get("homeAway") == "away")

            def _siera_for(side_c):
                probables = side_c.get("probables") or []
                if not probables:
                    return None
                name = probables[0].get("athlete", {}).get("fullName")
                if not name:
                    return None
                match_key = canonical_player(name)["match_key"]
                row = get_savant_pitcher_advanced_stats(match_key, season)
                if row is not None and len(row) > 0:
                    return row.iloc[0].get("SIERA")
                return None

            # home_opp_pitcher_siera = SIERA of the pitcher HOME's hitters face,
            # i.e. the AWAY team's probable starter (and vice versa).
            home_opp_pitcher_siera = _siera_for(away_c)
            away_opp_pitcher_siera = _siera_for(home_c)
    except Exception as e:
        print(f"  [warn] probable-pitcher SIERA lookup failed for {away_name} @ {home_name}: {e} "
              f"-- pitcher_quality_factor() will use its neutral default")

    return {
        "matchup": f"{away_name} @ {home_name}",
        "home_recent_runs": home_recent_runs,
        "away_recent_runs": away_recent_runs,
        "home_opp_pitcher_siera": home_opp_pitcher_siera,
        "away_opp_pitcher_siera": away_opp_pitcher_siera,
        "park_factor_full_game": park_factor,
        "market_f5_total_line": market_f5_total_line,
        "market_f5_total_odds": market_f5_total_odds,
        "_raw_game": game,                # kept for debugging field names on first real run
    }


def live_fetch_mlb_pitcher_k_prop(pitcher_player_id, pitcher_name, opponent_team_id, season,
                                   home_name=None, away_name=None):
    from data.name_registry import canonical_player

    logs = get_mlb_player_game_logs(pitcher_player_id, season, limit=8)
    if not logs:
        raise RuntimeError(f"No game logs found for pitcher {pitcher_player_id}")

    pitcher_display = canonical_player(
        logs[0].get("player", {}).get("full_name", pitcher_name)
    )["display"]

    recent_ks = sum(g.get("strikeouts", 0) for g in logs[:5])
    recent_bf = sum(g.get("batters_faced", 0) for g in logs[:5])
    recent_innings = [g.get("innings_pitched", 0) for g in logs[:5]]
    baseline_innings = [g.get("innings_pitched", 0) for g in logs]

    # Savant/FanGraphs advanced metrics -- public, no key, via pybaseball.
    # Wrapped in try/except: if pybaseball isn't installed yet or the pitcher name
    # match fails (name formatting mismatches are common), fall back to None rather
    # than crash the whole pipeline -- project_k_pct_advanced() already handles
    # missing csw_pct/swstr_pct gracefully by falling back to raw K% alone.
    csw_pct, swstr_pct = None, None
    try:
        # Use the canonicalized match_key (accent-stripped) for the Savant lookup --
        # pybaseball's .str.contains name match is exactly the kind of cross-source
        # join that breaks silently on accented characters (e.g. "Jose" vs "José").
        match_key = canonical_player(pitcher_name)["match_key"]
        savant_row = get_savant_pitcher_advanced_stats(match_key, season)
        if savant_row is not None and len(savant_row) > 0:
            row = savant_row.iloc[0]
            # NOTE: confirm these exact column names match your installed pybaseball
            # version by running print(savant_row.columns.tolist()) once locally --
            # FanGraphs/pybaseball column names for these fields have varied
            # (e.g. "CSW%" vs "CSW_pct") across versions.
            csw_pct = row.get("CSW%")
            swstr_pct = row.get("SwStr%")
    except Exception as e:
        print(f"  [warn] Savant advanced stats unavailable for {pitcher_display}: {e}")

    # Strikeout-prop market line -- The Odds API requires an event_id, which
    # requires matching this game to an odds-API event by team name first
    # (see find_odds_event). home_name/away_name are optional because not
    # every caller has them yet (e.g. if probable-pitcher discovery isn't
    # wired upstream); without them this just leaves the line as None like
    # before, same safe degrade as the rest of this file.
    market_k_line, market_k_odds = None, -110
    if home_name and away_name:
        try:
            game_odds_list = get_odds("baseball_mlb", markets="h2h")
            event = find_odds_event(game_odds_list, home_name, away_name)
            if event and event.get("id"):
                props = get_event_player_props(
                    "baseball_mlb", event["id"], markets="pitcher_strikeouts"
                )
                # Odds-API event-odds shape: {"bookmakers": [{"markets": [{"key":
                # "pitcher_strikeouts", "outcomes": [{"name": "Over", "description":
                # "<Pitcher Name>", "point": 6.5, "price": -115}, ...]}]}]}
                # NOT live-verified in this sandbox -- if this KeyErrors/finds
                # nothing on first real run, print(props) and adjust the walk below.
                for book in props.get("bookmakers", []):
                    for market in book.get("markets", []):
                        if market.get("key") != "pitcher_strikeouts":
                            continue
                        for outcome in market.get("outcomes", []):
                            desc = (outcome.get("description") or "").lower()
                            if pitcher_display.lower() in desc or desc in pitcher_display.lower():
                                if outcome.get("name") == "Over":
                                    market_k_line = outcome.get("point")
                                    market_k_odds = outcome.get("price", -110)
                        if market_k_line is not None:
                            break
                    if market_k_line is not None:
                        break
                if market_k_line is None:
                    print(f"  [info] No pitcher_strikeouts line found for {pitcher_display} "
                          f"in event {event['id']} -- check market key name for your books.")
            else:
                print(f"  [info] No odds event resolved for {away_name} @ {home_name} K-prop")
        except Exception as e:
            print(f"  [warn] K-prop market line fetch failed for {pitcher_display}: {e}")

    return {
        "matchup": f"{away_name} @ {home_name}" if (home_name and away_name) else "TBD",
        "pitcher": pitcher_display,
        "recent_ks": recent_ks, "recent_batters_faced": max(recent_bf, 1),
        "csw_pct": csw_pct, "swstr_pct": swstr_pct,
        "recent_innings": recent_innings, "baseline_innings": baseline_innings,
        "status_history": None,  # TODO: wire injury status endpoint if balldontlie exposes one for MLB
        "market_k_line": market_k_line, "market_k_odds": market_k_odds,
    }


def live_fetch_wnba_player_prop(player_id, season):
    from data.name_registry import canonical_player
    from data.fetch import get_wnba_team_injuries

    logs = get_wnba_player_game_logs(player_id, season, limit=8)
    if not logs:
        raise RuntimeError(f"No game logs found for player {player_id}")

    player_display = canonical_player(
        logs[0].get("player", {}).get("full_name", "Unknown")
    )["display"]

    recent_pts = [g.get("pts", 0) for g in logs[:4]]
    recent_minutes = [g.get("min", 0) for g in logs[:4]]
    baseline_minutes = [g.get("min", 0) for g in logs]
    season_avg_pts = sum(g.get("pts", 0) for g in logs) / len(logs)

    # Injury status -- FREE source (RotoWire scrape, off-ESPN per Wordsmith74
    # port; see data/fetch.get_wnba_team_injuries + data/rotowire_injuries.py).
    # Requires the team abbreviation + full roster names to safely filter the
    # league-wide scrape; if either is missing this just returns None and
    # process_wnba_prop() treats it as "no injury data" (zero adjustment),
    # never a crash.
    team_abbr = (logs[0].get("team") or {}).get("abbreviation")
    injury_data = None
    if team_abbr:
        try:
            roster_names = {
                (g.get("player", {}).get("full_name") or "").strip().lower()
                for g in logs if g.get("player", {}).get("full_name")
            }
            injury_data = get_wnba_team_injuries(team_abbr, roster_names=roster_names)
        except Exception as exc:
            print(f"  [warn] injury fetch failed for {team_abbr}: {exc} -- proceeding unadjusted")

    # Points-prop market line -- same event-id-matching approach as the MLB
    # K-prop wiring above; needs the opponent team name too, which isn't
    # available from game logs alone, so this is best-effort: skipped
    # (line stays None, same safe degrade as before) if matchup isn't TBD.
    market_pts_line, market_pts_odds = None, -110
    matchup = "TBD"
    try:
        from data.name_registry import canonical_team
        home_team_obj = canonical_team(team_abbr, "wnba") if team_abbr else None
        if home_team_obj:
            wnba_odds_list = get_odds("basketball_wnba", markets="h2h")
            # Without the opponent, match on this player's own team appearing
            # as either home or away in today's odds list.
            event = next(
                (o for o in wnba_odds_list
                 if home_team_obj["full"] in o.get("home_team", "")
                 or home_team_obj["full"] in o.get("away_team", "")),
                None,
            )
            if event and event.get("id"):
                matchup = f"{event.get('away_team')} @ {event.get('home_team')}"
                props = get_event_player_props(
                    "basketball_wnba", event["id"], markets="player_points"
                )
                # NOT live-verified in this sandbox -- same shape assumption as
                # the MLB pitcher_strikeouts walk above; adjust if it KeyErrors.
                for book in props.get("bookmakers", []):
                    for market in book.get("markets", []):
                        if market.get("key") != "player_points":
                            continue
                        for outcome in market.get("outcomes", []):
                            desc = (outcome.get("description") or "").lower()
                            if player_display.lower() in desc or desc in player_display.lower():
                                if outcome.get("name") == "Over":
                                    market_pts_line = outcome.get("point")
                                    market_pts_odds = outcome.get("price", -110)
                        if market_pts_line is not None:
                            break
                    if market_pts_line is not None:
                        break
                if market_pts_line is None:
                    print(f"  [info] No player_points line found for {player_display} "
                          f"in event {event['id']} -- check market key name for your books.")
    except Exception as e:
        print(f"  [warn] Points-prop market line fetch failed for {player_display}: {e}")

    return {
        "matchup": matchup,
        "player": player_display,
        "team_abbr": team_abbr,
        "recent_pts": recent_pts, "recent_minutes": recent_minutes,
        "season_avg_pts": round(season_avg_pts, 1),
        "baseline_minutes": baseline_minutes,
        "status_history": None,  # workload ramp-up still uses this; injuries are separate (below)
        "injury_data": injury_data,
        "market_pts_line": market_pts_line, "market_pts_odds": market_pts_odds,
    }


# ---------- MOCK raw inputs (fallback while USE_LIVE_DATA = False, or live fetch fails) ----------

def mock_fetch_mlb_f5_matchup():
    return {
        "matchup": "LAD @ SD",
        "home_recent_runs": [5, 3, 6, 4, 2],
        "away_recent_runs": [3, 4, 2, 5, 3],
        "home_opp_pitcher_siera": 3.95,   # SD starter facing LAD
        "away_opp_pitcher_siera": 4.55,   # LAD starter facing SD
        "park_factor_full_game": 1.05,
        "market_f5_total_line": 4.5,
        "market_f5_total_odds": -110,
    }


def mock_fetch_mlb_pitcher_k_prop():
    return {
        "matchup": "HOU @ SEA",
        "pitcher": "Framber Valdez",
        "recent_ks": 9, "recent_batters_faced": 24,
        "csw_pct": 0.305, "swstr_pct": 0.122,
        "recent_innings": [5.2, 6.0, 5.1, 6.1, 5.0],
        "baseline_innings": [6.0, 5.2, 6.1, 5.8, 6.0, 5.9, 6.2, 5.7],
        "status_history": [{"status": "Active"}, {"status": "Active"}],
        "market_k_line": 5.5,
        "market_k_odds": -120,
    }


def mock_fetch_wnba_player_prop():
    return {
        "matchup": "NY Liberty @ LV Aces",
        "player": "A'ja Wilson",
        "team_abbr": "LVA",
        "recent_pts": [22, 18, 25, 20], "recent_minutes": [30, 28, 32, 29],
        "season_avg_pts": 21.8,
        "baseline_minutes": [31, 30, 32, 29, 31, 30, 33, 31],
        "status_history": None,
        "injury_data": None,  # mock mode: no injury adjustment applied
        "market_pts_line": 21.5,
        "market_pts_odds": -110,
    }


# ---------- Pipeline stages per market ----------

def process_mlb_f5(raw):
    phase = detect_phase(TODAY, "mlb")

    park = f5_park_factor(raw["park_factor_full_game"])
    home_pitch_factor = pitcher_quality_factor(siera=raw["home_opp_pitcher_siera"])
    away_pitch_factor = pitcher_quality_factor(siera=raw["away_opp_pitcher_siera"])

    home_runs = (sum(raw["home_recent_runs"]) / len(raw["home_recent_runs"])) * 0.55 * home_pitch_factor * park
    away_runs = (sum(raw["away_recent_runs"]) / len(raw["away_recent_runs"])) * 0.55 * away_pitch_factor * park

    # Robust, uncertainty-aware edge check -- NOT a single point-estimate Monte
    # Carlo run. A backtest (models/backtest.py) showed that trusting one
    # simulation from one mean estimate produced a ~50-80% false-positive
    # rate on fairly-priced games, because it ignored how uncertain home_runs/
    # away_runs are as estimates. f5_mean_projection_std is a placeholder
    # uncertainty estimate (see sport_config.py) -- a real upgrade would scale
    # it per-team based on actual recent-game sample size rather than using
    # one fixed constant for every matchup.
    robust = f5_edge_with_uncertainty(
        home_runs, MLB["f5_mean_projection_std"],
        away_runs, MLB["f5_mean_projection_std"],
        raw["market_f5_total_line"], raw["market_f5_total_odds"],
        seed=hash(raw["matchup"]) % (10**6),
    )
    edge_pct = robust["mean_edge_pct"]
    side = robust["side"]

    # Still need one representative Monte Carlo draw for display fields
    # (model_number, model_prob) -- the robust check only decides WHETHER to
    # publish, the display numbers come from the point estimate as the most
    # representative single projection.
    sims = simulate_f5_game(home_runs, away_runs, seed=1)
    total_summary = summarize_f5_total(sims, raw["market_f5_total_line"])
    market_implied_over = _american_to_prob(raw["market_f5_total_odds"])

    pick = None
    if abs(edge_pct) >= MLB["edge_threshold_pct"] and robust["agreement_frac"] >= MLB["min_side_agreement_frac"]:
        pick = {
            "sport": "MLB F5", "market": "F5 Total", "matchup": raw["matchup"],
            "pick": f"{side.title()} {raw['market_f5_total_line']}",
            "line": f"{raw['market_f5_total_line']} ({raw['market_f5_total_odds']})",
            "model_number": total_summary["mean_total"],
            "model_prob": total_summary["over_prob"] if side == "over" else total_summary["under_prob"],
            "market_implied_prob": market_implied_over if side == "over" else 1 - market_implied_over,
            "edge_pct": round(edge_pct, 2),
            "side_agreement_frac": round(robust["agreement_frac"], 2),
            "confidence": max(MLB["confidence_floor"], min(MLB["confidence_cap"],
                               round(abs(edge_pct) * MLB["confidence_scale"] + MLB["confidence_floor"]))),
            "season_phase": phase,
            "market_type": "total", "side": side,
            "pick_time_line": raw["market_f5_total_line"], "current_line": raw["market_f5_total_line"],
        }
        pick["stake_pct_bankroll"] = round(
            kelly_stake(pick["model_prob"], raw["market_f5_total_odds"], MLB["kelly_fraction"]) * 100, 2
        )
    return pick


def process_mlb_k_prop(raw):
    phase = detect_phase(TODAY, "mlb")

    shrunk_k_pct = shrink_mlb_k_pct(raw["recent_ks"], raw["recent_batters_faced"])
    adv_k_pct = project_k_pct_advanced(csw_pct=raw["csw_pct"], swstr_pct=raw["swstr_pct"], raw_k_pct=shrunk_k_pct)

    workload = auto_adjust_workload_input(
        recent_values=raw["recent_innings"], baseline_values=raw["baseline_innings"],
        sport="mlb_pitcher", status_history=raw["status_history"],
    )
    avg_ip = workload["adjusted_value"]
    batters_faced_mean = avg_ip * 4.3

    if phase == "postseason":
        adj = adjust_for_postseason(adv_k_pct, "mlb_pitcher", postseason_sample_size=0)
        adv_k_pct = adj["adjusted_value"]

    k_sims = simulate_pitcher_ks(adv_k_pct, batters_faced_mean, seed=2)
    summary = summarize_over_under(k_sims, raw["market_k_line"])

    market_implied_over = _american_to_prob(raw["market_k_odds"])

    # Robust, uncertainty-aware edge check -- a naive single-point edge here
    # has the same false-positive problem the F5 path had (see backtest
    # history); redraw adv_k_pct and batters_faced_mean from their own
    # uncertainty before trusting an edge number.
    robust = k_prop_edge_with_uncertainty(
        adv_k_pct, MLB["k_pct_projection_std"],
        batters_faced_mean, MLB["bf_mean_projection_std"],
        raw["market_k_line"], raw["market_k_odds"],
        seed=hash(raw["matchup"] + raw["pitcher"]) % (10**6),
    )
    edge_pct = robust["mean_edge_pct"]
    side = robust["side"]

    pick = None
    if raw["recent_batters_faced"] < MLB["min_batters_faced_for_k_prop"]:
        print(f"  [skip] {raw['pitcher']}: sample too thin ({raw['recent_batters_faced']} BF < "
              f"{MLB['min_batters_faced_for_k_prop']} minimum) -- not trusting this K prop regardless of edge")
        return pick

    if abs(edge_pct) >= MLB["edge_threshold_pct"] and robust["agreement_frac"] >= MLB["min_side_agreement_frac"]:
        pick = {
            "sport": "MLB Ks", "market": "Pitcher Strikeouts", "matchup": raw["matchup"],
            "player": raw["pitcher"],
            "pick": f"{raw['pitcher']} {side.title()} {raw['market_k_line']}",
            "line": f"{raw['market_k_line']} ({raw['market_k_odds']})",
            "model_number": summary["mean"],
            "model_prob": summary["over_prob"] if side == "over" else summary["under_prob"],
            "market_implied_prob": market_implied_over if side == "over" else 1 - market_implied_over,
            "edge_pct": round(edge_pct, 2),
            "side_agreement_frac": round(robust["agreement_frac"], 2),
            "confidence": max(MLB["confidence_floor"], min(MLB["confidence_cap"],
                               round(abs(edge_pct) * MLB["confidence_scale"] + MLB["confidence_floor"]))),
            "season_phase": phase,
            "ramp_flag": workload["ramp_flag"],
            "market_type": "total", "side": side,
            "pick_time_line": raw["market_k_line"], "current_line": raw["market_k_line"],
        }
        pick["stake_pct_bankroll"] = round(
            kelly_stake(pick["model_prob"], raw["market_k_odds"], MLB["kelly_fraction"]) * 100, 2
        )
    return pick


def process_wnba_prop(raw):
    phase = detect_phase(TODAY, "wnba")

    raw_rate = sum(raw["recent_pts"]) / sum(raw["recent_minutes"])
    shrunk_rate = shrink_wnba_stat(raw_rate * 30, len(raw["recent_pts"]), raw["season_avg_pts"]) / 30

    workload = auto_adjust_workload_input(
        recent_values=raw["recent_minutes"], baseline_values=raw["baseline_minutes"],
        sport="wnba_player", status_history=raw["status_history"],
    )
    minutes_proj = workload["adjusted_value"]

    if phase == "postseason":
        adj = adjust_for_postseason(shrunk_rate, "wnba_player", is_starter_or_high_usage=True, postseason_sample_size=0)
        shrunk_rate = adj["adjusted_value"]

    pts_sims = simulate_wnba_stat(shrunk_rate, minutes_proj, stat_type="wnba_points", seed=3)
    summary = summarize_over_under(pts_sims, raw["market_pts_line"])

    market_implied_over = _american_to_prob(raw["market_pts_odds"])

    # Robust, uncertainty-aware edge check -- minutes volatility (foul trouble,
    # blowout garbage time, role changes) is the dominant real-world risk to
    # a WNBA points prop, so redrawing both rate and minutes from their own
    # uncertainty before trusting an edge number matters even more here than
    # in the MLB paths above.
    robust = wnba_edge_with_uncertainty(
        shrunk_rate, WNBA["rate_per_minute_projection_std"],
        minutes_proj, WNBA["minutes_projection_std"],
        raw["market_pts_line"], raw["market_pts_odds"],
        seed=hash(raw["matchup"] + raw["player"]) % (10**6),
    )
    edge_pct = robust["mean_edge_pct"]
    side = robust["side"]

    # Injury adjustment -- FREE source (RotoWire-free primary, ESPN fallback;
    # see data/fetch.get_wnba_team_injuries + models/injury_intel.py, ported
    # from Wordsmith74). subject_team_is_backed=True because this prop's own
    # team's injuries are what's in raw["injury_data"] -- an injury here hurts
    # THIS player's own prop (less help on the floor, more defensive attention
    # if a teammate is out and usage shifts, etc.), so it reduces edge rather
    # than helping it.
    injury = compute_injury_adjustment(raw.get("injury_data"), subject_team_is_backed=True)
    if injury["edge_adjustment"]:
        edge_pct = round(edge_pct + injury["edge_adjustment"], 2)

    pick = None
    if len(raw["recent_pts"]) < WNBA["min_games_for_player_prop"]:
        print(f"  [skip] {raw['player']}: sample too thin ({len(raw['recent_pts'])} games < "
              f"{WNBA['min_games_for_player_prop']} minimum) -- not trusting this prop regardless of edge")
        return pick

    if abs(edge_pct) >= WNBA["edge_threshold_pct"] and robust["agreement_frac"] >= WNBA["min_side_agreement_frac"]:
        pick = {
            "sport": "WNBA", "market": "Player Points", "matchup": raw["matchup"],
            "player": raw["player"],
            "pick": f"{raw['player']} {side.title()} {raw['market_pts_line']}",
            "line": f"{raw['market_pts_line']} ({raw['market_pts_odds']})",
            "model_number": summary["mean"],
            "model_prob": summary["over_prob"] if side == "over" else summary["under_prob"],
            "market_implied_prob": market_implied_over if side == "over" else 1 - market_implied_over,
            "edge_pct": round(edge_pct, 2),
            "side_agreement_frac": round(robust["agreement_frac"], 2),
            "confidence": max(WNBA["confidence_floor"], min(WNBA["confidence_cap"],
                               round(abs(edge_pct) * WNBA["confidence_scale"] + WNBA["confidence_floor"]))),
            "season_phase": phase,
            "ramp_flag": workload["ramp_flag"],
            "injury_adjustment_pct": injury["edge_adjustment"],
            "injury_note": injury["factor_text"],
            "market_type": "total", "side": side,
            "pick_time_line": raw["market_pts_line"], "current_line": raw["market_pts_line"],
        }
        pick["stake_pct_bankroll"] = round(
            kelly_stake(pick["model_prob"], raw["market_pts_odds"], WNBA["kelly_fraction"]) * 100, 2
        )
    return pick


def _american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


# ---------- Orchestration ----------

def run_pipeline():
    print("=== Running picks pipeline ===\n")
    preflight_ok = run_preflight_checks()
    if USE_LIVE_DATA and not preflight_ok:
        log("error", "pipeline", "Preflight FAILED with USE_LIVE_DATA=True -- aborting before "
                                  "any picks are generated rather than running on missing keys. "
                                  "Set the missing env var(s) above, or set USE_LIVE_DATA=False to "
                                  "run on mock data only.")
        out_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run_log.json"), "w") as f:
            json.dump(RUN_LOG, f, indent=2)
        return
    season = TODAY[:4]

    raw_picks = []

    # ---- MLB F5: loops over every real game today, no hardcoded team IDs ----
    if USE_LIVE_DATA:
        try:
            games = discover_mlb_f5_matchups()
            log("info", "MLB F5", f"{len(games)} game(s) found for {TODAY}")
            for game in games:
                try:
                    raw = live_fetch_mlb_f5_matchup(game, season)
                    if raw is None:
                        continue  # already logged why inside the fetch function
                    pick = process_mlb_f5(raw)
                    if pick:
                        raw_picks.append(pick)
                        log("info", "MLB F5", f"{raw['matchup']}: {pick['pick']} (edge {pick['edge_pct']}%)")
                    else:
                        log("info", "MLB F5", f"{raw['matchup']}: no edge cleared threshold -- skipped")
                except Exception as e:
                    log("error", "MLB F5", f"failed on one game ({type(e).__name__}: {e}) -- skipping just that game")
        except Exception as e:
            log("error", "MLB F5", f"live discovery failed entirely ({type(e).__name__}: {e}) -- falling back to one mock pick")
            pick = process_mlb_f5(mock_fetch_mlb_f5_matchup())
            if pick:
                raw_picks.append(pick)
    else:
        pick = process_mlb_f5(mock_fetch_mlb_f5_matchup())
        if pick:
            raw_picks.append(pick)
            log("info", "MLB F5", f"generated pick: {pick['pick']} (edge {pick['edge_pct']}%)")
        else:
            log("info", "MLB F5", "no edge cleared threshold -- skipped")

    # ---- MLB K props ----
    if USE_LIVE_DATA:
        try:
            pitchers = discover_mlb_probable_pitchers()
            log("info", "MLB K prop", f"{len(pitchers)} probable starter(s) resolved for {TODAY}")
            if not pitchers:
                log("warn", "MLB K prop", "no probable pitchers resolved -- falling back to one mock pick")
                pick = process_mlb_k_prop(mock_fetch_mlb_pitcher_k_prop())
                if pick:
                    raw_picks.append(pick)
            for p in pitchers:
                try:
                    raw = live_fetch_mlb_pitcher_k_prop(
                        p["pitcher_id"], p["pitcher_name"], None, season,
                        home_name=p["home_name"], away_name=p["away_name"],
                    )
                    pick = process_mlb_k_prop(raw)
                    if pick:
                        raw_picks.append(pick)
                        log("info", "MLB K prop", f"{raw['matchup']} ({raw['pitcher']}): "
                                                   f"{pick['pick']} (edge {pick['edge_pct']}%)")
                    else:
                        log("info", "MLB K prop", f"{p['pitcher_name']}: no edge cleared threshold -- skipped")
                except Exception as e:
                    log("error", "MLB K prop", f"failed on {p['pitcher_name']} "
                                                f"({type(e).__name__}: {e}) -- skipping just that pitcher")
        except Exception as e:
            log("error", "MLB K prop", f"live discovery failed entirely ({type(e).__name__}: {e}) -- "
                                        f"falling back to one mock pick")
            pick = process_mlb_k_prop(mock_fetch_mlb_pitcher_k_prop())
            if pick:
                raw_picks.append(pick)
    else:
        pick = process_mlb_k_prop(mock_fetch_mlb_pitcher_k_prop())
        if pick:
            raw_picks.append(pick)
            log("info", "MLB K prop", f"generated pick: {pick['pick']} (edge {pick['edge_pct']}%)")
        else:
            log("info", "MLB K prop", "no edge cleared threshold -- skipped")

    # ---- WNBA props ----
    if USE_LIVE_DATA:
        try:
            games = discover_wnba_matchups()
            targets = discover_wnba_player_props(games)
            log("info", "WNBA prop", f"{len(targets)} player prop target(s) resolved for {TODAY}")
            if not targets:
                log("warn", "WNBA prop", "no player targets resolved -- falling back to one mock pick")
                pick = process_wnba_prop(mock_fetch_wnba_player_prop())
                if pick:
                    raw_picks.append(pick)
            for t in targets:
                try:
                    raw = live_fetch_wnba_player_prop(t["player_id"], t["season"])
                    pick = process_wnba_prop(raw)
                    if pick:
                        raw_picks.append(pick)
                        log("info", "WNBA prop", f"{raw['matchup']} ({raw['player']}): "
                                                  f"{pick['pick']} (edge {pick['edge_pct']}%)")
                    else:
                        log("info", "WNBA prop", f"{raw['player']}: no edge cleared threshold -- skipped")
                except Exception as e:
                    log("error", "WNBA prop", f"failed on player {t['player_id']} "
                                               f"({type(e).__name__}: {e}) -- skipping just that player")
        except Exception as e:
            log("error", "WNBA prop", f"live discovery failed entirely ({type(e).__name__}: {e}) -- "
                                       f"falling back to one mock pick")
            pick = process_wnba_prop(mock_fetch_wnba_player_prop())
            if pick:
                raw_picks.append(pick)
    else:
        pick = process_wnba_prop(mock_fetch_wnba_player_prop())
        if pick:
            raw_picks.append(pick)
            log("info", "WNBA prop", f"generated pick: {pick['pick']} (edge {pick['edge_pct']}%)")
        else:
            log("info", "WNBA prop", "no edge cleared threshold -- skipped")

    log("info", "pipeline", f"{len(raw_picks)} raw picks generated. Running contradiction check...")
    cleaned, dropped_contradiction = filter_contradictions(raw_picks)
    if dropped_contradiction:
        log("warn", "pipeline", f"dropped {len(dropped_contradiction)} pick(s) on contradiction check")

    log("info", "pipeline", f"{len(cleaned)} picks after contradiction check. Running line movement check...")
    final, dropped_line_move = apply_line_movement_filter(cleaned)
    if dropped_line_move:
        log("warn", "pipeline", f"dropped {len(dropped_line_move)} pick(s) on line movement check")

    log("info", "pipeline", f"{len(final)} picks after line movement check. Applying per-sport daily caps...")
    final = _apply_daily_caps(final)
    log("info", "pipeline", f"{len(final)} picks after daily caps.")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "picks": final,
    }

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "picks.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Log every published pick to persistent history -- this is what makes a
    # real (not synthetic) track record possible later via models/grade_results.py.
    # MUST happen now, with pick-time odds locked in, before the market moves.
    if final:
        append_picks(final, generated_at=output["generated_at"])
        log("info", "pipeline", f"appended {len(final)} pick(s) to output/pick_history.jsonl")

    # Always write the run log, success or failure -- this is the file to
    # paste back if a GitHub Actions run produces something unexpected.
    log_path = os.path.join(out_dir, "run_log.json")
    n_errors = sum(1 for e in RUN_LOG if e["level"] == "error")
    n_warnings = sum(1 for e in RUN_LOG if e["level"] == "warn")
    with open(log_path, "w") as f:
        json.dump({
            "run_at": datetime.now(timezone.utc).isoformat(),
            "use_live_data": USE_LIVE_DATA,
            "n_errors": n_errors, "n_warnings": n_warnings,
            "n_picks_generated": len(final),
            "entries": RUN_LOG,
        }, f, indent=2)

    print(f"\nWrote {len(final)} final picks to {out_path}")
    print(f"Wrote run log ({n_errors} errors, {n_warnings} warnings) to {log_path}")
    if n_errors > 0:
        print("\n*** This run had errors -- paste output/run_log.json back for debugging. ***")
    return output


def _apply_daily_caps(picks):
    """
    Caps total published picks per sport per day (models.sport_config ->
    max_picks_per_day) -- top handicappers are selective, not exhaustive;
    publishing every pick that clears a thin edge threshold is itself a tell
    of an undisciplined model. MLB F5 and MLB Ks share the MLB cap (same
    sport, same bankroll); WNBA has its own, smaller cap.
    """
    sport_to_cfg = {"MLB F5": MLB, "MLB Ks": MLB, "WNBA": WNBA}
    by_cfg_group = {}
    for p in picks:
        cfg = sport_to_cfg.get(p.get("sport"))
        group_key = "MLB" if cfg is MLB else "WNBA" if cfg is WNBA else "OTHER"
        by_cfg_group.setdefault(group_key, []).append(p)

    capped = []
    for group_key, group_picks in by_cfg_group.items():
        cap = MLB["max_picks_per_day"] if group_key == "MLB" else (
            WNBA["max_picks_per_day"] if group_key == "WNBA" else len(group_picks)
        )
        group_picks.sort(key=lambda p: abs(p.get("edge_pct", 0)), reverse=True)
        kept = group_picks[:cap]
        dropped_n = len(group_picks) - len(kept)
        if dropped_n > 0:
            print(f"  [{group_key}] capped at {cap}/day -- dropped {dropped_n} lower-edge pick(s)")
        capped.extend(kept)
    return capped


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as fatal:
        # Last-resort guarantee: even if something escaped every try/except
        # above (e.g. a crash during preflight or in json.dump itself),
        # write SOMETHING to run_log.json so a GitHub Actions failure isn't
        # a pure black box -- this is the file to paste back.
        log("error", "fatal", f"uncaught exception: {type(fatal).__name__}: {fatal}")
        out_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run_log.json"), "w") as f:
            json.dump({
                "run_at": datetime.now(timezone.utc).isoformat(),
                "use_live_data": USE_LIVE_DATA,
                "fatal_error": True,
                "entries": RUN_LOG,
            }, f, indent=2)
        raise
