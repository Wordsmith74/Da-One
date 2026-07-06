"""
prop_grader.py

⚠️ NOT WIRED IN — appears superseded by core/historical_grader.py.

Called by main.py --mode grade -- but there is no main.py in this repo.
The grading path actually run by CI (.github/workflows/generate_daily_picks.yml,
`python3 -m core.historical_grader`) grades from output/pick_history.jsonl
using MLB Stats API / WNBA stats endpoints, not this file's ESPN-boxscore
approach, and writes results back to pick_history.jsonl -- not the `bets`
SQL table this file updates via UPDATE bets ... SET status='closed'.

Note: nothing in the current codebase calls grade_player_props() or any
other closer of the `bets` SQL table (see also score_grader.py,
core/results_tracker.py, core/market_scanner.py) -- rows written to `bets`
by run_pipeline.py's log_bet_dict() are opened but never closed by anything
that currently runs. If you want SQL-table consumers like
core/performance_tracker.py and core/slate_versioner.py to work, either this
grading path needs to be wired in, or historical_grader.py needs to also
update the `bets` row when it grades a pick -- these are two different
grading engines and should not both run against the same store.

Auto-grades player prop bets using ESPN game boxscore data.
Supports: pts, reb, ast, 3pm, stl, blk (WNBA/NBA) and
          H, HR, RBI, K (MLB — future use).

Called by main.py --mode grade alongside score_grader.
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# ESPN API helpers (duplicated minimally to keep graders self-contained)
# ---------------------------------------------------------------------------

def _espn_paths(sport: str) -> tuple[str, str]:
    s = sport.upper()
    if s == "MLB":  return ("baseball",   "mlb")
    if s == "WNBA": return ("basketball", "wnba")
    if s == "NBA":  return ("basketball", "nba")
    raise ValueError(f"Unsupported sport: {sport}")


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=12) as r:
        return json.load(r)


def _scoreboard(sport: str, date_ymd: str) -> list[dict]:
    league, sp = _espn_paths(sport)
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/{league}/{sp}/scoreboard"
        f"?dates={date_ymd}"
    )
    try:
        return _get_json(url).get("events", [])
    except Exception as exc:
        print(f"[prop_grader] scoreboard fetch failed ({sport} {date_ymd}): {exc}", flush=True)
        return []


def _game_summary(sport: str, event_id: str) -> dict:
    league, sp = _espn_paths(sport)
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/{league}/{sp}/summary"
        f"?event={event_id}"
    )
    try:
        return _get_json(url)
    except Exception as exc:
        print(f"[prop_grader] summary fetch failed ({sport} {event_id}): {exc}", flush=True)
        return {}


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------

def _event_abbrs(event: dict) -> set[str]:
    abbrs: set[str] = set()
    for comp in event.get("competitions", []):
        for c in comp.get("competitors", []):
            a = c.get("team", {}).get("abbreviation", "")
            if a:
                abbrs.add(a.upper())
    return abbrs


_FINAL_STATE_NAMES = {
    "STATUS_FINAL", "STATUS_FULL_TIME", "STATUS_END_PERIOD", "FINAL", "FULL_TIME",
}
_FINAL_DESCRIPTIONS = {
    "Final", "final", "FINAL", "F",
    "Completed", "completed", "COMPLETED",
    "Game Over", "game over",
    "Closed", "closed",
    "Full Time", "FT",
}


def _is_final(event: dict) -> bool:
    """
    Return True if the ESPN event is definitively finished.

    Checks four independent signals so that any one being true is enough:
      1. status.type.completed boolean  (primary ESPN flag)
      2. status.type.name in known final name set   (STATUS_FINAL, etc.)
      3. status.type.description in known final strings  (Final, Closed, etc.)
      4. status.type.state == "post"  (ESPN post-game state)
    """
    st_type = event.get("status", {}).get("type", {})
    if st_type.get("completed", False):
        return True
    if st_type.get("name", "").upper() in _FINAL_STATE_NAMES:
        return True
    if st_type.get("description", "") in _FINAL_DESCRIPTIONS:
        return True
    if st_type.get("state", "").lower() == "post":
        return True
    return False


# ---------------------------------------------------------------------------
# Player name matching
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z]", " ", name.lower()).strip()


def _name_match(target: str, candidate: str) -> bool:
    """
    Fuzzy last-name match.
    'A. Wilson' matches 'A'ja Wilson', 'N. Collier' matches 'Napheesa Collier'.
    """
    t_parts = _normalize_name(target).split()
    c_parts = _normalize_name(candidate).split()
    if not t_parts or not c_parts:
        return False
    # Last name must match
    if t_parts[-1] != c_parts[-1]:
        return False
    # If target has a first-name initial, check it too
    if len(t_parts) >= 2 and len(c_parts) >= 2:
        return t_parts[0][0] == c_parts[0][0]
    return True


# ---------------------------------------------------------------------------
# Market → ESPN stat column name
# ---------------------------------------------------------------------------

_BBALL_STAT_MAP: dict[str, str] = {
    "pts":        "PTS",
    "points":     "PTS",
    "reb":        "REB",
    "rebounds":   "REB",
    "ast":        "AST",
    "assists":    "AST",
    "3pm":        "3PT",   # ESPN column is "3PT" (made-attempted format)
    "3pt":        "3PT",
    "three":      "3PT",
    "stl":        "STL",
    "steals":     "STL",
    "blk":        "BLK",
    "blocks":     "BLK",
    "to":         "TO",
    "turnover":   "TO",
}

_MLB_STAT_MAP: dict[str, str] = {
    "hits":       "H",
    "hr":         "HR",
    "home_run":   "HR",
    "rbi":        "RBI",
    "strikeout":  "K",
    "k":          "K",
    "runs":       "R",
}


# ---------------------------------------------------------------------------
# Tank01 WNBA fallback — used when ESPN boxscore lookup returns nothing
# ---------------------------------------------------------------------------

# ESPN stat column → Tank01 WNBA player stat key
_T01_WNBA_STAT: dict[str, str] = {
    "PTS": "pts",
    "REB": "reb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "3PT": "3PM",   # made 3-pointers
    "TO":  "tov",
}


def _tank01_wnba_stat(
    player_name: str,
    date_ymd: str,
    team_field: str,
    stat_col: str,
) -> float | None:
    """
    Fallback: look up a WNBA player stat via Tank01 when ESPN fails.

    team_field  — value from wager_details["team"]; may be "MINvLV" or "MIN"
    date_ymd    — YYYYMMDD string (ET date the game was played)
    stat_col    — ESPN-style column name (PTS, REB, …)
    """
    try:
        from core.tank01 import get_wnba_games, get_wnba_boxscore, normalise_wnba_abbr

        t01_key = _T01_WNBA_STAT.get(stat_col.upper())
        if not t01_key:
            return None

        games = get_wnba_games(date_ymd)
        if not games:
            return None

        # Build set of normalised team abbreviations from the bet's team field
        raw_abbrs = [a.upper() for a in team_field.split("v")] if "v" in team_field else [team_field.upper()]
        norm_abbrs = {normalise_wnba_abbr(a) for a in raw_abbrs}

        # Find the matching game ID
        game_id: str | None = None
        for gid, gdata in games.items():
            away = normalise_wnba_abbr(gdata.get("away", ""))
            home = normalise_wnba_abbr(gdata.get("home", ""))
            if norm_abbrs & {away, home}:
                game_id = gid
                break

        if not game_id:
            print(f"[prop_grader][tank01] no WNBA game found for {team_field} on {date_ymd}", flush=True)
            return None

        boxscore = get_wnba_boxscore(game_id)
        if not boxscore:
            return None

        player_stats = boxscore.get("playerStats") or {}
        for _pid, pdata in player_stats.items():
            name = pdata.get("longName", "")
            if _name_match(player_name, name):
                raw = pdata.get(t01_key)
                if raw is None:
                    return float("nan")   # found player, no stat → DNP
                try:
                    return float(raw)
                except (ValueError, TypeError):
                    return None

        return None

    except Exception as exc:
        print(f"[prop_grader][tank01] fallback error: {exc}", flush=True)
        return None


def _market_to_stat(market: str, sport: str) -> str | None:
    m = market.lower().replace(" ", "_").replace("-", "_")
    sp = sport.upper()
    lookup = _BBALL_STAT_MAP if sp in ("WNBA", "NBA") else _MLB_STAT_MAP
    for keyword, stat in lookup.items():
        if keyword in m:
            return stat
    return None


# ---------------------------------------------------------------------------
# Boxscore player stat lookup
# ---------------------------------------------------------------------------

def _find_player_stat(summary: dict, player_name: str, stat_col: str) -> float | None:
    """
    Search ESPN game summary boxscore for a player's stat value.
    Returns float or None if not found.
    """
    boxscore = summary.get("boxscore", {})

    # Basketball: boxscore.players → [{team, statistics: [{names, athletes}]}]
    for team_data in boxscore.get("players", []):
        for stat_group in team_data.get("statistics", []):
            col_names: list[str] = stat_group.get("names", [])
            # Try exact match first, then case-insensitive
            try:
                idx = col_names.index(stat_col)
            except ValueError:
                col_upper = [c.upper() for c in col_names]
                try:
                    idx = col_upper.index(stat_col.upper())
                except ValueError:
                    continue

            for ath_entry in stat_group.get("athletes", []):
                ath  = ath_entry.get("athlete", {})
                name = ath.get("displayName") or ath.get("shortName", "")
                if _name_match(player_name, name):
                    stats = ath_entry.get("stats", [])
                    if not stats:
                        # Empty stats = DNP (Did Not Play) → signal void
                        return float("nan")
                    if idx < len(stats):
                        val = stats[idx]
                        # ESPN formats some stats as "made-attempted" (e.g. "2-4")
                        # For 3PT, FG, FT — take the made (left) count
                        if isinstance(val, str) and "-" in val:
                            val = val.split("-")[0]
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return None

    # MLB: boxscore.teams[*].players[*]  (different structure)
    for team_data in boxscore.get("teams", []):
        for player_data in team_data.get("players", []):
            ath  = player_data.get("athlete", {})
            name = ath.get("displayName") or ath.get("shortName", "")
            if not _name_match(player_name, name):
                continue
            stats_obj = player_data.get("stats", {})
            val = stats_obj.get(stat_col)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None

    return None


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def grade_player_props(sport: str, days_from: int = 2) -> int:
    """
    Grade all open player-prop bets for `sport`.
    Returns the number of props graded.
    """
    from core.results_tracker import _connect, _calculate_profit_loss, init_db
    from core.time_utils import now_utc, convert_to_est

    init_db()
    est_now = convert_to_est(now_utc())
    graded  = 0

    with _connect() as conn:
        open_props = conn.execute("""
            SELECT id, bet_id, sport, wager_details, sportsbook_odds, stake, timestamp
            FROM bets
            WHERE status = 'open'
              AND sport = ?
              AND timestamp >= datetime('now', ?)
              AND wager_details LIKE '%"player":%'
              AND wager_details NOT LIKE '%"player": null%'
              AND wager_details NOT LIKE '%"player":null%'
        """, (sport.upper(), f"-{days_from} days")).fetchall()

    if not open_props:
        return 0

    # Fetch scoreboard data per unique date
    date_events: dict[str, list[dict]] = {}
    for bet in open_props:
        try:
            dt  = datetime.fromisoformat(bet["timestamp"].replace("Z", "+00:00"))
            ymd = dt.astimezone(_EST).strftime("%Y%m%d")
        except Exception:
            ymd = est_now.strftime("%Y%m%d")
        if ymd not in date_events:
            date_events[ymd] = _scoreboard(sport, ymd)

    today_ymd = est_now.strftime("%Y%m%d")
    if today_ymd not in date_events:
        date_events[today_ymd] = _scoreboard(sport, today_ymd)

    # Cache summaries so we don't re-fetch the same game for multiple props
    summary_cache: dict[str, dict] = {}

    for bet in open_props:
        try:
            details   = json.loads(bet["wager_details"] or "{}")
            player    = details.get("player")
            if not player:
                continue

            team      = str(details.get("team", ""))
            market    = str(details.get("market", ""))
            direction = str(details.get("direction", "over")).lower()
            line      = float(details.get("sportsbook_line") or 0)
            bet_id    = bet["bet_id"]

            stat_col  = _market_to_stat(market, sport)
            if not stat_col:
                print(f"[prop_grader] Unknown market '{market}' for {bet_id} — skipping", flush=True)
                continue

            try:
                dt  = datetime.fromisoformat(bet["timestamp"].replace("Z", "+00:00"))
                ymd = dt.astimezone(_EST).strftime("%Y%m%d")
            except Exception:
                ymd = today_ymd

            events = date_events.get(ymd, [])

            # Match game — props use matchup team format "MINvPHX"
            # Also search ALL completed events for the player in case team was
            # logged with incorrect abbreviation (known edge case for player props)
            target  = {a.upper() for a in team.split("v")} if "v" in team else {team.upper()}
            candidates = [ev for ev in events if target & _event_abbrs(ev)]
            if not candidates:
                # Fallback: search all completed games on that date
                candidates = [ev for ev in events if _is_final(ev)]
            if not candidates:
                continue

            # Find the event that actually contains this player
            actual: float | None = None
            found_event_id: str  = ""
            for ev in candidates:
                if not _is_final(ev):
                    continue
                eid = ev["id"]
                if eid not in summary_cache:
                    summary_cache[eid] = _game_summary(sport, eid)
                val = _find_player_stat(summary_cache[eid], player, stat_col)
                if val is not None:
                    actual        = val
                    found_event_id = eid
                    break

            if actual is None and sport.upper() == "WNBA":
                actual = _tank01_wnba_stat(player, ymd, team, stat_col)
                if actual is not None:
                    print(f"[prop_grader][tank01] ✅ WNBA fallback resolved {player} {stat_col}", flush=True)

            if actual is None:
                print(
                    f"[prop_grader] Stat not found: {player} {stat_col} (searched {len(candidates)} event(s))",
                    flush=True,
                )
                continue

            import math
            if math.isnan(actual):
                # Player DNP — void the bet (grade as push, $0 P/L)
                outcome = "push"
                pl      = 0.0
                with _connect() as conn:
                    conn.execute("""
                        UPDATE bets
                        SET actual_outcome = ?, profit_loss = ?, status = 'closed'
                        WHERE id = ? AND status = 'open'
                    """, (outcome, pl, bet["id"]))
                print(f"[prop_grader] 🔄 {bet_id}  {player} DNP → VOID (push)", flush=True)
                graded += 1
                continue

            if actual > line:   outcome = "win"  if direction == "over" else "loss"
            elif actual < line: outcome = "loss" if direction == "over" else "win"
            else:               outcome = "push"

            stake = float(bet["stake"] or 100.0)
            odds  = int(bet["sportsbook_odds"])
            pl    = _calculate_profit_loss(outcome, odds, stake)

            with _connect() as conn:
                conn.execute("""
                    UPDATE bets
                    SET actual_outcome = ?, profit_loss = ?, status = 'closed'
                    WHERE id = ? AND status = 'open'
                """, (outcome, pl, bet["id"]))

            icon = "✅" if outcome == "win" else "❌" if outcome == "loss" else "🔄"
            print(
                f"[prop_grader] {icon} {bet_id}  {player} {stat_col}={actual} vs {line}"
                f" → {outcome.upper()}  P/L={pl:+.2f}",
                flush=True,
            )
            graded += 1

            # ── Fix 6: CLV closing line capture ──────────────────────────────
            try:
                from core.intelligence.clv_tracker import update_closing_line
                update_closing_line(
                    bet_id       = bet_id,
                    closing_odds = int(bet["sportsbook_odds"]),
                    closing_line = line,
                )
            except Exception as _clv_exc:
                print(f"[prop_grader] CLV update skipped for {bet_id}: {_clv_exc}", flush=True)

        except Exception as exc:
            print(f"[prop_grader] Error on {bet.get('bet_id', '?')}: {exc}", flush=True)

    return graded
