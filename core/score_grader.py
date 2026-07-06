"""
score_grader.py

⚠️ NOT WIRED IN — appears superseded by core/historical_grader.py.

Called by main.py --mode grade -- but there is no main.py in this repo.
See the identical note in core/prop_grader.py: this file is one of several
(along with prop_grader.py, core/results_tracker.py, core/market_scanner.py)
that can close a row in the `bets` SQL table, but nothing currently calls
any of them, so `bets` rows opened by run_pipeline.py's log_bet_dict() stay
'open' forever. The grading path CI actually runs
(`python3 -m core.historical_grader`) grades a separate store
(output/pick_history.jsonl) and does not touch the `bets` table at all.

Auto-grades open bets against ESPN final / live scores.
Handles full-game totals, spreads, and in-game markets:
  NRFI  — no run first inning
  YRFI  — yes run first inning
  F5    — first-5-innings total
  Q1    — first-quarter total
  H1    — first-half total

Called by main.py --mode grade, which is fired by the scheduler
every 10 minutes from 7–11 PM ET.
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# ESPN API helpers
# ---------------------------------------------------------------------------

def _espn_paths(sport: str) -> tuple[str, str]:
    s = sport.upper()
    if s == "MLB":  return ("baseball",    "mlb")
    if s == "WNBA": return ("basketball",  "wnba")
    if s == "NBA":  return ("basketball",  "nba")
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
        print(f"[score_grader] scoreboard fetch failed ({sport} {date_ymd}): {exc}", flush=True)
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
        print(f"[score_grader] summary fetch failed ({sport} {event_id}): {exc}", flush=True)
        return {}


# ---------------------------------------------------------------------------
# Event parsing helpers
# ---------------------------------------------------------------------------

# Engine abbreviation → ESPN abbreviation aliases.
# ESPN sometimes uses different short-codes than our odds-feed or engine.
_ABBR_ALIASES: dict[str, str] = {
    # Engine abbrev → ESPN abbrev  (for teams whose names differ between systems)
    # WNBA original franchises
    "WAS": "WSH",   # Washington Mystics  (our WAS → ESPN WSH)
    "LAS": "LA",    # LA Sparks           (our LAS → ESPN LA)
    "LVA": "LV",    # Las Vegas Aces      (our LVA → ESPN LV)
    "NYL": "NY",    # New York Liberty    (our NYL → ESPN NY)
    # WNBA 2026 expansion
    "GSV": "GS",    # Golden State Valkyries (our GSV → ESPN GS)
    # POR, TOR match ESPN directly — no alias needed
    # MLB — Athletics relocated to Sacramento; Odds API uses OAK, ESPN uses ATH
    "OAK": "ATH",   # Sacramento Athletics (our OAK → ESPN ATH)
    # MLB — White Sox: Odds API / engine uses CWS, ESPN uses CHW
    "CWS": "CHW",   # Chicago White Sox (our CWS → ESPN CHW)
}
# Reverse map so ESPN→engine lookups also work
_ABBR_ALIASES.update({v: k for k, v in list(_ABBR_ALIASES.items())})


def _event_abbrs(event: dict) -> set[str]:
    """All team abbreviations in an ESPN event, expanded with known aliases."""
    abbrs: set[str] = set()
    for comp in event.get("competitions", []):
        for c in comp.get("competitors", []):
            a = c.get("team", {}).get("abbreviation", "")
            if a:
                a = a.upper()
                abbrs.add(a)
                if a in _ABBR_ALIASES:
                    abbrs.add(_ABBR_ALIASES[a].upper())
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


def _final_scores(event: dict) -> tuple[int, int, bool]:
    """(home_score, away_score, is_final)"""
    comp     = event.get("competitions", [{}])[0]
    is_final = _is_final(event)
    home = away = 0
    for c in comp.get("competitors", []):
        try:
            score = int(c.get("score", 0) or 0)
        except (ValueError, TypeError):
            score = 0
        if c.get("homeAway") == "home":
            home = score
        else:
            away = score
    return home, away, is_final


def _inning_totals(sport: str, event_id: str, through_inning: int) -> tuple[int | None, bool]:
    """
    Returns (total_runs_through_inning, is_complete) via ESPN game summary.
    Tries multiple paths ESPN uses for linescore data.
    """
    data = _game_summary(sport, event_id)

    # Path 1: header.competitions[0].linescores
    for comp in data.get("header", {}).get("competitions", [{}]):
        linescores = comp.get("linescores", [])
        if linescores:
            return _sum_linescores(linescores, through_inning)

    # Path 2: boxscore.teams[*].linescores (some sports)
    for team in data.get("boxscore", {}).get("teams", []):
        linescores = team.get("linescores", [])
        if linescores:
            return _sum_linescores(linescores, through_inning)

    return None, False


def _sum_linescores(linescores: list[dict], through_inning: int) -> tuple[int, bool]:
    total   = 0
    max_per = 0
    for ls in linescores:
        per = ls.get("period", ls.get("id", 0))
        if isinstance(per, str):
            try: per = int(per)
            except ValueError: per = 0
        val = ls.get("value", ls.get("displayValue", 0))
        try: val = int(val)
        except (ValueError, TypeError): val = 0
        if per <= through_inning:
            total += val
        max_per = max(max_per, per)
    return total, max_per >= through_inning


# ---------------------------------------------------------------------------
# Market-type detection
# ---------------------------------------------------------------------------

def _market_type(market: str, bet_id: str) -> str:
    m = market.lower().replace(" ", "_").replace("-", "_")
    b = bet_id.lower()
    if "nrfi"       in m or "nrfi"       in b: return "nrfi"
    if "yrfi"       in m or "yrfi"       in b: return "yrfi"
    if "f5"         in m or "f5"         in b: return "f5"
    if "first_5"    in m or "1st_5"      in m: return "f5"
    if "q1"         in m or "q1"         in b: return "q1"
    if "1st_quarter" in m:                     return "q1"
    if "h1"         in m or "h1"         in b: return "h1"
    if "1st_half"   in m or "first_half" in m: return "h1"
    if "spread"     in m or "cover"      in m: return "spread"
    return "total"


# ---------------------------------------------------------------------------
# Grading logic
# ---------------------------------------------------------------------------

def _grade_over_under(actual: int | float, line: float, direction: str) -> str:
    d = direction.lower()
    if d == "over":
        if actual > line: return "win"
        if actual < line: return "loss"
    else:
        if actual < line: return "win"
        if actual > line: return "loss"
    return "push"


def _grade_event(
    bet_id: str,
    market: str,
    direction: str,
    line: float,
    team: str,
    sport: str,
    event: dict,
) -> str | None:
    """
    Return 'win' | 'loss' | 'push' | None (not yet gradeable).
    """
    home, away, is_final = _final_scores(event)
    total    = home + away
    mtype    = _market_type(market, bet_id)

    if mtype in ("total",):
        if not is_final:
            return None
        return _grade_over_under(total, line, direction)

    elif mtype in ("q1", "h1"):
        # Simplified: grade at full-game final; upgrade to period data later
        if not is_final:
            return None
        return _grade_over_under(total, line, direction)

    elif mtype == "f5":
        if is_final:
            runs5, done = _inning_totals(sport, event["id"], 5)
            if runs5 is not None:
                return _grade_over_under(runs5, line, direction)
            # No per-inning data available — skip rather than wrong-grade
            return None
        return None

    elif mtype == "nrfi":
        runs1, done = _inning_totals(sport, event["id"], 1)
        if runs1 is None or not done:
            return None
        return "win" if runs1 == 0 else "loss"

    elif mtype == "yrfi":
        runs1, done = _inning_totals(sport, event["id"], 1)
        if runs1 is None or not done:
            return None
        return "loss" if runs1 == 0 else "win"

    elif mtype == "spread":
        if not is_final:
            return None
        # Find team scores from competitors list
        comp = event.get("competitions", [{}])[0]
        target = {a.upper() for a in team.split("v")} if "v" in team else {team.upper()}
        our_score: int | None = None
        opp_score: int | None = None
        for c in comp.get("competitors", []):
            ta = c.get("team", {}).get("abbreviation", "").upper()
            try: sc = int(c.get("score", 0) or 0)
            except: sc = 0
            if ta in target:
                our_score = sc
            else:
                opp_score = sc
        if our_score is None or opp_score is None:
            return None
        adjusted = our_score + line
        if adjusted > opp_score:   return "win"
        if adjusted < opp_score:   return "loss"
        return "push"

    return None


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def grade_settled_bets(sport: str, days_from: int = 2) -> int:
    """
    Grade all open non-prop bets for `sport` from the last `days_from` days.
    Returns the number of bets graded.
    """
    from core.results_tracker import _connect, _calculate_profit_loss, init_db
    from core.time_utils import now_utc, convert_to_est

    init_db()
    est_now = convert_to_est(now_utc())
    graded  = 0

    with _connect() as conn:
        open_bets = conn.execute("""
            SELECT id, bet_id, sport, wager_details, sportsbook_odds, stake, timestamp
            FROM bets
            WHERE status = 'open'
              AND sport = ?
              AND timestamp >= datetime('now', ?)
        """, (sport.upper(), f"-{days_from} days")).fetchall()

    if not open_bets:
        return 0

    # Collect ESPN scoreboard data for all relevant dates
    date_events: dict[str, list[dict]] = {}
    for bet in open_bets:
        try:
            dt = datetime.fromisoformat(bet["timestamp"].replace("Z", "+00:00"))
            ymd = dt.astimezone(_EST).strftime("%Y%m%d")
        except Exception:
            ymd = est_now.strftime("%Y%m%d")
        if ymd not in date_events:
            date_events[ymd] = _scoreboard(sport, ymd)

    today_ymd = est_now.strftime("%Y%m%d")
    if today_ymd not in date_events:
        date_events[today_ymd] = _scoreboard(sport, today_ymd)

    for bet in open_bets:
        try:
            details   = json.loads(bet["wager_details"] or "{}")
            player    = details.get("player")
            if player:
                continue  # player props handled by prop_grader

            team      = str(details.get("team", ""))
            market    = str(details.get("market", ""))
            direction = str(details.get("direction", "over")).lower()
            line      = float(details.get("sportsbook_line") or 0)
            bet_id    = bet["bet_id"]

            try:
                dt  = datetime.fromisoformat(bet["timestamp"].replace("Z", "+00:00"))
                ymd = dt.astimezone(_EST).strftime("%Y%m%d")
            except Exception:
                ymd = today_ymd

            events = date_events.get(ymd, [])

            # Find matching ESPN event by team abbreviation
            target = {a.upper() for a in team.split("v")} if "v" in team else {team.upper()}
            matched = next((ev for ev in events if target & _event_abbrs(ev)), None)
            if not matched:
                continue

            outcome = _grade_event(bet_id, market, direction, line, team, sport, matched)
            if outcome is None:
                continue

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
            print(f"[score_grader] {icon} {bet_id} → {outcome.upper()}  P/L={pl:+.2f}", flush=True)
            graded += 1

            # ── Fix 6: CLV closing line capture ──────────────────────────────
            # Use the stored sportsbook odds as the closing-line proxy.
            # (Game has ended so the recorded odds are effectively the closing
            #  line for CLV computation purposes.)
            try:
                from core.intelligence.clv_tracker import update_closing_line
                update_closing_line(
                    bet_id       = bet_id,
                    closing_odds = int(bet["sportsbook_odds"]),
                    closing_line = line,
                )
            except Exception as _clv_exc:
                print(f"[score_grader] CLV update skipped for {bet_id}: {_clv_exc}", flush=True)

        except Exception as exc:
            print(f"[score_grader] Error on {bet.get('bet_id', '?')}: {exc}", flush=True)

    return graded
