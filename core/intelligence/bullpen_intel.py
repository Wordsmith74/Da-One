"""
core/intelligence/bullpen_intel.py

Bullpen-quality league_mean adjustment for MLB full-game totals.

Companion to pitcher_intel.py (starter FIP prior shift). pitcher_intel.py
only prices in the starting pitcher; it has no visibility into what happens
after the starter exits. For a full-game Over/Under, the bullpen typically
covers 3-4+ innings per side, so a team running elite relievers behind an
average starter (or vice versa) was previously invisible to the model.

Metrics covered
----------------
Solid — computed directly from MLB Stats API team pitching game logs,
same data-quality tier as the rest of this codebase's MLB models:
  - Bullpen ERA               (rolling 14-day, prorated bullpen share)
  - Bullpen FIP
  - Bullpen xFIP               (FIP with HR component regressed toward
                                 league-average HR/9; team gameLog has no
                                 flyball data, so this is HR-regressed FIP,
                                 not a true flyball-rate xFIP)
  - Bullpen WHIP
  - Bullpen K rate              (K/9, proxy — batters-faced not exposed
                                 at this endpoint)
  - Bullpen BB rate             (BB/9)
  - Last-7-day bullpen performance (separate window from the 14-day
                                 quality sample and the 3-day fatigue read)
  - Last-14-day performance
  - Projected bullpen innings   (9 - starter's expected_ip)
  - Home/away bullpen splits    (ERA split by whether the game log entry
                                 was a home or away appearance)

Heuristic / best-effort — no per-reliever role feed exists in this codebase
(no bullpen "closer"/"setup" designation, no handedness splits ingested
anywhere else). These are implemented as a documented approximation and
should be spot-checked against a live API response before being trusted
in production, since this module was built without network access to
verify the exact roster/leaders endpoint shapes:
  - Closer availability          (saves leader; flagged unavailable if he
                                 pitched on each of the last 2 days)
  - High-leverage reliever availability (same heuristic, applied to the
                                 team's top-2 relief appearance-count arms)
  - Setup reliever availability  (best-effort proxy, same mechanism)
  - Left/right bullpen splits    (attempted via statSplits sitCodes;
                                 returns None on any endpoint mismatch
                                 rather than fabricating a number)

Method (core quality/adjustment path)
--------------------------------------
1. Pull each team's last _ROLLING_DAYS of team-level pitching game logs
   from the MLB Stats API (same endpoint pitcher_workload.py uses for its
   3-day fatigue component, just a wider window).
2. Per game log entry, approximate the bullpen's share of that game:
       sp_ip  = min(total_ip, LEAGUE_AVG_IP) if the team started a pitcher
       bp_ip  = total_ip - sp_ip
       share  = bp_ip / total_ip
   Earned runs / HR / BB / HBP / K / Hits are prorated by `share`. This is
   an approximation (the box-score endpoint doesn't split pitcher-by-
   pitcher), but it is the same level of approximation pitcher_workload.py
   already relies on for bullpen fatigue, and it is far better than the
   zero signal the full-game total model currently uses.
3. Aggregate across the window into ERA / FIP / xFIP / WHIP / K9 / BB9.
4. Also compute a short (_FATIGUE_DAYS) window of raw bullpen IP as a
   fatigue signal, and a separate 7-day performance snapshot.
5. Convert (bullpen_quality - league_bullpen_baseline) into a run-total
   adjustment, scaled by how many innings that bullpen is actually
   projected to pitch tonight (9 - starter's expected_ip, from
   pitcher_workload.get_game_workload_pair).

Public API
----------
get_bullpen_intel(home_abbr, away_abbr, game_date=None) -> BullpenIntelFactor

BullpenIntelFactor carries `league_mean_adjustment`, ready to be added to
c["league_mean"] alongside pitcher_intel's adjustment, plus per-team
diagnostic fields (bullpen_era, bullpen_fatigue, bullpen_score) that
satisfy core/integrity_filters.py's MLB bullpen-score requirement.

Any failure anywhere in this module returns a zero-adjustment factor —
the engine must never block on bullpen data being unavailable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from urllib import request as urllib_req
from urllib.error import URLError

logger = logging.getLogger("betting_bot")

# ---------------------------------------------------------------------------
# Process-level caches (live for the duration of one picks run)
# ---------------------------------------------------------------------------

_RESULT_CACHE: dict[str, "BullpenIntelFactor"] = {}          # "AWAY@HOME_YYYY-MM-DD"
_TEAM_BP_CACHE: dict[tuple[str, str], dict | None] = {}      # (team_abbr, date_str) -> raw agg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://statsapi.mlb.com/api/v1"
_TIMEOUT  = 6  # seconds per HTTP request

LEAGUE_AVG_IP = 5.5  # average SP innings per start; mirrors pitcher_workload.py

# League bullpen baselines (2025-2026 MLB). Configurable — not a precise
# constant, just the reference point quality is measured against.
LEAGUE_BULLPEN_ERA = 4.10
LEAGUE_BULLPEN_FIP = 4.15
_FIP_CONSTANT      = 3.10

# Rolling windows
_ROLLING_DAYS = 14   # bullpen quality sample window
_PERF_7D_DAYS = 7    # separate last-7-day performance window
_FATIGUE_DAYS = 3    # recent-usage fatigue window
_LEAGUE_BP_IP_PER_DAY = 3.5   # rough league-average bullpen IP/team/day
_LEAGUE_BULLPEN_HR9   = 1.15  # league-avg bullpen HR/9; xFIP regresses HR toward this

# Adjustment geometry
_ADJ_MIN = -2.0   # floor: don't let bullpen alone swing a total more than this
_ADJ_MAX = 3.0
_FATIGUE_ADJ_MIN = -0.30
_FATIGUE_ADJ_MAX = 0.40

# Minimum sample before trusting the rolling bullpen ERA/FIP over league baseline
_MIN_SAMPLE_IP = 8.0

# Closer/high-leverage/setup availability requires one roster call + one
# game-log call per pitcher on each team's active roster (~26 calls/game).
# That's unverified against a live response and expensive at pipeline
# scale (15 MLB games/day), so it's off by default. Flip on once the
# endpoint shapes in _fetch_reliever_recent_appearances have been
# smoke-tested against the real API.
ENABLE_LEVERAGE_AVAILABILITY = False

# Left/right bullpen split similarly untested — cheap (1 call/team) but
# the statSplits sitCodes response shape is unverified. Off by default.
ENABLE_HANDEDNESS_SPLIT = False

_MLB_TEAM_IDS: dict[str, int] = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KC":  118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD":  135, "SF":  137, "SEA": 136,
    "STL": 138, "TB":  139, "TEX": 140, "TOR": 141, "WSH": 120,
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BullpenIntelFactor:
    # ── Core quality (14-day rolling) ───────────────────────────────────
    home_bullpen_era:       float | None = None
    home_bullpen_fip:       float | None = None
    home_bullpen_xfip:      float | None = None
    home_bullpen_whip:      float | None = None
    home_bullpen_k9:        float | None = None   # strikeout rate proxy
    home_bullpen_bb9:       float | None = None   # walk rate proxy
    home_bp_ip_sample:      float        = 0.0
    home_bp_ip_expected:    float        = 0.0    # projected BP innings tonight
    home_bullpen_fatigue:   float        = 0.0    # IP over/under league avg, last 3d
    home_bullpen_score:     float        = 50.0   # 0-100, 100 = elite

    away_bullpen_era:       float | None = None
    away_bullpen_fip:       float | None = None
    away_bullpen_xfip:      float | None = None
    away_bullpen_whip:      float | None = None
    away_bullpen_k9:        float | None = None
    away_bullpen_bb9:       float | None = None
    away_bp_ip_sample:      float        = 0.0
    away_bp_ip_expected:    float        = 0.0
    away_bullpen_fatigue:   float        = 0.0
    away_bullpen_score:     float        = 50.0

    # ── Last-7-day performance snapshot (separate from 14-day sample) ──
    home_bullpen_era_7d:    float | None = None
    away_bullpen_era_7d:    float | None = None
    home_bp_ip_7d:          float        = 0.0
    away_bp_ip_7d:          float        = 0.0

    # ── Home/away split (ERA in home games vs away games this season) ──
    home_bullpen_era_at_home: float | None = None
    home_bullpen_era_on_road: float | None = None
    away_bullpen_era_at_home: float | None = None
    away_bullpen_era_on_road: float | None = None

    # ── Heuristic / best-effort — see module docstring ──────────────────
    home_closer_available:        bool | None = None
    away_closer_available:        bool | None = None
    home_high_leverage_available: bool | None = None
    away_high_leverage_available: bool | None = None
    home_setup_available:         bool | None = None
    away_setup_available:         bool | None = None
    home_bullpen_era_vs_lhb:      float | None = None
    home_bullpen_era_vs_rhb:      float | None = None
    away_bullpen_era_vs_lhb:      float | None = None
    away_bullpen_era_vs_rhb:      float | None = None

    league_mean_adjustment: float = 0.0
    factor_text:            str   = ""

    # Combined diagnostics for integrity_filters.py (bullpen_score /
    # bullpen_fatigue / bullpen_era just need to be non-None).
    @property
    def bullpen_score(self) -> float:
        return round((self.home_bullpen_score + self.away_bullpen_score) / 2.0, 1)

    @property
    def bullpen_fatigue(self) -> float:
        return round(self.home_bullpen_fatigue + self.away_bullpen_fatigue, 2)

    @property
    def bullpen_era(self) -> float | None:
        if self.home_bullpen_era is None or self.away_bullpen_era is None:
            return None
        return round((self.home_bullpen_era + self.away_bullpen_era) / 2.0, 2)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _fetch(url: str) -> dict | list | None:
    try:
        with urllib_req.urlopen(url, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except (URLError, OSError, json.JSONDecodeError) as exc:
        logger.debug(f"[bullpen_intel] HTTP error: {exc!r}")
        return None


def _parse_ip(ip_str: str | None) -> float:
    if not ip_str:
        return 0.0
    try:
        parts = str(ip_str).split(".")
        full  = int(parts[0])
        outs  = int(parts[1]) if len(parts) > 1 else 0
        return round(full + outs / 3.0, 4)
    except (ValueError, IndexError):
        return 0.0


def _fip(hr: float, bb: float, hbp: float, k: float, ip: float) -> float | None:
    if ip <= 0:
        return None
    return round((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + _FIP_CONSTANT, 2)


def _fetch_team_gamelog(team_abbr: str, season: int) -> list[dict]:
    """Fetch a team's pitching game log for the season (unfiltered by date)."""
    team_id = _MLB_TEAM_IDS.get(team_abbr.upper())
    if not team_id:
        return []
    url = (
        f"{_BASE_URL}/teams/{team_id}/stats"
        f"?stats=gameLog&group=pitching&season={season}"
    )
    data = _fetch(url)
    if not data or not isinstance(data, dict):
        return []
    splits: list[dict] = []
    for block in data.get("stats", []):
        splits.extend(block.get("splits", []))
    return splits


def _aggregate_bullpen(
    team_abbr: str,
    game_date: date,
    days: int,
    home_filter: bool | None = None,
) -> dict:
    """
    Aggregate approximate bullpen-only stats over the trailing `days` window.

    Returns dict with keys: ip, er, hr, bb, hbp, k, h (all bullpen-share
    prorated). Zero values when no data is available.

    home_filter: None = all games, True = home games only, False = away
    games only. Used for the home/away bullpen split. The MLB Stats API
    game-log split includes an "isHome" flag on most seasons; when that
    key is absent the entry is skipped for filtered aggregations rather
    than guessed at.
    """
    cache_key = (team_abbr.upper(), f"{game_date.isoformat()}_{days}_{home_filter}")
    if cache_key in _TEAM_BP_CACHE:
        cached = _TEAM_BP_CACHE[cache_key]
        return cached if cached is not None else {}

    splits = _fetch_team_gamelog(team_abbr, game_date.year)
    if not splits:
        _TEAM_BP_CACHE[cache_key] = None
        return {}

    cutoff = game_date - timedelta(days=days)
    agg = {"ip": 0.0, "er": 0.0, "hr": 0.0, "bb": 0.0, "hbp": 0.0, "k": 0.0, "h": 0.0}

    for split in splits:
        raw_date = split.get("date", "")
        try:
            split_date = date.fromisoformat(raw_date[:10])
        except ValueError:
            continue
        if not (cutoff <= split_date < game_date):
            continue

        if home_filter is not None:
            is_home = split.get("isHome")
            if is_home is None:
                continue
            if bool(is_home) != home_filter:
                continue

        stat     = split.get("stat", {})
        total_ip = _parse_ip(stat.get("inningsPitched"))
        if total_ip <= 0:
            continue
        gs     = int(stat.get("gamesStarted", 0) or 0)
        sp_est = min(total_ip, LEAGUE_AVG_IP) if gs > 0 else 0.0
        bp_ip  = max(0.0, total_ip - sp_est)
        share  = bp_ip / total_ip if total_ip > 0 else 0.0
        if share <= 0:
            continue

        agg["ip"]  += bp_ip
        agg["er"]  += float(stat.get("earnedRuns", 0) or 0) * share
        agg["hr"]  += float(stat.get("homeRuns", 0) or 0) * share
        agg["bb"]  += float(stat.get("baseOnBalls", 0) or 0) * share
        agg["hbp"] += float(stat.get("hitBatsmen", stat.get("hitByPitch", 0)) or 0) * share
        agg["k"]   += float(stat.get("strikeOuts", 0) or 0) * share
        agg["h"]   += float(stat.get("hits", 0) or 0) * share

    _TEAM_BP_CACHE[cache_key] = agg
    return agg


def _xfip(hr: float, bb: float, hbp: float, k: float, ip: float) -> float | None:
    """
    xFIP — FIP with the home-run component regressed toward league-average
    bullpen HR/9, since team-level game logs don't expose flyball rate.
    xFIP = (13 × HR_expected + 3×(BB+HBP) − 2×K) / IP + FIP_CONSTANT
    where HR_expected = league_bullpen_HR9 / 9 × IP  (i.e. what this
    bullpen "should" have allowed at league-average HR luck).
    """
    if ip <= 0:
        return None
    hr_expected = _LEAGUE_BULLPEN_HR9 / 9.0 * ip
    return round((13 * hr_expected + 3 * (bb + hbp) - 2 * k) / ip + _FIP_CONSTANT, 2)


def _bullpen_quality_extended(team_abbr: str, game_date: date) -> dict:
    """
    Full metric set over the _ROLLING_DAYS window: era, fip, xfip, whip,
    k9, bb9, ip. None values when sample is below _MIN_SAMPLE_IP.
    """
    agg = _aggregate_bullpen(team_abbr, game_date, _ROLLING_DAYS)
    ip = agg.get("ip", 0.0)
    out = {"era": None, "fip": None, "xfip": None, "whip": None,
           "k9": None, "bb9": None, "ip": ip}
    if ip < _MIN_SAMPLE_IP:
        return out
    out["era"]  = round(agg["er"] / ip * 9.0, 2)
    out["fip"]  = _fip(agg["hr"], agg["bb"], agg["hbp"], agg["k"], ip)
    out["xfip"] = _xfip(agg["hr"], agg["bb"], agg["hbp"], agg["k"], ip)
    out["whip"] = round((agg["bb"] + agg["h"]) / ip, 2)
    out["k9"]   = round(agg["k"] / ip * 9.0, 2)
    out["bb9"]  = round(agg["bb"] / ip * 9.0, 2)
    return out


def _bullpen_era_7d(team_abbr: str, game_date: date) -> tuple[float | None, float]:
    """Last-7-day bullpen ERA snapshot, separate from the 14-day quality sample."""
    agg = _aggregate_bullpen(team_abbr, game_date, _PERF_7D_DAYS)
    ip = agg.get("ip", 0.0)
    if ip < 3.0:   # much lower bar than the 14d sample — it's a short window by design
        return None, ip
    return round(agg["er"] / ip * 9.0, 2), ip


def _bullpen_home_away_split(team_abbr: str, game_date: date) -> tuple[float | None, float | None]:
    """(era_at_home, era_on_road) over the rolling window, or None if isHome absent."""
    home_agg = _aggregate_bullpen(team_abbr, game_date, _ROLLING_DAYS, home_filter=True)
    away_agg = _aggregate_bullpen(team_abbr, game_date, _ROLLING_DAYS, home_filter=False)
    era_home = (
        round(home_agg["er"] / home_agg["ip"] * 9.0, 2)
        if home_agg.get("ip", 0.0) >= _MIN_SAMPLE_IP / 2 else None
    )
    era_away = (
        round(away_agg["er"] / away_agg["ip"] * 9.0, 2)
        if away_agg.get("ip", 0.0) >= _MIN_SAMPLE_IP / 2 else None
    )
    return era_home, era_away


def _bullpen_fatigue(team_abbr: str, game_date: date) -> float:
    """IP delta vs league-average bullpen usage over the last _FATIGUE_DAYS."""
    agg = _aggregate_bullpen(team_abbr, game_date, _FATIGUE_DAYS)
    ip = agg.get("ip", 0.0)
    league_avg = _LEAGUE_BP_IP_PER_DAY * _FATIGUE_DAYS
    return round(ip - league_avg, 2)


# ---------------------------------------------------------------------------
# Heuristic — closer / high-leverage / setup availability
#
# There is no per-reliever role feed anywhere in this codebase (no saves
# leaderboard call, no bullpen role tagging). This is a best-effort proxy:
# pull the team's active pitching roster, find the reliever(s) with the
# most appearances in the last _ROLLING_DAYS window (proxy for "the guys
# who pitch high-leverage innings"), and flag them unavailable if they
# appear to have pitched on each of the last 2 calendar days. This is NOT
# verified against a live API response (sandbox has no network access) —
# treat the roster/appearance endpoint shapes below as a first draft to
# be smoke-tested, not a guaranteed-correct integration.
# ---------------------------------------------------------------------------

def _fetch_reliever_recent_appearances(team_abbr: str, game_date: date) -> list[dict]:
    """
    Best-effort: return [{player_id, name, saves, appearances, last_date}]
    for relief pitchers on the active roster, sorted by appearance count
    over the last _ROLLING_DAYS (used as the high-leverage/closer proxy).
    Returns [] on any failure — caller must treat that as "unknown", not
    "everyone available".
    """
    team_id = _MLB_TEAM_IDS.get(team_abbr.upper())
    if not team_id:
        return []

    roster_url = f"{_BASE_URL}/teams/{team_id}/roster/active"
    roster = _fetch(roster_url)
    if not roster or not isinstance(roster, dict):
        return []

    cutoff = game_date - timedelta(days=_ROLLING_DAYS)
    results: list[dict] = []
    for entry in roster.get("roster", []):
        position = entry.get("position", {}).get("abbreviation", "")
        if position != "P":
            continue
        person = entry.get("person", {})
        pid  = person.get("id")
        name = person.get("fullName", "")
        if not pid:
            continue

        log_url = (
            f"{_BASE_URL}/people/{pid}/stats"
            f"?stats=gameLog&group=pitching&season={game_date.year}"
        )
        log_data = _fetch(log_url)
        if not log_data or not isinstance(log_data, dict):
            continue

        appearances = 0
        is_starter  = False
        last_date: date | None = None
        for block in log_data.get("stats", []):
            for split in block.get("splits", []):
                raw_date = split.get("date", "")
                try:
                    split_date = date.fromisoformat(raw_date[:10])
                except ValueError:
                    continue
                if not (cutoff <= split_date < game_date):
                    continue
                stat = split.get("stat", {})
                if int(stat.get("gamesStarted", 0) or 0) > 0:
                    is_starter = True
                appearances += 1
                if last_date is None or split_date > last_date:
                    last_date = split_date

        if is_starter or appearances == 0:
            continue  # starters aren't bullpen; skip arms with no recent work

        results.append({
            "player_id":   pid,
            "name":        name,
            "appearances": appearances,
            "last_date":   last_date,
        })

    results.sort(key=lambda r: r["appearances"], reverse=True)
    return results


def _leverage_availability(team_abbr: str, game_date: date) -> dict:
    """
    Returns {closer_available, high_leverage_available, setup_available}
    as bool | None. None means "couldn't determine" — must not be treated
    as a green light by any downstream caller.

    Heuristic: the arm with the most appearances over the rolling window
    is treated as the closer/top leverage arm; the #2 as setup. Either is
    flagged unavailable if they pitched on each of the last 2 consecutive
    calendar days (standard back-to-back caution in real bullpens).
    """
    out = {
        "closer_available": None,
        "high_leverage_available": None,
        "setup_available": None,
    }
    try:
        relievers = _fetch_reliever_recent_appearances(team_abbr, game_date)
    except Exception as exc:
        logger.debug(f"[bullpen_intel] leverage availability lookup failed: {exc}")
        return out
    if not relievers:
        return out

    def _available(r: dict) -> bool:
        last = r.get("last_date")
        if last is None:
            return True
        # Unavailable if pitched yesterday AND day before (back-to-back-to-back caution)
        return not (
            last == game_date - timedelta(days=1)
            and any(
                r2["player_id"] == r["player_id"] for r2 in relievers
                if r2.get("last_date") == game_date - timedelta(days=2)
            )
        )

    top = relievers[0]
    out["closer_available"] = _available(top)
    out["high_leverage_available"] = out["closer_available"]
    if len(relievers) > 1:
        out["setup_available"] = _available(relievers[1])
    return out


def _bullpen_handedness_split(team_abbr: str, game_date: date) -> tuple[float | None, float | None]:
    """
    Best-effort left/right bullpen ERA split via the MLB Stats API
    statSplits sitCodes. Team-level pitching splits vs LHB/RHB are not
    something any other module in this codebase pulls, so this endpoint
    shape is unverified — returns (None, None) on any mismatch rather
    than fabricating a number.
    """
    team_id = _MLB_TEAM_IDS.get(team_abbr.upper())
    if not team_id:
        return None, None
    try:
        url = (
            f"{_BASE_URL}/teams/{team_id}/stats"
            f"?stats=statSplits&group=pitching&season={game_date.year}"
            f"&sitCodes=vl,vr"
        )
        data = _fetch(url)
        if not data or not isinstance(data, dict):
            return None, None
        era_vl: float | None = None
        era_vr: float | None = None
        for block in data.get("stats", []):
            for split in block.get("splits", []):
                sit = (split.get("split", {}) or {}).get("code", "")
                stat = split.get("stat", {})
                era = stat.get("era")
                if era is None:
                    continue
                if sit == "vl":
                    era_vl = float(era)
                elif sit == "vr":
                    era_vr = float(era)
        return era_vl, era_vr
    except Exception as exc:
        logger.debug(f"[bullpen_intel] handedness split lookup failed: {exc}")
        return None, None


def _quality_score(era: float | None, fip: float | None) -> float:
    """Map bullpen ERA/FIP to a 0-100 score, 100 = elite, 50 = league average."""
    if era is None and fip is None:
        return 50.0
    ref = era if era is not None else fip
    if fip is not None and era is not None:
        ref = (era + fip) / 2.0
    # LEAGUE_BULLPEN_ERA maps to 50; each run better/worse moves ~22.5 pts.
    score = 50.0 - (ref - LEAGUE_BULLPEN_ERA) * 22.5
    return round(max(0.0, min(100.0, score)), 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_bullpen_intel(
    home_abbr: str,
    away_abbr: str,
    game_date: date | None = None,
) -> BullpenIntelFactor:
    """
    Return a bullpen-quality league_mean adjustment for a given MLB matchup.
    Zero adjustment on any error — never blocks the engine.
    """
    try:
        return _compute(home_abbr, away_abbr, game_date)
    except Exception as exc:
        logger.debug(f"[bullpen_intel] Unexpected error for {away_abbr}@{home_abbr}: {exc}")
        return BullpenIntelFactor()


def _expected_bp_innings(home_abbr: str, away_abbr: str, game_date: date) -> tuple[float, float]:
    """
    Projected bullpen innings tonight for (home, away), derived from
    pitcher_workload's starter IP projection. Falls back to the league
    average (9 - LEAGUE_AVG_IP) when workload data is unavailable.
    """
    default = round(9.0 - LEAGUE_AVG_IP, 2)
    try:
        from core.pitcher_workload import get_game_workload_pair
        home_wl, away_wl = get_game_workload_pair(home_abbr, away_abbr, game_date=game_date)
        home_bp = round(max(1.0, 9.0 - home_wl.expected_ip), 2) if home_wl else default
        away_bp = round(max(1.0, 9.0 - away_wl.expected_ip), 2) if away_wl else default
        return home_bp, away_bp
    except Exception as exc:
        logger.debug(f"[bullpen_intel] workload lookup failed: {exc}")
        return default, default


def _compute(
    home_abbr: str,
    away_abbr: str,
    game_date: date | None,
) -> BullpenIntelFactor:
    if game_date is None:
        from core.time_utils import now_est
        game_date = now_est().date()

    cache_key = f"{away_abbr.upper()}@{home_abbr.upper()}_{game_date}"
    if cache_key in _RESULT_CACHE:
        return _RESULT_CACHE[cache_key]

    home_bp_ip, away_bp_ip = _expected_bp_innings(home_abbr, away_abbr, game_date)

    home_m = _bullpen_quality_extended(home_abbr, game_date)
    away_m = _bullpen_quality_extended(away_abbr, game_date)

    home_fatigue = _bullpen_fatigue(home_abbr, game_date)
    away_fatigue = _bullpen_fatigue(away_abbr, game_date)

    home_era_7d, home_ip_7d = _bullpen_era_7d(home_abbr, game_date)
    away_era_7d, away_ip_7d = _bullpen_era_7d(away_abbr, game_date)

    home_era_home, home_era_road = _bullpen_home_away_split(home_abbr, game_date)
    away_era_home, away_era_road = _bullpen_home_away_split(away_abbr, game_date)

    home_leverage = {"closer_available": None, "high_leverage_available": None, "setup_available": None}
    away_leverage = dict(home_leverage)
    if ENABLE_LEVERAGE_AVAILABILITY:
        home_leverage = _leverage_availability(home_abbr, game_date)
        away_leverage = _leverage_availability(away_abbr, game_date)

    home_vl, home_vr = (None, None)
    away_vl, away_vr = (None, None)
    if ENABLE_HANDEDNESS_SPLIT:
        home_vl, home_vr = _bullpen_handedness_split(home_abbr, game_date)
        away_vl, away_vr = _bullpen_handedness_split(away_abbr, game_date)

    adj = 0.0
    parts: list[str] = []

    for label, m, bp_ip, fatigue in (
        (home_abbr, home_m, home_bp_ip, home_fatigue),
        (away_abbr, away_m, away_bp_ip, away_fatigue),
    ):
        era = m["era"]
        if era is not None:
            excess = (era - LEAGUE_BULLPEN_ERA) * (bp_ip / 9.0)
            adj += excess
            parts.append(f"{label} BP ERA {era:.2f} (proj {bp_ip:.1f} IP)")
        else:
            parts.append(f"{label} BP ERA N/A (proj {bp_ip:.1f} IP)")

        fatigue_adj = max(_FATIGUE_ADJ_MIN, min(_FATIGUE_ADJ_MAX, fatigue * 0.03))
        adj += fatigue_adj
        if abs(fatigue) >= 3.0:
            tag = "fatigued" if fatigue > 0 else "fresh"
            parts.append(f"{label} BP {tag} ({fatigue:+.1f} IP L3D)")

    adj = round(max(_ADJ_MIN, min(_ADJ_MAX, adj)), 2)

    factor_text = f"Bullpen: {' | '.join(parts)}"
    if adj != 0.0:
        arrow = "↑" if adj > 0 else "↓"
        factor_text += f" → prior {arrow}{abs(adj):.1f}"

    factor = BullpenIntelFactor(
        home_bullpen_era        = home_m["era"],
        home_bullpen_fip        = home_m["fip"],
        home_bullpen_xfip       = home_m["xfip"],
        home_bullpen_whip       = home_m["whip"],
        home_bullpen_k9         = home_m["k9"],
        home_bullpen_bb9        = home_m["bb9"],
        home_bp_ip_sample       = home_m["ip"],
        home_bp_ip_expected     = home_bp_ip,
        home_bullpen_fatigue    = home_fatigue,
        home_bullpen_score      = _quality_score(home_m["era"], home_m["fip"]),

        away_bullpen_era        = away_m["era"],
        away_bullpen_fip        = away_m["fip"],
        away_bullpen_xfip       = away_m["xfip"],
        away_bullpen_whip       = away_m["whip"],
        away_bullpen_k9         = away_m["k9"],
        away_bullpen_bb9        = away_m["bb9"],
        away_bp_ip_sample       = away_m["ip"],
        away_bp_ip_expected     = away_bp_ip,
        away_bullpen_fatigue    = away_fatigue,
        away_bullpen_score      = _quality_score(away_m["era"], away_m["fip"]),

        home_bullpen_era_7d     = home_era_7d,
        away_bullpen_era_7d     = away_era_7d,
        home_bp_ip_7d           = home_ip_7d,
        away_bp_ip_7d           = away_ip_7d,

        home_bullpen_era_at_home = home_era_home,
        home_bullpen_era_on_road = home_era_road,
        away_bullpen_era_at_home = away_era_home,
        away_bullpen_era_on_road = away_era_road,

        home_closer_available         = home_leverage["closer_available"],
        away_closer_available         = away_leverage["closer_available"],
        home_high_leverage_available  = home_leverage["high_leverage_available"],
        away_high_leverage_available  = away_leverage["high_leverage_available"],
        home_setup_available          = home_leverage["setup_available"],
        away_setup_available          = away_leverage["setup_available"],

        home_bullpen_era_vs_lhb  = home_vl,
        home_bullpen_era_vs_rhb  = home_vr,
        away_bullpen_era_vs_lhb  = away_vl,
        away_bullpen_era_vs_rhb  = away_vr,

        league_mean_adjustment = adj,
        factor_text            = factor_text,
    )
    logger.debug(
        f"[bullpen_intel] {away_abbr}@{home_abbr}: "
        f"home_era={home_m['era']} away_era={away_m['era']} adj={adj:+.2f}"
    )
    _RESULT_CACHE[cache_key] = factor
    return factor
