"""
Strikeout Matchup Adjustments — Enhanced Model
===============================================
Multi-layer K projection multiplier following the Strikeout Model Enhancement
Protocol.  Applied in player_props.py after Layer 1 (workload / IP scale).

Layer pipeline
--------------
  Layer 2A  Pitcher K% splits × lineup composition          strength=0.70
    Pitcher's own K% vs LHB and vs RHB, weighted by tonight's confirmed
    lineup L/R ratio.  Falls back to 50/50 when lineup not yet posted.

  Layer 2B  Per-batter K% vs pitcher hand                   strength=0.60
    For each confirmed lineup batter: their individual K% vs this pitcher's
    hand from MLB Stats API statSplits.  Averaged across available batters.
    Requires a confirmed lineup (≥ 3 batters with valid splits).

  Layer 4   30-day team K% vs pitcher hand (form)           strength=0.85/0.60
    Rolling 30-day team strikeout rate vs the pitcher's handedness.  Acts as
    a form adjustment and fallback when lineup is not yet posted.
    Downweighted 50 % when Layers 2A/2B already provide matchup data.

  Layer 5   Swinging Strike Rate (SwStr%)                   strength=0.40
    Pitcher's season SwStr% from Baseball Savant leaderboard CSV.

  Layer 6   CSW% (Called Strike + Whiff)                    strength=0.30
    Supporting sustainability indicator from Savant.

  Layer 7   Fastball velocity vs league average              strength=0.25
    Modest adjustment for velocity above/below 93.5 mph baseline.

Data sources
------------
  MLB Stats API    — pitcher/batter statSplits, schedule lineups (no auth)
  Baseball Savant  — SwStr%, CSW%, velocity (public CSV leaderboard)

Fail-safe: every layer returns a neutral contribution (1.0) on any error.
Final scale is clamped to [0.72, 1.40].
"""
from __future__ import annotations

import csv
import io
import json
import logging
from datetime import date, timedelta
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_MLB_BASE    = "https://statsapi.mlb.com/api/v1"
_SAVANT_BASE = "https://baseballsavant.mlb.com"
_TIMEOUT     = 6    # seconds — MLB Stats API
_SAVANT_TO   = 12   # seconds — Savant CSV can be slow

# ---------------------------------------------------------------------------
# Process-level caches (survive for the length of one scheduler run)
# ---------------------------------------------------------------------------
_HAND_CACHE:           dict[str, str]                        = {}  # pitcher_name.lower() → R/L/U
_TEAM_K_CACHE:         dict[tuple[str, str, str], float]     = {}  # (abbr, sit, date) → k_pct
_PITCHER_SPLITS_CACHE: dict[str, tuple[float, float] | None] = {}  # str(id) → (k_lhb, k_rhb)|None
_LINEUP_CACHE:         dict[str, tuple[list, list] | None]   = {}  # "abbr:date" → (lhb_ids,rhb_ids)|None
_BATTER_K_CACHE:       dict[str, float | None]               = {}  # "id:sit" → k_pct|None
_SAVANT_DATA:          dict[int, dict]                       = {}  # pitcher_id → stats
_SAVANT_LOADED: bool = False

# ---------------------------------------------------------------------------
# League baselines  (2024–2026 MLB composite)
# ---------------------------------------------------------------------------
_LEAGUE_K_VS_RHP  = 0.229   # batter K% facing a right-hander
_LEAGUE_K_VS_LHP  = 0.221   # batter K% facing a left-hander
_LEAGUE_K_VS_LHB  = 0.228   # pitcher K% facing left-handed batters
_LEAGUE_K_VS_RHB  = 0.222   # pitcher K% facing right-handed batters
_LEAGUE_SWSTR     = 0.115   # league-average swinging-strike rate
_LEAGUE_CSW       = 0.290   # league-average CSW%
_LEAGUE_VELO      = 93.5    # league-average SP fastball velocity (mph)

# Minimum PA/BF to trust a split sample
_MIN_PA = 40

# Final combined scale bounds
_MIN_SCALE = 0.72
_MAX_SCALE = 1.40

# ---------------------------------------------------------------------------
# Team abbreviation → MLB Stats API team ID
# ---------------------------------------------------------------------------
_MLB_TEAM_IDS: dict[str, int] = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KC":  118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD":  135, "SF":  137, "SEA": 136,
    "STL": 138, "TB":  139, "TEX": 140, "TOR": 141, "WSH": 120,
}


# ---------------------------------------------------------------------------
# Internal fetch helpers
# ---------------------------------------------------------------------------

def _fetch(url: str) -> dict | None:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=_TIMEOUT) as r:
            return json.loads(r.read())
    except (URLError, Exception) as exc:
        logger.debug(f"[strikeout_matchup] fetch failed: {exc}  url={url}")
        return None


def _fetch_text(url: str) -> str | None:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=_SAVANT_TO) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug(f"[strikeout_matchup] text fetch failed: {exc}  url={url}")
        return None


def _today() -> str:
    return date.today().isoformat()


def _days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).isoformat()


def _blend(raw_scale: float, strength: float) -> float:
    """Blend raw_scale towards 1.0 at the given strength (0–1)."""
    return 1.0 + (raw_scale - 1.0) * strength


# ---------------------------------------------------------------------------
# Layer 0: Pitcher throwing hand
# ---------------------------------------------------------------------------

def get_pitcher_hand(pitcher_name: str) -> str:
    """
    Return 'R', 'L', or 'U' (unknown) for the pitcher's throwing arm.
    Cached for the duration of the process.
    """
    key = pitcher_name.strip().lower()
    if key in _HAND_CACHE:
        return _HAND_CACHE[key]

    safe = pitcher_name.strip().replace(" ", "%20")
    data = _fetch(f"{_MLB_BASE}/people/search?names={safe}&sportId=1")
    hand = "U"
    if data:
        for p in data.get("people", []):
            ph = (p.get("pitchHand") or {}).get("code", "")
            if ph in ("R", "L"):
                hand = ph
                break

    _HAND_CACHE[key] = hand
    logger.debug(f"[strikeout_matchup] pitcher hand — {pitcher_name}: {hand}")
    return hand


# ---------------------------------------------------------------------------
# Layer 2A: Pitcher's own K% splits vs LHB / vs RHB
# ---------------------------------------------------------------------------

def _fetch_pitcher_splits(pitcher_id: int) -> tuple[float, float] | None:
    """
    Return (k_vs_lhb, k_vs_rhb) from the pitcher's current-season statSplits.
    Returns None when data is unavailable or below the minimum BF threshold.

    sit_code 'vl' = pitcher facing left-handed batters
    sit_code 'vr' = pitcher facing right-handed batters
    """
    cache_key = str(pitcher_id)
    if cache_key in _PITCHER_SPLITS_CACHE:
        return _PITCHER_SPLITS_CACHE[cache_key]

    season = date.today().year
    url = (
        f"{_MLB_BASE}/people/{pitcher_id}/stats"
        f"?stats=statSplits&group=pitching&season={season}&sitCodes=vl,vr"
    )
    data    = _fetch(url)
    result: tuple[float, float] | None = None

    if data:
        lhb_k: float | None = None
        rhb_k: float | None = None
        for block in data.get("stats", []):
            for split in block.get("splits", []):
                sit  = (split.get("split") or {}).get("code", "")
                stat = split.get("stat", {})
                try:
                    bf = float(stat.get("battersFaced") or 0)
                    ks = float(stat.get("strikeOuts") or 0)
                    if bf >= _MIN_PA:
                        k_pct = round(ks / bf, 4)
                        if sit == "vl":
                            lhb_k = k_pct
                        elif sit == "vr":
                            rhb_k = k_pct
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
        if lhb_k is not None and rhb_k is not None:
            result = (lhb_k, rhb_k)
            logger.debug(
                f"[strikeout_matchup] pitcher splits id={pitcher_id}: "
                f"K%vsLHB={lhb_k:.1%}  K%vsRHB={rhb_k:.1%}"
            )

    _PITCHER_SPLITS_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Layer 2B: Confirmed lineup with bat-side for each batter
# ---------------------------------------------------------------------------

def _fetch_lineup_handedness(
    opp_abbr: str,
    game_date: str,
) -> tuple[list[int], list[int]] | None:
    """
    Return (lhb_ids, rhb_ids) from the confirmed starting lineup for opp_abbr.
    Returns None when the lineup has not yet been posted by MLB.

    Uses /schedule?hydrate=lineups&teamId={opp_team_id}.
    Switch hitters are counted as RHB for initial pass.
    """
    cache_key = f"{opp_abbr}:{game_date}"
    if cache_key in _LINEUP_CACHE:
        return _LINEUP_CACHE[cache_key]

    opp_team_id = _MLB_TEAM_IDS.get(opp_abbr.upper())
    if not opp_team_id:
        _LINEUP_CACHE[cache_key] = None
        return None

    url = (
        f"{_MLB_BASE}/schedule?sportId=1&date={game_date}"
        f"&hydrate=lineups&teamId={opp_team_id}"
    )
    data = _fetch(url)
    if not data:
        _LINEUP_CACHE[cache_key] = None
        return None

    lhb_ids: list[int] = []
    rhb_ids:  list[int] = []

    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            lineups = game.get("lineups")
            if not lineups:
                continue

            home_id = (game.get("teams", {})
                       .get("home", {}).get("team", {}).get("id"))
            away_id = (game.get("teams", {})
                       .get("away", {}).get("team", {}).get("id"))

            if home_id == opp_team_id:
                batters = lineups.get("homeBatters", [])
            elif away_id == opp_team_id:
                batters = lineups.get("awayBatters", [])
            else:
                continue

            for batter in batters:
                bid  = batter.get("id")
                side = (batter.get("batSide") or {}).get("code", "")
                if not bid:
                    continue
                if side == "L":
                    lhb_ids.append(bid)
                else:
                    rhb_ids.append(bid)
            break  # found the game — no need to check others

    if not lhb_ids and not rhb_ids:
        _LINEUP_CACHE[cache_key] = None
        return None

    result = (lhb_ids, rhb_ids)
    _LINEUP_CACHE[cache_key] = result
    logger.debug(
        f"[strikeout_matchup] lineup {opp_abbr} {game_date}: "
        f"{len(lhb_ids)} LHB, {len(rhb_ids)} RHB"
    )
    return result


# ---------------------------------------------------------------------------
# Layer 3: Per-batter K% vs pitcher hand
# ---------------------------------------------------------------------------

def _fetch_batter_k_vs_hand(batter_id: int, sit_code: str) -> float | None:
    """
    Return batter's K% vs pitchers of a given hand from MLB statSplits.
    sit_code: 'vr' = vs RHP,  'vl' = vs LHP
    """
    cache_key = f"{batter_id}:{sit_code}"
    if cache_key in _BATTER_K_CACHE:
        return _BATTER_K_CACHE[cache_key]

    season = date.today().year
    url = (
        f"{_MLB_BASE}/people/{batter_id}/stats"
        f"?stats=statSplits&group=hitting&season={season}&sitCodes={sit_code}"
    )
    data   = _fetch(url)
    result: float | None = None

    if data:
        for block in data.get("stats", []):
            for split in block.get("splits", []):
                stat = split.get("stat", {})
                try:
                    pa = float(stat.get("plateAppearances") or 0)
                    ks = float(stat.get("strikeOuts") or 0)
                    if pa >= _MIN_PA:
                        result = round(ks / pa, 4)
                except (TypeError, ValueError, ZeroDivisionError):
                    continue

    _BATTER_K_CACHE[cache_key] = result
    return result


def _get_lineup_avg_k(
    lhb_ids: list[int],
    rhb_ids:  list[int],
    pitcher_hand: str,
) -> float | None:
    """
    Average K% across the confirmed lineup's batters vs this pitcher's hand.

    sit_code depends on pitcher's hand (batters face that hand):
      pitcher=R  →  sit_code='vr'  (batters are facing a right-hander)
      pitcher=L  →  sit_code='vl'  (batters are facing a left-hander)

    Returns None when fewer than 3 batters have valid splits (too thin a sample).
    """
    sit_code = "vr" if pitcher_hand == "R" else "vl"
    k_pcts: list[float] = []
    for bid in lhb_ids + rhb_ids:
        k = _fetch_batter_k_vs_hand(bid, sit_code)
        if k is not None:
            k_pcts.append(k)

    if len(k_pcts) < 3:
        return None

    avg = round(sum(k_pcts) / len(k_pcts), 4)
    logger.debug(
        f"[strikeout_matchup] per-batter avg K% vs {'R' if pitcher_hand == 'R' else 'L'}HP: "
        f"{avg:.1%}  (n={len(k_pcts)} batters)"
    )
    return avg


# ---------------------------------------------------------------------------
# Layer 4: 30-day team K% vs pitcher hand
# ---------------------------------------------------------------------------

def _fetch_split_k_pct(
    team_abbr: str,
    sit_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> float | None:
    team_id = _MLB_TEAM_IDS.get(team_abbr.upper())
    if not team_id:
        return None

    season = date.today().year
    url = (
        f"{_MLB_BASE}/teams/{team_id}/stats"
        f"?stats=statSplits&group=hitting&season={season}&sitCodes={sit_code}"
    )
    if start_date:
        url += f"&startDate={start_date}&endDate={end_date or _today()}"

    data = _fetch(url)
    if not data:
        return None

    for block in data.get("stats", []):
        for split in block.get("splits", []):
            stat = split.get("stat", {})
            try:
                pa = float(stat.get("plateAppearances") or 0)
                ks = float(stat.get("strikeOuts") or 0)
                if pa >= _MIN_PA:
                    return round(ks / pa, 4)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
    return None


def get_team_k_pct_vs_hand(
    team_abbr: str,
    pitcher_hand: str,
) -> tuple[float, str]:
    """
    Return (k_pct, source) for the opposing team's K% vs pitchers of the given hand.
    Source priority: '30d' → 'season' → 'fallback' (league average).
    """
    sit_code  = "vr" if pitcher_hand == "R" else "vl"
    cache_key = (team_abbr.upper(), sit_code, _today())

    if cache_key in _TEAM_K_CACHE:
        return (_TEAM_K_CACHE[cache_key], "cached")

    k_pct  = _fetch_split_k_pct(team_abbr, sit_code, _days_ago(30), _today())
    source = "30d"

    if k_pct is None:
        k_pct  = _fetch_split_k_pct(team_abbr, sit_code)
        source = "season"

    if k_pct is None:
        k_pct  = _LEAGUE_K_VS_RHP if pitcher_hand == "R" else _LEAGUE_K_VS_LHP
        source = "fallback"

    _TEAM_K_CACHE[cache_key] = k_pct
    logger.debug(
        f"[strikeout_matchup] {team_abbr} K% vs {'R' if pitcher_hand == 'R' else 'L'}HP "
        f"({source}): {k_pct:.1%}"
    )
    return (k_pct, source)


# ---------------------------------------------------------------------------
# Layers 5–7: Baseball Savant  (SwStr%, CSW%, velocity)
# ---------------------------------------------------------------------------

def _load_savant_leaderboard() -> None:
    """
    Fetch the Savant custom leaderboard CSV and populate _SAVANT_DATA.
    Called at most once per process; silently skips on any network/parse error.
    """
    global _SAVANT_LOADED
    if _SAVANT_LOADED:
        return
    _SAVANT_LOADED = True  # mark even on failure so we don't retry on every pick

    season = date.today().year
    url = (
        f"{_SAVANT_BASE}/leaderboard/custom"
        f"?year={season}&type=pitcher&filter=&sort=4&sortDir=desc&min=q"
        f"&selections=k_percent,whiff_percent,csw,fastball_avg_speed&csv=true"
    )
    text = _fetch_text(url)
    if not text or "<html" in text[:200].lower():
        logger.debug("[strikeout_matchup] Savant leaderboard unavailable or returned HTML")
        return

    try:
        reader = csv.DictReader(io.StringIO(text))
        count  = 0
        for row in reader:
            try:
                pid = int(row.get("player_id") or 0)
                if not pid:
                    continue

                def _pct(val: str | None) -> float:
                    v = float(val or 0)
                    return v / 100.0 if v > 1.0 else v  # handle 0-100 or 0-1 format

                whiff = _pct(row.get("whiff_percent"))
                csw   = _pct(row.get("csw"))
                velo  = float(row.get("fastball_avg_speed") or 0)

                if whiff or csw or velo:
                    _SAVANT_DATA[pid] = {
                        "whiff_pct": round(whiff, 4),
                        "csw_pct":   round(csw,   4),
                        "velo":      round(velo,   1),
                    }
                    count += 1
            except (ValueError, TypeError):
                continue

        logger.debug(f"[strikeout_matchup] Savant leaderboard loaded: {count} pitchers")
    except Exception as exc:
        logger.debug(f"[strikeout_matchup] Savant CSV parse error: {exc}")


def _get_savant_stats(pitcher_id: int | None) -> dict | None:
    """Return Savant stats for pitcher_id, or None if unavailable."""
    if not pitcher_id:
        return None
    _load_savant_leaderboard()
    return _SAVANT_DATA.get(pitcher_id)


# ---------------------------------------------------------------------------
# Main entry point — full multi-layer matchup scale
# ---------------------------------------------------------------------------

def get_k_matchup_scale(
    pitcher_name: str,
    opp_abbr: str,
    pitcher_id: int | None = None,
    game_date: str | None = None,
) -> float:
    """
    Return a combined K projection multiplier from the full 7-layer pipeline.

    Parameters
    ----------
    pitcher_name : pitcher display name (for hand lookup)
    opp_abbr     : opposing team abbreviation
    pitcher_id   : MLB Stats API person ID — enables splits + Savant lookup
    game_date    : "YYYY-MM-DD" for lineup lookup; defaults to today

    Layers applied multiplicatively, each strength-blended towards 1.0:
      2A  Pitcher K% splits × lineup L/R composition   strength=0.70
      2B  Per-batter K% vs pitcher hand                 strength=0.60
      4   30-day team K% vs pitcher hand                strength=0.85/0.60
              (halved when 2A/2B already provide matchup data)
      5   SwStr% from Baseball Savant                   strength=0.40
      6   CSW%   from Baseball Savant                   strength=0.30
      7   Velocity vs league average                    strength=0.25

    Returns 1.0 (no adjustment) on any unrecoverable error.
    Final scale is clamped to [0.72, 1.40].
    """
    if not opp_abbr:
        return 1.0

    try:
        game_date = game_date or _today()

        # ── Layer 0: pitcher hand ────────────────────────────────────────
        hand = get_pitcher_hand(pitcher_name)
        if hand == "U":
            hand = None  # keep going; team K% layer still works without it

        # ── Layer 2A: pitcher splits × lineup handedness ─────────────────
        splits_scale = 1.0
        lineup_lhb:  list[int] = []
        lineup_rhb:  list[int] = []
        have_matchup_data = False

        if hand and pitcher_id:
            pitcher_splits = _fetch_pitcher_splits(pitcher_id)
            lineup_info    = _fetch_lineup_handedness(opp_abbr, game_date)

            if pitcher_splits is not None:
                k_vs_lhb, k_vs_rhb = pitcher_splits

                if lineup_info:
                    lineup_lhb, lineup_rhb = lineup_info
                    n_lhb  = len(lineup_lhb)
                    n_rhb  = len(lineup_rhb)
                    total  = max(n_lhb + n_rhb, 1)
                    # Weighted K rate based on actual tonight's lineup composition
                    lineup_k_rate = (n_lhb * k_vs_lhb + n_rhb * k_vs_rhb) / total
                else:
                    # Lineup not posted — average the two splits (50/50 assumption)
                    lineup_k_rate = (k_vs_lhb + k_vs_rhb) / 2.0

                league_avg_k = (_LEAGUE_K_VS_LHB + _LEAGUE_K_VS_RHB) / 2.0
                raw          = lineup_k_rate / league_avg_k
                splits_scale = round(_blend(raw, 0.70), 4)
                have_matchup_data = True
                logger.debug(
                    f"[strikeout_matchup] 2A splits×lineup {pitcher_name}: "
                    f"K_rate={lineup_k_rate:.1%} league={league_avg_k:.1%} "
                    f"raw={raw:.3f} → scale={splits_scale:.4f}"
                )

        # ── Layer 2B: per-batter K% vs pitcher hand ──────────────────────
        batter_k_scale = 1.0
        if hand and (lineup_lhb or lineup_rhb):
            avg_batter_k = _get_lineup_avg_k(lineup_lhb, lineup_rhb, hand)
            if avg_batter_k is not None:
                league_batter_k = _LEAGUE_K_VS_RHP if hand == "R" else _LEAGUE_K_VS_LHP
                raw             = avg_batter_k / league_batter_k
                batter_k_scale  = round(_blend(raw, 0.60), 4)
                have_matchup_data = True
                logger.debug(
                    f"[strikeout_matchup] 2B per-batter K {pitcher_name}: "
                    f"avg={avg_batter_k:.1%} league={league_batter_k:.1%} "
                    f"raw={raw:.3f} → scale={batter_k_scale:.4f}"
                )

        # ── Layer 4: 30-day team K% (form / fallback) ────────────────────
        team_k_scale = 1.0
        if hand:
            opp_k_pct, src = get_team_k_pct_vs_hand(opp_abbr, hand)
            league_avg_k   = _LEAGUE_K_VS_RHP if hand == "R" else _LEAGUE_K_VS_LHP
            base_strength  = 0.85 if src in ("30d", "cached") else 0.60
            # Downweight when Layers 2A/2B already captured matchup specifics
            strength = base_strength * (0.50 if have_matchup_data else 1.00)
            raw      = opp_k_pct / league_avg_k
            team_k_scale = round(_blend(raw, strength), 4)
            logger.debug(
                f"[strikeout_matchup] 4 team K {pitcher_name} vs {opp_abbr}: "
                f"opp={opp_k_pct:.1%} league={league_avg_k:.1%} "
                f"raw={raw:.3f} strength={strength:.0%} → scale={team_k_scale:.4f} [{src}]"
            )

        # ── Layers 5–7: Savant (SwStr%, CSW%, velocity) ──────────────────
        swstr_scale = 1.0
        csw_scale   = 1.0
        velo_scale  = 1.0

        savant = _get_savant_stats(pitcher_id)
        if savant:
            # Layer 5: SwStr%
            if savant["whiff_pct"] > 0:
                raw = savant["whiff_pct"] / _LEAGUE_SWSTR
                swstr_scale = round(_blend(raw, 0.40), 4)
                logger.debug(
                    f"[strikeout_matchup] 5 SwStr% {pitcher_name}: "
                    f"{savant['whiff_pct']:.1%} vs league {_LEAGUE_SWSTR:.1%} "
                    f"→ scale={swstr_scale:.4f}"
                )
            # Layer 6: CSW%
            if savant["csw_pct"] > 0:
                raw = savant["csw_pct"] / _LEAGUE_CSW
                csw_scale = round(_blend(raw, 0.30), 4)
                logger.debug(
                    f"[strikeout_matchup] 6 CSW% {pitcher_name}: "
                    f"{savant['csw_pct']:.1%} vs league {_LEAGUE_CSW:.1%} "
                    f"→ scale={csw_scale:.4f}"
                )
            # Layer 7: velocity (skip unusually low values — probably missing data)
            if savant["velo"] > 80:
                velo_delta = savant["velo"] - _LEAGUE_VELO
                # ~1.5 % K-rate change per mph deviation (empirical estimate)
                raw = 1.0 + (velo_delta / _LEAGUE_VELO) * 1.5
                velo_scale = round(_blend(raw, 0.25), 4)
                logger.debug(
                    f"[strikeout_matchup] 7 velo {pitcher_name}: "
                    f"{savant['velo']} mph vs league {_LEAGUE_VELO} mph "
                    f"→ scale={velo_scale:.4f}"
                )

        # ── Combine all layers multiplicatively then clamp ────────────────
        combined = (
            splits_scale
            * batter_k_scale
            * team_k_scale
            * swstr_scale
            * csw_scale
            * velo_scale
        )
        final = round(max(_MIN_SCALE, min(_MAX_SCALE, combined)), 4)

        logger.debug(
            f"[strikeout_matchup] FINAL {pitcher_name} vs {opp_abbr}: "
            f"2A={splits_scale:.4f} × 2B={batter_k_scale:.4f} × "
            f"4={team_k_scale:.4f} × 5={swstr_scale:.4f} × "
            f"6={csw_scale:.4f} × 7={velo_scale:.4f} "
            f"= {combined:.4f} → clamped={final:.4f}"
        )
        return final

    except Exception as exc:
        logger.debug(f"[strikeout_matchup] {pitcher_name} vs {opp_abbr}: {exc}")
        return 1.0
