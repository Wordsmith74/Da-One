"""
models/weather_intel.py — MLB ballpark weather → K-rate adjustment

Data source: Open-Meteo (https://open-meteo.com) — free, no API key required.
Only used for strikeout_matchup.py's Layer 10.

Honesty note on the effect size
--------------------------------
Weather's best-established effect in baseball is on batted-ball distance
(wind/temperature/altitude → home runs, park factors) — that's a real,
well-documented signal. Its effect on swing-and-miss / strikeout rate is
much weaker and less studied: the only semi-established mechanism is that
cold air can dull a pitcher's grip and fastball/breaking-ball spin, which
can cost a tick of velocity or break. There's no comparable mechanism for
wind affecting whiffs, so wind is fetched and returned (for use elsewhere,
e.g. an eventual totals/park-factor consumer) but is NOT used to adjust the
K projection here. Only temperature is used, and only at a low blend
strength, to avoid overstating a soft signal.

Domes / retractable roofs
--------------------------
8 of 30 MLB parks are enclosed or retractable and are closed the large
majority of the time (AC parks in hot climates, or fixed domes). Rather
than pull outdoor conditions that don't apply, these are hardcoded as
"enclosed" and skipped entirely — returns neutral with reason "dome".
This is a simplification (e.g. Rogers Centre/T-Mobile Park do occasionally
play with the roof open) but a false-neutral is much safer than a
false-weather-penalty applied to a climate-controlled game.

Fail-safe: every public function returns a neutral value on any error,
missing coordinates, dome, or unavailable forecast (e.g. game date is
outside Open-Meteo's ~16-day forecast window — common in backtest.py runs
against past dates, which need the /archive endpoint instead; not wired in
here since backtest correctness for weather is a smaller concern than not
breaking live picks).
"""
from __future__ import annotations

import json
import logging
from datetime import date
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_MLB_BASE      = "https://statsapi.mlb.com/api/v1"
_OPEN_METEO    = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT       = 5   # seconds — keep short, weather is a nice-to-have layer

# League-neutral baseline for the (weak) temperature → K-rate heuristic.
_BASELINE_TEMP_F = 70.0

# Below this, cold-grip effect kicks in; above this, treated as neutral-to-
# warm (no bonus applied — "warm helps whiffs" is even less established
# than "cold hurts them", so we only model the downside).
_COLD_THRESHOLD_F = 55.0

# ---------------------------------------------------------------------------
# Ballpark coordinates + enclosed-roof flag
# ---------------------------------------------------------------------------
# (lat, lon, enclosed). Coordinates are approximate (park-level, not exact
# home-plate GPS) — fine for pulling a city-grid weather forecast.
_MLB_VENUES: dict[str, tuple[float, float, bool]] = {
    "ARI": (33.4455, -112.0667, True),   # Chase Field — retractable, closed most of season
    "ATL": (33.8907, -84.4677, False),
    "BAL": (39.2838, -76.6217, False),
    "BOS": (42.3467, -71.0972, False),
    "CHC": (41.9484, -87.6553, False),
    "CWS": (41.8299, -87.6338, False),
    "CIN": (39.0975, -84.5071, False),
    "CLE": (41.4962, -81.6852, False),
    "COL": (39.7559, -104.9942, False),
    "DET": (42.3390, -83.0485, False),
    "HOU": (29.7570, -95.3555, True),    # Minute Maid Park — retractable, usually closed
    "KC":  (39.0517, -94.4803, False),
    "LAA": (33.8003, -117.8827, False),
    "LAD": (34.0739, -118.2400, False),
    "MIA": (25.7781, -80.2196, True),    # loanDepot Park — retractable, usually closed
    "MIL": (43.0280, -87.9712, True),    # American Family Field — retractable
    "MIN": (44.9817, -93.2776, False),
    "NYM": (40.7571, -73.8458, False),
    "NYY": (40.8296, -73.9262, False),
    "OAK": (37.7516, -122.2005, False),
    "PHI": (39.9061, -75.1665, False),
    "PIT": (40.4469, -80.0057, False),
    "SD":  (32.7073, -117.1566, False),
    "SF":  (37.7786, -122.3893, False),
    "SEA": (47.5914, -122.3325, True),   # T-Mobile Park — retractable, often closed
    "STL": (38.6226, -90.1928, False),
    "TB":  (27.7683, -82.6534, True),    # Tropicana Field — fixed dome
    "TEX": (32.7473, -97.0827, True),    # Globe Life Field — retractable, AC closed
    "TOR": (43.6414, -79.3894, True),    # Rogers Centre — retractable
    "WSH": (38.8730, -77.0074, False),
}

_MLB_TEAM_IDS: dict[str, int] = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KC":  118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD":  135, "SF":  137, "SEA": 136,
    "STL": 138, "TB":  139, "TEX": 140, "TOR": 141, "WSH": 120,
}

_MIN_SCALE = 0.90
_MAX_SCALE = 1.05  # weather is a minor layer — keep its own sub-clamp tight

# ---------------------------------------------------------------------------
# Process-level caches
# ---------------------------------------------------------------------------
_HOME_ABBR_CACHE: dict[str, str | None] = {}          # "teamA:teamB:date" → home abbr
_WEATHER_CACHE:   dict[str, dict | None] = {}          # "abbr:date" → weather dict


def _fetch(url: str, timeout: int = _TIMEOUT) -> dict | None:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (URLError, Exception) as exc:
        logger.debug(f"[weather_intel] fetch failed: {exc!r}  url={url}")
        return None


def _get_home_abbr(team_abbr: str, opp_abbr: str, game_date: str) -> str | None:
    """
    Determine which of the two teams is playing at home on game_date, so we
    know which park's weather applies. One schedule lookup, cached.
    """
    cache_key = f"{team_abbr}:{opp_abbr}:{game_date}"
    if cache_key in _HOME_ABBR_CACHE:
        return _HOME_ABBR_CACHE[cache_key]

    team_id = _MLB_TEAM_IDS.get(team_abbr.upper())
    result: str | None = None

    if team_id:
        data = _fetch(f"{_MLB_BASE}/schedule?sportId=1&date={game_date}&teamId={team_id}")
        if data:
            for date_block in data.get("dates", []):
                for game in date_block.get("games", []):
                    home = (game.get("teams", {}).get("home", {})
                            .get("team", {}).get("abbreviation", ""))
                    away = (game.get("teams", {}).get("away", {})
                            .get("team", {}).get("abbreviation", ""))
                    if home and (home.upper() in (team_abbr.upper(), opp_abbr.upper())):
                        result = home.upper()
                        break
                    if away and (away.upper() in (team_abbr.upper(), opp_abbr.upper())
                                 and home):
                        result = home.upper()
                        break
                if result:
                    break

    _HOME_ABBR_CACHE[cache_key] = result
    return result


def get_weather(team_abbr: str, opp_abbr: str, game_date: str | None = None) -> dict | None:
    """
    Return {"temp_f": float, "wind_mph": float, "is_dome": bool, "home_abbr": str}
    for the park hosting this game, or None if unavailable/enclosed/out of
    forecast range. Never raises.
    """
    try:
        game_date = game_date or date.today().isoformat()
        home_abbr = _get_home_abbr(team_abbr, opp_abbr, game_date)
        if not home_abbr:
            return None

        venue = _MLB_VENUES.get(home_abbr)
        if not venue:
            return None
        lat, lon, enclosed = venue
        if enclosed:
            logger.debug(f"[weather_intel] {home_abbr} is enclosed — skipping fetch")
            return {"temp_f": None, "wind_mph": None, "is_dome": True, "home_abbr": home_abbr}

        cache_key = f"{home_abbr}:{game_date}"
        if cache_key in _WEATHER_CACHE:
            return _WEATHER_CACHE[cache_key]

        url = (
            f"{_OPEN_METEO}?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,wind_speed_10m_max"
            f"&temperature_unit=fahrenheit&wind_speed_unit=mph"
            f"&timezone=auto&start_date={game_date}&end_date={game_date}"
        )
        data = _fetch(url)
        result: dict | None = None

        if data:
            daily = data.get("daily", {})
            tmax = (daily.get("temperature_2m_max") or [None])[0]
            tmin = (daily.get("temperature_2m_min") or [None])[0]
            wind = (daily.get("wind_speed_10m_max") or [None])[0]
            if tmax is not None and tmin is not None:
                result = {
                    "temp_f":   round((tmax + tmin) / 2.0, 1),
                    "wind_mph": round(wind, 1) if wind is not None else None,
                    "is_dome":  False,
                    "home_abbr": home_abbr,
                }
            else:
                logger.debug(
                    f"[weather_intel] no forecast for {home_abbr} {game_date} "
                    "(likely outside Open-Meteo's ~16-day forecast window)"
                )

        _WEATHER_CACHE[cache_key] = result
        return result

    except Exception as exc:
        logger.debug(f"[weather_intel] get_weather failed: {exc!r}")
        return None


def get_k_weather_scale(team_abbr: str, opp_abbr: str, game_date: str | None = None) -> float:
    """
    Return a small multiplier on projected strikeouts from ballpark
    temperature. Neutral (1.0) for domes, missing data, out-of-forecast-
    range dates, or any error. Deliberately does NOT use wind — see module
    docstring for why. Sub-clamped to [0.90, 1.05]: this is a soft, minor
    layer and shouldn't swing a K projection on its own.
    """
    try:
        wx = get_weather(team_abbr, opp_abbr, game_date)
        if not wx or wx.get("is_dome") or wx.get("temp_f") is None:
            return 1.0

        temp_f = wx["temp_f"]
        if temp_f >= _COLD_THRESHOLD_F:
            return 1.0  # no established "warm helps whiffs" effect — stay neutral

        # Cold grip/velo penalty: modest linear falloff below 55°F, roughly
        # -1% K-rate per 10°F below the threshold, blended toward 1.0 at a
        # low strength since this is a soft heuristic, not a measured split.
        delta = _COLD_THRESHOLD_F - temp_f
        raw = 1.0 - (delta / 10.0) * 0.01
        strength = 0.30
        scale = 1.0 + (raw - 1.0) * strength
        scale = round(max(_MIN_SCALE, min(_MAX_SCALE, scale)), 4)
        logger.debug(
            f"[weather_intel] {wx.get('home_abbr')} {game_date}: "
            f"{temp_f}°F → K scale={scale:.4f}"
        )
        return scale

    except Exception as exc:
        logger.debug(f"[weather_intel] get_k_weather_scale failed: {exc!r}")
        return 1.0
