"""
api_connector.py

Handles all inbound data from external sports/odds APIs.

Timestamp contract
------------------
Every timestamp that enters the engine through this module is immediately
normalized in two steps:
  1. localize_utc()    — stamp naive datetimes as UTC (APIs often omit tzinfo)
  2. convert_to_est()  — convert to America/New_York for any display logic

Internal engine objects always carry UTC-aware datetimes; EST is only applied
at the point of display (Telegram messages, reports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.time_utils import (
    UTC,
    convert_to_est,
    format_est,
    format_est_short,
    is_in_future,
    is_within_hours,
    localize_utc,
    now_utc,
    format_utc_iso,
)


# ---------------------------------------------------------------------------
# Canonical game data structure
# ---------------------------------------------------------------------------

@dataclass
class GameInfo:
    """
    Normalised representation of a single game received from an external API.

    All datetime fields are stored UTC-aware internally. Call .game_time_est
    or .display_time for the America/New_York representation.
    """

    game_id:       str
    sport:         str
    home_team:     str
    away_team:     str
    game_time_utc: datetime          # always UTC-aware after normalisation
    venue:         str = ""
    raw_metrics:   dict[str, Any] = field(default_factory=dict)
    raw_odds:      dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Enforce UTC on construction — guard against callers passing naive dt
        self.game_time_utc = localize_utc(self.game_time_utc)

    # ── Display helpers (EST) ────────────────────────────────────────────────

    @property
    def game_time_est(self) -> datetime:
        """Game start time converted to America/New_York."""
        return convert_to_est(self.game_time_utc)

    @property
    def display_time(self) -> str:
        """Human-readable EST time for Telegram messages, e.g. '7:30 PM ET'."""
        return format_est_short(self.game_time_utc)

    @property
    def display_date(self) -> str:
        """Full date in EST, e.g. 'Saturday, May 31 2026'."""
        return format_est(self.game_time_utc, "%A, %B %d %Y")

    @property
    def display_datetime(self) -> str:
        """Full date + time in EST, e.g. 'Saturday, May 31 2026  07:30 PM ET'."""
        return format_est(self.game_time_utc)

    # ── Status helpers ───────────────────────────────────────────────────────

    @property
    def is_upcoming(self) -> bool:
        """True if the game has not yet started (UTC comparison)."""
        return is_in_future(self.game_time_utc)

    @property
    def is_imminent(self) -> bool:
        """True if the game starts within the next 3 hours (UTC comparison)."""
        return is_within_hours(self.game_time_utc, hours=3.0)

    def minutes_until(self) -> float:
        """Minutes from now (UTC) until game start. Negative if already started."""
        from core.time_utils import utc_diff_minutes
        return utc_diff_minutes(self.game_time_utc, now_utc())


# ---------------------------------------------------------------------------
# Timestamp normalisation — called immediately on raw API payloads
# ---------------------------------------------------------------------------

def normalize_api_timestamp(raw: str | datetime | int | float) -> datetime:
    """
    Accept any timestamp format returned by external APIs and return a
    UTC-aware datetime.

    Supported formats:
      - ISO-8601 string:  '2026-05-31T23:00:00Z' or '2026-05-31T19:00:00-04:00'
      - Naive ISO string: '2026-05-31T23:00:00'  (assumed UTC)
      - Unix timestamp:   1748732400  (int or float, assumed UTC)
      - datetime object:  naive (assumed UTC) or already aware

    Args:
        raw: The timestamp value from the API response.

    Returns:
        UTC-aware datetime.

    Raises:
        ValueError: If the format cannot be parsed.
    """
    if isinstance(raw, (int, float)):
        dt = datetime.fromtimestamp(raw, tz=UTC)
        return dt

    if isinstance(raw, datetime):
        return localize_utc(raw)

    if isinstance(raw, str):
        raw = raw.strip()
        # Handle trailing 'Z' (UTC indicator)
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
            return localize_utc(dt)
        except ValueError:
            pass
        # Fallback: try common formats
        for fmt in (
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(raw, fmt)
                return localize_utc(dt)
            except ValueError:
                continue
        raise ValueError(
            f"Cannot parse timestamp '{raw}'. Expected ISO-8601 string, "
            "Unix int/float, or datetime object."
        )

    raise TypeError(f"Unsupported timestamp type: {type(raw).__name__}")


# ---------------------------------------------------------------------------
# Raw API response parser — entry point for inbound data
# ---------------------------------------------------------------------------

def parse_game_from_api(raw_game: dict[str, Any], sport: str) -> GameInfo:
    """
    Convert a raw API response dict into a normalised GameInfo.

    Timestamp fields are normalised to UTC immediately on entry.

    This function is the adapter layer between whatever shape an external
    API returns and the internal GameInfo contract. Add sport-specific
    field mappings here as data connectors are built.

    Args:
        raw_game: Raw dict from an API response.
        sport:    Sport identifier ('WNBA', 'NBA', 'MLB').

    Returns:
        GameInfo with all datetimes UTC-aware.
    """
    sport = sport.upper()

    # ── Timestamp: normalise immediately regardless of API format ────────────
    raw_ts = (
        raw_game.get("game_time")
        or raw_game.get("start_time")
        or raw_game.get("commence_time")     # The Odds API field name
        or raw_game.get("gameDate")          # NBA Stats API field name
        or raw_game.get("timestamp")
    )
    if raw_ts is None:
        raise ValueError(
            f"No recognizable timestamp field in game payload: {list(raw_game.keys())}"
        )

    game_time_utc = normalize_api_timestamp(raw_ts)

    # ── Team names ───────────────────────────────────────────────────────────
    home = (
        raw_game.get("home_team")
        or raw_game.get("homeTeam", {}).get("abbreviation", "")
        or raw_game.get("home", "")
    )
    away = (
        raw_game.get("away_team")
        or raw_game.get("awayTeam", {}).get("abbreviation", "")
        or raw_game.get("away", "")
    )

    return GameInfo(
        game_id=str(raw_game.get("id") or raw_game.get("game_id") or ""),
        sport=sport,
        home_team=str(home),
        away_team=str(away),
        game_time_utc=game_time_utc,
        venue=str(raw_game.get("venue") or raw_game.get("arena") or ""),
        raw_metrics=raw_game.get("metrics", {}),
        raw_odds=raw_game.get("odds", {}),
    )


def parse_games_from_api(raw_games: list[dict[str, Any]], sport: str) -> list[GameInfo]:
    """
    Parse a list of raw API game dicts, silently skipping any that fail parsing.

    Args:
        raw_games: List of raw game dicts from the API.
        sport:     Sport identifier.

    Returns:
        List of successfully parsed GameInfo objects.
    """
    games: list[GameInfo] = []
    for raw in raw_games:
        try:
            games.append(parse_game_from_api(raw, sport))
        except (ValueError, KeyError, TypeError) as exc:
            print(f"[api_connector] Skipping malformed game entry: {exc}")
    return games


# ---------------------------------------------------------------------------
# Game-time gate — used by DecisionOrchestrator integration
# ---------------------------------------------------------------------------

def filter_upcoming_games(
    games: list[GameInfo],
    within_hours: float = 24.0,
) -> list[GameInfo]:
    """
    Return only games that start in the future and within `within_hours`.

    All comparisons use UTC to prevent DST-induced time-jump bugs.

    Args:
        games:        List of GameInfo objects.
        within_hours: Cutoff window in hours from now (UTC).

    Returns:
        Filtered and time-sorted list of GameInfo objects.
    """
    upcoming = [g for g in games if is_within_hours(g.game_time_utc, within_hours)]
    return sorted(upcoming, key=lambda g: g.game_time_utc)
