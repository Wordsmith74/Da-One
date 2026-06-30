"""
time_utils.py

Centralized datetime / timezone utilities for the entire engine.

Rules enforced here:
  - All internal storage and comparisons use UTC-aware datetimes.
  - All user-facing display (Telegram messages, reports) converts to EST/EDT
    via convert_to_est() so viewers always see America/New_York time,
    regardless of the server's local clock or DST transitions.
  - Naive datetimes are never passed around; localize_utc() is the one
    place to stamp incoming timestamps with UTC when the source doesn't
    provide tzinfo.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Canonical timezone objects — import these instead of re-constructing
# ---------------------------------------------------------------------------

UTC = timezone.utc
EST = ZoneInfo("America/New_York")   # handles EST ↔ EDT automatically


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_to_est(dt: datetime) -> datetime:
    """
    Convert any datetime to America/New_York (EST winter / EDT summer).

    If the input is timezone-naive it is assumed to be UTC and stamped
    accordingly before conversion — this prevents silent wrong-time bugs
    when external APIs return naive timestamps.

    Args:
        dt: Any datetime object (aware or naive).

    Returns:
        A timezone-aware datetime in America/New_York.

    Examples:
        >>> from datetime import datetime, timezone
        >>> utc_dt = datetime(2026, 1, 15, 18, 0, tzinfo=timezone.utc)
        >>> est_dt = convert_to_est(utc_dt)
        >>> est_dt.strftime("%H:%M %Z")
        '13:00 EST'
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(EST)


def localize_utc(dt: datetime) -> datetime:
    """
    Stamp a naive datetime as UTC, or normalise an aware one to UTC.

    Use this as the first step when receiving timestamps from external
    APIs that don't include timezone info.

    Args:
        dt: Naive or aware datetime.

    Returns:
        UTC-aware datetime.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def now_utc() -> datetime:
    """Return the current moment as a UTC-aware datetime."""
    return datetime.now(UTC)


def now_est() -> datetime:
    """Return the current moment in America/New_York."""
    return datetime.now(EST)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_est(
    dt: datetime,
    fmt: str = "%A, %B %d %Y  %I:%M %p ET",
) -> str:
    """
    Convert dt to EST and return it as a display string.

    Default format: 'Saturday, May 31 2026  07:30 PM ET'

    Args:
        dt:  Any datetime (naive assumed UTC).
        fmt: strftime format string. 'ET' suffix is appended automatically
             via the %Z token if you include it in fmt.

    Returns:
        Human-readable EST string.
    """
    return convert_to_est(dt).strftime(fmt)


def format_est_short(dt: datetime) -> str:
    """Return a compact EST time string, e.g. '7:30 PM ET'."""
    return convert_to_est(dt).strftime("%-I:%M %p ET")


def format_est_date(dt: datetime) -> str:
    """Return just the date portion in EST, e.g. 'Saturday, May 31 2026'."""
    return convert_to_est(dt).strftime("%A, %B %d %Y")


def format_utc_iso(dt: datetime) -> str:
    """Return an ISO-8601 UTC string suitable for DB storage."""
    return localize_utc(dt).isoformat()


# ---------------------------------------------------------------------------
# Comparison helpers (all comparisons must stay in UTC)
# ---------------------------------------------------------------------------

def is_in_future(dt: datetime) -> bool:
    """
    Return True if dt is strictly after the current UTC moment.

    Both operands are normalised to UTC so DST shifts never affect the result.
    """
    return localize_utc(dt) > now_utc()


def is_within_hours(dt: datetime, hours: float) -> bool:
    """
    Return True if dt is between now and now + hours (UTC).

    Useful for filtering games that are starting soon.
    """
    from datetime import timedelta
    now = now_utc()
    target = localize_utc(dt)
    return now <= target <= now + timedelta(hours=hours)


def utc_diff_minutes(dt_a: datetime, dt_b: datetime) -> float:
    """
    Return the signed difference (dt_a − dt_b) in minutes, using UTC.

    Args:
        dt_a: First datetime (naive assumed UTC).
        dt_b: Second datetime (naive assumed UTC).

    Returns:
        Float minutes. Positive → dt_a is later than dt_b.
    """
    delta = localize_utc(dt_a) - localize_utc(dt_b)
    return delta.total_seconds() / 60
