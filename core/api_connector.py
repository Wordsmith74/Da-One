"""
api_connector.py

Small, dependency-free helpers for normalising data coming back from
external odds providers (The Odds API, PropLine, etc.) into the internal
representations the rest of the engine expects.

Currently this just handles timestamp normalisation, but it's the natural
home for any future "make provider X's payload look like provider Y's"
glue code.
"""

from __future__ import annotations

from datetime import datetime, timezone


def normalize_api_timestamp(raw: str | datetime) -> datetime:
    """
    Convert a raw timestamp from an odds provider into a UTC-aware datetime.

    Providers (The Odds API, PropLine) return ISO-8601 strings, almost
    always in the form ``"2026-06-01T19:00:00Z"``. This accepts that form
    (and the ``+00:00`` offset form) and always returns a timezone-aware
    UTC datetime, never a naive one -- callers (e.g. core.time_utils.
    convert_to_est) assume tz-aware input.

    Args:
        raw: An ISO-8601 timestamp string, or an already-parsed datetime.

    Returns:
        UTC-aware datetime.

    Raises:
        ValueError: if *raw* is empty or cannot be parsed.
    """
    if isinstance(raw, datetime):
        dt = raw
    else:
        if not raw:
            raise ValueError("normalize_api_timestamp: empty timestamp")
        # datetime.fromisoformat doesn't accept a trailing 'Z' before 3.11
        # semantics solidified -- normalise it to +00:00 for safety across
        # the Python versions this engine might run under.
        s = raw.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
