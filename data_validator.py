"""
data_validator.py

Data Integrity Validation — Rule 3 of the data resilience protocol.

Before any payload is passed to the Bayesian inference engine, the
validator checks:

  1. The response is non-empty.
  2. The top-level structure contains the expected keys for the endpoint
     type (schedule / injuries / statistics).
  3. Nested records are not uniformly null/empty — a response that
     returns an empty list is treated as "structurally incomplete" and
     triggers the next source in the waterfall.

Public API
----------
  validate_schedule(data)    → ValidationResult
  validate_injuries(data)    → ValidationResult
  validate_statistics(data)  → ValidationResult
  validate_generic(data, required_keys)  → ValidationResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("betting_bot")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid:          bool
    reason:         str                  # human-readable summary
    missing_fields: list[str] = field(default_factory=list)

    def log(self, context: str = "") -> None:
        prefix = f"[VALIDATE]{' ' + context if context else ''}"
        if self.valid:
            logger.debug(f"{prefix} OK — {self.reason}")
        else:
            logger.warning(
                f"{prefix} FAILED — {self.reason}"
                + (
                    f"  missing={self.missing_fields}"
                    if self.missing_fields
                    else ""
                )
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(reason: str, missing: list[str] | None = None) -> ValidationResult:
    return ValidationResult(valid=False, reason=reason, missing_fields=missing or [])


def _ok(reason: str = "data complete") -> ValidationResult:
    return ValidationResult(valid=True, reason=reason)


# ---------------------------------------------------------------------------
# Endpoint-specific validators
# ---------------------------------------------------------------------------

def validate_schedule(data: dict[str, Any] | None) -> ValidationResult:
    """
    Validate an ESPN schedule response.

    Required structure
    ------------------
    {
      "events": [
        {
          "date": "<ISO string>",
          "status": {"type": {"completed": <bool>}},
          ...
        },
        ...
      ]
    }
    """
    if not data:
        return _fail("empty or null response", ["root"])

    events = data.get("events")
    if events is None:
        return _fail("missing top-level 'events' key", ["events"])

    if not isinstance(events, list):
        return _fail(f"'events' is not a list (got {type(events).__name__})", ["events"])

    if len(events) == 0:
        return _fail("'events' list is empty — no schedule data for this period", ["events"])

    # Spot-check first event for critical sub-fields
    missing: list[str] = []
    first = events[0]
    if not isinstance(first, dict):
        return _fail("first event is not a dict", ["events[0]"])
    if "date" not in first:
        missing.append("events[0].date")
    if "status" not in first:
        missing.append("events[0].status")

    if missing:
        return _fail("incomplete event structure", missing)

    return _ok(f"{len(events)} event(s) present")


def validate_injuries(data: dict[str, Any] | None) -> ValidationResult:
    """
    Validate an ESPN injuries response.

    Accepts either a completely empty injuries list (no injuries for this
    team) or a populated list.  The key check is that the top-level
    structure is a dict and the 'injuries' key is present (even if []).
    """
    if not data:
        return _fail("empty or null response", ["root"])

    if not isinstance(data, dict):
        return _fail(f"response is not a dict (got {type(data).__name__})", ["root"])

    # 'injuries' key may be absent on endpoints that return no data
    injuries = data.get("injuries")
    if injuries is None:
        # Some ESPN endpoints wrap injuries under a different key;
        # treat an entirely missing key as a structural failure.
        # An empty list [] is acceptable (no injuries reported).
        return _fail("'injuries' key absent from response", ["injuries"])

    if not isinstance(injuries, list):
        return _fail(
            f"'injuries' is not a list (got {type(injuries).__name__})",
            ["injuries"],
        )

    # Spot-check first entry if present
    if injuries:
        first = injuries[0]
        if not isinstance(first, dict):
            return _fail("first injury record is not a dict", ["injuries[0]"])
        if "athlete" not in first:
            return _fail("injury record missing 'athlete' field", ["injuries[0].athlete"])

    return _ok(f"{len(injuries)} injury record(s) present")


def validate_statistics(data: dict[str, Any] | None) -> ValidationResult:
    """
    Validate an ESPN statistics response.

    ESPN statistics endpoints nest stats inside various wrapper keys.
    This validator checks that the response is non-empty and contains
    at least one recognisable stat-bearing structure.
    """
    if not data:
        return _fail("empty or null response", ["root"])

    if not isinstance(data, dict):
        return _fail(f"response is not a dict (got {type(data).__name__})", ["root"])

    # ESPN statistics responses vary by sport but always contain one of:
    # "statistics", "splits", "categories", or "athletes"
    stat_keys = {"statistics", "splits", "categories", "athletes", "results"}
    found = stat_keys & set(data.keys())

    if not found:
        return _fail(
            "no recognised statistics key found in response "
            f"(expected one of: {sorted(stat_keys)})",
            sorted(stat_keys - set(data.keys())),
        )

    # Check the first found key is not empty
    key = next(iter(found))
    value = data[key]
    if isinstance(value, (list, dict)) and not value:
        return _fail(f"'{key}' present but empty — no stat data available", [key])

    return _ok(f"statistics key '{key}' present and non-empty")


def validate_generic(
    data: dict[str, Any] | None,
    required_keys: list[str],
) -> ValidationResult:
    """
    Generic validator: checks that the response is a non-empty dict
    containing all keys in *required_keys*.
    """
    if not data:
        return _fail("empty or null response", ["root"])

    if not isinstance(data, dict):
        return _fail(f"response is not a dict (got {type(data).__name__})", ["root"])

    missing = [k for k in required_keys if k not in data]
    if missing:
        return _fail(f"missing required keys: {missing}", missing)

    return _ok("all required keys present")
