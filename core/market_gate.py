"""
core/market_gate.py — System Scope Definition Layer
=====================================================

This is NOT a filter step.  It defines which markets *exist* to the model.
Any candidate whose (sport, market) pair is not in ALLOWED_MARKETS is treated
as if it was never ingested: no simulation, no edge/confidence scoring, no
gatekeeper entry, no audit trail beyond a brief rejection log.

Allowed scope
─────────────
MLB:  Moneyline · Run Line (Spread) · Game Total · Pitcher Strikeouts
WNBA: Game Total · Player Assists · Player Rebounds · Moneyline  (unchanged)

NBA:  (no markets in scope — all candidates blocked before modeling)

Normalization aliases
─────────────────────
Raw labels coming from the odds feed or display strings are normalised to the
internal keys used throughout the engine.  All comparisons use these keys.
"""

from __future__ import annotations

from typing import Any

from core.decision_gatekeeper import market_normalized  # re-use existing normalizer

# ---------------------------------------------------------------------------
# Allowed markets (internal normalized keys)
# ---------------------------------------------------------------------------

ALLOWED_MARKETS: dict[str, frozenset[str]] = {
    # MLB scope: pitcher strikeouts only. Moneyline / run line / game total
    # were removed from scope per explicit request -- see git history around
    # 2026-07-10 for the graded sample that motivated it (run_line 37.5% win
    # rate n=8, moneyline weakest market overall, game_total's own floor was
    # only ever derived from an n=2 sample). Candidates for those three
    # markets are now blocked here exactly like an out-of-scope NBA
    # candidate would be -- no simulation, no gatekeeper entry, no picks.
    # core/game_markets.py's _MARKET_BUNDLE["MLB"] and the MLB game-total
    # fetch in run_pipeline.py are ALSO disabled (belt and suspenders / to
    # stop spending API credits on candidates that would just be blocked
    # here anyway) -- this entry is the authoritative one either way.
    "MLB": frozenset({
        "pitcher_strikeouts",   # Pitcher Ks prop      (Bayesian, core/player_props.py)
    }),
    "WNBA": frozenset({
        "game_total",           # Full-game total     (odds_client.py fetch_todays_candidates)
        "player_assists",       # Assists prop         (Bayesian)
        "player_rebounds",      # Rebounds prop        (Bayesian)
        "moneyline",            # Full-game moneyline  (precomputed)
    }),
    # NBA: empty — every NBA candidate is blocked before modeling
}

# ---------------------------------------------------------------------------
# Alias table — raw display names / market-key variants → normalized key
# ---------------------------------------------------------------------------

MARKET_ALIASES: dict[str, str] = {
    # Full-game total (odds_client.py candidates carry market="Totals" with
    # no market_key field, so this only reaches market_normalized()'s plain
    # lowercase/underscore form -- must stay distinct from the F5 aliases
    # below, which are a different bet type).
    "totals":                    "game_total",
    "total":                     "game_total",
    "game_total":                "game_total",
    # F5 variants
    "f5":                        "first_5_total",
    "first 5 innings":           "first_5_total",
    "5 inning line":             "first_5_total",
    "f5 total":                  "first_5_total",
    "first_5_total":             "first_5_total",
    "f5 moneyline":              "first_5_ml",
    "first_5_ml":                "first_5_ml",
    "f5 run line":               "first_5_rl",
    "first_5_rl":                "first_5_rl",
    # NRFI / YRFI
    "nrfi":                      "nrfi",
    "no run 1st inning":         "nrfi",
    "no run first inning":       "nrfi",
    "yrfi":                      "yrfi",
    "yes run first inning":      "yrfi",
    # Pitcher Ks
    "strikeouts":                "pitcher_strikeouts",
    "ks":                        "pitcher_strikeouts",
    "k prop":                    "pitcher_strikeouts",
    "pitcher strikeouts":        "pitcher_strikeouts",
    "pitcher_strikeouts":        "pitcher_strikeouts",
    # WNBA props
    "assists":                   "player_assists",
    "ast":                       "player_assists",
    "player_assists":            "player_assists",
    "rebounds":                  "player_rebounds",
    "reb":                       "player_rebounds",
    "player_rebounds":           "player_rebounds",
    # Moneyline
    "moneyline":                 "moneyline",
    "ml":                        "moneyline",
    "h2h":                       "moneyline",
    # Run line / spread — MLB has no separate spread market from the run
    # line (see src/clients/oddspapi_client.py), so all these names alias
    # to the same internal key.
    "run_line":                  "run_line",
    "runline":                   "run_line",
    "rl":                        "run_line",
    "spread":                    "run_line",
    "spreads":                   "run_line",
}


# ---------------------------------------------------------------------------
# Core gate helpers
# ---------------------------------------------------------------------------

def _resolve_market_key(candidate: dict[str, Any]) -> str:
    """
    Extract the normalized market key from a candidate dict.

    Candidates from game_markets.py carry a ``market_key`` field with an
    already-normalized internal key (e.g. "first_5_total").  Candidates from
    player_props.py carry only a ``market`` display name (e.g. "Strikeouts").
    """
    # 1. Prefer the internal key set by game_markets.py
    mkt_key = candidate.get("market_key", "")
    if mkt_key:
        alias = MARKET_ALIASES.get(mkt_key.strip().lower(), mkt_key)
        return alias

    # 2. Fall back to display name via existing normalizer + alias table
    raw = candidate.get("market", "")
    if not raw:
        return ""
    via_normalizer = market_normalized(raw)          # handles "Strikeouts" → "pitcher_strikeouts"
    return MARKET_ALIASES.get(via_normalizer, via_normalizer)


def is_market_allowed(sport: str, candidate: dict[str, Any]) -> bool:
    """
    Return True if the candidate's market is in scope for *sport*.

    Parameters
    ----------
    sport:      "MLB", "NBA", "WNBA"
    candidate:  raw candidate dict (must have at least a ``market`` or ``market_key`` field)
    """
    key = _resolve_market_key(candidate)
    return key in ALLOWED_MARKETS.get(sport.upper(), frozenset())


def filter_candidates(
    candidates: list[dict[str, Any]],
    sport: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Partition *candidates* into (allowed, blocked) lists.

    Returns
    -------
    allowed  — candidates that may proceed to modeling
    blocked  — candidates that must not enter the simulation pipeline
    """
    allowed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for c in candidates:
        (allowed if is_market_allowed(sport, c) else blocked).append(c)
    return allowed, blocked


def log_market_filter_summary(
    sport: str,
    total: int,
    allowed: int,
    blocked: int,
    blocked_reasons: dict[str, int],
) -> None:
    """
    Print the per-run market filter summary (requirement §6 of the patch spec).
    Uses print() so it appears in the same stdout stream as the rest of the engine.
    """
    lines = [
        f"[market_gate] MARKET FILTER SUMMARY [{sport}]:",
        f"  total candidates received : {total}",
        f"  allowed markets passed    : {allowed}",
        f"  blocked markets rejected  : {blocked}",
    ]
    if blocked_reasons:
        lines.append("  rejection reasons:")
        for reason, count in sorted(blocked_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"    MARKET_NOT_ALLOWED({reason}): {count}")
    print("\n".join(lines), flush=True)
