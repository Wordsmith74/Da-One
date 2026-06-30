"""
Market Governance — Publication Whitelist & Priority Ranking
============================================================

Defines which markets may generate official public picks (Telegram, MiniApp,
Discord, website) and the priority order when multiple markets score similarly.

PUBLICATION_MARKETS
    Only these markets may appear in any public-facing output.
    Scope is defined by core/market_gate.py (System Scope Definition Layer);
    the whitelist here is the publication-facing reflection of that scope.

MARKET_PRIORITY
    Lower number = higher priority.  When picks from multiple markets score
    similarly, the engine prefers markets with lower priority numbers for the
    Nuke / Diamond slots.

Approved markets (normalized internal keys):
    MLB  : pitcher_strikeouts, first_5_total, first_5_ml, first_5_rl, nrfi, yrfi
    WNBA : player_assists, player_rebounds, moneyline

Priority order (per spec):
    1  MLB  pitcher_strikeouts
    2  MLB  first_5_total
    3  MLB  first_5_ml
    4  MLB  first_5_rl
    5  WNBA player_assists
    6  WNBA player_rebounds
    7  WNBA moneyline
    8  MLB  nrfi
    9  MLB  yrfi
"""

from core.decision_gatekeeper import market_normalized

# ---------------------------------------------------------------------------
# Publication whitelist — market keys are normalized (lowercase_underscore)
# ---------------------------------------------------------------------------

PUBLICATION_MARKETS: dict[str, frozenset[str]] = {
    "MLB": frozenset({
        "pitcher_strikeouts",
        "first_5_total",
        "first_5_ml",
        "first_5_rl",
        "nrfi",
        "yrfi",
    }),
    "WNBA": frozenset({
        "player_assists",
        "player_rebounds",
        "moneyline",
    }),
    # NBA: no markets in publication scope
}

# ---------------------------------------------------------------------------
# Priority ranking — lower number = higher priority (used as tiebreaker)
# ---------------------------------------------------------------------------

MARKET_PRIORITY: dict[tuple[str, str], int] = {
    ("MLB",  "pitcher_strikeouts"): 1,
    ("MLB",  "first_5_total"):      2,
    ("MLB",  "first_5_ml"):         3,
    ("MLB",  "first_5_rl"):         4,
    ("WNBA", "player_assists"):     5,
    ("WNBA", "player_rebounds"):    6,
    ("WNBA", "moneyline"):          7,
    ("MLB",  "nrfi"):               8,
    ("MLB",  "yrfi"):               9,
}

_DEFAULT_PRIORITY = 99  # any market not in the ranking table


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_publication_eligible(sport: str, market: str) -> bool:
    """
    Return True if this (sport, market) pair may appear in any public output.

    Both arguments are normalised internally so callers may pass raw strings
    (e.g. 'Strikeouts') or normalised ones ('pitcher_strikeouts').
    """
    mkt = market_normalized(market)
    return mkt in PUBLICATION_MARKETS.get(sport.upper(), frozenset())


def publication_priority(sport: str, market: str) -> int:
    """
    Return the Tier-1 market priority rank for this (sport, market) pair.

    Lower is better (1 = highest priority).  Non-publication markets and
    unlisted publication markets return _DEFAULT_PRIORITY (99).
    """
    mkt = market_normalized(market)
    return MARKET_PRIORITY.get((sport.upper(), mkt), _DEFAULT_PRIORITY)
