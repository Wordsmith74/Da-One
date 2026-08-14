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
    MLB  : pitcher_strikeouts, nrfi, yrfi
           -- moneyline, run_line, game_total removed from scope 2026-08-05.
              First cut 2026-07-10 on thin evidence (run_line 37.5% win n=8,
              moneyline called weakest market overall); confirmed by
              output/threshold_recommendations.json (generated 2026-08-04,
              n=31 graded): MLB market_class=game -> "insufficient_data" --
              no edge/confidence cutoff across all three combined cleared
              breakeven ROI at n >= 15. Final graded numbers by then:
              moneyline 35.3% win / -0.52 units (n=34), run_line 25.0% win /
              -0.16 units (n=12), game_total never grew past its original n=2.
              first_5_total, first_5_ml, first_5_rl remain OUT of scope too
              (see core/market_gate.py). pitcher_strikeouts uses its own tier
              thresholds (core/decision_gatekeeper.py's
              _MARKET_TIER_THRESHOLDS), not the sport-wide MLB thresholds.
              nrfi/yrfi were re-added to scope alongside the first-inning
              tiered-handicapping wiring (models/nrfi_handicapper.py,
              data/statcast_first_inning.py, etc.) -- see
              core/market_gate.py's ALLOWED_MARKETS comment for why they were
              caught by the 2026-07-10/08-05 removal in the first place even
              though that removal's own evidence never covered first-inning
              markets. This whitelist has to be updated in lockstep with
              ALLOWED_MARKETS -- a market can clear the scope gate, generate
              a real candidate, survive the gatekeeper's edge/confidence
              floors, and still get silently dropped right here at the very
              last step if this table isn't kept in sync (this is exactly
              what happened before this update: real, approved MLB yrfi
              picks were generated and then dropped at publication with no
              error, just a one-line warning easy to miss in a long log).
    WNBA : player_assists, player_rebounds, moneyline, game_total  (unchanged)

Priority order (per spec):
    4  MLB  pitcher_strikeouts
    5  WNBA player_assists
    6  WNBA player_rebounds
    7  WNBA moneyline
    8  MLB  nrfi
    9  MLB  yrfi
    11 WNBA game_total
"""

from core.decision_gatekeeper import market_normalized

# ---------------------------------------------------------------------------
# Publication whitelist — market keys are normalized (lowercase_underscore)
# ---------------------------------------------------------------------------

PUBLICATION_MARKETS: dict[str, frozenset[str]] = {
    # F5 remains out of scope (see core/market_gate.py). MLB moneyline /
    # run_line / game_total removed from scope 2026-08-05 -- see this
    # module's docstring for the evidence. market_gate.py's ALLOWED_MARKETS
    # is the authoritative scope check (blocks these before simulation);
    # this whitelist is kept in sync with it so it can't silently re-open
    # publication for a market that scope no longer allows, OR silently
    # keep blocking a market that scope now allows (see docstring's note on
    # nrfi/yrfi -- the latter failure mode is exactly what happened here
    # until this update).
    "MLB": frozenset({
        "pitcher_strikeouts",
        "nrfi",
    }),
    "WNBA": frozenset({
        "player_assists",
        "player_rebounds",
        "moneyline",
        "game_total",
    }),
    # NBA: no markets in publication scope
}

# ---------------------------------------------------------------------------
# Priority ranking — lower number = higher priority (used as tiebreaker)
# ---------------------------------------------------------------------------

MARKET_PRIORITY: dict[tuple[str, str], int] = {
    ("MLB",  "pitcher_strikeouts"): 4,
    ("WNBA", "player_assists"):     5,
    ("WNBA", "player_rebounds"):    6,
    ("WNBA", "moneyline"):          7,
    ("MLB",  "nrfi"):               8,
    ("WNBA", "game_total"):         11,
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
