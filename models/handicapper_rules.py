"""
Professional handicapper discipline, encoded as functions.

This is the answer to "what do the top profitable MLB/WNBA handicappers
actually do before placing a bet" -- not folklore, the recurring, well-documented
practices that separate long-run-profitable bettors from break-even/losing ones:

  1. Track closing line value (CLV), not just win/loss. Beating the closing
     line consistently is the most reliable indicator of real long-run edge --
     more reliable than a small-sample win rate. -> clv_pct() / log_clv()
  2. Never bet on too-small a sample. -> enforced upstream via
     sport_config.MLB['min_batters_faced_for_k_prop'] and
     WNBA['min_games_for_player_prop'] (see ramp_detection.py / run_pipeline.py)
  3. Size bets with a fraction of Kelly, never full Kelly or flat-unit-by-feel.
     -> kelly_stake()
  4. Don't bet too many games a day -- selectivity beats volume.
     -> sport_config.*['max_picks_per_day'] (enforced in run_pipeline._apply_daily_caps)
  5. Shop/compare against the closing/consensus line, don't anchor to one book.
     -> needs a real multi-book feed; stubbed here as compare_to_consensus()
        until a multi-book odds source is wired in (currently single-book via
        The Odds API's first returned price -- see data/fetch.py TODOs)
  6. Avoid betting into steam/injury-news line moves after the fact.
     -> models/line_movement.py
  7. Don't parlay/stack correlated outcomes from the same game as if independent.
     -> models/contradiction_check.py (conservative version: drops outright
        conflicts; true correlation-aware parlay pricing is out of scope here)
  8. Fade public betting percentages only when there's an independent model
     edge too -- public fade alone isn't a strategy, it's a slogan.
     -> public_fade_signal() is informational/diagnostic only, NEVER a
        standalone trigger to generate or boost a pick
  9. Respect situational spots known to move the number (umpire crew tendencies,
     bullpen fatigue, back-to-backs/3-in-4-nights for WNBA) -- partially covered
     by ramp_detection.py (workload) and season_context.py (phase); a full
     situational-spot database is future work, not faked here.
  10. Keep a written record / backtest before trusting any model with real
      money. -> models/backtest.py
"""


def kelly_stake(model_prob, american_odds, kelly_fraction):
    """
    Fractional Kelly stake as a fraction of bankroll.
    model_prob: your model's win probability for the side you'd bet (0-1)
    american_odds: the odds for that side
    kelly_fraction: 0.20-0.25 typical for serious bettors (full Kelly is too
    volatile in practice even when the edge estimate is correct, and your
    edge estimate is never perfectly correct) -- pull from sport_config, not
    a hardcoded default, so MLB and WNBA can use different fractions.
    """
    if american_odds > 0:
        b = american_odds / 100
    else:
        b = 100 / abs(american_odds)

    p = model_prob
    q = 1 - p
    full_kelly = (b * p - q) / b
    if full_kelly <= 0:
        return 0.0
    return full_kelly * kelly_fraction


def clv_pct(pick_time_odds, closing_odds):
    """
    Closing Line Value: how much better (or worse) your bet's odds were vs.
    the closing line, in implied-probability terms. Positive CLV (you got a
    worse number for the side than the closing market settled on -- i.e. you
    were getting paid more than the market eventually thought was fair) is
    the standard long-run signal of real edge, independent of whether any
    individual bet wins.
    """
    def implied(odds):
        return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

    pick_implied = implied(pick_time_odds)
    close_implied = implied(closing_odds)
    if pick_implied == 0:
        return 0.0
    return (close_implied - pick_implied) / pick_implied * 100


def public_fade_signal(public_bet_pct, public_money_pct):
    """
    DIAGNOSTIC ONLY -- returns a label, never feeds into pick generation
    directly. Heavy public money with light public ticket count (or vice
    versa) is the classic signal sharps look at, but using it alone as a bet
    trigger is gambler folklore, not a model. Real use: cross-reference
    against a pick that ALREADY cleared the edge threshold, as one more piece
    of context for the person placing the bet -- not a generator of new picks.
    """
    if public_bet_pct is None or public_money_pct is None:
        return "no_data"
    gap = public_money_pct - public_bet_pct
    if gap >= 15:
        return "sharp_money_likely_opposite_public"
    if gap <= -15:
        return "square_money_inflated_ticket_count"
    return "no_strong_signal"


def umpire_zone_factor(ump_name, ump_strike_zone_db=None):
    """
    STUB -- returns a neutral factor and an explicit "no_data" reason string.

    Real umpire crew tendencies (tight vs. hitter-friendly strike zones,
    typically expressed as extra called strikes/game above or below league
    average) require a maintained crew-assignment + zone-history feed --
    there is no free, reliable, day-of umpire-assignment API to wire in
    here. Sites that publish this (e.g. Umpire Scorecards) don't offer a
    stable public API contract to build a betting pipeline on top of.

    This function exists purely so the contract is in place, matching
    compare_to_consensus() below: wire a real `ump_strike_zone_db` lookup
    (crew chief for tonight's game → their K-boost/K-suppress zone history)
    before trusting this for real money. Until then this is NOT called
    anywhere in the pick-generation pipeline -- do not wire it into
    strikeout_matchup.py or player_props.py without a real data source
    behind it, per rule 9 above (situational-spot database is future work,
    not faked here).
    """
    if not ump_name or not ump_strike_zone_db:
        return {"factor": 1.0, "reason": "no_data"}
    zone_history = ump_strike_zone_db.get(ump_name)
    if zone_history is None:
        return {"factor": 1.0, "reason": "no_data"}
    return {"factor": 1.0, "reason": "STUB -- lookup found but not implemented"}


def compare_to_consensus(book_odds_list):
    """
    STUB: real line-shopping needs multiple books' prices for the same
    market, which data/fetch.py doesn't pull yet (The Odds API call in this
    build takes the first returned price). This function exists so the
    contract is in place -- wire in regions='us,us2' / loop all bookmakers in
    the Odds API response before trusting this for real money.
    """
    if not book_odds_list:
        return None
    best = max(book_odds_list, key=lambda o: o if o > 0 else -100 / o if o != 0 else 0)
    return {"best_price": best, "n_books_compared": len(book_odds_list),
            "note": "STUB -- only as good as the books actually passed in"}
