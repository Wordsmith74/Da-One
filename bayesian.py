"""
Bayesian shrinkage for small-sample rate stats.

Why: a pitcher who struck out 9 of 24 batters faced in his last outing has a
37.5% K rate on paper, but n=24 is noisy. Top handicappers never bet recent
small-sample rates raw -- they shrink toward a believable prior (league/season
average) weighted by how much data backs the recent number. This is the
single biggest gap between a profitable model and someone betting "hot streaks".

Method: empirical-Bayes-style shrinkage toward a league-average prior using a
pseudo-count (the prior is treated as if it were observed over a fixed number
of "prior" opportunities). This is a standard, well-understood approximation
(similar in spirit to a Beta-Binomial posterior mean) -- not a from-scratch
Bayesian model, just the shrinkage idea applied transparently.

IMPORTANT: all priors/weights below are pulled from models/sport_config.py,
NOT hardcoded here. MLB and WNBA shrink toward different things (league
average vs. the player's own season average) using different pseudo-counts --
mixing them up is exactly the kind of cross-sport bug this file is designed
to avoid. Edit sport_config.py, not the defaults in these function signatures.
"""
from models.sport_config import MLB, WNBA

MLB_LEAGUE_AVG_K_PCT = MLB["league_avg_k_pct"]
PRIOR_WEIGHT_MLB_BF = MLB["prior_weight_batters_faced"]
OWN_SEASON_K_PCT_WEIGHT = MLB["own_season_k_pct_weight"]
PRIOR_WEIGHT_F5_GAMES = MLB["prior_weight_f5_games"]
PRIOR_WEIGHT_WNBA_GAMES = WNBA["prior_weight_games"]

# How much a confirmed shift event (pitch-mix change, role change, mechanical
# tweak, rotation/lineup change) should discount trust in the *prior*
# relative to the fresh recent sample. 0.25 means the prior counts for only
# a quarter of its normal pseudo-count weight -- NOT zero, because even a
# real-shift player/team doesn't fully erase its baseline overnight.
KNOWN_SHIFT_EVENT_PRIOR_DISCOUNT = 0.25


def adjust_recent_ks_for_opponent(recent_ks, opp_k_rate_allowed, league_avg_k_pct=MLB_LEAGUE_AVG_K_PCT):
    """
    Removes the opponent-quality confound from a recent K sample BEFORE
    shrink_mlb_k_pct runs. Shrinkage corrects for SAMPLE SIZE noise -- it
    has no way to know whether a pitcher's recent starts happened to come
    against unusually punchout-prone or contact-heavy lineups. A pitcher
    who faced three weak-contact lineups in a row will look like he's
    "trending up" even with zero real change in his own stuff; shrinking
    that toward league average doesn't fix it, because the inflated number
    has enough sample size to resist being pulled down.

    Rescales recent_ks by (league_avg_k_pct / opp_k_rate_allowed) -- an
    easy-strikeout slate (opponent K rate above league average) pulls
    recent_ks down before shrinkage sees it; a contact-heavy slate pulls it
    up. This is a normalization, not a new prior -- shrink_mlb_k_pct still
    runs as-is afterward on the adjusted count.

    opp_k_rate_allowed: opponent lineup's own K rate (0-1 float), e.g. from
        data.fetch.get_mlb_team_k_rate_allowed. If None or non-positive
        (data source unavailable or degraded), returns recent_ks UNCHANGED
        -- this must degrade to a no-op, never fall back to a guessed ratio.
    """
    if opp_k_rate_allowed is None or opp_k_rate_allowed <= 0:
        return recent_ks
    adjustment_ratio = league_avg_k_pct / opp_k_rate_allowed
    return recent_ks * adjustment_ratio


def shrink_mlb_k_pct(recent_ks, recent_batters_faced, league_avg_k_pct=MLB_LEAGUE_AVG_K_PCT,
                      prior_weight_bf=PRIOR_WEIGHT_MLB_BF, own_season_k_pct=None,
                      own_season_weight=OWN_SEASON_K_PCT_WEIGHT, known_shift_event=False):
    """
    Shrinks a pitcher's recent K% toward a prior, weighted by sample size.

    recent_ks: strikeouts over the recent sample
    recent_batters_faced: batters faced over that same sample
    own_season_k_pct: pitcher's own season-long K% (0-1 float), if available.
        When provided (and no shift event), the prior is a blend of this and
        league_avg_k_pct (weighted by own_season_weight) rather than league
        average alone -- own-season is a better personal prior once enough
        season-long data exists, mirroring why WNBA shrinks toward the
        player's own average instead of a league average.
    known_shift_event: True when there's a confirmed, non-noise reason
        recent K% may have genuinely moved (pitch-mix change, role change,
        mechanical tweak). NOT for opponent-quality streaks -- that's a
        confound that belongs upstream (normalize recent_ks against
        opponent K-rate-allowed before calling this; don't discount the
        prior to paper over it). When True: the prior's pseudo-count weight
        is discounted so recent data dominates, AND own_season_k_pct (if
        given) is dropped in favor of league_avg_k_pct alone, since a
        pitcher's pre-shift season average is itself contaminated by the
        change.
    Returns: shrunk K% (0-1 float)
    """
    if recent_batters_faced <= 0:
        return league_avg_k_pct

    if own_season_k_pct is not None and not known_shift_event:
        prior = (own_season_k_pct * own_season_weight
                 + league_avg_k_pct * (1 - own_season_weight))
    else:
        prior = league_avg_k_pct

    weight = prior_weight_bf * KNOWN_SHIFT_EVENT_PRIOR_DISCOUNT if known_shift_event else prior_weight_bf

    shrunk = (
        (recent_ks + prior * weight)
        / (recent_batters_faced + weight)
    )
    return shrunk


def shrink_mlb_f5_runs(recent_f5_runs_per_game, n_games, season_avg_f5_runs_per_game,
                        prior_weight_games=PRIOR_WEIGHT_F5_GAMES, known_shift_event=False):
    """
    Shrinks a team's recent first-5-innings runs/game toward its season F5
    scoring average, weighted by sample size. Mirrors shrink_wnba_stat's
    mechanism (shrink toward the team's OWN season average, not a league
    average) -- F5 totals previously had NO shrinkage at all, so a noisy
    2-run/14-run back-to-back swing was taken at face value.

    recent_f5_runs_per_game: recent F5 runs/game rate
    n_games: number of games in the recent sample
    season_avg_f5_runs_per_game: team's season-long F5 runs/game average
    known_shift_event: True for a confirmed non-noise cause (rotation
        change, key bat traded in/out, lineup change) -- discounts the
        prior so recent data dominates instead of being pulled toward a
        baseline that may no longer reflect the team.
    """
    if n_games <= 0:
        return season_avg_f5_runs_per_game

    weight = prior_weight_games * KNOWN_SHIFT_EVENT_PRIOR_DISCOUNT if known_shift_event else prior_weight_games

    shrunk = (
        (recent_f5_runs_per_game * n_games + season_avg_f5_runs_per_game * weight)
        / (n_games + weight)
    )
    return shrunk


def shrink_wnba_stat(recent_stat_per_30, n_games, season_avg_stat_per_30,
                      prior_weight_games=PRIOR_WEIGHT_WNBA_GAMES):
    # NOTE: WNBA shrinks toward the PLAYER'S OWN season average, not a league
    # average -- deliberately different mechanism from the MLB function above.
    """
    Shrinks a recent per-30-minute rate toward the player's OWN season average
    (not league average -- season average is a much better personal prior than
    league average for an established player's own scoring rate).

    recent_stat_per_30: recent rate (already normalized to per-30-min)
    n_games: number of games in the recent sample
    season_avg_stat_per_30: player's season-long per-30-min average
    """
    if n_games <= 0:
        return season_avg_stat_per_30
    shrunk = (
        (recent_stat_per_30 * n_games + season_avg_stat_per_30 * prior_weight_games)
        / (n_games + prior_weight_games)
    )
    return shrunk
