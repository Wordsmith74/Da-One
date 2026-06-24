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
PRIOR_WEIGHT_WNBA_GAMES = WNBA["prior_weight_games"]


def shrink_mlb_k_pct(recent_ks, recent_batters_faced, league_avg_k_pct=MLB_LEAGUE_AVG_K_PCT,
                      prior_weight_bf=PRIOR_WEIGHT_MLB_BF):
    """
    Shrinks a pitcher's recent K% toward the league average, weighted by sample size.

    recent_ks: strikeouts over the recent sample
    recent_batters_faced: batters faced over that same sample
    Returns: shrunk K% (0-1 float)
    """
    if recent_batters_faced <= 0:
        return league_avg_k_pct
    raw_k_pct = recent_ks / recent_batters_faced
    shrunk = (
        (recent_ks + league_avg_k_pct * prior_weight_bf)
        / (recent_batters_faced + prior_weight_bf)
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
