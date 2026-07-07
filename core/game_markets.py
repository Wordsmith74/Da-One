"""
core/game_markets.py

Approved Market Coverage & Scan Priority Framework — expanded game market fetching.

Scans ALL approved markets for each game before any NO-PLAY decision:

  MLB  : Full Game ML, Full Game Run Line (Spread)
         -- Full Game Total comes from core/odds_client.py separately.
         -- F5 Total/ML/RL, NRFI/YRFI, and Team Totals are OUT of scope.
  NBA  : Q1/H1 Total, Q1/H1 ML, Q1/H1 Spread,
         Team Totals (home + away), Full Game ML, Full Game Spread
  WNBA : Team Totals (home + away), Full Game ML, Full Game Spread

Total-based markets (first_5, team_total, q1, h1) produce Bayesian-ready
candidates with scaled historical data and flow through engine.analyze().

Moneyline and spread markets carry precomputed_edge / precomputed_confidence /
precomputed_model_prob and bypass the NUTS sampler in main.py.

Single Odds API request per sport per session.  Results cached in the same
slate cache layer used by odds_client, under key "{SPORT}_EXPANDED".
"""

from __future__ import annotations

import math
import os
from typing import Any

from core.market_intelligence import (
    detect_sharp_action,
    detect_steam_move,
    detect_reverse_line_movement,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sport / market configuration
# ─────────────────────────────────────────────────────────────────────────────

_SPORT_KEY: dict[str, str] = {
    "MLB":  "baseball_mlb",
    "NBA":  "basketball_nba",
    "WNBA": "basketball_wnba",
}

# Odds API market bundle per sport.
# Scope limited to the System Scope Definition (core/market_gate.py):
#   MLB  — full-game moneyline (h2h) + full-game run line/spread (spreads) only.
#          F5 total/ML/RL and NRFI/YRFI (totals_1st_1_innings) intentionally
#          removed from scope -- MLB's full-game total already comes from
#          core/odds_client.py's fetch_todays_candidates (markets=totals),
#          so it doesn't need to be requested again here.
#   WNBA — full-game moneyline only (spreads and team-totals removed) -- unchanged
#   NBA  — no markets in scope (empty string → fetch_expanded_game_candidates returns [])
_MARKET_BUNDLE: dict[str, str] = {
    "MLB":  "h2h,spreads",
    "NBA":  "",   # blocked: no game markets in scope
    "WNBA": "h2h",
}

# Scale factors for sub-game totals relative to the full-game total
_TOTAL_SCALE: dict[str, float] = {
    "totals_first_5_innings": 0.52,
    "totals_q1":              0.245,
    "totals_h1":              0.490,
}

# market_key → stored market label (market_key field on candidate)
# NOTE: totals_1st_1_innings is NOT mapped to a single label here -- it
# splits into two distinct market_keys ("nrfi" / "yrfi") per side, set
# directly on the candidate by _process_nrfi_yrfi() (see that function),
# since core/market_gate.py's ALLOWED_MARKETS/MARKET_ALIASES already expect
# those two keys specifically, not a shared "totals_1st_1_innings" key with
# a direction field the way F5 total works.
_MARKET_LABEL: dict[str, str] = {
    "h2h":                     "moneyline",
    "spreads":                 "run_line",
    "team_totals":             "team_total",
    "totals_first_5_innings":  "first_5_total",
    "h2h_first_5_innings":     "first_5_ml",
    "spreads_first_5_innings": "first_5_rl",
    "totals_q1":               "q1_total",
    "totals_h1":               "h1_total",
    "h2h_q1":                  "q1_ml",
    "h2h_h1":                  "h1_ml",
    "spreads_q1":              "q1_spread",
    "spreads_h1":              "h1_spread",
}

# market_label → human-readable display name
_MARKET_DISPLAY: dict[str, str] = {
    "moneyline":    "Moneyline",
    "run_line":     "Run Line",
    "team_total":   "Team Total",
    "first_5_total":"F5 Total",
    "first_5_ml":   "F5 Moneyline",
    "first_5_rl":   "F5 Run Line",
    "q1_total":     "Q1 Total",
    "h1_total":     "H1 Total",
    "q1_ml":        "Q1 Moneyline",
    "h1_ml":        "H1 Moneyline",
    "q1_spread":    "Q1 Spread",
    "h1_spread":    "H1 Spread",
    "nrfi":         "NRFI",
    "yrfi":         "YRFI",
}

# Full game total priors per sport
_GAME_TOTAL_PRIOR: dict[str, dict[str, float]] = {
    "MLB":  {"mean": 8.5,   "std": 1.5},
    "NBA":  {"mean": 222.0, "std": 12.0},
    "WNBA": {"mean": 165.0, "std": 8.0},
}

# Home-field advantage (raw win-prob boost for home team)
_HOME_ADV: dict[str, float] = {
    "MLB":  0.035,
    "NBA":  0.060,
    "WNBA": 0.045,
}

# Market types that use scaled historical data through the Bayesian engine
_SCALED_TOTAL_MARKETS = frozenset({
    "totals_first_5_innings", "totals_q1", "totals_h1",
})

# Market types that use pre-computed win/cover probability (bypass NUTS)
_PRECOMPUTED_MARKETS = frozenset({
    "h2h", "h2h_first_5_innings", "h2h_q1", "h2h_h1",
    "spreads", "spreads_first_5_innings", "spreads_q1", "spreads_h1",
})

# ─────────────────────────────────────────────────────────────────────────────
# Probability helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun, maximum error 7.5e-8)."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    p = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    cdf = 1.0 - 0.3989422804 * math.exp(-x * x / 2.0) * p
    return cdf if x >= 0 else 1.0 - cdf


def _american_to_implied(odds: int) -> float:
    """American odds → raw implied probability (includes vig)."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _devig(p1: float, p2: float) -> tuple[float, float]:
    """Remove vig from a two-outcome market."""
    total = p1 + p2
    if total <= 0:
        return 0.5, 0.5
    return p1 / total, p2 / total


def _win_prob(
    home_hist: list[float], away_hist: list[float], sport: str
) -> tuple[float, float]:
    """
    (home_win_prob, away_win_prob) via normal approximation of run differential.
    home_hist / away_hist are per-game run/point totals for each team.
    """
    prior = _GAME_TOTAL_PRIOR.get(sport, {"mean": 8.5, "std": 1.5})

    if home_hist:
        h_avg = sum(home_hist) / len(home_hist)
        h_var = (sum((x - h_avg) ** 2 for x in home_hist) / max(1, len(home_hist) - 1)) if len(home_hist) > 1 else 4.0
    else:
        h_avg = prior["mean"] * 0.5
        h_var = (prior["std"] * 0.65) ** 2

    if away_hist:
        a_avg = sum(away_hist) / len(away_hist)
        a_var = (sum((x - a_avg) ** 2 for x in away_hist) / max(1, len(away_hist) - 1)) if len(away_hist) > 1 else 4.0
    else:
        a_avg = prior["mean"] * 0.5
        a_var = (prior["std"] * 0.65) ** 2

    diff_std = math.sqrt(max(0.5, h_var + a_var))
    z = (h_avg - a_avg) / diff_std
    raw_home = _ncdf(z)
    home = min(0.93, max(0.07, raw_home + _HOME_ADV.get(sport, 0.04)))
    return home, 1.0 - home


def _spread_cover_prob(
    home_hist: list[float], away_hist: list[float], spread: float, sport: str
) -> tuple[float, float]:
    """
    (home_cover_prob, away_cover_prob) for the given spread.
    spread is from the home perspective: -1.5 means home favored by 1.5.
    Home covers if (home_score - away_score) > -spread.
    """
    prior = _GAME_TOTAL_PRIOR.get(sport, {"mean": 8.5, "std": 1.5})

    if home_hist:
        h_avg = sum(home_hist) / len(home_hist)
        h_var = (sum((x - h_avg) ** 2 for x in home_hist) / max(1, len(home_hist) - 1)) if len(home_hist) > 1 else 4.0
    else:
        h_avg = prior["mean"] * 0.5
        h_var = (prior["std"] * 0.65) ** 2

    if away_hist:
        a_avg = sum(away_hist) / len(away_hist)
        a_var = (sum((x - a_avg) ** 2 for x in away_hist) / max(1, len(away_hist) - 1)) if len(away_hist) > 1 else 4.0
    else:
        a_avg = prior["mean"] * 0.5
        a_var = (prior["std"] * 0.65) ** 2

    diff_std  = math.sqrt(max(0.5, h_var + a_var))
    threshold = -spread          # home must beat this differential to cover
    z = (h_avg - a_avg - threshold) / diff_std
    home_cover = _ncdf(z)
    return home_cover, 1.0 - home_cover


def _edge_pct(model_prob: float, odds: int) -> float:
    """
    Kelly / EV edge expressed as a percentage:
        edge = model_prob × decimal_odds - 1  (×100 to get %)
    """
    decimal = (1.0 + odds / 100.0) if odds >= 0 else (1.0 + 100.0 / abs(odds))
    ev = model_prob * decimal - 1.0
    return round(ev * 100.0, 2)


def _precomp_confidence(book_count: int, hist_len: int, model_prob: float) -> float:
    """
    Confidence for pre-computed (non-Bayesian) moneyline/spread candidates.
    Deliberately capped at 82 so they cannot auto-qualify as Diamond without
    additional corroborating signals.
    """
    base         = 68.0
    book_boost   = min(8.0, book_count * 1.5)
    hist_boost   = min(4.0, hist_len * 0.06)
    conviction   = abs(model_prob - 0.5) * 2.0  # 0 = coin-flip, 1 = certain
    conv_boost   = conviction * 5.0
    return round(min(82.0, base + book_boost + hist_boost + conv_boost), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Team history helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_team_histories(
    sport: str, home_team: str, away_team: str, game_seed: float,
    as_of_date: str | None = None,
) -> tuple[list[float], list[float]]:
    """Return (home_runs_per_game, away_runs_per_game) histories."""
    from core.odds_client import _synthetic_history, _abbrev  # type: ignore[attr-defined]

    prior     = _GAME_TOTAL_PRIOR.get(sport, {"mean": 8.5, "std": 1.5})
    half_mean = prior["mean"] * 0.5
    half_std  = prior["std"]  * 0.65

    home_hist: list[float] = []
    away_hist: list[float] = []

    # These imports were previously pointed at a nonexistent
    # `core.intelligence.game_logs` module, so both branches silently
    # ImportError'd and every history here was always synthetic.
    # get_mlb_game_totals_history actually lives in core.odds_client;
    # get_team_game_totals actually lives in data.game_logs.
    if sport == "MLB":
        try:
            from core.odds_client import get_mlb_game_totals_history
            home_hist = get_mlb_game_totals_history(home_team, as_of_date=as_of_date) or []
            away_hist = get_mlb_game_totals_history(away_team, as_of_date=as_of_date) or []
        except Exception:
            pass
    elif sport in ("NBA", "WNBA"):
        try:
            from data.game_logs import get_team_game_totals
            home_hist = get_team_game_totals(sport, _abbrev(home_team), as_of_date=as_of_date) or []
            away_hist = get_team_game_totals(sport, _abbrev(away_team), as_of_date=as_of_date) or []
        except Exception:
            pass

    if not home_hist:
        home_hist = _synthetic_history(game_seed,     half_mean, half_std, sport=sport)
    if not away_hist:
        away_hist = _synthetic_history(game_seed + 1, half_mean, half_std, sport=sport)

    return home_hist, away_hist


def _combined_hist(h1: list[float], h2: list[float]) -> list[float]:
    """Interleave two per-team histories into a combined game-total history."""
    combined: list[float] = []
    for pair in zip(h1, h2):
        combined.extend(pair)
    longer = h1 if len(h1) >= len(h2) else h2
    combined.extend(longer[min(len(h1), len(h2)):])
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Candidate builder
# ─────────────────────────────────────────────────────────────────────────────

def _base_candidate(
    *,
    bet_id: str, game_label: str,
    away_team: str, home_team: str,
    away_abbr: str, home_abbr: str,
    team: str, mkt_key: str, direction: str,
    sportsbook_line: float, american_odds: int,
    bookmaker_source: str, book_count: int,
    historical_data: list[float], league_mean: float, league_std: float,
    factor: str, game_time_et: Any,
) -> dict[str, Any]:
    label   = _MARKET_LABEL.get(mkt_key, mkt_key.replace("_", " ").title())
    display = _MARKET_DISPLAY.get(label, label)
    return {
        "bet_id":           bet_id,
        "game_id":          game_label,
        "away_team":        away_team,
        "home_team":        home_team,
        "full_team_name":   home_team if team == home_abbr else away_team,
        "team":             team,
        "market":           display,
        "market_key":       label,
        "player":           None,
        "direction":        direction,
        "sportsbook_line":  sportsbook_line,
        "opening_line":     sportsbook_line,
        "american_odds":    american_odds,
        "bookmaker_source": bookmaker_source,
        "book_count":       book_count,
        "consensus_line":   sportsbook_line,
        "line_dispersion":  0.0,
        "historical_data":  historical_data,
        "league_mean":      league_mean,
        "league_std":       league_std,
        "context":          "regular",
        "volatility_index": None,
        "recent_n":         5,
        "factor":           factor,
        "game_time_utc":    game_time_et,
        "mis_score":        40.0,
        "book_lines":       [],
        "sharp_signal":     "no_sharp",
        "sharp_label":      "No Sharp Action",
        "sharp_book_count": 0,
        "steam_detected":   False,
        "rlm_detected":     False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-market-type processors
# ─────────────────────────────────────────────────────────────────────────────

def _process_scaled_total(
    candidates: list[dict[str, Any]],
    mkt_key: str, book_outcomes_list: list,
    home_abbr: str, away_abbr: str,
    home_team: str, away_team: str,
    comb_hist: list[float],
    game_label: str, game_time_et: Any, sport: str,
) -> None:
    """OVER/UNDER candidates for first-5/Q1/H1 totals (scaled history → Bayesian)."""
    from core.odds_client import _best_side  # type: ignore[attr-defined]

    scale       = _TOTAL_SCALE[mkt_key]
    prior       = _GAME_TOTAL_PRIOR[sport]
    scaled_std  = prior["std"] * scale
    hist_scaled = [x * scale for x in comb_hist]
    eff_mean    = (sum(hist_scaled) / len(hist_scaled)) if hist_scaled else prior["mean"] * scale

    # ── Workload adjustment for MLB F5 totals ─────────────────────────────
    # Adjusts eff_mean up or down based on projected starter quality in
    # first 5 innings.  Deep starters suppress runs; short outings bring in
    # the bullpen (higher ERA) earlier, raising expected F5 scoring.
    if sport == "MLB" and mkt_key == "totals_first_5_innings":
        try:
            from core.pitcher_workload import (
                get_game_workload_pair,
                get_f5_workload_adjustment,
            )
            _wl_h, _wl_a = get_game_workload_pair(home_abbr, away_abbr)
            if _wl_h and _wl_a:
                _f5_adj = get_f5_workload_adjustment(_wl_h, _wl_a)
                if _f5_adj != 0.0:
                    eff_mean = round(eff_mean + _f5_adj, 2)
                    print(
                        f"[game_markets] F5 workload adj "
                        f"{away_abbr}@{home_abbr}: {_f5_adj:+.2f} "
                        f"→ eff_mean={eff_mean:.2f}",
                        flush=True,
                    )
        except Exception:
            pass   # workload is always optional; never block candidate build

    label       = _MARKET_LABEL[mkt_key]
    display     = _MARKET_DISPLAY.get(label, label)

    over_lines:  list[tuple[float, int, str]] = []
    under_lines: list[tuple[float, int, str]] = []

    for bk_title, outcomes in book_outcomes_list:
        if bk_title == "Matchbook":  # EU-only, consistently stale vs US market
            continue
        for out in outcomes:
            name  = (out.get("name") or "").lower()
            pt    = out.get("point")
            price = out.get("price")
            if pt is None or price is None:
                continue
            if name == "over":
                over_lines.append((float(pt), int(price), bk_title))
            elif name == "under":
                under_lines.append((float(pt), int(price), bk_title))

    # Stale-line drift limits for scaled sub-game totals
    _SCALED_DRIFT: dict[str, float] = {"MLB": 0.5, "NBA": 1.0, "WNBA": 0.5}
    _scaled_drift_lim = _SCALED_DRIFT.get(sport.upper(), 0.5)

    for direction, lines in (("over", over_lines), ("under", under_lines)):
        result = _best_side(lines)
        if result is None:
            continue
        best_line, best_odds, best_book, n_books = result

        # Consensus deviation gate: reject lines too far from the market median
        if lines:
            _c_pts = sorted(pt for pt, _, _ in lines)
            _consensus = _c_pts[len(_c_pts) // 2]
            if _consensus and abs(best_line - _consensus) > _scaled_drift_lim:
                _within = [(l, p, b) for l, p, b in lines
                           if abs(l - _consensus) <= _scaled_drift_lim]
                _fb = _best_side(_within) if _within else None
                if _fb:
                    best_line, best_odds, best_book, n_books = _fb
                else:
                    print(
                        f"[game_markets] STALE LINE rejected: "
                        f"{away_team}@{home_team} {label} {direction} "
                        f"best={best_line} consensus={_consensus} "
                        f"drift={abs(best_line - _consensus):.2f} > {_scaled_drift_lim}",
                        flush=True,
                    )
                    continue

        d_lbl  = "O" if direction == "over" else "U"
        bet_id = f"{home_abbr}_{label}_{direction}"
        factor = (
            f"{away_team} @ {home_team} — {display} {best_line} {d_lbl} "
            f"({n_books} books, best: {best_book})"
        )
        cand = _base_candidate(
            bet_id=bet_id, game_label=game_label,
            away_team=away_team, home_team=home_team,
            away_abbr=away_abbr, home_abbr=home_abbr,
            team=home_abbr, mkt_key=mkt_key, direction=direction,
            sportsbook_line=best_line, american_odds=best_odds,
            bookmaker_source=best_book, book_count=n_books,
            historical_data=hist_scaled, league_mean=eff_mean, league_std=scaled_std,
            factor=factor, game_time_et=game_time_et,
        )
        candidates.append(cand)


def _process_team_total(
    candidates: list[dict[str, Any]],
    book_outcomes_list: list,
    home_abbr: str, away_abbr: str,
    home_team: str, away_team: str,
    home_hist: list[float], away_hist: list[float],
    game_label: str, game_time_et: Any, sport: str,
) -> None:
    """OVER/UNDER candidates for team-specific totals."""
    from core.odds_client import _best_side  # type: ignore[attr-defined]

    prior = _GAME_TOTAL_PRIOR[sport]
    label = _MARKET_LABEL["team_totals"]
    display = _MARKET_DISPLAY.get(label, label)

    # team_totals outcomes have a "description" field with the full team name
    team_lines: dict[str, dict[str, list[tuple[float, int, str]]]] = {
        "home": {"over": [], "under": []},
        "away": {"over": [], "under": []},
    }
    for bk_title, outcomes in book_outcomes_list:
        if bk_title == "Matchbook":  # EU-only, consistently stale vs US market
            continue
        for out in outcomes:
            name  = (out.get("name") or "").lower()
            desc  = out.get("description", "")
            pt    = out.get("point")
            price = out.get("price")
            if pt is None or price is None or name not in ("over", "under"):
                continue
            side = "home" if desc == home_team else ("away" if desc == away_team else None)
            if side is None:
                continue
            team_lines[side][name].append((float(pt), int(price), bk_title))

    # Stale-line drift limits for team totals
    _TT_DRIFT: dict[str, float] = {"MLB": 0.5, "NBA": 1.0, "WNBA": 0.5}
    _tt_drift_lim = _TT_DRIFT.get(sport.upper(), 0.5)

    # WNBA Regime Adjustment: compute game-specific Contextual Expected Total
    # once per game so both team-total candidates share the same CET baseline.
    _cet: float   = 0.0
    _regime: str  = "neutral"
    _vol_mult_tt: float = 1.0
    if sport.upper() == "WNBA":
        try:
            from core.wnba_regime import compute_wnba_cet
            _cet, _regime, _vol_mult_tt = compute_wnba_cet(home_hist, away_hist)
            print(
                f"[game_markets] WNBA team-total CET: "
                f"{away_abbr}@{home_abbr}  cet={_cet}  "
                f"regime={_regime}  vol_mult={_vol_mult_tt}",
                flush=True,
            )
        except Exception as _cet_exc:
            print(
                f"[game_markets] CET fallback for WNBA: {_cet_exc}", flush=True,
            )

    for side, hist, abbr, team_name in (
        ("home", home_hist, home_abbr, home_team),
        ("away", away_hist, away_abbr, away_team),
    ):
        if sport.upper() == "WNBA" and _cet > 0:
            # CET/2 = expected per-team points in this specific scoring environment
            eff_mean = round(_cet / 2.0, 2)
            eff_std  = round(prior["std"] * 0.65 * _vol_mult_tt, 2)
        else:
            eff_mean = (sum(hist) / len(hist)) if hist else prior["mean"] * 0.5
            eff_std  = prior["std"] * 0.65
        for direction in ("over", "under"):
            side_lines = team_lines[side][direction]
            result = _best_side(side_lines)
            if result is None:
                continue
            best_line, best_odds, best_book, n_books = result

            # Consensus deviation gate
            if side_lines:
                _c_pts = sorted(pt for pt, _, _ in side_lines)
                _consensus = _c_pts[len(_c_pts) // 2]
                if _consensus and abs(best_line - _consensus) > _tt_drift_lim:
                    _within = [(l, p, b) for l, p, b in side_lines
                               if abs(l - _consensus) <= _tt_drift_lim]
                    _fb = _best_side(_within) if _within else None
                    if _fb:
                        best_line, best_odds, best_book, n_books = _fb
                    else:
                        print(
                            f"[game_markets] STALE TEAM TOTAL rejected: "
                            f"{team_name} {direction} best={best_line} "
                            f"consensus={_consensus} "
                            f"drift={abs(best_line - _consensus):.2f} > {_tt_drift_lim}",
                            flush=True,
                        )
                        continue

            d_lbl  = "O" if direction == "over" else "U"
            bet_id = f"{abbr}_{label}_{direction}"
            factor = (
                f"{team_name} team total {best_line} {d_lbl} "
                f"(in {away_team} @ {home_team}, {n_books} books, best: {best_book})"
            )
            cand = _base_candidate(
                bet_id=bet_id, game_label=game_label,
                away_team=away_team, home_team=home_team,
                away_abbr=away_abbr, home_abbr=home_abbr,
                team=abbr, mkt_key="team_totals", direction=direction,
                sportsbook_line=best_line, american_odds=best_odds,
                bookmaker_source=best_book, book_count=n_books,
                historical_data=hist, league_mean=eff_mean, league_std=eff_std,
                factor=factor, game_time_et=game_time_et,
            )
            if sport.upper() == "WNBA" and _cet > 0:
                cand["cet"]            = _cet
                cand["scoring_regime"] = _regime
            candidates.append(cand)


def _process_moneyline(
    candidates: list[dict[str, Any]],
    mkt_key: str, book_outcomes_list: list,
    home_abbr: str, away_abbr: str,
    home_team: str, away_team: str,
    home_hist: list[float], away_hist: list[float],
    game_label: str, game_time_et: Any, sport: str,
) -> None:
    """Pre-computed moneyline candidates for home and away sides."""
    from core.odds_client import _best_side  # type: ignore[attr-defined]

    label   = _MARKET_LABEL.get(mkt_key, mkt_key)
    display = _MARKET_DISPLAY.get(label, label)

    home_lines: list[tuple[float, int, str]] = []
    away_lines: list[tuple[float, int, str]] = []

    for bk_title, outcomes in book_outcomes_list:
        for out in outcomes:
            name  = out.get("name") or ""
            price = out.get("price")
            pt    = float(out.get("point") or 0)
            if price is None:
                continue
            if name == home_team:
                home_lines.append((pt, int(price), bk_title))
            elif name == away_team:
                away_lines.append((pt, int(price), bk_title))

    home_win_model, away_win_model = _win_prob(home_hist, away_hist, sport)

    # Regress partial-game MLs toward 50/50 (less runs = more uncertainty)
    if mkt_key == "h2h_first_5_innings":
        home_win_model = 0.5 + (home_win_model - 0.5) * 0.65
    elif mkt_key == "h2h_q1":
        home_win_model = 0.5 + (home_win_model - 0.5) * 0.55
    elif mkt_key == "h2h_h1":
        home_win_model = 0.5 + (home_win_model - 0.5) * 0.75
    away_win_model = 1.0 - home_win_model

    # Fair probabilities after devig (when both sides available)
    fair_home = home_win_model
    fair_away = away_win_model
    if home_lines and away_lines:
        best_h = sorted(home_lines, key=lambda x: x[1], reverse=True)[0][1]
        best_a = sorted(away_lines, key=lambda x: x[1], reverse=True)[0][1]
        fair_home, fair_away = _devig(_american_to_implied(best_h), _american_to_implied(best_a))

    for side, lines, team_abbr, team_name, model_prob, fair_impl in (
        ("home", home_lines, home_abbr, home_team, home_win_model, fair_home),
        ("away", away_lines, away_abbr, away_team, away_win_model, fair_away),
    ):
        result = _best_side(lines)
        if result is None:
            continue
        _, best_odds, best_book, n_books = result
        edge       = _edge_pct(model_prob, best_odds)
        confidence = _precomp_confidence(n_books, len(home_hist) + len(away_hist), model_prob)

        # ── Sharp action / steam / reverse-line-movement (moneyline) ───────
        # These three detectors already existed in core/market_intelligence.py
        # and were wired into player props, but never into game markets --
        # moneyline picks always carried "no sharp coverage" implicitly.
        # No public bet-% feed exists in this codebase (confirmed -- there's
        # no market data source for it anywhere), so, same as player_props.py,
        # this uses the sharp-vs-recreational-book line-disagreement proxy
        # documented in detect_sharp_action()/detect_reverse_line_movement().
        # For a point-spread/prop line, "line" is literally the number bet on;
        # for a moneyline there's no such number, so each book's price is
        # converted to implied win probability. detect_sharp_action's /
        # detect_reverse_line_movement's thresholds (0.25 / 0.5) were
        # calibrated for point-line granularity, not raw 0-1 probability --
        # so the probability is expressed in PERCENTAGE POINTS (x100) here,
        # not a 0-1 fraction, to land in the same scale those thresholds
        # actually mean something at (e.g. "books disagree by half a
        # percentage point" is a real tight-consensus signal; "books
        # disagree by 0.5 of a 0-1 probability" would almost never happen
        # and would make every check trivially pass or trivially fail).
        # direction="under" reuses the generic detector's "higher = confirm"
        # branch, since a higher implied probability for this side is the
        # moneyline equivalent of a sharp book posting a "harder" number.
        _book_lines_ml = [
            {"book": bk, "line": _american_to_implied(price) * 100.0}
            for (_pt, price, bk) in lines
        ]
        _all_probs_ml  = [_american_to_implied(price) * 100.0 for (_pt, price, _bk) in lines]
        _sharp_action  = detect_sharp_action(_book_lines_ml, direction="under")
        _steam_detected = detect_steam_move(_all_probs_ml, n_books, sport)
        _rlm_detected   = detect_reverse_line_movement(
            _sharp_action.get("sharp_consensus_line"),
            _sharp_action.get("rec_consensus_line"),
            direction="under",
        )

        bet_id = f"{team_abbr}_{label}_{side}"
        factor = (
            f"{away_team} @ {home_team} — {display} {team_name} "
            f"({best_odds:+d}, {n_books} books, best: {best_book}) "
            f"| model={model_prob:.1%}  fair_impl={fair_impl:.1%}"
            f" | {_sharp_action['signal_label']}"
            + (" | STEAM" if _steam_detected else "")
            + (" | RLM" if _rlm_detected else "")
        )
        cand = _base_candidate(
            bet_id=bet_id, game_label=game_label,
            away_team=away_team, home_team=home_team,
            away_abbr=away_abbr, home_abbr=home_abbr,
            team=team_abbr, mkt_key=mkt_key, direction=side,
            sportsbook_line=float(best_odds),  # odds serve as the "line" for ML
            american_odds=best_odds,
            bookmaker_source=best_book, book_count=n_books,
            historical_data=[], league_mean=0.5, league_std=0.15,
            factor=factor, game_time_et=game_time_et,
        )
        cand["precomputed_edge"]       = edge
        cand["precomputed_confidence"] = confidence
        cand["precomputed_model_prob"] = model_prob
        cand["sharp_signal"]           = _sharp_action["signal_type"]
        cand["sharp_label"]            = _sharp_action["signal_label"]
        cand["sharp_book_count"]       = _sharp_action["sharp_book_count"]
        cand["steam_detected"]         = _steam_detected
        cand["rlm_detected"]           = _rlm_detected
        candidates.append(cand)


def _process_nrfi_yrfi(
    candidates: list[dict[str, Any]],
    book_outcomes_list: list,
    home_abbr: str, away_abbr: str,
    home_team: str, away_team: str,
    home_hist: list[float], away_hist: list[float],
    game_label: str, game_time_et: Any, sport: str,
) -> None:
    """
    Pre-computed NRFI/YRFI candidates -- bypasses the NUTS sampler exactly
    like _process_moneyline (a Poisson closed-form model_prob stands in for
    the Bayesian engine here, the same role _win_prob() plays for h2h).

    market_key is set DIRECTLY to "nrfi" / "yrfi" on each candidate (not
    derived from _MARKET_LABEL by mkt_key, since both sides come from the
    same raw odds-API market totals_1st_1_innings) -- these are the exact
    internal keys core/market_gate.py's ALLOWED_MARKETS and
    core/decision_gatekeeper.py's _MARKET_ENTRY_FLOORS already expect.

    Lambda projection: models.nrfi_handicapper.project_combined_first_inning_lambda()
    implements the tiered framework (pitcher / lineup / environment) from
    NRFI_YRFI_F5_Elite_Handicapping_Reference.md. No first-inning-specific
    splits feed (FBF OBP, first-inning ERA, platoon top-4 stats, umpire
    zone history) is wired into this repo yet -- see that module's docstring
    for why those tiers are left at their honest neutral default rather than
    faked from season-long stats. Until that feed exists, the ONLY
    real per-game differentiator applied here is a coarse, clearly-labeled
    fallback: each team's own full-game run-scoring history, scaled down to
    a first-inning share and shrunk toward the league baseline via
    models.bayesian.shrink_mlb_nrfi_lambda -- structurally the same
    "scale the full-game number down" approach _process_scaled_total uses
    for F5, at the much smaller first-inning share of scoring.
    """
    import math

    from core.odds_client import _best_side  # type: ignore[attr-defined]
    from models.bayesian import shrink_mlb_nrfi_lambda
    from models.nrfi_handicapper import (
        NRFI_COMBINED_LAMBDA_BASELINE,
        project_combined_first_inning_lambda,
    )
    from models.sport_config import MLB as _MLB_CFG

    league_prior_per_team = NRFI_COMBINED_LAMBDA_BASELINE / 2.0
    full_game_half_mean = _GAME_TOTAL_PRIOR.get(sport, {"mean": 8.5, "std": 1.5})["mean"] * 0.5

    def _team_first_inning_lambda_fallback(hist: list[float]) -> float:
        if not hist:
            return league_prior_per_team
        recent_avg_full = sum(hist) / len(hist)
        # Scale this team's full-game run rate down to the first-inning
        # share implied by the league baseline, then shrink toward that
        # baseline by sample size -- a hot- or cold-scoring recent stretch
        # in a small sample shouldn't move the first-inning number as much
        # as it would move a full-game total (doc: first-inning splits are
        # noisier per sample than full-game/F5 numbers).
        share = league_prior_per_team / full_game_half_mean if full_game_half_mean else 0.0
        recent_first_inning_rate = recent_avg_full * share
        return shrink_mlb_nrfi_lambda(
            recent_first_inning_rate, len(hist), league_prior_per_team,
        )

    home_fallback_lambda = _team_first_inning_lambda_fallback(home_hist)
    away_fallback_lambda = _team_first_inning_lambda_fallback(away_hist)

    # project_combined_first_inning_lambda's tier inputs are left empty
    # (see docstring above) -- but we seed the league prior per side with
    # the fallback lambda so the tiered framework's dampening band still
    # applies around a real, per-game-varying anchor rather than a flat
    # league constant every night.
    projection = project_combined_first_inning_lambda(
        home_team_inputs=None, away_team_inputs=None,
        league_combined_lambda=home_fallback_lambda + away_fallback_lambda,
    )
    home_lambda = projection["home_lambda"]
    away_lambda = projection["away_lambda"]
    combined_lambda = projection["combined_lambda"]

    nrfi_model_prob = math.exp(-combined_lambda)
    yrfi_model_prob = 1.0 - nrfi_model_prob
    # Sanity-check against the doc's own stated prior -- a per-game
    # projection this far from the league baseline needs the causal tiers
    # above, not just the coarse fallback, before it should be trusted at
    # the extremes. Clamp keeps a noisy fallback from producing an
    # implausible near-certain NRFI/YRFI read on its own.
    nrfi_model_prob = max(0.45, min(0.92, nrfi_model_prob))
    yrfi_model_prob = 1.0 - nrfi_model_prob

    nrfi_lines: list[tuple[float, int, str]] = []
    yrfi_lines: list[tuple[float, int, str]] = []

    for bk_title, outcomes in book_outcomes_list:
        if bk_title == "Matchbook":
            continue
        for out in outcomes:
            name  = (out.get("name") or "").lower()
            price = out.get("price")
            if price is None:
                continue
            if name == "under":     # Under 0.5 == no run == NRFI
                nrfi_lines.append((0.5, int(price), bk_title))
            elif name == "over":    # Over 0.5 == a run scores == YRFI
                yrfi_lines.append((0.5, int(price), bk_title))

    for side, lines, model_prob in (
        ("nrfi", nrfi_lines, nrfi_model_prob),
        ("yrfi", yrfi_lines, yrfi_model_prob),
    ):
        result = _best_side(lines)
        if result is None:
            continue
        best_line, best_odds, best_book, n_books = result
        edge       = _edge_pct(model_prob, best_odds)
        confidence = _precomp_confidence(n_books, len(home_hist) + len(away_hist), model_prob)

        bet_id = f"{home_abbr}_{side}"
        factor = (
            f"{away_team} @ {home_team} — {side.upper()} "
            f"({best_odds:+d}, {n_books} books, best: {best_book}) "
            f"| model={model_prob:.1%} combined_lambda={combined_lambda:.3f} "
            f"(home={home_lambda:.3f}, away={away_lambda:.3f})"
        )
        cand = _base_candidate(
            bet_id=bet_id, game_label=game_label,
            away_team=away_team, home_team=home_team,
            away_abbr=away_abbr, home_abbr=home_abbr,
            team=home_abbr, mkt_key="totals_1st_1_innings", direction=side,
            sportsbook_line=0.5, american_odds=best_odds,
            bookmaker_source=best_book, book_count=n_books,
            historical_data=[], league_mean=combined_lambda, league_std=0.15,
            factor=factor, game_time_et=game_time_et,
        )
        # market_key / market OVERRIDE -- see function docstring: these two
        # keys must be exactly "nrfi"/"yrfi", not derived from mkt_key.
        cand["market_key"]             = side
        cand["market"]                 = side.upper()
        cand["precomputed_edge"]       = edge
        cand["precomputed_confidence"] = confidence
        cand["precomputed_model_prob"] = model_prob
        candidates.append(cand)


def _process_spread(
    candidates: list[dict[str, Any]],
    mkt_key: str, book_outcomes_list: list,
    home_abbr: str, away_abbr: str,
    home_team: str, away_team: str,
    home_hist: list[float], away_hist: list[float],
    game_label: str, game_time_et: Any, sport: str,
) -> None:
    """Pre-computed spread/run-line candidates for home and away sides."""
    from core.odds_client import _best_side  # type: ignore[attr-defined]

    label   = _MARKET_LABEL.get(mkt_key, mkt_key)
    display = _MARKET_DISPLAY.get(label, label)

    home_lines: list[tuple[float, int, str]] = []
    away_lines: list[tuple[float, int, str]] = []

    for bk_title, outcomes in book_outcomes_list:
        for out in outcomes:
            name  = out.get("name") or ""
            price = out.get("price")
            pt    = out.get("point")
            if price is None or pt is None:
                continue
            if name == home_team:
                home_lines.append((float(pt), int(price), bk_title))
            elif name == away_team:
                away_lines.append((float(pt), int(price), bk_title))

    # Stale line guard (2026-07-07 fix — was missing here even though the
    # comment in core/odds_client.py claimed this file already had it).
    # _best_side() picks whichever book offers the *best odds* for a side,
    # with no check that its point value agrees with the rest of the
    # market. A single off-market/stale book (e.g. still showing a team
    # at -1.5 after the market consensus has moved to +1.5) can have the
    # best odds precisely because it's mispriced, and would otherwise get
    # selected outright -- producing a pick with the wrong spread sign.
    # Reject the top-odds pick when its point drifts too far from the
    # cross-book consensus (median) for that side, retrying among only
    # the in-consensus books first. Mirrors the guard in
    # core/odds_client.py's total_over/total_under handling.
    _STALE_DRIFT_THRESHOLD: dict[str, float] = {
        "MLB": 0.5, "NBA": 1.0, "WNBA": 0.5,
    }
    _drift_limit = _STALE_DRIFT_THRESHOLD.get(sport.upper(), 0.75)

    for side, lines, team_abbr, team_name in (
        ("home", home_lines, home_abbr, home_team),
        ("away", away_lines, away_abbr, away_team),
    ):
        result = _best_side(lines)
        if result is None:
            continue
        spread, best_odds, best_book, n_books = result

        if lines:
            _all_pts = sorted(pt for pt, _, _ in lines)
            _consensus_line = _all_pts[len(_all_pts) // 2]
            if abs(spread - _consensus_line) > _drift_limit:
                _within = [
                    (ln, od, bk) for ln, od, bk in lines
                    if abs(ln - _consensus_line) <= _drift_limit
                ]
                _fb = _best_side(_within) if _within else None
                if _fb:
                    spread, best_odds, best_book, n_books = _fb
                else:
                    print(
                        f"[game_markets] STALE LINE rejected: {away_team}@{home_team} "
                        f"{side} best={spread} consensus={_consensus_line} "
                        f"drift={abs(spread - _consensus_line):.2f} > {_drift_limit}",
                        flush=True,
                    )
                    continue

        # Compute cover probability from the team's perspective
        if side == "home":
            model_prob, _ = _spread_cover_prob(home_hist, away_hist, spread, sport)
        else:
            # away spread is positive (e.g. +1.5); convert to home perspective
            _, model_prob = _spread_cover_prob(home_hist, away_hist, -spread, sport)

        fair_home = model_prob
        fair_away = 1.0 - model_prob
        if home_lines and away_lines:
            best_h = sorted(home_lines, key=lambda x: x[1], reverse=True)[0][1]
            best_a = sorted(away_lines, key=lambda x: x[1], reverse=True)[0][1]
            fair_home, fair_away = _devig(
                _american_to_implied(best_h), _american_to_implied(best_a)
            )
        fair_impl = fair_home if side == "home" else fair_away

        edge       = _edge_pct(model_prob, best_odds)
        confidence = _precomp_confidence(n_books, len(home_hist) + len(away_hist), model_prob)

        bet_id = f"{team_abbr}_{label}_{side}"
        factor = (
            f"{away_team} @ {home_team} — {display} {team_name} {spread:+g} "
            f"({best_odds:+d}, {n_books} books, best: {best_book}) "
            f"| cover_prob={model_prob:.1%}  fair_impl={fair_impl:.1%}"
        )
        cand = _base_candidate(
            bet_id=bet_id, game_label=game_label,
            away_team=away_team, home_team=home_team,
            away_abbr=away_abbr, home_abbr=home_abbr,
            team=team_abbr, mkt_key=mkt_key, direction=side,
            sportsbook_line=spread,
            american_odds=best_odds,
            bookmaker_source=best_book, book_count=n_books,
            historical_data=[], league_mean=0.5, league_std=0.15,
            factor=factor, game_time_et=game_time_et,
        )
        cand["precomputed_edge"]       = edge
        cand["precomputed_confidence"] = confidence
        cand["precomputed_model_prob"] = model_prob
        candidates.append(cand)


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def fetch_expanded_game_candidates(
    sport: str,
    as_of_date: str | None = None,
    snapshot_time: str = "10:00:00",
) -> list[dict[str, Any]]:
    """
    Fetch ALL approved game markets for *sport* beyond the base game total.

    Single Odds API request per sport per session.  Results are cached under
    the key "{SPORT}_EXPANDED" so repeated pipeline runs share one credit.

    Returns
    -------
    list[dict]
        Candidate dicts compatible with the main.py simulation pipeline.
        Total-based candidates have ``historical_data`` populated and flow
        through engine.analyze().  Moneyline/spread candidates carry
        ``precomputed_edge`` and bypass the NUTS sampler.
    """
    from datetime import date as _date, datetime as _datetime
    from core.odds_client import (  # type: ignore[attr-defined]
        _fetch, _fetch_historical, _et_day_bounds_utc, _abbrev, _validate_game,
    )
    from core.api_connector import normalize_api_timestamp
    from core.time_utils import convert_to_est, now_est
    from core.slate_cache import read_slate, write_slate

    sport_up    = sport.upper()
    api_sport   = _SPORT_KEY.get(sport_up)
    markets_str = _MARKET_BUNDLE.get(sport_up)
    if not api_sport or not markets_str:
        return []

    if as_of_date:
        today_et     = _date.fromisoformat(as_of_date)
        date_str     = as_of_date
        snapshot_iso = f"{as_of_date}T{snapshot_time}Z"
    else:
        today_et     = now_est().date()
        date_str     = today_et.isoformat()
        snapshot_iso = None

    cache_key = f"{sport_up}_EXPANDED"

    # Live slate cache is swapped for an isolated per-date replay cache dir
    # during replay — same reasoning as fetch_todays_candidates: shares one
    # API credit across pipeline stages for a replay date, without ever
    # touching or being served by the live cache.
    cache_dir = os.path.join("data", "slate_cache_replay") if as_of_date else None
    cached = read_slate(cache_key, date_str, cache_dir=cache_dir)
    if cached is not None:
        print(
            f"[game_markets] {sport_up} expanded markets from cache "
            f"({len(cached)} candidates).",
            flush=True,
        )
        return cached

    commence_from, commence_to = _et_day_bounds_utc(today_et)
    params = (
        f"regions=us,eu&markets={markets_str}&oddsFormat=american&dateFormat=iso"
        f"&commenceTimeFrom={commence_from}&commenceTimeTo={commence_to}"
    )

    try:
        if as_of_date is not None:
            raw_games = _fetch_historical(f"sports/{api_sport}/odds/", params, snapshot_iso)
        else:
            raw_games = _fetch(f"sports/{api_sport}/odds/", params)
    except Exception as exc:
        print(f"[game_markets] {sport_up} expanded fetch failed: {exc}", flush=True)
        return []

    if not isinstance(raw_games, list):
        return []

    candidates: list[dict[str, Any]] = []

    for game in raw_games:
        ok, reason = _validate_game(game)
        if not ok:
            continue

        home_team = game["home_team"]
        away_team = game["away_team"]
        game_id   = game["id"]

        try:
            game_time_utc = normalize_api_timestamp(game["commence_time"])
        except Exception:
            continue

        game_et = convert_to_est(game_time_utc)
        if game_et.date() != today_et:
            continue
        _cutoff_now = now_est() if as_of_date is None else convert_to_est(
            _datetime.fromisoformat(snapshot_iso.replace("Z", "+00:00"))
        )
        if game_et <= _cutoff_now:
            continue

        home_abbr  = _abbrev(home_team)
        away_abbr  = _abbrev(away_team)
        game_label = f"{away_abbr}@{home_abbr}_{sport_up}_{date_str}"
        game_seed  = float(sum(ord(ch) for ch in game_id[:8]) % 100_000)

        home_hist, away_hist = _get_team_histories(
            sport_up, home_team, away_team, game_seed, as_of_date=as_of_date,
        )
        comb_hist = _combined_hist(home_hist, away_hist)

        # Collect per-market-key bookmaker lines
        mkt_books: dict[str, list] = {}
        for bk in game.get("bookmakers", []):
            bk_title = bk.get("title") or bk.get("key", "Unknown")
            if bk_title == "Matchbook":  # EU-only, consistently stale vs US market
                continue
            for mkt in bk.get("markets", []):
                mkt_key = mkt.get("key", "")
                if mkt_key not in mkt_books:
                    mkt_books[mkt_key] = []
                mkt_books[mkt_key].append((bk_title, mkt.get("outcomes", [])))

        game_start = len(candidates)

        for mkt_key, book_list in mkt_books.items():
            if not book_list:
                continue

            if mkt_key in _SCALED_TOTAL_MARKETS:
                _process_scaled_total(
                    candidates, mkt_key, book_list,
                    home_abbr, away_abbr, home_team, away_team,
                    comb_hist, game_label, game_et, sport_up,
                )
            elif mkt_key == "team_totals":
                _process_team_total(
                    candidates, book_list,
                    home_abbr, away_abbr, home_team, away_team,
                    home_hist, away_hist,
                    game_label, game_et, sport_up,
                )
            elif mkt_key == "totals_1st_1_innings":
                _process_nrfi_yrfi(
                    candidates, book_list,
                    home_abbr, away_abbr, home_team, away_team,
                    home_hist, away_hist,
                    game_label, game_et, sport_up,
                )
            elif mkt_key in _PRECOMPUTED_MARKETS:
                if mkt_key.startswith("h2h"):
                    _process_moneyline(
                        candidates, mkt_key, book_list,
                        home_abbr, away_abbr, home_team, away_team,
                        home_hist, away_hist,
                        game_label, game_et, sport_up,
                    )
                else:
                    _process_spread(
                        candidates, mkt_key, book_list,
                        home_abbr, away_abbr, home_team, away_team,
                        home_hist, away_hist,
                        game_label, game_et, sport_up,
                    )

        n_game = len(candidates) - game_start
        print(
            f"[game_markets] {sport_up} {away_abbr}@{home_abbr}: "
            f"{n_game} expanded candidate(s) from "
            f"{len([k for k in mkt_books if k != 'totals'])} market type(s).",
            flush=True,
        )

    # Args were previously swapped (real signature is
    # write_slate(key, date_str, candidates)) — every write here was
    # silently corrupted. In replay mode this writes to the isolated
    # replay cache dir, never the live cache.
    write_slate(cache_key, date_str, candidates, cache_dir=cache_dir)
    print(
        f"[game_markets] {sport_up}: {len(candidates)} expanded candidates total "
        f"(cached for session).",
        flush=True,
    )
    return candidates
