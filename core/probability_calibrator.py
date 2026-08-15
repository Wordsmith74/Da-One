"""
core/probability_calibrator.py -- Empirical probability calibration (Fix #2)

Problem this fixes
-------------------
model_prob (and everything downstream of it -- confidence_score from
composite_confidence_score.py, calibrate_edge() in edge_calibrator.py) is a
*model output*, not a calibrated probability. None of those numbers are fit
against realized outcomes: composite_confidence_score.py's five factors use
fixed hand-tuned weights and a hardcoded per-market "efficiency prior"
(e.g. pitcher_strikeouts = 62.0, picked by inspection, not fit), and
edge_calibrator.calibrate_edge() is a fixed linear rescale of the raw
Bayesian edge, also not fit to outcomes.

Audit of output/pick_history.jsonl confirmed the consequence: MLB
pitcher_strikeouts picks scored at 70-90% "confidence" won at ~51% actual
--roughly a coin flip. The model's *ranking* may still be informative (a
picked-as-70% play may really be better than a picked-as-55% play), but the
*scale* is wrong, and every downstream consumer of model_prob --
kelly_stake() above all -- was sizing bets off that wrong scale.

Fix
---
Fit a monotonic (isotonic) mapping from raw model_prob -> empirical win
rate, using graded history, via pooled-adjacent-violators (PAV). Isotonic
regression is the standard tool for exactly this failure mode: it preserves
the model's ranking (it can only stretch/compress the scale, never invert
which picks look better than which) while correcting the absolute
probability to match what actually happened.

Fit separately per (sport, market_class) group -- "77%" from the MLB
strikeout model and "77%" from the WNBA rebounds model can be biased in
different directions and by different amounts; pooling them would launder
market-specific miscalibration into every market's numbers.

Kelly staking must ALWAYS consume the calibrated probability, never raw
model_prob directly -- see run_pipeline.py wiring (search "CALIBRATED
PROBABILITY" there).

No sklearn dependency -- requirements.txt doesn't have it, and PAV is ~30
lines without it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .grading_utils import market_normalized, read_jsonl
from .edge_calibrator import is_game_market

logger = logging.getLogger("probability_calibrator")

DEFAULT_PICK_HISTORY_PATH = "output/pick_history.jsonl"
DEFAULT_CALIBRATION_PATH = "output/probability_calibration.json"

# Below this many graded (win/loss) picks in a group, a fitted curve isn't
# trustworthy -- fall back to the identity mapping (raw model_prob passed
# through unchanged) and flag the group as unfitted rather than silently
# guessing. Matches the spirit of threshold_optimizer.MIN_SAMPLE_SIZE (15)
# but set a bit higher here since isotonic regression has more degrees of
# freedom than a single threshold cut and overfits faster on small n.
MIN_SAMPLE_SIZE = 30


# ---------------------------------------------------------------------------
# Pooled Adjacent Violators isotonic regression
# ---------------------------------------------------------------------------

def _pav_isotonic(probs: list[float], outcomes: list[int]) -> list[tuple[float, float]]:
    """
    Fit a monotonic non-decreasing step function mapping raw probability ->
    empirical win rate via the pooled-adjacent-violators algorithm.

    probs: raw model probabilities (any order).
    outcomes: realized 0/1 outcomes, aligned with probs (1 = win).

    Returns a list of (raw_prob, calibrated_prob) points sorted ascending
    by raw_prob, meant to be linearly interpolated between at lookup time
    (see _interp). Each point's raw_prob is the mean raw probability of the
    pooled block it represents (rather than a block edge), so the fitted
    curve is a smoother, more representative step-through-blocks line.
    """
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    x = [probs[i] for i in order]
    y = [float(outcomes[i]) for i in order]

    # Each block tracks [sum_of_y, count, list_of_x] so we can merge blocks
    # (pool) whenever an earlier block's average win rate exceeds a later
    # block's -- that's the "violation" of monotonicity PAV removes.
    blocks: list[list] = [[yi, 1, [xi]] for xi, yi in zip(x, y)]

    i = 0
    while i < len(blocks) - 1:
        avg_i = blocks[i][0] / blocks[i][1]
        avg_next = blocks[i + 1][0] / blocks[i + 1][1]
        if avg_i > avg_next:
            merged = [
                blocks[i][0] + blocks[i + 1][0],
                blocks[i][1] + blocks[i + 1][1],
                blocks[i][2] + blocks[i + 1][2],
            ]
            blocks[i:i + 2] = [merged]
            i = max(0, i - 1)  # re-check the newly merged block against its left neighbor
        else:
            i += 1

    points: list[tuple[float, float]] = []
    for sum_y, count, xs in blocks:
        avg_y = sum_y / count
        mean_x = sum(xs) / len(xs)
        points.append((mean_x, avg_y))
    return points


def _interp(points: list[tuple[float, float]], x: float) -> float:
    """Piecewise-linear interpolation over PAV output.

    Outside the observed [min_x, max_x] range there is, by definition, zero
    graded evidence -- so instead of clamping to whichever boundary point is
    nearest (which used to mean a *single* graded pick's win/loss outcome
    got applied as a hard floor/ceiling to every query beyond it -- see
    CalibrationCurve.apply()'s docstring), we return the raw x unchanged.
    That keeps this function's job strictly to interpolation; the
    identity-fallback decision for out-of-range x lives in apply().
    """
    if not points:
        return x
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return points[-1][1]


# ---------------------------------------------------------------------------
# Curve object + fit/save/load/apply
# ---------------------------------------------------------------------------

@dataclass
class CalibrationCurve:
    sport: str
    market_class: str          # "game" | "prop"
    n: int
    points: list                # list[tuple[float, float]]; empty when unfitted
    fitted: bool                # False -> apply() is identity (insufficient n)
    train_window: tuple = (None, None)   # (earliest, latest) generated_at in the fit set
    x_min: Optional[float] = None   # min raw_prob actually seen in the fit set
    x_max: Optional[float] = None   # max raw_prob actually seen in the fit set

    def apply(self, raw_prob: float) -> float:
        if not self.fitted:
            return raw_prob
        # Bug fix (extrapolation clamp): a query outside [x_min, x_max] has
        # zero graded evidence behind it. _interp() used to clamp these to
        # the y-value of the nearest training point -- but PAV's boundary
        # points are frequently built from a single graded pick (confirmed
        # via output/pick_history.jsonl: MLB props' and both sports' game
        # markets' leftmost anchor was n=1, one loss, y=0.0). That meant
        # e.g. every MLB pitcher_strikeouts "over" candidate, whose raw
        # probabilities structurally land in the 30s-40s (below the fit
        # set's observed minimum of ~48.6%), was being told "0% true
        # probability" off the back of one unrelated pick's outcome, not
        # off any evidence about probabilities in that range. Match the
        # module's own "don't calibrate what you don't have evidence for"
        # principle (already applied at the group/MIN_SAMPLE_SIZE level)
        # by falling back to identity for out-of-range queries too, rather
        # than extrapolating via boundary-clamp.
        if self.x_min is not None and raw_prob < self.x_min:
            return raw_prob
        if self.x_max is not None and raw_prob > self.x_max:
            return raw_prob
        return _interp(self.points, raw_prob)


def _group_key(sport_raw: str, market_raw: str) -> tuple[str, str]:
    sport_key = (sport_raw or "UNKNOWN").strip().split()[0].upper() if sport_raw else "UNKNOWN"
    mkt_norm = market_normalized(market_raw or "")
    # nrfi/yrfi get their OWN market_class rather than falling into "game"
    # or "prop" -- is_game_market()'s substring list (totals/moneyline/
    # h2h/run_line/etc) doesn't match "nrfi"/"yrfi" at all, so before this
    # fix they silently fell into "prop" and were calibrated against
    # whatever else MLB's prop bucket contained. In practice that bucket is
    # dominated by pitcher_strikeouts (207 of ~207 graded MLB prop picks
    # at last fit, vs 0 graded nrfi/yrfi picks), so nrfi/yrfi bets were
    # being calibrated entirely off strikeout-prop history -- a market with
    # a completely different probability structure. Confirmed by
    # (MLB, nrfi/yrfi) calibration-veto rejections showing raw model_prob
    # in the low-50s mapping to a "true" probability read straight off that
    # strikeout curve. Giving first-inning markets their own group means
    # they fall back to the safe raw_prob identity mapping (via
    # calibrate_probability's own MIN_SAMPLE_SIZE guard) until nrfi/yrfi
    # accumulate enough graded history of their own to fit a curve that
    # actually reflects how THIS market performs -- same "don't calibrate
    # what you don't have evidence for" principle the rest of this module
    # already follows for every other group.
    if mkt_norm in ("nrfi", "yrfi"):
        market_class = "first_inning"
    else:
        market_class = "game" if is_game_market(mkt_norm) else "prop"
    return sport_key, market_class


def _normalize_prob_scale(prob: float) -> float:
    """
    output/pick_history.jsonl mixes two scales for model_prob: the
    earliest records (2026-06-30, before ~12:44 UTC) store it 0-1
    (e.g. 0.7756), but every record after that stores it 0-100
    (e.g. 86.7) -- almost certainly a since-changed call site that used to
    divide by 100 before logging and, at some point, stopped. Fitting a
    single isotonic curve across both scales unmodified produces a curve
    spanning x=0.78 to x=100 in the same fit, which is meaningless.
    Treat anything > 1.0 as the 0-100 scale and rescale down; anything
    <= 1.0 is assumed already 0-1. This is a heuristic, not a fix for the
    underlying logging inconsistency -- see the note in run_pipeline.py /
    data/cache_history.py; that inconsistency should be fixed at the
    source too, not just compensated for here.
    """
    return prob / 100.0 if prob > 1.0 else prob


def fit_calibration_curves(records: list[dict]) -> dict:
    """
    Fit one isotonic curve per (sport, market_class) group from graded
    records (actual_result in {"win","loss"} -- pushes carry no probability
    information and are excluded).

    Returns {(sport, market_class): CalibrationCurve}.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        result = r.get("actual_result")
        if result not in ("win", "loss"):
            continue
        prob = r.get("model_prob")
        if prob is None:
            continue
        try:
            prob = _normalize_prob_scale(float(prob))
        except (TypeError, ValueError):
            continue
        key = _group_key(r.get("sport", ""), r.get("market", ""))
        groups.setdefault(key, []).append({
            "prob": prob,
            "y": 1 if result == "win" else 0,
            "generated_at": r.get("generated_at"),
        })

    curves: dict[tuple[str, str], CalibrationCurve] = {}
    for (sport, market_class), recs in groups.items():
        n = len(recs)
        times = sorted(r["generated_at"] for r in recs if r.get("generated_at"))
        window = (times[0], times[-1]) if times else (None, None)
        if n < MIN_SAMPLE_SIZE:
            curves[(sport, market_class)] = CalibrationCurve(
                sport=sport, market_class=market_class, n=n,
                points=[], fitted=False, train_window=window,
            )
            logger.info(
                "[calibration] %s/%s: only %d graded picks (need >= %d) -- "
                "identity fallback, raw model_prob passed through unmodified.",
                sport, market_class, n, MIN_SAMPLE_SIZE,
            )
            continue
        fit_probs = [r["prob"] for r in recs]
        points = _pav_isotonic(fit_probs, [r["y"] for r in recs])
        curves[(sport, market_class)] = CalibrationCurve(
            sport=sport, market_class=market_class, n=n,
            points=points, fitted=True, train_window=window,
            x_min=min(fit_probs), x_max=max(fit_probs),
        )
    return curves


def save_calibration_curves(curves: dict, path: str = DEFAULT_CALIBRATION_PATH) -> None:
    payload = {
        f"{sport}|{market_class}": {
            "sport": c.sport, "market_class": c.market_class, "n": c.n,
            "fitted": c.fitted,
            "points": [list(p) for p in c.points],
            "train_window": list(c.train_window),
            "x_min": c.x_min, "x_max": c.x_max,
        }
        for (sport, market_class), c in curves.items()
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))


def load_calibration_curves(path: str = DEFAULT_CALIBRATION_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    curves = {}
    for key, v in raw.items():
        sport, market_class = key.split("|", 1)
        curves[(sport, market_class)] = CalibrationCurve(
            sport=v["sport"], market_class=v["market_class"], n=v["n"],
            points=[tuple(pt) for pt in v["points"]], fitted=v["fitted"],
            train_window=tuple(v.get("train_window", (None, None))),
            x_min=v.get("x_min"), x_max=v.get("x_max"),
        )
    return curves


# Process-level cache so run_pipeline.py doesn't re-read the calibration
# file from disk on every single candidate -- loaded once per run, first
# time calibrate_probability() is called without an explicit curves dict.
_CACHED_CURVES: Optional[dict] = None


def calibrate_probability(raw_prob: float, sport: str, market: str,
                           curves: Optional[dict] = None) -> float:
    """
    Public entry point used at pick time. Looks up the fitted curve for
    this (sport, market_class) and applies it to raw_prob.

    Falls back to raw_prob unmodified if no calibration file exists yet, or
    the group didn't have enough graded samples to fit -- calibration can
    only correct what it has evidence for. This is a deliberate silent-safe
    default: an engine with zero graded history should behave exactly as it
    does today until enough picks have been graded to calibrate against.
    """
    global _CACHED_CURVES
    if curves is None:
        if _CACHED_CURVES is None:
            _CACHED_CURVES = load_calibration_curves()
        curves = _CACHED_CURVES
    key = _group_key(sport, market)
    curve = curves.get(key)
    if curve is None:
        return raw_prob
    # Defensive: normalize scale here too, not just at fit time, in case a
    # future call site regresses back to passing 0-100 -- see
    # _normalize_prob_scale's docstring for why this mixed-scale bug
    # exists in the historical data in the first place.
    normalized = _normalize_prob_scale(raw_prob)
    return curve.apply(normalized)


def refit_and_save(pick_history_path: str = DEFAULT_PICK_HISTORY_PATH,
                    output_path: str = DEFAULT_CALIBRATION_PATH) -> dict:
    """
    Refit curves from the full graded history and persist to disk. Run this
    on the same cadence as optimize_thresholds.yml (it reads the same
    pick_history.jsonl) so the calibration curve keeps pace as more picks
    get graded. Also invalidates the in-process cache so a long-running
    process picks up the refit immediately rather than on next restart.
    """
    global _CACHED_CURVES
    records = read_jsonl(pick_history_path)
    curves = fit_calibration_curves(records)
    save_calibration_curves(curves, output_path)
    _CACHED_CURVES = curves
    return {
        f"{sport}/{mc}": {"n": c.n, "fitted": c.fitted, "train_window": c.train_window}
        for (sport, mc), c in curves.items()
    }


if __name__ == "__main__":
    summary = refit_and_save()
    print(f"Fitted calibration curves from {DEFAULT_PICK_HISTORY_PATH}:\n")
    for key, info in sorted(summary.items()):
        status = "fitted" if info["fitted"] else "IDENTITY (insufficient graded data)"
        print(f"  {key:20s} n={info['n']:4d}  {status}")
    print(f"\nSaved to {DEFAULT_CALIBRATION_PATH}")
