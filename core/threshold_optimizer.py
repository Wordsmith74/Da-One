"""
core/threshold_optimizer.py -- Data-driven Nuke/Diamond/Gold threshold search.

This is the missing 5th step of the grading workflow:

    1. Find every ungraded pick             -> core.historical_grader.run()
    2. Determine the official result        -> core.historical_grader.grade_pick()
    3. Write the grades back                -> core.historical_grader.run()
    4. Run calibration across graded bets    -> core.calibration.generate_summary()
    5. Output the best thresholds to replace
       Gold / Diamond / Nuke                -> THIS MODULE

Steps 1-4 already existed and run live (core/historical_grader.py). This
module reads the same graded records calibration.py already reports on and
searches for edge_pct / confidence / consensus (side_agreement_frac)
cutoffs that would have produced the best realized ROI, structured as a
direct drop-in replacement for decision_gatekeeper.SPORT_TIER_THRESHOLDS.

Method
------
For each (sport, market_class) group where market_class is "game" or
"prop" (via core.edge_calibrator.is_game_market):

  1. Build candidate cut points from the *observed* quantiles of edge_pct
     and confidence in that group (keeps the grid small and keeps every
     candidate threshold anchored to a real bet that was actually made --
     no interpolated/synthetic cutoffs).
  2. Score every (edge_cut, confidence_cut) combination: subset = picks
     with edge_pct >= edge_cut AND confidence >= confidence_cut. Compute
     n, win_pct, roi for the subset. Discard combinations below
     MIN_SAMPLE_SIZE -- we will not recommend a threshold no bet volume
     can support.
  3. Greedily peel off three nested tiers from loosest to strictest:
       Gold    = the loosest surviving combination whose ROI clears
                 GOLD_ROI_FLOOR (default breakeven, roi >= 0).
       Diamond = the loosest combination that is >= Gold's cuts on both
                 axes and beats Gold's ROI by at least TIER_STEP_ROI.
       Nuke    = the loosest combination that is >= Diamond's cuts on
                 both axes and beats Diamond's ROI by at least
                 TIER_STEP_ROI.
     "Loosest surviving" is used at each step (rather than the single
     best-ROI combo) to avoid hand-picking an overfit spike -- ties in
     ROI are broken toward more sample / looser thresholds.
  4. If side_agreement_frac ("consensus") has enough coverage in the
     group, it is folded in as a third axis the same way; otherwise it is
     left out of that group's recommendation and flagged as
     "consensus_not_evaluated" (insufficient data -- not silently ignored).

Groups without enough graded volume are reported as
"insufficient_data" rather than guessing -- this tool will not emit a
threshold it cannot support with evidence.

Run as a script:
    python -m core.threshold_optimizer --pick-history output/pick_history.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .grading_utils import market_normalized, read_jsonl
from .edge_calibrator import is_game_market

logger = logging.getLogger("threshold_optimizer")

DEFAULT_PICK_HISTORY_PATH = "output/pick_history.jsonl"
DEFAULT_OUTPUT_PATH = "output/threshold_recommendations.json"

# Minimum graded picks a candidate cutoff must have behind it to be
# considered at all. Below this, win%/ROI is noise, not signal.
MIN_SAMPLE_SIZE = 15
# Smaller floor for the strictest tier (Nuke), since by construction the
# eligible pool shrinks as thresholds rise -- still a real floor, not zero.
MIN_SAMPLE_SIZE_NUKE = 8

# How many quantile-based candidate cut points to test per axis. Candidates
# are pulled from observed data, not synthesized, so this just controls
# search resolution, not the actual thresholds available.
N_CANDIDATES = 10

# ROI a threshold must clear to be usable as the Gold (entry) floor.
GOLD_ROI_FLOOR = 0.0  # breakeven or better
# Minimum ROI improvement required to justify each stricter tier existing
# as a separate tier at all (otherwise Diamond/Nuke add no information
# over the tier below them).
TIER_STEP_ROI = 0.05  # +5 percentage points of ROI


@dataclass
class TierCandidate:
    edge_threshold: float
    confidence_threshold: float
    consensus_threshold: Optional[float]
    n: int
    wins: int
    losses: int
    pushes: int
    win_pct: Optional[float]
    roi: float
    profit_units: float


@dataclass
class GroupRecommendation:
    sport: str
    market_class: str  # "game" | "prop"
    n_graded: int
    status: str  # "ok" | "insufficient_data"
    consensus_evaluated: bool
    current_thresholds: Optional[dict]
    recommended: Optional[dict[str, Optional[dict]]]  # {"Nuke":..., "Diamond":..., "Gold":...}
    note: str


def _quantile_points(values: list[float], n: int) -> list[float]:
    """Pick up to n distinct cut points spread across the observed range,
    anchored to real observed values (so every candidate threshold
    corresponds to at least one actual graded bet)."""
    uniq = sorted(set(values))
    if len(uniq) <= n:
        return uniq
    step = (len(uniq) - 1) / (n - 1)
    return sorted({uniq[round(i * step)] for i in range(n)})


def _wlp(records: list[dict]) -> tuple[int, int, int, float, float]:
    wins = losses = pushes = 0
    profit = 0.0
    staked = 0.0
    for r in records:
        result = r.get("actual_result")
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        elif result == "push":
            pushes += 1
        else:
            continue
        profit += r.get("profit_units") or 0.0
        staked += (r.get("stake_pct_bankroll") or 1.0) / 100.0
    roi = profit / staked if staked else 0.0
    return wins, losses, pushes, roi, profit


def _score(records: list[dict], edge_cut: float, conf_cut: float,
           cons_cut: Optional[float]) -> Optional[TierCandidate]:
    subset = [
        r for r in records
        if abs(r["_edge_abs"]) >= edge_cut and r["_confidence"] >= conf_cut
        and (cons_cut is None or (r["_consensus"] is not None and r["_consensus"] >= cons_cut))
    ]
    if not subset:
        return None
    wins, losses, pushes, roi, profit = _wlp(subset)
    n = wins + losses + pushes
    win_pct = round(wins / (wins + losses), 4) if (wins + losses) else None
    return TierCandidate(
        edge_threshold=edge_cut,
        confidence_threshold=conf_cut,
        consensus_threshold=cons_cut,
        n=n, wins=wins, losses=losses, pushes=pushes,
        win_pct=win_pct, roi=round(roi, 4), profit_units=round(profit, 2),
    )


def _peel_tiers(records: list[dict], edge_candidates: list[float],
                conf_candidates: list[float],
                cons_candidates: list[Optional[float]]) -> dict[str, Optional[TierCandidate]]:
    """Greedily find loosest-surviving Gold, then a stricter Diamond that
    clears TIER_STEP_ROI over Gold, then a stricter Nuke over Diamond."""
    all_candidates = []
    for e in edge_candidates:
        for c in conf_candidates:
            for k in cons_candidates:
                cand = _score(records, e, c, k)
                if cand is None:
                    continue
                min_n = MIN_SAMPLE_SIZE
                all_candidates.append(cand)

    # Sort loosest-first: ascending on both thresholds (ties broken by n desc)
    def looseness_key(c: TierCandidate):
        return (c.edge_threshold, c.confidence_threshold, c.consensus_threshold or 0.0, -c.n)

    all_candidates.sort(key=looseness_key)

    gold = None
    for c in all_candidates:
        if c.n >= MIN_SAMPLE_SIZE and c.roi >= GOLD_ROI_FLOOR:
            gold = c
            break
    if gold is None:
        return {"Gold": None, "Diamond": None, "Nuke": None}

    diamond = None
    for c in all_candidates:
        if (c.edge_threshold >= gold.edge_threshold and c.confidence_threshold >= gold.confidence_threshold
                and (c.edge_threshold, c.confidence_threshold) != (gold.edge_threshold, gold.confidence_threshold)
                and c.n >= MIN_SAMPLE_SIZE and c.roi >= gold.roi + TIER_STEP_ROI):
            diamond = c
            break

    nuke = None
    if diamond is not None:
        for c in all_candidates:
            if (c.edge_threshold >= diamond.edge_threshold and c.confidence_threshold >= diamond.confidence_threshold
                    and (c.edge_threshold, c.confidence_threshold) != (diamond.edge_threshold, diamond.confidence_threshold)
                    and c.n >= MIN_SAMPLE_SIZE_NUKE and c.roi >= diamond.roi + TIER_STEP_ROI):
                nuke = c
                break

    return {"Gold": gold, "Diamond": diamond, "Nuke": nuke}


# Mirrors decision_gatekeeper.SPORT_TIER_THRESHOLDS for the "current" side
# of the before/after comparison. Duplicated here (not imported) so this
# tool works standalone even if decision_gatekeeper's import chain (which
# pulls in market_weights, confidence_caps, market_agreement, etc.) fails
# in an environment that only has pick_history.jsonl available.
_CURRENT_SPORT_TIER_THRESHOLDS = {
    "MLB":  {"Nuke": (3.5, 85.0), "Diamond": (2.0, 78.0), "Gold": (1.0, 68.0)},
    "NBA":  {"Nuke": (4.0, 85.0), "Diamond": (2.5, 78.0), "Gold": (1.5, 68.0)},
    "WNBA": {"Nuke": (3.5, 85.0), "Diamond": (2.0, 78.0), "Gold": (1.0, 68.0)},
}


def optimize(pick_history_path: str = DEFAULT_PICK_HISTORY_PATH) -> dict[str, Any]:
    records = read_jsonl(pick_history_path)
    graded = [r for r in records if r.get("actual_result") in ("win", "loss", "push")]

    # Normalize the fields we need onto private keys so we don't mutate the
    # persisted schema, and so missing/odd values degrade to "excluded"
    # rather than crashing the search.
    for r in graded:
        try:
            r["_edge_abs"] = abs(float(r.get("edge_pct")))
        except (TypeError, ValueError):
            r["_edge_abs"] = None
        try:
            r["_confidence"] = float(r.get("confidence"))
        except (TypeError, ValueError):
            r["_confidence"] = None
        saf = r.get("side_agreement_frac")
        try:
            r["_consensus"] = float(saf) if saf is not None else None
        except (TypeError, ValueError):
            r["_consensus"] = None
        raw_sport = (r.get("sport") or "").strip()
        r["_sport_key"] = raw_sport.split()[0].upper() if raw_sport else "UNKNOWN"
        r["_market_class"] = "game" if is_game_market(market_normalized(r.get("market", ""))) else "prop"

    usable = [r for r in graded if r["_edge_abs"] is not None and r["_confidence"] is not None]

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in usable:
        groups[(r["_sport_key"], r["_market_class"])].append(r)

    group_reports: list[GroupRecommendation] = []

    for (sport, market_class), recs in sorted(groups.items()):
        n = len(recs)
        current = _CURRENT_SPORT_TIER_THRESHOLDS.get(sport)
        current_out = (
            {k: {"edge_pct": v[0], "confidence": v[1]} for k, v in current.items()}
            if current else None
        )

        if n < MIN_SAMPLE_SIZE:
            group_reports.append(GroupRecommendation(
                sport=sport, market_class=market_class, n_graded=n,
                status="insufficient_data", consensus_evaluated=False,
                current_thresholds=current_out, recommended=None,
                note=f"Only {n} graded picks (need >= {MIN_SAMPLE_SIZE}). "
                     f"Keeping existing thresholds until more results are graded.",
            ))
            continue

        edge_candidates = _quantile_points([r["_edge_abs"] for r in recs], N_CANDIDATES)
        conf_candidates = _quantile_points([r["_confidence"] for r in recs], N_CANDIDATES)

        cons_values = [r["_consensus"] for r in recs if r["_consensus"] is not None]
        consensus_evaluated = len(cons_values) >= MIN_SAMPLE_SIZE
        cons_candidates: list[Optional[float]] = (
            [None] + _quantile_points(cons_values, N_CANDIDATES) if consensus_evaluated else [None]
        )

        tiers = _peel_tiers(recs, edge_candidates, conf_candidates, cons_candidates)

        if tiers["Gold"] is None:
            group_reports.append(GroupRecommendation(
                sport=sport, market_class=market_class, n_graded=n,
                status="insufficient_data", consensus_evaluated=consensus_evaluated,
                current_thresholds=current_out, recommended=None,
                note="No edge/confidence cutoff in this group cleared breakeven ROI "
                     f"with n >= {MIN_SAMPLE_SIZE}. Keeping existing thresholds.",
            ))
            continue

        recommended = {
            tier: (asdict(cand) if cand else None) for tier, cand in tiers.items()
        }
        note_bits = []
        if tiers["Diamond"] is None:
            note_bits.append("No Diamond cutoff beat Gold's ROI by the required margin -- "
                              "recommend collapsing Diamond into Gold for this group.")
        if tiers["Diamond"] is not None and tiers["Nuke"] is None:
            note_bits.append("No Nuke cutoff beat Diamond's ROI by the required margin -- "
                              "recommend collapsing Nuke into Diamond for this group.")
        if not consensus_evaluated:
            note_bits.append(f"consensus_not_evaluated: only {len(cons_values)} picks had "
                              f"side_agreement_frac populated (need >= {MIN_SAMPLE_SIZE}).")

        group_reports.append(GroupRecommendation(
            sport=sport, market_class=market_class, n_graded=n,
            status="ok", consensus_evaluated=consensus_evaluated,
            current_thresholds=current_out, recommended=recommended,
            note=" ".join(note_bits) if note_bits else "OK",
        ))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_total_picks": len(records),
        "n_graded": len(graded),
        "n_usable_for_search": len(usable),
        "min_sample_size": MIN_SAMPLE_SIZE,
        "min_sample_size_nuke": MIN_SAMPLE_SIZE_NUKE,
        "gold_roi_floor": GOLD_ROI_FLOOR,
        "tier_step_roi": TIER_STEP_ROI,
        "groups": [asdict(g) for g in group_reports],
    }
    return payload


def print_report(payload: dict[str, Any]) -> None:
    print(f"=== Threshold Recommendations ({payload['n_graded']} graded / "
          f"{payload['n_total_picks']} total picks) ===\n")
    for g in payload["groups"]:
        print(f"--- {g['sport']} / {g['market_class']} markets (n={g['n_graded']}) ---")
        if g["status"] == "insufficient_data":
            print(f"  INSUFFICIENT DATA: {g['note']}")
            if g["current_thresholds"]:
                print(f"  Keeping current: {g['current_thresholds']}")
            print()
            continue

        cur = g["current_thresholds"] or {}
        for tier in ("Nuke", "Diamond", "Gold"):
            rec = g["recommended"].get(tier)
            cur_t = cur.get(tier)
            cur_str = f"(was edge>={cur_t['edge_pct']}% conf>={cur_t['confidence']})" if cur_t else "(no prior threshold)"
            if rec is None:
                print(f"  {tier:<8}: -- no supported cutoff {cur_str}")
                continue
            cons = f" consensus>={rec['consensus_threshold']}" if rec["consensus_threshold"] is not None else ""
            print(
                f"  {tier:<8}: edge>={rec['edge_threshold']}% conf>={rec['confidence_threshold']}{cons}  "
                f"{cur_str}  ->  n={rec['n']} win%={rec['win_pct']} roi={rec['roi']:.2%}"
            )
        if g["note"] and g["note"] != "OK":
            print(f"  note: {g['note']}")
        print()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Search graded picks for optimal tier thresholds.")
    parser.add_argument("--pick-history", default=DEFAULT_PICK_HISTORY_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    payload = optimize(args.pick_history)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print_report(payload)
    print(f"(full detail written to {args.output})")


if __name__ == "__main__":
    main()
