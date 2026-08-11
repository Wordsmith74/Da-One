"""
run_grading_workflow.py -- One command for the full grade -> calibrate ->
re-threshold loop.

    1. Find every ungraded pick             (core.historical_grader.run)
    2. Determine the official result          "
    3. Write the grades back                  "
    4. Run calibration across all graded bets  "
    5. Output best Nuke/Diamond/Gold thresholds (core.threshold_optimizer)

Run:
    python3 run_grading_workflow.py
    python3 run_grading_workflow.py --pick-history output/pick_history.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from core import historical_grader
from core.calibration import print_summary
from core import threshold_optimizer
from core import probability_calibrator


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Grade, calibrate, and re-threshold picks in one pass.")
    parser.add_argument("--pick-history", default=historical_grader.DEFAULT_PICK_HISTORY_PATH)
    parser.add_argument("--reject-log", default=historical_grader.DEFAULT_REJECT_LOG_PATH)
    parser.add_argument("--threshold-output", default=threshold_optimizer.DEFAULT_OUTPUT_PATH)
    parser.add_argument("--walk-forward-output", default=threshold_optimizer.DEFAULT_WALK_FORWARD_OUTPUT_PATH)
    parser.add_argument("--calibration-output", default=probability_calibrator.DEFAULT_CALIBRATION_PATH)
    args = parser.parse_args()

    print("### Step 1-4: grading ungraded picks + calibration summary ###\n")
    summary = historical_grader.run(args.pick_history, args.reject_log)
    print_summary(summary)

    print("\n### Step 5: searching graded history for best tier thresholds ###\n")
    payload = threshold_optimizer.optimize(args.pick_history)
    Path(args.threshold_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.threshold_output, "w") as f:
        json.dump(payload, f, indent=2)
    threshold_optimizer.print_report(payload)
    print(f"(full detail written to {args.threshold_output})")

    # Fix #3 wiring: the previous version of this file called optimize()
    # directly, never threshold_optimizer.main() -- so walk_forward_validate()
    # (added in core/threshold_optimizer.py) never actually ran in the
    # scheduled job even though it existed in the file. Calling it
    # explicitly here, in the one place that's actually on the cron path.
    print("\n### Step 6: walk-forward out-of-sample validation ###\n")
    wf_payload = threshold_optimizer.walk_forward_validate(args.pick_history)
    Path(args.walk_forward_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.walk_forward_output, "w") as f:
        json.dump(wf_payload, f, indent=2)
    threshold_optimizer.print_walk_forward_report(wf_payload)
    print(f"(full detail written to {args.walk_forward_output})")
    print(
        "\nOnly deploy tiers marked 'validated' above to "
        "decision_gatekeeper.SPORT_TIER_THRESHOLDS -- 'failed_walk_forward' means "
        "the in-sample recommendation from step 5 did not hold up on picks it "
        "wasn't fit to."
    )

    # Fix #2 wiring: core/probability_calibrator.py existed and was called
    # at pick time (run_pipeline.py's calibrate_probability()), but nothing
    # ever called refit_and_save() to actually produce/update
    # output/probability_calibration.json -- so calibrate_probability()
    # was silently falling back to identity (raw model_prob unchanged) on
    # every single pick, forever, because the calibration file never
    # existed. This is the fix: refit after every grading pass, same
    # cadence as the threshold search above, using the same freshly graded
    # history.
    print("\n### Step 7: refitting probability calibration curves ###\n")
    calib_summary = probability_calibrator.refit_and_save(
        pick_history_path=args.pick_history, output_path=args.calibration_output,
    )
    for key, info in sorted(calib_summary.items()):
        status = "fitted" if info["fitted"] else "IDENTITY (insufficient graded data)"
        print(f"  {key:20s} n={info['n']:4d}  {status}")
    print(f"(calibration curves written to {args.calibration_output})")


if __name__ == "__main__":
    main()
