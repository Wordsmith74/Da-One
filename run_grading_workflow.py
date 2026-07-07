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


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Grade, calibrate, and re-threshold picks in one pass.")
    parser.add_argument("--pick-history", default=historical_grader.DEFAULT_PICK_HISTORY_PATH)
    parser.add_argument("--reject-log", default=historical_grader.DEFAULT_REJECT_LOG_PATH)
    parser.add_argument("--threshold-output", default=threshold_optimizer.DEFAULT_OUTPUT_PATH)
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


if __name__ == "__main__":
    main()
