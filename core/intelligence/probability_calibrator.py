"""
Deprecated location. The real implementation lives at
core/probability_calibrator.py (that's the only path anything imports --
see run_pipeline.py and run_grading_workflow.py). This used to be a
byte-identical hand-copied duplicate of that file with no import pointing
at it, which is pure drift risk: nothing enforced the two staying in sync.
Re-exporting instead of deleting so any existing `from
core.intelligence.probability_calibrator import ...` doesn't break.
"""
from core.probability_calibrator import *  # noqa: F401,F403
from core.probability_calibrator import (  # noqa: F401
    calibrate_probability,
    refit_and_save,
    load_calibration_curves,
    save_calibration_curves,
    DEFAULT_CALIBRATION_PATH,
    MIN_SAMPLE_SIZE,
)
