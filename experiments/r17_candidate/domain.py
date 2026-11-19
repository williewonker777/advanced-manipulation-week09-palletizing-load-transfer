"""Known-payload routing. This guard never inspects state, phase or a test identifier."""
import math

import numpy as np

CALIBRATED_RANGE_KG = (2.86, 8.0)


def supported_payload(mass, com):
    m = float(mass)
    center = np.asarray(com, dtype=float)
    if not math.isfinite(m) or m < 0 or center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("known finite nonnegative payload and three-axis COM required")
    return CALIBRATED_RANGE_KG[0] <= m <= CALIBRATED_RANGE_KG[1]


def check_calibration_ranges(height, early):
    """Do not simply extend fitted polynomials by changing their declared support."""
    for source in (height, early.get('dynamics') if isinstance(early, dict) else None):
        if not isinstance(source, dict):
            raise ValueError("explicit height and early calibrated models required")
        limits = np.asarray(source.get('range', []), dtype=float)
        if limits.shape != (2,) or not np.array_equal(limits, CALIBRATED_RANGE_KG):
            raise ValueError("calibration does not cover the frozen payload contract")
