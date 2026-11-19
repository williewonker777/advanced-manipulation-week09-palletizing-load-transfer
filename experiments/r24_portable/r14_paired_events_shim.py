"""Restricted exact false-event matcher used by the R22/R23 replay scorers."""
from __future__ import annotations

import numpy as np


FALSE_EVENT_TOLERANCE_S = 0.020


def match_false_events(candidate, reference, tolerance_s=FALSE_EVENT_TOLERANCE_S):
    """Exact source-compatible subset of tools.r14_paired_events."""
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if (candidate.ndim != 1 or reference.ndim != 1
            or not np.isfinite(candidate).all() or not np.isfinite(reference).all()):
        raise ValueError("finite one-dimensional false-event times required")
    unused = list(map(float, reference))
    pairs = []
    for event in map(float, candidate):
        choices = [(abs(event - other), index, other)
                   for index, other in enumerate(unused)
                   if abs(event - other) <= tolerance_s + 1e-12]
        if not choices:
            return {"passed": False, "pairs": pairs,
                    "unmatched_candidate": event,
                    "tolerance_ms": 1000.0 * tolerance_s}
        delta, index, other = min(choices)
        pairs.append({"candidate_s": event, "reference_s": other,
                      "delta_ms": 1000.0 * delta})
        unused.pop(index)
    return {"passed": True, "pairs": pairs, "unmatched_candidate": None,
            "tolerance_ms": 1000.0 * tolerance_s}
