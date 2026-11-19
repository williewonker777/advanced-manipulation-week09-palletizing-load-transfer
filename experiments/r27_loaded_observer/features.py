"""Pure causal feature extraction shared by fitting and runtime tests."""
from __future__ import annotations

import numpy as np


def normal_scores(rows, *, grip_stability=64, onset_lag=8, sustain_lag=32):
    """Return eligible normal-only onset and long-displacement scores."""
    data = np.asarray(rows, float)
    if data.ndim != 2 or data.shape[1] != 7 or not np.isfinite(data).all():
        raise ValueError("expected finite observer rows [tick,effective,grip_run,native,fx,fy,fz]")
    ticks = data[:, 0].astype(int)
    forces = data[:, 4:7]
    onset, sustain = [], []
    for index in range(sustain_lag, len(data)):
        if ticks[index] - ticks[index - sustain_lag] != sustain_lag:
            continue
        if int(data[index, 2]) < grip_stability:
            continue
        onset.append(float(np.linalg.norm(forces[index] - forces[index - onset_lag])))
        sustain.append(float(np.linalg.norm(forces[index] - forces[index - sustain_lag])))
    return np.asarray(onset), np.asarray(sustain)
