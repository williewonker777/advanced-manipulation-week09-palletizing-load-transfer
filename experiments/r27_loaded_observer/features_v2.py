"""Normal-only scores under observer-effective settled unsupported eligibility."""
from __future__ import annotations

import numpy as np


def normal_scores(rows, *, grip_stability=96, onset_lag=8, sustain_lag=32):
    data = np.asarray(rows, float)
    if data.ndim != 2 or data.shape[1] != 8 or not np.isfinite(data).all():
        raise ValueError("expected v2 rows [tick,effective,grip_run,native,table,fx,fy,fz]")
    ticks = data[:, 0].astype(int)
    forces = data[:, 5:8]
    onset, sustain = [], []
    for index in range(sustain_lag, len(data)):
        history = slice(index - sustain_lag, index + 1)
        if ticks[index] - ticks[index - sustain_lag] != sustain_lag:
            continue
        if int(data[index, 2]) < grip_stability or bool(np.any(data[history, 4])):
            continue
        onset.append(float(np.linalg.norm(forces[index] - forces[index - onset_lag])))
        sustain.append(float(np.linalg.norm(forces[index] - forces[index - sustain_lag])))
    return np.asarray(onset), np.asarray(sustain)
