"""Normal-only bounds for a direction-neutral horizontal transient channel."""
import json
from pathlib import Path

import numpy as np

from experiments.r20_horizontal.trace_detector import Detector
from sim.runner import run_episode
from my_torque import Controller


MASSES = (0.0, 2.3, 5.7, 8.0)
COMS = ((0.0, 0.0, 0.0), (0.02, -0.01, 0.03),
        (0.025, -0.02, 0.03),
        (-0.025, 0.02, 0.03), (0.015, 0.015, -0.01))


def main(path):
    rows = []
    for mass in MASSES:
        for com in COMS:
            detector = Detector()
            result, _ = run_episode(Controller(), box_mass=mass, box_com=com,
                                    detector=detector, collision=None)
            innovation = np.vstack([x["innovation"] for x in detector.trace])
            residual = np.vstack([x["residual"] for x in detector.trace])
            times = np.array([x["t"] for x in detector.trace])
            grip = np.array([x["grip"] for x in detector.trace])
            # Direction-neutral horizontal magnitudes, from the already causal
            # R10 observer.  Windows remain reported separately rather than
            # being hidden by a phase-specific hand-tuned threshold.
            hi = np.linalg.norm(innovation[:, :2], axis=1)
            hr = np.linalg.norm(residual[:, :2], axis=1)
            impulse = {}
            for width in (8, 16, 24, 32):
                cs = np.vstack((np.zeros((1, 2)), np.cumsum(innovation[:, :2], axis=0)))
                summed = cs[width:] - cs[:-width]
                values = np.r_[np.zeros(width - 1), np.linalg.norm(summed, axis=1)]
                impulse[width] = values
            windows = {"all_grip": grip,
                       "early": (times >= 1.35) & (times < 1.75),
                       "carry": (times >= 1.75) & (times < 3.65),
                       "release": (times >= 3.65) & (times < 4.15)}
            rows.append({
                "mass": mass, "com": list(com),
                "false_alarms": result["false_alarms"],
                "max": {name: {"innovation_h": float(hi[sel].max()),
                                "residual_h": float(hr[sel].max()),
                                "innovation_xy": np.abs(innovation[sel, :2]).max(axis=0).tolist(),
                                "residual_xy": np.abs(residual[sel, :2]).max(axis=0).tolist(),
                                **{f"impulse_{width}": float(values[sel].max())
                                   for width, values in impulse.items()}}
                        for name, sel in windows.items()},
            })
    payload = {"schema": "R20_HORIZONTAL_NORMAL.v1", "rows": rows}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True)
    main(p.parse_args().out)
