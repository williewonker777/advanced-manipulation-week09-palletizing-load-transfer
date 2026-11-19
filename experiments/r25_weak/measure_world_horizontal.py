"""Measure preregistered signed world-horizontal evidence on disclosed twins."""
from collections import deque
from pathlib import Path
import argparse
import json

import numpy as np

from sim import config as C
from experiments.r15_delayed_wrench.observer import LAG, NormalForce
from experiments.r24_composed.detector import Detector as R24
from experiments.r24_holdout import evaluator as lab
from experiments.r25_weak.world_horizontal import unexplained_horizontal


class Probe(R24):
    def reset(self):
        super().reset()
        self.inverse25 = NormalForce(self.nom, self.nom_box, dt=C.DT,
                                     viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)
        self.support25 = deque(maxlen=LAG + 1)
        self.samples25 = []

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, alarm = super().update(t, th, thd, tau_m, tau_s, grip)
        allowed = bool(self._shared.last_table_allowed)
        table = np.asarray(self._shared.last_table_wrench[:3], float).copy()
        self.support25.append(np.r_[table, float(allowed)])
        sample = self.inverse25.update(t, th, tau_s, grip)
        if sample is None or len(self.support25) != self.support25.maxlen:
            self.samples25.append([t, np.nan, np.nan, np.nan, np.nan, allowed, grip,
                                   *table])
        else:
            raw = np.asarray(sample.force_n[:2], float)
            delayed = self.support25[0]
            corrected = unexplained_horizontal(raw, delayed[:3], bool(delayed[3]),
                                                self._shared.friction_mu)
            self.samples25.append([t, raw[0], raw[1], corrected[0], corrected[1],
                                   allowed, grip, *table])
        return residual, alarm


def summarize(values):
    values = np.asarray(values, float)
    finite = np.isfinite(values[:, 3:5]).all(axis=1) & values[:, 6].astype(bool)
    norms = np.linalg.norm(values[finite, 3:5], axis=1)
    return {"eligible_samples": int(finite.sum()),
            "corrected_norm_max_n": float(norms.max(initial=0.0)),
            "corrected_norm_p999_n": float(np.quantile(norms, .999)) if len(norms) else 0.0}


def main(args):
    plan = json.loads(args.plan.read_text())
    case = next(row for row in plan["single"] if row["id"] == args.case)
    rows = {}
    arrays = {}
    twins = (("none", None),) if args.none_only else (("none", None), ("force", case["impact"]))
    for twin, impact in twins:
        probe = Probe()
        _, log = lab.run_episode(case, plan["variants"][args.mechanics], lambda: probe,
                                 (impact, None))
        data = np.asarray(probe.samples25, float)
        arrays[twin] = data
        rows[twin] = summarize(data)
        if impact is not None:
            start = lab.certified.actual_start(log["t"], impact)[0]
            event = data[start:start + plan["event_window_ms"]]
            rows[twin]["event_corrected_norm_max_n"] = float(
                np.nanmax(np.linalg.norm(event[:, 3:5], axis=1)))
            at = start + int(np.nanargmax(np.linalg.norm(event[:, 3:5], axis=1)))
            rows[twin]["event_peak_tick"] = at
            rows[twin]["event_peak_signed_n"] = data[at, 3:5].tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.raw.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    args.out.write_text(json.dumps({"case": args.case, "mechanics": args.mechanics,
                                    "fields": ["t", "raw_x", "raw_y", "corrected_x",
                                               "corrected_y", "support_allowed", "grip",
                                               "table_x", "table_y", "table_z"],
                                    "twins": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case", default="s24_transfer_offaxis_mid")
    parser.add_argument("--mechanics", default="stock")
    parser.add_argument("--none-only", action="store_true")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
