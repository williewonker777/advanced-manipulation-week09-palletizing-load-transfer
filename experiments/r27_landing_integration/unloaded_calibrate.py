"""Fit table-normal resolution from the preregistered mass-zero normal only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r26_delivery_pair.controller import Controller
from sim.runner import run_episode


class Probe(C1):
    latest = None
    def reset(self):
        super().reset(); self.normal = []; Probe.latest = self
    def update(self, t, th, thd, tau_m, tau_s, grip):
        result = super().update(t, th, thd, tau_m, tau_s, grip)
        self.normal.append(float(max(0.0, self._shared.last_table_wrench[2]))
                           if grip and self._shared.last_table_allowed else 0.0)
        return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():p.error('fresh --out required')
    metrics,log=run_episode(Controller(),box_mass=0.0,detector=Probe())
    maximum=float(np.max(Probe.latest.normal))
    model={"schema":"r27-unresolved-payload-resolution.v1","fit_episodes":1,
           "fit_mass_kg":0.0,"fit_max_table_normal_N":maximum,
           "sensor_reserve_N":0.5,"table_normal_resolution_N":maximum+0.5}
    a.out.write_text(json.dumps(model,indent=2,allow_nan=False)+'\n')
    print(json.dumps(model,indent=2))


if __name__=='__main__':main()
