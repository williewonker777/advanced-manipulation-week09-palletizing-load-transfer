"""Support ownership with a calibrated unresolved-payload physical branch."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.r15_delayed_wrench.observer import NormalForce
from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r27_landing_integration.candidate import GripForceEpisode
from sim import config as C


FORCE_MODEL = Path(__file__).with_name("model.json")
RESOLUTION_MODEL = Path(__file__).with_name("unloaded_resolution.json")
GRAVITY_M_S2 = 9.81


class Detector(C1):
    def reset(self):
        super().reset()
        force = json.loads(FORCE_MODEL.read_bytes())
        resolution = json.loads(RESOLUTION_MODEL.read_bytes())
        if (force.get("schema") != "r27-gripped-world-force-bound.v1"
                or resolution.get("schema") != "r27-unresolved-payload-resolution.v1"):
            raise ValueError("invalid frozen R27 models")
        self.normal_resolution = float(resolution["table_normal_resolution_N"])
        if not np.isfinite(self.normal_resolution) or self.normal_resolution <= 0:
            raise ValueError("positive finite normal resolution required")
        self.grip_force = NormalForce(self.nom,self.nom_box,dt=C.DT,
            viscous_spring=C.D,jts_hz=C.JTS_CUTOFF)
        self.grip_episode = GripForceEpisode(force["horizontal_bound_N"],
            force["release_fraction"],force["onset_consecutive_samples"],
            self._shared.release_persistence)
        self.grip_force_value=None;self.grip_force_extra=False
        self.load_ownership=False;self.load_branch="unknown"

    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,incumbent=super().update(t,th,thd,tau_m,tau_s,grip)
        sample=self.grip_force.update(t,th,tau_s,bool(grip))
        self.grip_force_value=None if sample is None else sample.force_n.copy()
        payload_weight=float(self.box_mass)*GRAVITY_M_S2
        support=getattr(self.ctrl,"shared_estimator",None)
        if not np.isfinite(payload_weight) or payload_weight < 0:
            owned=False;self.load_branch="unknown"
        elif payload_weight <= self.normal_resolution:
            table_normal=(float(max(0.0,self._shared.last_table_wrench[2]))
                          if self._shared.last_table_allowed else 0.0)
            owned=np.isfinite(table_normal) and table_normal <= self.normal_resolution
            self.load_branch="unresolved"
        else:
            owned=bool(support is not None and support.lambda_support_valid
                       and np.isfinite(support.lambda_support)
                       and support.lambda_support >= 0.5)
            self.load_branch="resolved" if owned else "unknown"
        self.load_ownership=bool(grip) and bool(owned)
        self.grip_force_extra=self.grip_episode.feed(
            self.grip_force_value,self.load_ownership,incumbent)
        return residual,bool(incumbent or self.grip_force_extra)
