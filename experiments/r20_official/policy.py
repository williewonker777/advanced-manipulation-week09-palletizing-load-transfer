"""Official-evaluator adapter. Never installed as a default course module."""
import json
from functools import lru_cache
from pathlib import Path

import my_detector
from experiments.r16_payload_contract.controller import attach
from experiments.r17_candidate.domain import supported_payload
from experiments.r17_candidate.policy import Controller as Staged
from experiments.r19_impulse_end.detector import Detector as Impulse

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def calibrations():
    # Created by the authenticated assembly; I/O outside the control loop.
    return tuple(json.loads((ROOT / name).read_bytes()) for name in ('height.json', 'early.json'))


class Controller(Staged):
    def __init__(self):
        super().__init__(*calibrations())


class Detector:
    def reset(self):
        registered = isinstance(self.ctrl, Controller)
        supported = supported_payload(self.box_mass, self.box_com)
        if registered and (self.ctrl._active is None or self.ctrl.loaded_policy != supported):
            raise ValueError("initialized matching control policy required")
        if registered and supported:
            detector = Impulse()
            detector.model_path = ROOT / 'results/r17/release_observation_model.json'
        else:
            detector = my_detector.Detector()
        attach(self, detector)
        detector.ctrl = self.ctrl._active if registered else self.ctrl
        detector.reset()
        self._active = detector
        self.update = detector.update
