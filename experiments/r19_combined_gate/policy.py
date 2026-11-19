"""Integrate immutable R19 V2 only inside R17's declared payload domain."""
from pathlib import Path

import my_detector
from experiments.r16_payload_contract.controller import attach
from experiments.r17_candidate.domain import supported_payload
from experiments.r17_candidate.policy import Controller
from experiments.r19_event_state.physical_detector import Detector as PhysicalEvent


class Detector:
    model_path = Path(__file__).resolve().parents[2] / "results/r17/release_observation_model.json"

    def reset(self):
        in_range = supported_payload(self.box_mass, self.box_com)
        registered = isinstance(self.ctrl, Controller)
        if registered and (self.ctrl._active is None or self.ctrl.loaded_policy != in_range):
            raise ValueError("consistent initialized controller and known payload required")
        if registered and in_range:
            impl = PhysicalEvent()
            impl.model_path = self.model_path
        else:
            impl = my_detector.Detector()
        attach(self, impl)
        impl.ctrl = self.ctrl._active if registered else self.ctrl
        impl.reset()
        self._active = impl
        self.update = impl.update
