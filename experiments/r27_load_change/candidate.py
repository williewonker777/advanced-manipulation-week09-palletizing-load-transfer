"""Optional C3 with a normal-only calibrated single-stream planar change observer."""
import json
from pathlib import Path

from experiments.r26_yaw_acquisition.candidate import Detector as Baseline
from experiments.r27_load_change.event import Event, sensor


class Detector(Baseline):
    def reset(self):
        super().reset()
        model = json.loads(Path(__file__).with_name('model.json').read_bytes())
        if model['schema'] != 'r27-normal-only-direction-change.v1':
            raise ValueError('wrong directional model')
        self.direction_change = Event(model['bounds_N'])

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, native = super().update(t, th, thd, tau_m, tau_s, grip)
        return residual, self.direction_change.feed(sensor(self, grip), native)
