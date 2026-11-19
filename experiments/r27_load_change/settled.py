"""R26 plus an optional two-observer, settled-change notification on unreported loading."""
import json
from pathlib import Path

from experiments.r26_yaw_acquisition.candidate import Detector as Baseline
from experiments.r27_load_change.projection import Projection
from experiments.r27_load_change.settled_event import Settled, channels


class Detector(Baseline):
    def reset(self):
        super().reset()
        self.r27_projection=Projection(self.box_mass,self.box_com)
        model=json.loads(Path(__file__).with_name('settled_model.json').read_bytes())
        if model['schema']!='r27-two-stream-settled-normal.v1':
            raise ValueError('wrong settled-change calibration')
        self.r27_settled=Settled(model)

    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,native=super().update(t,th,thd,tau_m,tau_s,grip)
        return residual,self.r27_settled.feed(channels(self,grip),native)
