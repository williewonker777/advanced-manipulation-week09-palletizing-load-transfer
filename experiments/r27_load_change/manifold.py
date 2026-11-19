"""Only add a settled notification if two measured forces also exclude the local normal set."""
import json
from pathlib import Path

from experiments.r26_yaw_acquisition.candidate import Detector as Base
from experiments.r27_load_change.manifold_event import Coordinates, FeasibleNormals
from experiments.r27_load_change.projection import Projection
from experiments.r27_load_change.settled_event import Settled, channels


class Detector(Base):
    def reset(self):
        super().reset()
        folder=Path(__file__).resolve().parent
        self.r27_projection=Projection(self.box_mass,self.box_com)
        self.r27_gate=Settled(json.loads((folder/'settled_model.json').read_bytes()))
        self.r27_coords=Coordinates()
        self.r27_normals=FeasibleNormals(json.loads((folder/'manifold_model.json').read_bytes()))
        self.r27_certificate=False

    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,native=super().update(t,th,thd,tau_m,tau_s,grip)
        pair=channels(self,grip)
        coords=self.r27_coords.feed(th,pair is not None)
        extra_or_native=self.r27_gate.feed(pair,native)
        if self.r27_gate.notified:
            # The exact two 40-tick mean world forces on which the notified plateau is based.
            levels=[window.b/len(window.present) for window in self.r27_gate.windows.windows]
            self.r27_certificate=self.r27_normals.unexpected(coords,levels,self.r27_gate.axis)
        elif not extra_or_native or native:
            self.r27_certificate=False
        return residual,bool(native or (self.r27_certificate and extra_or_native))
