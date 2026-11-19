"""Opt-in physically measured departures from local two-wrench normal expectations."""
import hashlib,json
from pathlib import Path
from experiments.r26_yaw_acquisition.candidate import Detector as C3
from experiments.r29_force_tube.stream import Streams
from experiments.r29_force_tube.tube import Notice,Region,Windows,loads
HERE=Path(__file__).resolve().parent


class Detector(C3):
    def reset(self):
        super().reset()
        model=json.loads((HERE/'model.json').read_bytes())
        if model.get('schema')!='r29-calibrated-two-wrench-tube.v1' or model['protocol_sha256']!=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest():
            raise ValueError('force-tube model or registration differs')
        self.tube_regions={int(p):Region(s)for p,s in model['phases'].items()}
        self.tube_streams=Streams(self.box_mass,self.box_com)
        self.tube_weight=loads(self.box_mass,self.box_com)
        self.tube_window=Windows();self.tube_event=Notice()

    def update(self,t,th,thd,motor,jts,grip):
        residual,native=super().update(t,th,thd,motor,jts,grip)
        phase,pair=self.tube_streams.read(self,t,th,thd,jts,grip)
        vector=self.tube_window.feed(phase,pair,self.tube_weight)
        margin=None if vector is None else self.tube_regions[phase].margin(*vector)
        return residual,self.tube_event.feed(margin,native)
