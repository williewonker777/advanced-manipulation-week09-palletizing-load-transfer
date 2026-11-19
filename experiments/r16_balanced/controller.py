"""Blend two independently actuated release-state directions with normal-only coefficients."""
import json
from pathlib import Path
import numpy as np
from experiments.r16_normal_phase.controller import Controller as Phase
from experiments.r16_release.controller import Controller as Relief, release_motion
from sim import trajectory as T


class Controller(Relief, Phase):
    gains = None
    def reset(self):
        super().reset()
        if self.gains is None:
            doc=json.loads((Path(__file__).parent/'coefficients.json').read_text())
            if doc.get('schema')!='r16-balanced-normal-fit.v1':raise ValueError('wrong pressure fit')
            self.relief_gain,self.phase_gain=map(float,(doc['relief'],doc['phase']))
        else:self.relief_gain,self.phase_gain=map(float,self.gains)
        if not (0.<=self.relief_gain<=1.+1e-12 and 0.<=self.phase_gain<=1.+1e-12):
            raise ValueError('unbounded fit')
        if self._correction is not None:
            self._correction=tuple(self.phase_gain*value for value in self._correction)

    def _reference(self,t):
        q,v,a=super()._reference(t)
        if self.relief_gain==1. or self.relief_start is None:return q,v,a
        shape=np.array(release_motion(t,self.relief_start,T.T_RELEASE,2.*self.parameters.release_return))
        if not np.any(shape):return q,v,a
        adjust=(self.relief_gain-1.)*shape
        axis=self.relief_axis
        return q+adjust[0]*axis,v+adjust[1]*axis,a+adjust[2]*axis
