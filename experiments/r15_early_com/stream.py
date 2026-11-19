"""Bounded causal prefix view of the frozen R14 public Stream."""
import numpy as np
from experiments.r14_online.stream import Stream as Full
from tools.r13_grasp_estimator import CENTRES

PREFIX_MASK=(CENTRES>=1840)&(CENTRES<=2440)
PREFIX_CENTRES=CENTRES[PREFIX_MASK]
DEADLINE_INDEX=int(PREFIX_CENTRES[-1]+23)

class Stream(Full):
 def finish_prefix(self):
  if self.count<=DEADLINE_INDEX:raise ValueError('prefix incomplete before 2.447s')
  slots=np.flatnonzero(PREFIX_MASK)
  if self._next_centre<=slots[-1] or self.latch is None or np.any(~self.grip[PREFIX_CENTRES]):raise ValueError('carried prefix centres incomplete')
  return {'mass':np.array(self.mass),'y':self._y[PREFIX_MASK].copy(),'jac':self._jac[PREFIX_MASK].copy(),'stiffness':self._stiffness[PREFIX_MASK].copy(),'prior_bottom_mm':self._prior_bottom_mm.copy(),'prior_com_mm':self._prior_com_mm.copy(),'vertical':self._vertical.copy(),'centres':PREFIX_CENTRES.copy(),'completed_at_s':DEADLINE_INDEX*.001}
