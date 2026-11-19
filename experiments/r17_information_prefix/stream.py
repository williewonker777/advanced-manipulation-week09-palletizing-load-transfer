"""One-pass measured signatures and bounded carried prefix on the public 1 ms inputs."""
import numpy as np

from experiments.r14_online.stream import Stream
from experiments.r16_robust_prefix.robust import PRE_CENTRES, USED_CENTRES
from sim import config as C, trajectory as T
from sim.nominal import Nominal
from tools.r13_grasp_estimator import DT, HALF, WEIGHT


class InformationStream(Stream):
    def __init__(self, mass, com):
        super().__init__(mass, com)
        self._pre_model = Nominal()
        self._pre_slot = 0
        self._pre_stiff = np.zeros((len(PRE_CENTRES), 6))
        self._pre_resid = np.zeros_like(self._pre_stiff)
        self._pre_track = np.zeros_like(self._pre_stiff)
        refs = [T.reference(float(k)*DT) for k in PRE_CENTRES]
        self._taught = np.stack([ref[0] for ref in refs])
        velocity = np.stack([ref[1] for ref in refs])
        drag = np.tanh(velocity/C.FRICTION_EPS)
        self._velocity = velocity-velocity.mean(0)
        self._drag = drag-drag.mean(0)
        self._signature = None

    def observe(self, t, encoder, jts, grip):
        super().observe(t, encoder, jts, grip)
        i = self._pre_slot
        if i == len(PRE_CENTRES) or self.count-1 != PRE_CENTRES[i]+HALF+3:
            return
        c = int(PRE_CENTRES[i])
        ix = np.arange(c-HALF-1, c+HALF+2)
        # Only the measured support for this local difference stencil.
        enc = self.encoder[ix]; disp = self.disp[ix]
        q, qd, qdd = self._local_kinematics(enc-disp, HALF+1)
        dq, dvd, dad = self._local_kinematics(disp, HALF+1)
        obs = np.einsum('s,sj->j', WEIGHT, self.tau[ix[1:-1]])
        inv = self._pre_model.inverse_dynamics
        base = inv(q, qd, qdd)
        plus = inv(q-.05*dq, qd-.05*dvd, qdd-.05*dad)
        minus = inv(q+.05*dq, qd+.05*dvd, qdd+.05*dad)
        self._pre_stiff[i] = (plus-minus)/.10
        self._pre_resid[i] = obs-base
        self._pre_track[i] = np.einsum('s,sj->j',WEIGHT,enc[1:-1])-self._taught[i]
        self._pre_slot += 1
        if self._pre_slot == len(PRE_CENTRES):
            dc = np.broadcast_to(np.eye(6),(len(PRE_CENTRES),6,6))
            design=np.concatenate((self._pre_stiff[:,:,None],dc),axis=2).reshape(-1,7)
            scales=np.r_[.065,np.full(6,.15)]
            coef=np.linalg.solve(design.T@design+np.diag(1/scales**2),design.T@self._pre_resid.ravel())
            err=self._pre_track-self._pre_track.mean(0)
            dots=[np.sum(err*b)/(np.sum(b*b)+1e-12) for b in (self._velocity,self._drag)]
            self._signature=np.r_[coef[0],dots,np.sqrt(np.mean(err**2))]

    def information(self, last_centre):
        centres=USED_CENTRES[USED_CENTRES<=last_centre]
        n=len(centres)
        if (not n or centres[-1]!=last_centre or self.count<=last_centre+23
                or self._next_centre<n or self._signature is None or not np.all(self.grip[centres])):
            raise ValueError('causal centre or pregrasp history incomplete')
        return {'mass':np.array(self.mass),'y':self._y[:n].copy(),'jac':self._jac[:n].copy(),
                'stiffness':self._stiffness[:n].copy(),'prior_com_mm':self._prior_com_mm.copy(),
                'signature':self._signature.copy(),'centres':centres.copy(),
                'completed_at_s':(last_centre+23)*DT}


def feature(public_path, last_centre):
    with np.load(public_path,allow_pickle=False) as r:
        s=InformationStream(float(r['mass']),r['com'])
        count=last_centre+24
        rec=[r[k][:count] for k in ('t','encoder','jts','grip')]
        if any(len(v)!=count for v in rec):
            raise ValueError('public record ends before frozen decision')
        for t,q,jts,g in zip(*rec,strict=True):
            s.observe(t,q,jts,g)
    return s.information(last_centre)
