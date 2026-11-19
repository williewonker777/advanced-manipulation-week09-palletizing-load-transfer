"""C2 phase-local vertical-velocity intervention for causal identification."""
from __future__ import annotations
import numpy as np
import r1_servo,r1_torque
from sim import trajectory as T

START=3.720
END=3.880
CROSS=3.845
DURATION=END-START
X_CROSS=(CROSS-START)/DURATION
MAX_JOINT_VELOCITY_OFFSET=.12
MAX_JOINT_ACCELERATION_OFFSET=8.0
DAMPING=1e-6


def bump_terms(t,cross_velocity):
    """Return vertical position/velocity/acceleration offsets in SI units.

    delta_z = H * 64*x^3*(1-x)^3*(x_cross-x), with H selected so
    d(delta_z)/dt at CROSS equals ``cross_velocity``. Position, velocity, and
    acceleration are zero at both phase endpoints.
    """
    if t<=START or t>=END or cross_velocity==0.: return 0.,0.,0.
    x=(float(t)-START)/DURATION; c=X_CROSS
    b=64*x**3*(1-x)**3
    bp=64*(3*x**2-12*x**3+15*x**4-6*x**5)
    bpp=64*(6*x-36*x**2+60*x**3-30*x**4)
    bc=64*c**3*(1-c)**3
    height=-float(cross_velocity)*DURATION/bc
    f=b*(c-x); fp=bp*(c-x)-b; fpp=bpp*(c-x)-2*bp
    return height*f,height*fp/DURATION,height*fpp/DURATION**2


class _ApproachIntervention:
    CROSS_VELOCITY=0.0
    def reset(self):
        super().reset(); self.intervention_scale=self._select_scale()
        self.intervention_cutoff=bool(self.intervention_scale<1.-1e-12)
        self.max_commanded_dz=0.;self.max_commanded_dvz=0.;self.max_joint_velocity_offset=0.;self.max_joint_acceleration_offset=0.
    def _base_reference(self,t,q_d=None,qd_d=None,qdd_d=None):
        if q_d is None:
            q_d,qd_d,qdd_d=T.reference(t)
        return (r1_servo.Controller._reference(self,t,q_d,qd_d,qdd_d) if self.MODE=='servo'
                else r1_torque.Controller._reference(self,t))
    def _joint_offsets(self,t,scale=1.):
        dz,dvz,daz=bump_terms(t,self.CROSS_VELOCITY*scale)
        q,_,_=self._base_reference(t,*T.reference(t))
        J=self.nom.jacobian(q); pinv=J.T@np.linalg.inv(J@J.T+DAMPING*np.eye(3))
        return pinv@np.array([0.,0.,dz]),pinv@np.array([0.,0.,dvz]),pinv@np.array([0.,0.,daz]),(dz,dvz,daz)
    def _select_scale(self):
        maxv=maxa=0.
        for t in np.linspace(START,END,161):
            _,dv,da,_=self._joint_offsets(float(t),1.)
            maxv=max(maxv,float(np.max(np.abs(dv))));maxa=max(maxa,float(np.max(np.abs(da))))
        return float(min(1.,MAX_JOINT_VELOCITY_OFFSET/(maxv+1e-12),MAX_JOINT_ACCELERATION_OFFSET/(maxa+1e-12)))
    def _reference(self,t,*args):
        q,qd,qdd=self._base_reference(t,*args)
        dq,dv,da,cart=self._joint_offsets(t,self.intervention_scale)
        self.max_commanded_dz=max(self.max_commanded_dz,abs(cart[0]));self.max_commanded_dvz=max(self.max_commanded_dvz,abs(cart[1]))
        self.max_joint_velocity_offset=max(self.max_joint_velocity_offset,float(np.max(np.abs(dv))))
        self.max_joint_acceleration_offset=max(self.max_joint_acceleration_offset,float(np.max(np.abs(da))))
        return q+dq,qd+dv,qdd+da


class Servo(_ApproachIntervention,r1_servo.Controller): pass
class Torque(_ApproachIntervention,r1_torque.Controller): pass
