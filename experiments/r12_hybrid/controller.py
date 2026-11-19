"""Contact-armed normal-impedance reduction plus partial gravity support.

Unlike R8 contact_v4/load shaping and R9/R10 additive force candidates, this
controller selectively removes the world-vertical component of the motor
position-loop torque while retaining its orthogonal/tangential component and
all damping. A bumpless C2 blend simultaneously changes modeled payload support.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import r1_torque
from sim import config as C,trajectory as T

@dataclass(frozen=True)
class Parameters:
    contact_threshold:float=.05
    placement_start:float=3.780
    blend_duration:float=.020
    release_return:float=.040
    robot_support_fraction:float=.45
    normal_position_scale:float=.20


def smoothstep(s):
    s=float(np.clip(s,0,1));return 10*s**3-15*s**4+6*s**5

def scale_vertical(torque,jz,scale):
    jz=np.asarray(jz,float);torque=np.asarray(torque,float);projection=jz*(float(jz@torque)/(float(jz@jz)+1e-12))
    return torque+(float(scale)-1.)*projection

class Controller(r1_torque.Controller):
    parameters=Parameters()
    def reset(self):
        super().reset();p=self.parameters
        if not(0<=p.contact_threshold<.5 and 3.7<=p.placement_start<T.T_RELEASE and p.blend_duration>0 and p.release_return>0 and 0<p.robot_support_fraction<1 and 0<p.normal_position_scale<=1):raise ValueError('invalid hybrid parameters')
        self.hybrid_active=False;self.hybrid_start=None;self.hybrid_alpha=0.;self.commanded_load=0.;self.normal_position_scale=1.;self.minimum_normal_position_scale_seen=1.;self.max_alpha=0.;self.pre_release_load=None;self.release_load=None
    def _alpha(self,t):
        p=self.parameters
        if not self.hybrid_active and p.placement_start<=t<T.T_RELEASE:
            valid=self.shared_estimator.lambda_support_valid;support=float(np.clip(1-self.shared_estimator.lambda_support,0,1)) if valid else 0.
            if support>=p.contact_threshold:self.hybrid_active=True;self.hybrid_start=float(t)
        if not self.hybrid_active:return 0.
        if t<T.T_RELEASE:return smoothstep((t-self.hybrid_start)/p.blend_duration)
        return 1-smoothstep((t-T.T_RELEASE)/p.release_return)
    def compute(self,t,th,thd,tau_s,grip,q_d,qd_d,qdd_d):
        q_ref,qd_ref,qdd_ref=r1_torque.Controller._reference(self,t)
        base=self._estimate_load(t,th,thd,tau_s,grip)
        if self.UNLOAD_START<t<T.T_RELEASE:
            s=np.clip((t-self.UNLOAD_START)/(T.T_RELEASE-self.UNLOAD_START),0,1);base=min(base,1-self.UNLOAD_AMOUNT*smoothstep(s))
        alpha=self._alpha(t);p=self.parameters
        load=(1-alpha)*base+alpha*p.robot_support_fraction
        if t>=T.T_RELEASE:load=alpha*p.robot_support_fraction
        self.hybrid_alpha=float(alpha);self.max_alpha=max(self.max_alpha,float(alpha));self.commanded_load=float(load);self.normal_position_scale=float(1-alpha*(1-p.normal_position_scale))
        self.minimum_normal_position_scale_seen=min(self.minimum_normal_position_scale_seen,self.normal_position_scale)
        if T.T_RELEASE-.0015<=t<T.T_RELEASE:self.pre_release_load=float(load)
        if t>=T.T_RELEASE and self.release_load is None:self.release_load=float(load)
        tau_empty=self.nom.inverse_dynamics(q_ref,qd_ref,qdd_ref);tau_full=self.nom_box.inverse_dynamics(q_ref,qd_ref,qdd_ref);tau_link=tau_empty+load*(tau_full-tau_empty)
        theta=q_ref+self.inv_k*tau_link
        self.v_motor += .2*(thd-self.v_motor)
        if self.tau_link_prev is None:theta_dot,theta_ddot=qd_ref.copy(),qdd_ref.copy()
        else:
            raw=(tau_link-self.tau_link_prev)/self.dt;old=self.tau_dot.copy();self.tau_dot += .04*(raw-self.tau_dot)
            theta_dot=np.clip(qd_ref+self.inv_k*self.tau_dot,qd_ref-.35,qd_ref+.35);theta_ddot=np.clip(qdd_ref+self.inv_k*(self.tau_dot-old)/self.dt,-25.,25.)
        self.tau_link_prev=tau_link.copy();friction=C.VISCOUS*theta_dot+C.COULOMB*np.tanh(theta_dot/C.FRICTION_EPS);ff=self.nom.B*theta_ddot+tau_link+friction
        position=self.nom.B*(self.wn**2*(theta-th));damping=self.nom.B*(2*self.wn*(theta_dot-self.v_motor));jz=self.nom.jacobian(q_ref)[2]
        position=scale_vertical(position,jz,self.normal_position_scale)
        return ff+position+damping+self.k_tau*(tau_s-tau_link)
