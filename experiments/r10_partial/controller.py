"""Contact-armed compliant vertical load sharing without payload unloading.

R1's modeled payload command is retained.  Confirmed public support arms a
bounded downward TCP equilibrium wrench through the nominal Jacobian.  The
wrench follows a finite-jerk S-curve, is held continuously across weld opening,
and then returns smoothly to zero, avoiding a new commanded torque step when
the payload constraint disappears.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import r1_servo, r1_torque
from sim import config as C, trajectory as T


@dataclass(frozen=True)
class Parameters:
    support_fraction: float = .68
    contact_threshold: float = .01
    minimum_duration: float = .039
    velocity_ramp: float = .010
    maximum_fraction_rate: float = 25.0
    return_duration: float = .060


def s_curve(t, duration, ramp):
    """Unit progress/rate with raised-cosine velocity ramps and finite jerk."""
    t=float(np.clip(t,0.0,duration)); ramp=min(float(ramp),duration/2)
    area=duration-ramp
    peak=1.0/area
    if t<ramp:
        u=t/ramp
        progress=peak*(.5*t-ramp*np.sin(np.pi*u)/(2*np.pi))
        rate=.5*peak*(1-np.cos(np.pi*u))
    elif t<=duration-ramp:
        progress=peak*(.5*ramp+t-ramp)
        rate=peak
    else:
        tail=duration-t; u=tail/ramp
        remaining=peak*(.5*tail-ramp*np.sin(np.pi*u)/(2*np.pi))
        progress=1.0-remaining
        rate=.5*peak*(1-np.cos(np.pi*u))
    return float(np.clip(progress,0,1)),float(rate)


class _PartialSupport:
    parameters=Parameters()

    def reset(self):
        super().reset(); p=self.parameters
        if not (0<p.support_fraction<1 and 0<=p.contact_threshold<.25
                and .02<=p.minimum_duration<.1 and 0<p.velocity_ramp<=p.minimum_duration/2
                and p.maximum_fraction_rate>0 and p.return_duration>=2*p.velocity_ramp):
            raise ValueError('invalid partial-support parameters')
        self.plan_active=False; self.plan_rejected=False; self.plan_start=None
        self.plan_duration=0.; self.press_fraction=0.; self.press_rate=0.
        self.max_press_rate=0.; self.release_press_fraction=None
        self.release_press_rate=None

    def _reference_r1(self,t,q_d,qd_d,qdd_d):
        return (r1_servo.Controller._reference(self,t,q_d,qd_d,qdd_d) if self.MODE=='servo'
                else r1_torque.Controller._reference(self,t))

    def _press(self,t):
        p=self.parameters
        # The same support geometry exists at pickup. Arm only in the declared
        # placement approach phase; this is commanded-path state, not a copied
        # force label or simulator contact time.
        placement_phase = t >= T.T_RELEASE-.120
        if not self.plan_active and not self.plan_rejected and placement_phase and t<T.T_RELEASE:
            support=(float(np.clip(1-self.shared_estimator.lambda_support,0,1))
                     if self.shared_estimator.lambda_support_valid else 0.)
            if support>=p.contact_threshold:
                duration=T.T_RELEASE-t
                peak=p.support_fraction/(duration-p.velocity_ramp) if duration>p.velocity_ramp else np.inf
                if duration>=p.minimum_duration and peak<=p.maximum_fraction_rate:
                    self.plan_active=True; self.plan_start=float(t); self.plan_duration=float(duration)
                else:
                    self.plan_rejected=True
        if not self.plan_active:
            return 0.,0.
        if t<T.T_RELEASE:
            progress,rate=s_curve(t-self.plan_start,self.plan_duration,p.velocity_ramp)
            value,derivative=p.support_fraction*progress,p.support_fraction*rate
        else:
            progress,rate=s_curve(t-T.T_RELEASE,p.return_duration,p.velocity_ramp)
            value,derivative=p.support_fraction*(1-progress),-p.support_fraction*rate
            if self.release_press_fraction is None:
                self.release_press_fraction=float(value); self.release_press_rate=float(derivative)
        self.max_press_rate=max(self.max_press_rate,abs(float(derivative)))
        return float(value),float(derivative)

    def compute(self,t,th,thd,tau_s,grip,q_d,qd_d,qdd_d):
        q_ref,qd_ref,qdd_ref=self._reference_r1(t,q_d,qd_d,qdd_d)
        load=self._estimate_load(t,th,thd,tau_s,grip)
        if self.UNLOAD_START<t<T.T_RELEASE:
            s=np.clip((t-self.UNLOAD_START)/(T.T_RELEASE-self.UNLOAD_START),0,1)
            load=min(load,1-self.UNLOAD_AMOUNT*(10*s**3-15*s**4+6*s**5))
        fraction,rate=self._press(t); self.press_fraction,self.press_rate=fraction,rate
        tau_empty=self.nom.inverse_dynamics(q_ref,qd_ref,qdd_ref)
        tau_full=self.nom_box.inverse_dynamics(q_ref,qd_ref,qdd_ref)
        # Downward world force applied at the public nominal TCP. Existing
        # elastic-joint compensation turns this into a compliant equilibrium.
        press_tau=self.nom.jacobian(q_ref).T@np.array([0.,0.,-fraction*self.box_mass*9.81])
        tau_link=tau_empty+load*(tau_full-tau_empty)+press_tau
        theta=q_ref+self.inv_k*tau_link
        if self.tau_link_prev is None:
            theta_dot,theta_ddot=qd_ref.copy(),qdd_ref.copy()
        else:
            raw=(tau_link-self.tau_link_prev)/self.dt; old=self.tau_dot.copy()
            self.tau_dot += .04*(raw-self.tau_dot)
            theta_dot=np.clip(qd_ref+self.inv_k*self.tau_dot,qd_ref-.35,qd_ref+.35)
            theta_ddot=np.clip(qdd_ref+self.inv_k*(self.tau_dot-old)/self.dt,-25.,25.)
        self.tau_link_prev=tau_link.copy()
        friction=C.VISCOUS*theta_dot+C.COULOMB*np.tanh(theta_dot/C.FRICTION_EPS)
        ff=self.nom.B*theta_ddot+tau_link+friction
        if self.MODE=='servo': return theta,theta_dot,theta_dot,ff
        self.v_motor += .2*(thd-self.v_motor)
        pd=self.nom.B*(self.wn**2*(theta-th)+2*self.wn*(theta_dot-self.v_motor))
        return ff+pd+self.k_tau*(tau_s-tau_link)


class Servo(_PartialSupport,r1_servo.Controller): pass
class Torque(_PartialSupport,r1_torque.Controller): pass
