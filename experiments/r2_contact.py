"""Second-round contact laws, implemented independently from cited principles.

Only public nominal models and measured controller inputs are used. Spatial
vectors here are [linear, angular], unlike the angular-first pRNE module.
These are experimental elastic adaptations, not transferred rigid theorems.
"""
import importlib
import numpy as np
import mujoco
from sim import config as C, trajectory as T

SMALL = 0.0006894392593624343
STROKE = 0.0026785554844834247
START, CONTACT, EARLY = 3.6, 3.8465, 3.809
VMAX = STROKE * 1.875 / (CONTACT - START)
AMAX = STROKE * (10.0 / np.sqrt(3.0)) / (CONTACT - START)**2


def skew(x):
    a, b, c = x
    return np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]])


def so3_exp(phi):
    angle = np.linalg.norm(phi)
    A = skew(phi)
    if angle < 1e-7:
        return np.eye(3) + A + .5*A@A
    return np.eye(3) + np.sin(angle)/angle*A + (1-np.cos(angle))/angle**2*(A@A)


def so3_log(R):
    angle = np.arccos(np.clip((np.trace(R)-1)*.5, -1., 1.))
    a = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
    if angle < 1e-7:
        return .5*a
    if np.pi-angle < 1e-5:
        vals, vecs = np.linalg.eigh((R+np.eye(3))*.5)
        axis = vecs[:, np.argmax(vals)]
        if axis@a < 0:
            axis = -axis
        return angle*axis
    return angle/(2*np.sin(angle))*a


def se3_log(R, p):
    phi = so3_log(R)
    a = np.linalg.norm(phi)
    A = skew(phi)
    coefficient = 1/12 if a < 1e-6 else (1-.5*a/np.tan(.5*a))/a**2
    rho = (np.eye(3)-.5*A+coefficient*A@A)@p
    return np.r_[rho, phi]


def adjoint(R, p):
    return np.block([[R, skew(p)@R], [np.zeros((3, 3)), R]])


def dexp_minus(x):
    """Right-trivialized exponential differential, convergent matrix series."""
    v, w = x[:3], x[3:]
    ad = np.block([[skew(w), skew(v)], [np.zeros((3, 3)), skew(w)]])
    out, term = np.eye(6), np.eye(6)
    for k in range(1, 20):
        term = term@(-ad)/(k+1)
        out += term
        if np.max(np.abs(term)) < 1e-15:
            break
    return out


def normal_lie_error(R, p, Rref, pref, velocity, reference_velocity):
    """Scalar log error in the fixed *desired* frame and its power dual.

    E=T_des^-1 T_actual, xi=log(E), e=-n_des.T xi.  A body
    perturbation of the actual pose has dxi=Jr(xi)^-1 delta_actual,
    hence -de/d(delta)=Jr(xi)^-T n_des.  Using this reversed group
    error avoids dropping an Adjoint from the original inverse convention.
    The moving desired-frame normal derivative is included in e_dot.
    """
    Re, pe = Rref.T@R, Rref.T@(p-pref)
    xi = se3_log(Re, pe)
    Q = np.block([[R.T, np.zeros((3, 3))], [np.zeros((3, 3)), R.T]])
    Qref = np.block([[Rref.T, np.zeros((3, 3))], [np.zeros((3, 3)), Rref.T]])
    V, Vr = Q@velocity, Qref@reference_velocity
    normal = np.r_[Rref.T@np.array([0., 0., 1.]), np.zeros(3)]
    normal_d = np.r_[-np.cross(Vr[3:], normal[:3]), np.zeros(3)]
    differential = dexp_minus(xi)
    xi_d = np.linalg.solve(differential, V-np.linalg.solve(adjoint(Re, pe), Vr))
    error = -float(normal@xi)
    error_d = -float(normal@xi_d+normal_d@xi)
    world_dual = Q.T@np.linalg.solve(differential.T, normal)
    return error, error_d, world_dual


def profile(s, name='quintic'):
    """Return position, ds derivative and d2s derivative on [0,1]."""
    s = float(np.clip(s, 0., 1.))
    if name in ('seventh_joint', 'SO3_R3_seventh'):
        return (35*s**4-84*s**5+70*s**6-20*s**7,
                140*s**3-420*s**4+420*s**5-140*s**6,
                420*s**2-1680*s**3+2100*s**4-840*s**5)
    if name == 'S_curve_joint':
        # Four constant-jerk pieces (+32,-32,-32,+32): bounded jerk,
        # C2 position, not a claim of continuous jerk like the seventh order.
        h, hd, hdd = 0., 0., 0.
        for a, coefficient in [(0., 1.), (.25, -2.), (.75, 2.), (1., -1.)]:
            z = max(0., s-a)
            h += coefficient*z**3*32/6
            hd += coefficient*z**2*16
            hdd += coefficient*z*32
        return h, hd, hdd
    return (10*s**3-15*s**4+6*s**5,
            30*s**2-60*s**3+30*s**4,
            60*s-180*s**2+120*s**3)


def lift_command(ctrl, qref, vref, aref, tau_link, th, thd, tau_s):
    """Exactly the frozen R1 elastic motor tracker, with an explicit link law."""
    theta = qref + ctrl.inv_k*tau_link
    if ctrl.tau_link_prev is None:
        theta_d, theta_dd = vref.copy(), aref.copy()
    else:
        raw = (tau_link-ctrl.tau_link_prev)/ctrl.dt
        old = ctrl.tau_dot.copy()
        ctrl.tau_dot += .04*(raw-ctrl.tau_dot)
        theta_d = vref+ctrl.inv_k*ctrl.tau_dot
        theta_dd = aref+ctrl.inv_k*(ctrl.tau_dot-old)/ctrl.dt
        theta_d = np.clip(theta_d, vref-.35, vref+.35)
        theta_dd = np.clip(theta_dd, -25., 25.)
    ctrl.tau_link_prev = tau_link.copy()
    friction = C.VISCOUS*theta_d+C.COULOMB*np.tanh(theta_d/C.FRICTION_EPS)
    out = ctrl.nom.B*theta_dd+tau_link+friction
    if ctrl.MODE == 'servo':
        return np.array([theta, theta_d, theta_d, out])
    ctrl.v_motor += .2*(thd-ctrl.v_motor)
    pd = ctrl.nom.B*(ctrl.wn**2*(theta-th)+2*ctrl.wn*(theta_d-ctrl.v_motor))
    return out+pd+ctrl.k_tau*(tau_s-tau_link)


def make_controller(mode, params):
    base = importlib.import_module('r1_'+mode).Controller
    config = dict(params)

    class ContactController(base):
        R2_PARAMETERS = config
        R2_STABILITY_STATE = ('normal_x', 'normal_v', 'feedback_force')

        def reset(self):
            super().reset()
            self.normal_x, self.normal_v, self.feedback_force = 0., 0., 0.
            self.family = config['family']
            self.geometry = config.get('geometry', 'euclidean')
            self.fraction = float(config.get('fraction', .5))
            self.feedback_gain = float(config.get('feedback_gain', 0.))
            self.error_scale = float(config.get('error_scale', SMALL))
            q = T.reference(T.T_RELEASE)[0]
            J, p, R = self.shared_estimator._point_jacobian(q)
            self.normal_direction = np.linalg.solve(J, np.array([0., 0., 1., 0., 0., 0.]))
            self.contact_J, self.contact_point, self.contact_rotation = J.copy(), p.copy(), R.copy()
            M = self.nom_box.mass_matrix(q)
            self.normal_mass = float(self.normal_direction@M@self.normal_direction)
            self.normal_k = max(.5*self.box_mass*9.81/self.error_scale, 1e-9)
            self.normal_d = 2*np.sqrt(self.normal_mass*self.normal_k)
            # Critically damped exact ZOH. No forward-Euler contact oscillator.
            w = np.sqrt(self.normal_k/self.normal_mass)
            Ac = np.array([[0., 1.], [-w*w, -2*w]])
            self.admit_A = np.exp(-w*self.dt)*(np.eye(2)+(Ac+w*np.eye(2))*self.dt)
            self.admit_B = np.linalg.solve(Ac, (self.admit_A-np.eye(2))@np.array([0., 1/self.normal_mass]))
            self.q0, self.qf = T.reference(START)[0], T.reference(T.T_RELEASE)[0]
            _, self.p0, self.R0 = self.shared_estimator._point_jacobian(self.q0)
            _, self.pf, self.Rf = self.shared_estimator._point_jacobian(self.qf)
            self.rotation_delta = so3_log(self.R0.T@self.Rf)

        def _r1_reference(self, t, q, v, a):
            if mode == 'servo':
                return super()._reference(t, q, v, a)
            return super()._reference(t)

        def _interpolation_reference(self, t, original):
            if not START <= t < T.T_RELEASE:
                return original
            variant = config.get('variant', 'seventh_joint')
            duration = EARLY-START
            s = (t+self.dt-START)/duration
            h, hd, hdd = profile(s, variant)
            if s >= 1.:
                hd, hdd = 0., 0.
            hd, hdd = hd/duration, hdd/duration**2
            dq = self.qf-self.q0
            q = self.q0+h*dq
            if variant != 'SO3_R3_seventh':
                return q, hd*dq, hdd*dq
            target_p = self.p0+h*(self.pf-self.p0)
            target_R = self.R0@so3_exp(h*self.rotation_delta)
            for _ in range(8):
                J, p, R = self.shared_estimator._point_jacobian(q)
                error = np.r_[target_p-p, R@so3_log(R.T@target_R)]
                if np.max(np.abs(error)) < 1e-11:
                    break
                q = q+np.linalg.solve(J, error)
            J = self.shared_estimator._point_jacobian(q)[0]
            twist = np.r_[hd*(self.pf-self.p0), self.R0@self.rotation_delta*hd]
            accel = np.r_[hdd*(self.pf-self.p0), self.R0@self.rotation_delta*hdd]
            v = np.linalg.solve(J, twist)
            eps = 1e-5
            Jp = self.shared_estimator._point_jacobian(q+eps*v)[0]
            Jm = self.shared_estimator._point_jacobian(q-eps*v)[0]
            a = np.linalg.solve(J, accel-(Jp-Jm)/(2*eps)@v)
            return q, v, a

        def _support_wrench(self, q, force):
            J, _, R = self.shared_estimator._point_jacobian(q)
            # Resultant at COM's horizontal projection. Respect the known
            # rectangular bottom footprint in its own frame (COP constraint).
            com_from_bottom = np.asarray(self.box_com)-np.array([0., 0., C.BOX_SIZE[2]*.5])
            cop = np.r_[np.linalg.solve(R[:2, :2], (R@com_from_bottom)[:2]), 0.]
            cop[:2] = np.clip(cop[:2], -np.asarray(C.BOX_SIZE[:2])*.5,
                             np.asarray(C.BOX_SIZE[:2])*.5)
            lever = R@cop
            f = np.array([0., 0., force])
            world = np.r_[f, np.cross(lever, f)]
            # Rotation-only coadjoint at the SAME bottom origin. Translation
            # from flange to bottom is already in _point_jacobian.
            Q = np.block([[R.T, np.zeros((3, 3))], [np.zeros((3, 3)), R.T]])
            body_J, body_wrench = Q@J, Q@world
            return body_J.T@body_wrench

        def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
            qr, vr, ar = self._r1_reference(t, q_d, qd_d, qdd_d)
            if self.family == 'C6':
                qr, vr, ar = self._interpolation_reference(t, (qr, vr, ar))
            load = self._estimate_load(t, th, thd, tau_s, grip)
            if self.UNLOAD_START < t < T.T_RELEASE:
                s = (t-self.UNLOAD_START)/(T.T_RELEASE-self.UNLOAD_START)
                load = min(load, 1-self.UNLOAD_AMOUNT*profile(s)[0])
            packet = self.shared_estimator
            empty = self.nom.inverse_dynamics(qr, vr, ar)
            full = self.nom_box.inverse_dynamics(qr, vr, ar)
            tau_link = empty+load*(full-empty)
            active = bool(grip and START < t < T.T_RELEASE and self.box_mass > 0)
            phase = profile((t-START)/(CONTACT-START))[0] if active else 0.
            desired_force = self.fraction*self.box_mass*9.81*phase
            if self.family == 'C1' and active:
                measured = float(packet.last_wrench[2])
                if not packet.detected:
                    self.feedback_force = float(np.clip(desired_force+
                        self.feedback_gain*(desired_force-measured), 0., self.box_mass*9.81))
                # Full payload minus planned table wrench, blended from R1.
                # F_des already contains phase: do NOT attenuate it twice.
                tau_link = (1-phase)*tau_link+phase*full-self._support_wrench(qr, self.feedback_force)
            elif self.family == 'C2':
                if mode == 'servo':
                    force = desired_force-float(packet.last_wrench[2]) if active else 0.
                    state = self.admit_A@np.array([self.normal_x, self.normal_v])+self.admit_B*force
                    vel = float(np.clip(state[1], self.normal_v-AMAX*self.dt,
                                        self.normal_v+AMAX*self.dt))
                    vel = float(np.clip(vel, -VMAX, VMAX))
                    # Exact unbounded ZOH propagation, then a bounded
                    # kinematic projection if rate/acceleration limits bind.
                    proposed = state[0]+self.dt*.5*(vel-state[1])
                    x = float(np.clip(proposed, 0., STROKE))
                    if packet.detected and active:
                        x, vel = self.normal_x, 0.
                    if x in (0., STROKE) and (x-self.normal_x)*vel <= 0.:
                        vel = 0.
                    acc = (vel-self.normal_v)/self.dt
                    # Compare the same pure world normal target in Euclidean
                    # and log/dexp coordinates; equal rotation makes the two
                    # mathematically identical, a useful negative ablation.
                    direction = self.normal_direction
                    if self.geometry == 'se3':
                        R = self.contact_rotation
                        lam = se3_log(np.eye(3), R.T@np.array([0., 0., -x]))
                        Q = np.block([[R.T, np.zeros((3, 3))], [np.zeros((3, 3)), R.T]])
                        body_normal = np.r_[R.T@np.array([0., 0., 1.]), np.zeros(3)]
                        direction = np.linalg.solve(Q@self.contact_J, dexp_minus(lam)@body_normal)
                    self.normal_x, self.normal_v = x, vel
                    qr, vr, ar = qr-direction*x, vr-direction*vel, ar-direction*acc
                    empty = self.nom.inverse_dynamics(qr, vr, ar)
                    full = self.nom_box.inverse_dynamics(qr, vr, ar)
                    tau_link = empty+load*(full-empty)
                elif active:
                    q, v = packet.q_prev, packet.v
                    J, p, R = packet._point_jacobian(q)
                    Jr, pr, Rr = packet._point_jacobian(qr)
                    if self.geometry == 'euclidean':
                        error, error_d = pr[2]-p[2], float((Jr@vr-J@v)[2])
                        torque_direction = J[2]
                    else:
                        error, error_d, world_dual = normal_lie_error(R, p, Rr, pr, J@v, Jr@vr)
                        torque_direction = J.T@world_dual
                    normal_force = phase*(self.normal_k*error+self.normal_d*error_d)-desired_force
                    normal_force = np.clip(normal_force, -self.box_mass*9.81, self.box_mass*9.81)
                    if not packet.detected:
                        self.feedback_force = float(normal_force)
                    tau_link = tau_link+torque_direction*self.feedback_force
            return lift_command(self, qr, vr, ar, tau_link, th, thd, tau_s)

    if config['family'] == 'C1' or (config['family'] == 'C2' and mode == 'torque'):
        ContactController.R2_STABILITY_STATE = ('feedback_force',)
    elif config['family'] == 'C2':
        ContactController.R2_STABILITY_STATE = ('normal_x', 'normal_v')
    else:
        ContactController.R2_STABILITY_STATE = ()
    ContactController.__name__ = 'R2'+config['family']+mode.title()
    return ContactController
