"""Bounded normal-only reference correction for the preregistered C_normal arm.

The correction is a first-order, local tangent construction.  A constant joint
direction is computed at the taught release configuration from the full WORLD
6D bottom-centre Jacobian.  Consequently it preserves the other five twist
components only to first order.  The wrench is the delayed JTS-derived estimate
already produced by the frozen v1 shared estimator; elastic deflection, model
error, estimator filtering, and contact delay therefore prevent this outer loop
from being an exact force controller or a stability guarantee for the elastic
plant.  These limitations are why the correction is small and explicitly
bounded while the v1 inner controller and reference shaping remain unchanged.
Because the joint direction is held constant, the added joint position,
velocity, and acceleration are exact derivatives of one another, but the
resulting Cartesian motion is normal-only only under the taught-point Jacobian
linearization; nonlinear forward kinematics introduce higher-order tangential
and rotational motion.

Two boundary qualifications are explicit.  First, an alarm immediately holds
the displacement instead of honoring the acceleration slew for that one
transition; collision freeze has priority over that nominal bound.  Second, a
fixed 0.10 s C2 quintic cannot, for every state inside the independently
registered displacement / velocity / acceleration box, both reach rest and
remain inside all three bounds.  The implementation preserves C2 matching and
the fixed return time rather than silently clipping mutually dependent
derivatives.  Such an infeasible extreme is regression-tested and must be
treated as a candidate limitation during selection, not as a plant guarantee.

The construction is an original adaptation of normal admittance and wrench /
Jacobian duality.  Public background references for the experiment are Hogan
(1985), Albu-Schaeffer et al. (2007), van Steen et al. (2024), and *Lie Group
Robotics* manuscript (2024), sections 2.2, 6.1, and 7.2.
"""

import numpy as np


def normal_joint_direction(jacobian):
    """Return ``a`` solving ``J a = [0, 0, 1, 0, 0, 0]`` in WORLD axes."""
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.shape[0] != 6:
        raise ValueError("the bottom-centre Jacobian must have six rows")
    normal_twist = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    if jacobian.shape[1] == 6:
        try:
            return np.linalg.solve(jacobian, normal_twist)
        except np.linalg.LinAlgError:
            pass
    return np.linalg.pinv(jacobian, rcond=1e-8) @ normal_twist


def _quintic_boundary(x0, v0, a0, x1, v1, a1, duration):
    """Coefficients of a quintic matching position, velocity, acceleration."""
    duration = float(duration)
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    c0 = float(x0)
    c1 = float(v0)
    c2 = 0.5 * float(a0)
    matrix = np.array(
        [
            [duration**3, duration**4, duration**5],
            [3.0 * duration**2, 4.0 * duration**3, 5.0 * duration**4],
            [6.0 * duration, 12.0 * duration**2, 20.0 * duration**3],
        ]
    )
    rhs = np.array(
        [
            float(x1) - (c0 + c1 * duration + c2 * duration**2),
            float(v1) - (c1 + 2.0 * c2 * duration),
            float(a1) - 2.0 * c2,
        ]
    )
    return np.r_[c0, c1, c2, np.linalg.solve(matrix, rhs)]


def _evaluate_polynomial(coefficients, time):
    t = float(time)
    c = np.asarray(coefficients, dtype=float)
    position = sum(c[i] * t**i for i in range(6))
    velocity = sum(i * c[i] * t ** (i - 1) for i in range(1, 6))
    acceleration = sum(i * (i - 1) * c[i] * t ** (i - 2) for i in range(2, 6))
    return float(position), float(velocity), float(acceleration)


class NormalAdmittance:
    """Scalar preregistered admittance with saturation and release recovery."""

    def __init__(
        self,
        dt,
        release_time,
        mass,
        start_before_release=0.12,
        support_fraction=0.85,
        damping=12000.0,
        filter_time=0.010,
        displacement_limit=0.0015,
        velocity_limit=0.01,
        acceleration_limit=0.5,
        return_time=0.10,
    ):
        self.dt = float(dt)
        self.release_time = float(release_time)
        self.start_time = self.release_time - float(start_before_release)
        self.mass = max(float(mass), 0.0)
        self.support_fraction = float(support_fraction)
        self.damping = float(damping)
        self.filter_time = float(filter_time)
        self.displacement_limit = float(displacement_limit)
        self.velocity_limit = float(velocity_limit)
        self.acceleration_limit = float(acceleration_limit)
        self.return_time = float(return_time)
        self.reset()

    def reset(self):
        self.displacement = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.last_time = None
        self.initial_force = None
        self.return_coefficients = None

    @staticmethod
    def _smoothstep5(s):
        s = float(np.clip(s, 0.0, 1.0))
        return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5

    def _active_step(self, time, estimated_force, alarm):
        if self.initial_force is None:
            self.initial_force = float(estimated_force)
        phase = (float(time) - self.start_time) / (self.release_time - self.start_time)
        blend = self._smoothstep5(phase)
        final_target = self.support_fraction * self.mass * 9.81
        desired_force = self.initial_force + blend * (final_target - self.initial_force)
        raw_velocity = (float(estimated_force) - desired_force) / self.damping

        if alarm and (raw_velocity < 0.0 or self.velocity < 0.0):
            # Collision evidence has priority over smoothness: never command an
            # additional downward displacement while an alarm is asserted.
            new_velocity = 0.0
        else:
            alpha = min(self.dt / self.filter_time, 1.0)
            filtered = self.velocity + alpha * (raw_velocity - self.velocity)
            dv_limit = self.acceleration_limit * self.dt
            new_velocity = float(
                np.clip(filtered, self.velocity - dv_limit, self.velocity + dv_limit)
            )
            new_velocity = float(
                np.clip(new_velocity, -self.velocity_limit, self.velocity_limit)
            )

        proposed = self.displacement + new_velocity * self.dt
        clipped = float(
            np.clip(proposed, -self.displacement_limit, self.displacement_limit)
        )
        if clipped != proposed:
            # Saturation anti-windup: discard velocity into the active bound.
            new_velocity = 0.0
        old_velocity = self.velocity
        self.displacement = clipped
        self.velocity = new_velocity
        self.acceleration = (new_velocity - old_velocity) / self.dt

    def _return_state(self, time):
        # Matching x/v/a at both ends makes the scalar correction C2.  Do not
        # independently clip these outputs: that would destroy q/qd/qdd
        # consistency.  The documented fixed-time feasibility limitation is
        # instead exposed to candidate selection and its regression test.
        if self.return_coefficients is None:
            self.return_coefficients = _quintic_boundary(
                self.displacement,
                self.velocity,
                self.acceleration,
                0.0,
                0.0,
                0.0,
                self.return_time,
            )
        elapsed = float(time) - self.release_time
        if elapsed >= self.return_time:
            self.displacement = self.velocity = self.acceleration = 0.0
        else:
            self.displacement, self.velocity, self.acceleration = _evaluate_polynomial(
                self.return_coefficients, max(elapsed, 0.0)
            )

    def step(self, time, estimated_force, alarm=False):
        """Advance once and return correction displacement, velocity, acceleration."""
        time = float(time)
        if self.last_time is not None and time == self.last_time:
            return self.displacement, self.velocity, self.acceleration
        if time < self.start_time:
            self.displacement = self.velocity = self.acceleration = 0.0
        elif time < self.release_time:
            self._active_step(time, estimated_force, bool(alarm))
        else:
            self._return_state(time)
        self.last_time = time
        return self.displacement, self.velocity, self.acceleration


class NormalReferenceMixin:
    """Mixin that layers C_normal over a frozen v1 controller implementation."""

    NORMAL_START_BEFORE_RELEASE = 0.12
    NORMAL_SUPPORT_FRACTION = 0.85
    NORMAL_DAMPING = 12000.0
    NORMAL_FILTER_TIME = 0.010
    NORMAL_DISPLACEMENT_LIMIT = 0.0015
    NORMAL_VELOCITY_LIMIT = 0.01
    NORMAL_ACCELERATION_LIMIT = 0.5
    NORMAL_RETURN_TIME = 0.10

    def reset(self):
        super().reset()
        q_release = self.NORMAL_TRAJECTORY.reference(self.NORMAL_TRAJECTORY.T_RELEASE)[0]
        jacobian = self.shared_estimator._point_jacobian(q_release)[0]
        self.normal_direction = normal_joint_direction(jacobian)
        self.normal_admittance = NormalAdmittance(
            dt=self.dt,
            release_time=self.NORMAL_TRAJECTORY.T_RELEASE,
            mass=self.box_mass,
            start_before_release=self.NORMAL_START_BEFORE_RELEASE,
            support_fraction=self.NORMAL_SUPPORT_FRACTION,
            damping=self.NORMAL_DAMPING,
            filter_time=self.NORMAL_FILTER_TIME,
            displacement_limit=self.NORMAL_DISPLACEMENT_LIMIT,
            velocity_limit=self.NORMAL_VELOCITY_LIMIT,
            acceleration_limit=self.NORMAL_ACCELERATION_LIMIT,
            return_time=self.NORMAL_RETURN_TIME,
        )
        self.normal_state = (0.0, 0.0, 0.0)

    def _normal_update(self, t, th, thd, tau_s, grip):
        packet = self.shared_estimator.update(t, th, thd, tau_s, grip)
        table_wrench = getattr(packet, "last_table_wrench", np.zeros(6))
        self.normal_state = self.normal_admittance.step(
            t, float(table_wrench[2]), alarm=bool(getattr(packet, "detected", False))
        )

    def _normal_reference(self, q, qd, qdd):
        displacement, velocity, acceleration = self.normal_state
        return (
            np.asarray(q) + self.normal_direction * displacement,
            np.asarray(qd) + self.normal_direction * velocity,
            np.asarray(qdd) + self.normal_direction * acceleration,
        )
