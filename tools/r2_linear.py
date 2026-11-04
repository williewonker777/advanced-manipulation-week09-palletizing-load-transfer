"""Pure-NumPy linear analysis kernel for the R2 elastic-joint controllers.

This module is analysis infrastructure, not an episode runner.  Its coordinates are

    z = [theta_e(J1..J3), q_link(J1..J6)]                 [rad]
    x = [z, zdot]                                         [rad, rad/s]

and an input is the six motor torques in joint order [N m].  Clocks, counters,
references, controller parameters, saturation flags, and random/noise state are
*not* dynamical state.  A nonlinear controller adapter must expose all of its
actual filters, integrators, observers, and held commands in ``controller_state``.

The runner map below deliberately mirrors ``sim.runner`` ordering: sense first,
form theta_dot from the current and previous encoder samples, compute a new
command, apply the command delayed from the previous cycle, then advance the
exact 1 ms ZOH plant.  Encoder quantization/noise are intentionally excluded
from Jacobians (their point derivatives are not meaningful), but the 1 ms
finite-difference transfer remains explicit.  The JTS low-pass and one-cycle
delay are also explicit dynamical states.

All stability and margin results are local sampled analyses.  Contact mode
boundaries, saturation, quantization, and untested structured uncertainty mean
they are not nonlinear or hybrid-system proofs.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


Array = np.ndarray


def _array(value, shape=None, name="array"):
    out = np.asarray(value, dtype=float)
    if shape is not None and out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def selector_matrices():
    """Return E (J1--J3) and R (J4--J6) selectors, both 6x3."""
    eye = np.eye(6)
    return eye[:, :3], eye[:, 3:]


def _elastic_values(value, name):
    """Accept either compact J1--J3 values or a six-axis config vector."""
    out = np.asarray(value, dtype=float)
    if out.shape == (6,):
        out = out[:3]
    return _array(out, (3,), name)


def _contact_jacobian(jacobian):
    """Embed a link-coordinate contact J (..x6) into z, or accept ..x9."""
    if jacobian is None:
        return None
    J = np.asarray(jacobian, dtype=float)
    if J.ndim == 1:
        J = J.reshape(1, -1)
    if J.ndim != 2:
        raise ValueError("contact Jacobian must be a vector or matrix")
    if J.shape[1] == 6:
        J = np.hstack((np.zeros((J.shape[0], 3)), J))
    if J.shape[1] != 9:
        raise ValueError("contact Jacobian must have 6 link or 9 generalized columns")
    return J


@dataclass(frozen=True)
class ElasticPlant:
    """Continuous linear plant and the matrices used to construct it."""

    A: Array
    B: Array
    H: Array
    K: Array
    D: Array
    G: Array
    L: Array
    E: Array
    R: Array


def _quadratic_term(jacobian, coefficient, size, name):
    """Return J.T C J for scalar/diagonal/full C, accepting a vector J."""
    if jacobian is None:
        return np.zeros((size, size))
    J = np.asarray(jacobian, dtype=float)
    if J.ndim == 1:
        J = J.reshape(1, -1)
    if J.ndim != 2 or J.shape[1] != size:
        raise ValueError(f"{name} Jacobian must have {size} columns")
    c = np.asarray(coefficient, dtype=float)
    if c.ndim == 0:
        C = float(c) * np.eye(J.shape[0])
    elif c.ndim == 1:
        if c.shape != (J.shape[0],):
            raise ValueError(f"{name} diagonal has incompatible shape {c.shape}")
        C = np.diag(c)
    else:
        C = _array(c, (J.shape[0], J.shape[0]), f"{name} coefficient")
    return J.T @ C @ J


def build_elastic_plant(
    mass_matrix,
    motor_inertia,
    stiffness,
    joint_damping,
    *,
    viscous=None,
    coulomb=None,
    velocity=None,
    friction_eps=0.02,
    gravity_derivative=None,
    bias_velocity_derivative=None,
    contact_jacobian=None,
    contact_stiffness=0.0,
    contact_damping=0.0,
):
    """Build the 9-DOF continuous linearized elastic plant.

    ``mass_matrix`` is the nominal *link* inertia M(q), with no additional
    armature.  Only rigid J4--J6 armature is added to its link block.  Elastic
    motor inertias form the separate leading 3x3 block.

    ``gravity_derivative`` is d(g)/dq and ``bias_velocity_derivative`` is
    d(h)/d(qdot), both 6x6 in link coordinates.  Contact J may be expressed in
    the 9 generalized coordinates.  ``contact_damping`` uses the same J.
    Friction is differentiated exactly for
    b*v + c*tanh(v/eps), then mapped by G diag(df/dv) G.T.
    """
    M = _array(mass_matrix, (6, 6), "mass_matrix")
    Bm = _array(motor_inertia, (6,), "motor_inertia")
    Ke = _elastic_values(stiffness, "stiffness")
    De = _elastic_values(joint_damping, "joint_damping")
    if np.any(Bm <= 0.0) or np.any(Ke < 0.0) or np.any(De < 0.0):
        raise ValueError("inertias must be positive; stiffness/damping nonnegative")
    E, R = selector_matrices()
    G = np.vstack((E.T, R @ R.T))
    L = np.hstack((np.eye(3), -E.T))
    H = np.zeros((9, 9))
    H[:3, :3] = np.diag(Bm[:3])
    H[3:, 3:] = M + R @ np.diag(Bm[3:]) @ R.T
    K = L.T @ np.diag(Ke) @ L
    D = L.T @ np.diag(De) @ L

    if gravity_derivative is not None:
        Kg = _array(gravity_derivative, (6, 6), "gravity_derivative")
        K[3:, 3:] += Kg
    if bias_velocity_derivative is not None:
        Dv = _array(
            bias_velocity_derivative, (6, 6), "bias_velocity_derivative"
        )
        D[3:, 3:] += Dv
    contact_jacobian = _contact_jacobian(contact_jacobian)
    K += _quadratic_term(contact_jacobian, contact_stiffness, 9, "contact stiffness")
    if contact_jacobian is not None and np.any(np.asarray(contact_damping) != 0.0):
        D += _quadratic_term(
            contact_jacobian, contact_damping, 9, "contact damping"
        )

    viscous = np.zeros(6) if viscous is None else _array(viscous, (6,), "viscous")
    coulomb = np.zeros(6) if coulomb is None else _array(coulomb, (6,), "coulomb")
    velocity = np.zeros(6) if velocity is None else _array(velocity, (6,), "velocity")
    if friction_eps <= 0.0:
        raise ValueError("friction_eps must be positive")
    sech2 = 1.0 / np.cosh(velocity / friction_eps) ** 2
    friction_derivative = viscous + coulomb * sech2 / friction_eps
    D += G @ np.diag(friction_derivative) @ G.T

    if not np.allclose(H, H.T, atol=1e-12):
        raise ValueError("generalized inertia is not symmetric")
    HinvK = np.linalg.solve(H, K)
    HinvD = np.linalg.solve(H, D)
    HinvG = np.linalg.solve(H, G)
    A = np.block([[np.zeros((9, 9)), np.eye(9)], [-HinvK, -HinvD]])
    B = np.vstack((np.zeros((9, 6)), HinvG))
    return ElasticPlant(A=A, B=B, H=H, K=K, D=D, G=G, L=L, E=E, R=R)


def matrix_exponential(matrix, *, tolerance=1e-15, max_terms=160):
    """NumPy-only scaling/squaring Taylor matrix exponential.

    Scaling to a one-norm <= 0.5 makes the Taylor tail small; squaring restores
    the requested interval.  This is intended for the small dense R2 analysis
    matrices, not as a general replacement for a production Schur/Pade expm.
    """
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square")
    if A.size == 0:
        return A.copy()
    norm = float(np.linalg.norm(A, 1))
    squarings = max(0, int(np.ceil(np.log2(norm / 0.5)))) if norm > 0.5 else 0
    scaled = A / (2.0**squarings)
    result = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for k in range(1, max_terms + 1):
        term = term @ scaled / k
        result += term
        if np.linalg.norm(term, 1) <= tolerance * max(1.0, np.linalg.norm(result, 1)):
            break
    else:
        raise RuntimeError("matrix exponential Taylor series did not converge")
    for _ in range(squarings):
        result = result @ result
    return result


def exact_zoh(A, B, dt=0.001):
    """Exact discrete (Ad, Bd) for xdot=A x+B u with held input."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("B row count must match A")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    n, m = B.shape
    augmented = np.zeros((n + m, n + m))
    augmented[:n, :n] = A
    augmented[:n, n:] = B
    exponential = matrix_exponential(augmented * dt)
    return exponential[:n, :n], exponential[:n, n:]


def plant_step(Ad, Bd, state, held_torque):
    """Advance the exact-ZOH plant under one cycle of held motor torque."""
    Ad = _array(Ad, (18, 18), "Ad")
    Bd = _array(Bd, (18, 6), "Bd")
    state = _array(state, (18,), "plant state")
    held_torque = _array(held_torque, (6,), "held_torque")
    return Ad @ state + Bd @ held_torque


def motor_coordinates(x):
    """Return six motor positions and velocities from the 18-state plant."""
    x = _array(x, (18,), "plant state")
    z, zd = x[:9], x[9:]
    theta = z[3:9].copy()
    theta_dot = zd[3:9].copy()
    theta[:3], theta_dot[:3] = z[:3], zd[:3]
    return theta, theta_dot


def smooth_friction(velocity, viscous, coulomb, friction_eps=0.02):
    velocity = np.asarray(velocity, dtype=float)
    return np.asarray(viscous) * velocity + np.asarray(coulomb) * np.tanh(
        velocity / friction_eps
    )


def true_joint_torque(
    plant: ElasticPlant,
    x,
    applied_torque,
    *,
    stiffness,
    joint_damping,
    motor_inertia,
    viscous,
    coulomb,
    friction_eps=0.02,
):
    """Ideal JTS torque at a linearization state, before sensor dynamics.

    Elastic axes use K(theta-q)+D(theta_dot-q_dot).  Rigid axes use
    u-friction-B_r*qdd_r, matching the sensor location in ``sim.plant``.
    """
    x = _array(x, (18,), "plant state")
    u = _array(applied_torque, (6,), "applied_torque")
    Ke = _elastic_values(stiffness, "stiffness")
    De = _elastic_values(joint_damping, "joint_damping")
    Bm = _array(motor_inertia, (6,), "motor_inertia")
    theta, theta_dot = motor_coordinates(x)
    q, qdot = x[3:9], x[12:18]
    out = np.empty(6)
    out[:3] = Ke * (theta[:3] - q[:3]) + De * (theta_dot[:3] - qdot[:3])
    acceleration = plant.A @ x + plant.B @ u
    qdd = acceleration[12:18]
    friction = smooth_friction(theta_dot, viscous, coulomb, friction_eps)
    out[3:] = u[3:] - friction[3:] - Bm[3:] * qdd[3:]
    return out


def jts_sensor_step(true_torque, filter_state, delay_state, *, cutoff=300.0, dt=0.001):
    """JTS first-order filter plus one-cycle delay.

    Returns ``(measurement_now, filter_next, delay_next)``.  Offset, drift, and
    random noise are affine/exogenous and therefore excluded from pole analysis.
    """
    tau = _array(true_torque, (6,), "true_torque")
    filt = _array(filter_state, (6,), "filter_state")
    delayed = _array(delay_state, (6,), "delay_state")
    if cutoff <= 0.0 or dt <= 0.0:
        raise ValueError("cutoff and dt must be positive")
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff * dt)
    filt_next = filt + alpha * (tau - filt)
    return delayed.copy(), filt_next, filt_next.copy()


def plant_measurement_step(
    plant,
    state,
    applied_torque,
    jts_filter,
    jts_delay,
    *,
    stiffness,
    joint_damping,
    motor_inertia,
    viscous,
    coulomb,
    friction_eps=0.02,
    jts_cutoff=300.0,
    dt=0.001,
):
    """Produce the pre-controller measurement and advance only JTS states.

    The returned tuple is ``(observation, filter_next, delay_next, ideal_jts)``;
    observation ordering is ``[theta(6), theta_dot(6), delayed_jts(6)]``.
    Encoder quantization, offsets, drift, and noise are excluded because they
    are exogenous/nondifferentiable rather than pole-producing states.
    """
    state = _array(state, (18,), "plant state")
    ideal = true_joint_torque(
        plant,
        state,
        applied_torque,
        stiffness=stiffness,
        joint_damping=joint_damping,
        motor_inertia=motor_inertia,
        viscous=viscous,
        coulomb=coulomb,
        friction_eps=friction_eps,
    )
    jts, filter_next, delay_next = jts_sensor_step(
        ideal, jts_filter, jts_delay, cutoff=jts_cutoff, dt=dt
    )
    theta, theta_dot = motor_coordinates(state)
    return np.concatenate((theta, theta_dot, jts)), filter_next, delay_next, ideal


def servo_drive_step(
    command,
    theta,
    theta_dot,
    velocity_filter,
    integral,
    *,
    kp,
    kv,
    ki,
    filter_coefficient=0.2,
    dt=0.001,
):
    """One exact discrete drive update in ``sim.drive`` order.

    ``command`` rows are ``[pos, vel_unused, vel_ff, tau_ff]``.  The returned
    torque uses the updated velocity filter and updated PI integral.
    """
    cmd = _array(command, (4, 6), "command")
    theta = _array(theta, (6,), "theta")
    theta_dot = _array(theta_dot, (6,), "theta_dot")
    vf = _array(velocity_filter, (6,), "velocity_filter")
    integ = _array(integral, (6,), "integral")
    kv = np.broadcast_to(np.asarray(kv, dtype=float), (6,))
    ki = np.broadcast_to(np.asarray(ki, dtype=float), (6,))
    if not 0.0 <= filter_coefficient <= 1.0:
        raise ValueError("filter_coefficient must be in [0, 1]")
    vf_next = vf + filter_coefficient * (theta_dot - vf)
    velocity_error = cmd[2] + kp * (cmd[0] - theta) - vf_next
    integral_next = integ + dt * velocity_error
    torque = cmd[3] + kv * velocity_error + ki * integral_next
    return torque, vf_next, integral_next


@dataclass(frozen=True)
class RunnerState:
    """Complete dynamical state at a pre-sense 1 ms cycle boundary."""

    plant: Array
    controller: Array
    pending_command: Array
    drive_velocity_filter: Array
    drive_integral: Array
    jts_filter: Array
    jts_delay: Array
    applied_torque: Array
    encoder_previous: Array


def runner_state_sizes(controller_size, mode="torque"):
    """Named flat-state slices for adapters and numerical Jacobians."""
    command_size = 24 if mode == "servo" else 6
    sizes = (
        ("plant", 18),
        ("controller", int(controller_size)),
        ("pending_command", command_size),
        ("drive_velocity_filter", 6),
        ("drive_integral", 6),
        ("jts_filter", 6),
        ("jts_delay", 6),
        ("applied_torque", 6),
        ("encoder_previous", 6),
    )
    start = 0
    out = {}
    for name, size in sizes:
        out[name] = slice(start, start + size)
        start += size
    out["size"] = start
    return out


def controller_adapter_contract(controller_size, mode="torque"):
    """Machine-readable contract for an external restored-state adapter.

    The adapter owns extraction/restoration of every numeric controller state
    (filters, integrators, observers, previous samples, held values).  It must
    not put time, sample counters, references, gains, model parameters, or mode
    labels in that vector; those enter ``reference``/``parameters`` exogenously.
    """
    return {
        "controller_state_size": int(controller_size),
        "runner_state": runner_state_sizes(controller_size, mode),
        "observation_order": (
            "theta_encoder[6]",
            "(theta_encoder-current - theta_encoder-previous)/dt[6]",
            "jts_delayed[6]",
        ),
        "callback": "(controller_state, observation, reference, parameters) -> (state_next, command)",
        "servo_command": "[pos, vel_unused, vel_ff, tau_ff] with shape (4,6)",
        "torque_command": "motor torque with shape (6,)",
        "excluded_from_state": (
            "clock",
            "sample/event counters",
            "references",
            "controller/model parameters",
            "noise",
        ),
        "initialization": (
            "set encoder_previous=current encoder for the first-cycle zero velocity "
            "used by sim.runner; restored snapshots must retain the actual previous sample"
        ),
        "queue_ordering": (
            "sense; theta_dot=(encoder-current - encoder-previous)/dt; compute new "
            "command; servo drive uses current sampled encoder/theta_dot and previous "
            "command, while direct torque applies previous torque command; exact plant step"
        ),
    }


def pack_runner_state(state: RunnerState, mode="torque"):
    pending_shape = (4, 6) if mode == "servo" else (6,)
    pieces = [
        _array(state.plant, (18,), "plant"),
        np.asarray(state.controller, dtype=float).reshape(-1),
        _array(state.pending_command, pending_shape, "pending_command").reshape(-1),
        _array(state.drive_velocity_filter, (6,), "drive_velocity_filter"),
        _array(state.drive_integral, (6,), "drive_integral"),
        _array(state.jts_filter, (6,), "jts_filter"),
        _array(state.jts_delay, (6,), "jts_delay"),
        _array(state.applied_torque, (6,), "applied_torque"),
        _array(state.encoder_previous, (6,), "encoder_previous"),
    ]
    return np.concatenate(pieces)


def unpack_runner_state(flat, controller_size, mode="torque"):
    flat = np.asarray(flat, dtype=float).reshape(-1)
    slices = runner_state_sizes(controller_size, mode)
    if flat.size != slices["size"]:
        raise ValueError(f"flat runner state must have size {slices['size']}")
    pending = flat[slices["pending_command"]].copy()
    if mode == "servo":
        pending = pending.reshape(4, 6)
    return RunnerState(
        plant=flat[slices["plant"]].copy(),
        controller=flat[slices["controller"]].copy(),
        pending_command=pending,
        drive_velocity_filter=flat[slices["drive_velocity_filter"]].copy(),
        drive_integral=flat[slices["drive_integral"]].copy(),
        jts_filter=flat[slices["jts_filter"]].copy(),
        jts_delay=flat[slices["jts_delay"]].copy(),
        applied_torque=flat[slices["applied_torque"]].copy(),
        encoder_previous=flat[slices["encoder_previous"]].copy(),
    )


def exact_runner_step(
    state: RunnerState,
    controller_step: Callable[[Array, Array, object, object], tuple],
    plant: ElasticPlant,
    *,
    Ad,
    Bd,
    mode="torque",
    reference=None,
    parameters=None,
    stiffness,
    joint_damping,
    motor_inertia,
    viscous,
    coulomb,
    friction_eps=0.02,
    jts_cutoff=300.0,
    dt=0.001,
    drive_kp=12.0,
    drive_kv=None,
    drive_ki=None,
    drive_filter=0.2,
):
    """Advance the deterministic linear-analysis runner by one cycle.

    ``controller_step(xc, observation, reference, parameters)`` must return
    ``(xc_next, command)``.  Observation is ``[theta, theta_dot, jts]`` (18).
    Reference and parameters are exogenous and excluded from the state.
    """
    x = _array(state.plant, (18,), "plant")
    Ad = _array(Ad, (18, 18), "Ad")
    Bd = _array(Bd, (18, 6), "Bd")
    observation, jts_filter_next, jts_delay_next, _ = plant_measurement_step(
        plant,
        x,
        state.applied_torque,
        state.jts_filter,
        state.jts_delay,
        stiffness=stiffness,
        joint_damping=joint_damping,
        motor_inertia=motor_inertia,
        viscous=viscous,
        coulomb=coulomb,
        friction_eps=friction_eps,
        jts_cutoff=jts_cutoff,
        dt=dt,
    )
    theta = observation[:6]
    encoder_previous = _array(state.encoder_previous, (6,), "encoder_previous")
    theta_dot = (theta - encoder_previous) / dt
    # sim.runner sends this sampled encoder finite difference to both the
    # controller and Drive; instantaneous motor velocity remains internal to
    # the ideal-JTS calculation above.
    observation = np.concatenate((theta, theta_dot, observation[12:18]))
    controller_next, command = controller_step(
        np.asarray(state.controller, dtype=float).copy(), observation, reference, parameters
    )
    controller_next = np.asarray(controller_next, dtype=float).reshape(-1)

    if mode == "servo":
        if drive_kv is None or drive_ki is None:
            raise ValueError("servo mode requires drive_kv and drive_ki")
        command = _array(command, (4, 6), "new servo command")
        torque, vf_next, integral_next = servo_drive_step(
            state.pending_command,
            theta,
            theta_dot,
            state.drive_velocity_filter,
            state.drive_integral,
            kp=drive_kp,
            kv=drive_kv,
            ki=drive_ki,
            filter_coefficient=drive_filter,
            dt=dt,
        )
    elif mode == "torque":
        command = _array(command, (6,), "new torque command")
        torque = _array(state.pending_command, (6,), "pending torque")
        vf_next = _array(state.drive_velocity_filter, (6,), "drive_velocity_filter").copy()
        integral_next = _array(state.drive_integral, (6,), "drive_integral").copy()
    else:
        raise ValueError("mode must be 'servo' or 'torque'")
    plant_next = plant_step(Ad, Bd, x, torque)
    return RunnerState(
        plant=plant_next,
        controller=controller_next,
        pending_command=command,
        drive_velocity_filter=vf_next,
        drive_integral=integral_next,
        jts_filter=jts_filter_next,
        jts_delay=jts_delay_next,
        applied_torque=torque.copy(),
        encoder_previous=theta.copy(),
    )


@dataclass(frozen=True)
class FeedbackStateSpace:
    """Discrete controller xc+=Ac xc+Bc y; command=Cc xc+Dc y."""

    Ac: Array
    Bc: Array
    Cc: Array
    Dc: Array


def build_delayed_feedback_closed_loop(Ad, Bd, Cy, Dyu, feedback):
    """Close a discrete plant/controller with one command-cycle delay.

    The augmented state is ``[x_plant, x_controller, pending_command,
    previous_applied_torque]``.  At sense time
    ``y=Cy*x+Dyu*previous_applied_torque``; the pending command is then applied
    to the plant, copied into previous-applied state, and replaced by the new
    controller output.  Controller signs must already be present in Bc/Dc.

    This realizes the direct-torque (B) queue including rigid-JTS input
    feedthrough.  It must not be used to delay an entire servo (A) feedback
    loop: servo drive feedback sees the current encoder and the previous 4x6
    command, as implemented by ``exact_runner_step``.
    """
    Ad, Bd, Cy, Dyu = map(lambda a: np.asarray(a, dtype=float), (Ad, Bd, Cy, Dyu))
    fb = feedback
    nx, nu = Bd.shape
    ny, nc = Cy.shape[0], np.asarray(fb.Ac).shape[0]
    if Ad.shape != (nx, nx) or Cy.shape != (ny, nx) or Dyu.shape != (ny, nu):
        raise ValueError("incompatible plant/output shapes")
    Ac = _array(fb.Ac, (nc, nc), "Ac")
    Bc = _array(fb.Bc, (nc, ny), "Bc")
    Cc = _array(fb.Cc, (nu, nc), "Cc")
    Dc = _array(fb.Dc, (nu, ny), "Dc")
    return np.block(
        [
            [Ad, np.zeros((nx, nc)), Bd, np.zeros((nx, nu))],
            [Bc @ Cy, Ac, np.zeros((nc, nu)), Bc @ Dyu],
            [Dc @ Cy, Cc, np.zeros((nu, nu)), Dc @ Dyu],
            [np.zeros((nu, nx)), np.zeros((nu, nc)), np.eye(nu), np.zeros((nu, nu))],
        ]
    )


@dataclass(frozen=True)
class JacobianResult:
    A: Array
    B: Optional[Array]
    diagnostics: dict


def linearize_discrete_map(
    map_function,
    state,
    input_value=None,
    *,
    state_step=1e-6,
    input_step=1e-6,
    sensitivity_ratio=0.5,
):
    """Central-difference a complete discrete map with step diagnostics.

    The callable is ``f(x)`` or ``f(x,u)`` and must return a flat next state.
    Diagnostics compare h against ``sensitivity_ratio*h``; a large relative
    change flags nonsmoothness, omitted state, or a poor finite-difference step.
    """
    x0 = np.asarray(state, dtype=float).reshape(-1)
    u0 = None if input_value is None else np.asarray(input_value, dtype=float).reshape(-1)
    if state_step <= 0.0 or input_step <= 0.0 or not 0.0 < sensitivity_ratio < 1.0:
        raise ValueError("steps must be positive and sensitivity_ratio in (0,1)")

    def evaluate(x, u):
        y = map_function(x) if u is None else map_function(x, u)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != x0.size:
            raise ValueError("discrete map output size must equal state size")
        return y

    def jacobian(step_x, step_u):
        A = np.empty((x0.size, x0.size))
        for j in range(x0.size):
            h = step_x * max(1.0, abs(x0[j]))
            delta = np.zeros_like(x0)
            delta[j] = h
            A[:, j] = (evaluate(x0 + delta, u0) - evaluate(x0 - delta, u0)) / (2.0 * h)
        B = None
        if u0 is not None:
            B = np.empty((x0.size, u0.size))
            for j in range(u0.size):
                h = step_u * max(1.0, abs(u0[j]))
                delta = np.zeros_like(u0)
                delta[j] = h
                B[:, j] = (evaluate(x0, u0 + delta) - evaluate(x0, u0 - delta)) / (2.0 * h)
        return A, B

    A, B = jacobian(state_step, input_step)
    A2, B2 = jacobian(state_step * sensitivity_ratio, input_step * sensitivity_ratio)

    def relative(first, second):
        return float(np.linalg.norm(first - second) / max(1.0, np.linalg.norm(second)))

    diagnostics = {
        "state_step": float(state_step),
        "input_step": float(input_step),
        "sensitivity_ratio": float(sensitivity_ratio),
        "A_relative_step_change": relative(A, A2),
        "B_relative_step_change": None if B is None else relative(B, B2),
        "warning": "local finite-difference result; inspect step sensitivity and mode boundaries",
    }
    return JacobianResult(A=A2, B=B2, diagnostics=diagnostics)


def closed_loop_poles(matrix, stability_tolerance=1e-10):
    A = np.asarray(matrix, dtype=float)
    poles = np.linalg.eigvals(A)
    radius = float(np.max(np.abs(poles))) if poles.size else 0.0
    return {
        "poles": poles,
        "spectral_radius": radius,
        "stable": bool(radius < 1.0 - stability_tolerance),
    }


def _sampled_boundary(builder, nominal, direction, tested_limit, tolerance, iterations):
    """Find the first sampled stable/unstable boundary from a stable nominal."""
    stable = lambda gain: closed_loop_poles(builder(gain))["stable"]
    if not stable(nominal):
        return {"status": "nominal_unstable", "gain": None}
    if direction == "lower":
        endpoint = tested_limit
    else:
        endpoint = tested_limit
    if stable(endpoint):
        return {"status": "stable_to_tested_limit", "gain": float(endpoint)}
    stable_gain, unstable_gain = nominal, endpoint
    for _ in range(iterations):
        middle = 0.5 * (stable_gain + unstable_gain)
        if stable(middle):
            stable_gain = middle
        else:
            unstable_gain = middle
        if abs(unstable_gain - stable_gain) <= tolerance * max(1.0, abs(stable_gain)):
            break
    return {
        "status": "sampled_boundary",
        "stable_gain": float(stable_gain),
        "unstable_gain": float(unstable_gain),
    }


def structured_gain_margins(
    matrix_builder: Callable[[Array], Array],
    axes=6,
    *,
    lower_test=0.0,
    upper_test=16.0,
    tolerance=1e-6,
    iterations=80,
):
    """Sample scalar and per-axis feedback-gain scale boundaries.

    ``matrix_builder(scales)`` receives one scale per axis.  The nominal vector
    is all ones.  Lower/upper results are numerical mode-local boundaries, not
    classical SISO margins or a structured-singular-value proof.
    """
    nominal = np.ones(int(axes))

    def scalar_builder(value):
        return matrix_builder(np.full(int(axes), float(value)))

    result = {
        "scalar": {
            "lower": _sampled_boundary(
                scalar_builder, 1.0, "lower", lower_test, tolerance, iterations
            ),
            "upper": _sampled_boundary(
                scalar_builder, 1.0, "upper", upper_test, tolerance, iterations
            ),
        },
        "per_axis": [],
        "feedforward_margin": "N/A: affine feed-forward does not close the local feedback loop",
        "qualification": (
            "sampled local gain-scale boundaries only; saturation, contact-mode "
            "switches, and unstructured/multi-axis combinations remain; this is not a proof"
        ),
    }
    for axis in range(int(axes)):
        def axis_builder(value, axis=axis):
            scales = nominal.copy()
            scales[axis] = value
            return matrix_builder(scales)

        result["per_axis"].append(
            {
                "axis": axis,
                "lower": _sampled_boundary(
                    axis_builder, 1.0, "lower", lower_test, tolerance, iterations
                ),
                "upper": _sampled_boundary(
                    axis_builder, 1.0, "upper", upper_test, tolerance, iterations
                ),
            }
        )
    return result


def analyze_closed_loop(matrix, *, margin_builder=None, axes=6):
    """Return poles and optionally sampled structured feedback margins."""
    report = closed_loop_poles(matrix)
    report["feedforward_margin"] = (
        "N/A: affine feed-forward does not change local closed-loop poles"
    )
    report["proof_scope"] = (
        "local fixed-mode analysis; contact boundaries and nonlinear limits are not a proof"
    )
    if margin_builder is not None:
        report["gain_margins"] = structured_gain_margins(margin_builder, axes=axes)
    return report

