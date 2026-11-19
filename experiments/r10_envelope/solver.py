"""Nominal-only constrained solver for a C2 local phase envelope."""
from __future__ import annotations

from math import comb

import numpy as np

import r1_servo
from sim import config as C, trajectory as T
from sim.nominal import Nominal


CLOCK_START = 3.60
CLOCK_END = 3.88
BASIS_COUNT = 5
TCP_BUDGET_MM = 0.15
CLOCK_RATE_MIN = 0.95
CLOCK_RATE_MAX = 1.05
CLOCK_ACCELERATION_MAX = 1.0
COEFFICIENT_ABS_MAX_S = 0.0005
MEANINGFUL_HEADROOM_GAIN_NM = 5.0
ACTIVATION_PEAK_NM = 330.0
CRITICAL_PAYLOADS = (
    (8.0, (-0.03, 0.02, -0.02)),
    (8.0, (-0.03, 0.02, 0.015)),
)


def _basis_polynomial(index):
    left = index + 3
    right = BASIS_COUNT - 1 - index + 3
    coefficients = np.zeros(left + right + 1)
    for power in range(right + 1):
        coefficients[left + power] = (-1)**power * comb(right, power)
    maximum_at = left/(left + right)
    maximum = maximum_at**left * (1.0-maximum_at)**right
    return coefficients/maximum


POLYNOMIALS = tuple(_basis_polynomial(index) for index in range(BASIS_COUNT))
FIRST = tuple(np.polynomial.polynomial.polyder(poly, 1) for poly in POLYNOMIALS)
SECOND = tuple(np.polynomial.polynomial.polyder(poly, 2) for poly in POLYNOMIALS)


def basis(t):
    """Return basis value and time derivatives at scalar/array time."""
    values = np.asarray(t, float)
    x = (values-CLOCK_START)/(CLOCK_END-CLOCK_START)
    inside = (values > CLOCK_START) & (values < CLOCK_END)
    x = np.clip(x, 0.0, 1.0)
    duration = CLOCK_END-CLOCK_START
    b = np.stack([np.polynomial.polynomial.polyval(x, poly) for poly in POLYNOMIALS], axis=-1)
    bd = np.stack([np.polynomial.polynomial.polyval(x, poly) for poly in FIRST], axis=-1)/duration
    bdd = np.stack([np.polynomial.polynomial.polyval(x, poly) for poly in SECOND], axis=-1)/duration**2
    b *= np.expand_dims(inside, -1)
    bd *= np.expand_dims(inside, -1)
    bdd *= np.expand_dims(inside, -1)
    return b, bd, bdd


def clock(t, coefficients):
    b, bd, bdd = basis(t)
    coefficients = np.asarray(coefficients, float)
    return (np.asarray(t, float) + b@coefficients,
            1.0 + bd@coefficients,
            bdd@coefficients)


def configured(mass, com):
    controller = r1_servo.Controller()
    controller.nom, controller.nom_box = Nominal(), Nominal()
    controller.nom_box.set_payload(mass, com)
    controller.box_mass, controller.box_com = mass, com
    controller.reset()
    return controller


def warped_reference(controller, t, coefficients):
    warped, rate, acceleration = clock(float(t), coefficients)
    warped, rate, acceleration = float(warped), float(rate), float(acceleration)
    q, qd, qdd = controller._reference(warped, *T.reference(warped))
    return q, qd*rate, qdd*rate**2 + qd*acceleration


def planned_peaks(mass, com, coefficients):
    controller = configured(mass, com)
    previous_link = None
    filtered_link_rate = np.zeros(6)
    peaks = np.zeros(6)
    for t in np.arange(3.40, 3.891, C.DT):
        q, qd, qdd = warped_reference(controller, float(t), coefficients)
        link = controller.nom_box.inverse_dynamics(q, qd, qdd)
        if previous_link is None:
            theta_dot, theta_ddot = qd.copy(), qdd.copy()
        else:
            raw_rate = (link-previous_link)/C.DT
            old_rate = filtered_link_rate.copy()
            filtered_link_rate += 0.04*(raw_rate-filtered_link_rate)
            theta_dot = np.clip(qd+controller.inv_k*filtered_link_rate,
                                qd-0.35, qd+0.35)
            theta_ddot = np.clip(
                qdd+controller.inv_k*(filtered_link_rate-old_rate)/C.DT, -25.0, 25.0)
        friction = C.VISCOUS*theta_dot + C.COULOMB*np.tanh(theta_dot/C.FRICTION_EPS)
        feedforward = controller.nom.B*theta_ddot + link + friction
        peaks = np.maximum(peaks, np.abs(feedforward))
        previous_link = link
    return peaks


def constraints(coefficients):
    coefficients = np.asarray(coefficients, float)
    times = np.linspace(CLOCK_START, CLOCK_END, 1401)
    warped, rate, acceleration = clock(times, coefficients)
    model = Nominal()
    tcp = []
    reference = configured(*CRITICAL_PAYLOADS[0])
    for t, u in zip(times, warped):
        q0 = reference._reference(float(t), *T.reference(float(t)))[0]
        q1 = reference._reference(float(u), *T.reference(float(u)))[0]
        tcp.append(1e3*np.linalg.norm(model.tcp_position(q1)-model.tcp_position(q0)))
    return {
        "max_tcp_deviation_mm": float(max(tcp)),
        "clock_rate_min": float(np.min(rate)),
        "clock_rate_max": float(np.max(rate)),
        "clock_acceleration_abs_max": float(np.max(np.abs(acceleration))),
        "max_phase_shift_ms": float(1e3*np.max(np.abs(warped-times))),
        "endpoint_exact": bool(np.array_equal(clock(CLOCK_START, coefficients),
                                               (np.asarray(CLOCK_START), np.asarray(1.0), np.asarray(0.0)))
                               and np.array_equal(clock(CLOCK_END, coefficients),
                                                  (np.asarray(CLOCK_END), np.asarray(1.0), np.asarray(0.0)))),
    }


def feasible(coefficients):
    values = constraints(coefficients)
    return (values["max_tcp_deviation_mm"] <= TCP_BUDGET_MM + 1e-12
            and values["clock_rate_min"] >= CLOCK_RATE_MIN - 1e-12
            and values["clock_rate_max"] <= CLOCK_RATE_MAX + 1e-12
            and values["clock_acceleration_abs_max"] <= CLOCK_ACCELERATION_MAX + 1e-12
            and np.max(np.abs(coefficients)) <= COEFFICIENT_ABS_MAX_S + 1e-15)


def project(coefficients):
    coefficients = np.clip(np.asarray(coefficients, float),
                           -COEFFICIENT_ABS_MAX_S, COEFFICIENT_ABS_MAX_S)
    if feasible(coefficients):
        return coefficients
    low, high = 0.0, 1.0
    for _ in range(24):
        middle = 0.5*(low+high)
        if feasible(middle*coefficients):
            low = middle
        else:
            high = middle
    return low*coefficients


def objective(coefficients):
    peaks = [planned_peaks(mass, com, coefficients)
             for mass, com in CRITICAL_PAYLOADS]
    return max(row[1] for row in peaks), peaks


def solve():
    coefficients = np.zeros(BASIS_COUNT)
    best, _ = objective(coefficients)
    evaluations = 1
    steps = (0.0002, 0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625)
    for step in steps:
        for _ in range(3):
            improved = False
            for index in range(BASIS_COUNT):
                candidates = []
                for direction in (-1.0, 1.0):
                    proposal = coefficients.copy()
                    proposal[index] += direction*step
                    proposal = project(proposal)
                    value, peaks = objective(proposal)
                    evaluations += 1
                    candidates.append((value, float(np.dot(proposal, proposal)), proposal, peaks))
                value, _, proposal, _ = min(candidates, key=lambda item: (item[0], item[1]))
                if value < best-1e-8:
                    coefficients, best, improved = proposal, value, True
            if not improved:
                break
    baseline_value, baseline_peaks = objective(np.zeros(BASIS_COUNT))
    final_value, final_peaks = objective(coefficients)
    tradeoff = []
    for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
        value, _ = objective(scale*coefficients)
        tradeoff.append({"scale": scale, "worst_j2_peak_nm": value,
                         **constraints(scale*coefficients)})
    non_j2_ok = all(np.all(final[[0, 2, 3, 4, 5]]
                           <= base[[0, 2, 3, 4, 5]]*1.05 + 1e-9)
                    for final, base in zip(final_peaks, baseline_peaks))
    return {
        "coefficients_s": coefficients.tolist(),
        "constraints": constraints(coefficients),
        "baseline_worst_j2_peak_nm": baseline_value,
        "optimized_worst_j2_peak_nm": final_value,
        "planned_headroom_gain_nm": baseline_value-final_value,
        "baseline_axis_peaks_nm": [row.tolist() for row in baseline_peaks],
        "optimized_axis_peaks_nm": [row.tolist() for row in final_peaks],
        "non_j2_new_peak_gate": non_j2_ok,
        "coordinate_evaluations": evaluations,
        "terminal_step_s": steps[-1],
        "tradeoff_fixed_constraints": tradeoff,
        "feasible_for_plant": bool(feasible(coefficients)
                                   and baseline_value-final_value >= MEANINGFUL_HEADROOM_GAIN_NM
                                   and non_j2_ok),
    }
