#!/usr/bin/env python3
"""Focused pRNE mathematical and detector-isolation tests."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.r2_prne import PRNE, make_detector
from sim.nominal import Nominal


def _cases():
    rng = np.random.default_rng(369601)
    for mass in (0.0, 2.95, 7.45):
        nominal = Nominal()
        nominal.set_payload(mass, (0.013, -0.009, 0.019))
        for _ in range(3):
            yield nominal, rng.uniform(-1.1, 1.1, 6), rng.uniform(-0.8, 0.8, 6)


def verify_math():
    rows = []
    maximum_bias = maximum_vjp = maximum_direct = 0.0
    maximum_identity = 0.0
    for nominal, q, velocity in _cases():
        prne = PRNE(nominal)
        matrix = prne.matrix(q, velocity)
        bias = nominal.bias(q, velocity) - nominal.gravity(q)
        bias_error = float(np.max(np.abs(prne.action(q, velocity, velocity) - bias)))
        y = np.linspace(-0.7, 0.8, 6)
        u = np.linspace(0.6, -0.5, 6)
        vjp_error = float(abs(y @ prne.action(q, velocity, u)
                              - u @ prne.transpose(q, velocity, y)))
        sweep = []
        for step in (1e-4, 1e-5, 1e-6, 1e-7):
            mdot = (nominal.mass_matrix(q + step * velocity)
                    - nominal.mass_matrix(q - step * velocity)) / (2.0 * step)
            absolute = float(np.linalg.norm(
                mdot - matrix - matrix.T, ord=np.inf))
            normalized = absolute / max(1.0, float(np.linalg.norm(mdot, ord=np.inf)))
            sweep.append({"h": step, "absolute_inf": absolute,
                          "normalized_inf": normalized})
        selected = min(sweep, key=lambda row: row["normalized_inf"])
        best = selected["normalized_inf"]
        step = selected["h"]
        selected_mdot = (nominal.mass_matrix(q + step * velocity)
                         - nominal.mass_matrix(q - step * velocity)) / (2.0 * step)
        direct_oracle = selected_mdot @ velocity - bias
        direct_error = float(np.max(np.abs(
            prne.transpose(q, velocity, velocity) - direct_oracle)))
        rows.append({"payload_mass_kg": float(nominal.m.body_mass[nominal.m.body("box").id]),
                     "q": q.tolist(), "velocity": velocity.tolist(),
                     "bias_max_abs": bias_error, "vjp_inner_product_abs": vjp_error,
                     "direct_transpose_max_abs": direct_error,
                     "mdot_sweep": sweep, "selected_h": selected["h"],
                     "selected_vs_best_factor": selected["normalized_inf"] / max(best, 1e-30)})
        maximum_bias = max(maximum_bias, bias_error)
        maximum_vjp = max(maximum_vjp, vjp_error)
        maximum_direct = max(maximum_direct, direct_error)
        maximum_identity = max(maximum_identity, selected["normalized_inf"])
    protocol = ROOT / "results/v2_protocol/PROTOCOL_R2.json"
    return {"schema": "R2_PRNE_MATH.v1", "official": False,
            "candidate_episodes": 0,
            "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            "cases": rows,
            "summary": {"cases": len(rows), "maximum_bias_max_abs": maximum_bias,
                        "maximum_vjp_inner_product_abs": maximum_vjp,
                        "maximum_direct_transpose_max_abs": maximum_direct,
                        "maximum_selected_mdot_normalized_inf": maximum_identity,
                        "all_protocol_gates_pass": bool(
                            maximum_bias <= 1e-6 + 1e-5 * 100.0
                            and maximum_vjp <= 1e-9 + 1e-7 * 100.0
                            and maximum_direct <= 1e-6
                            and maximum_identity <= 1e-4
                            and all(r["selected_vs_best_factor"] <= 3.0 for r in rows))},
            "qualification": "finite sampled identities for the public six-link nominal model; not a nonlinear plant stability proof"}


def test_math_contract():
    result = verify_math()
    assert result["summary"]["all_protocol_gates_pass"], result["summary"]


def test_transpose_does_not_call_basis_action():
    nominal = Nominal()
    prne = PRNE(nominal)
    prne.action = lambda *args: (_ for _ in ()).throw(AssertionError("basis action used"))
    result = prne.transpose(np.zeros(6), np.ones(6) * 0.1, np.ones(6))
    assert result.shape == (6,)


def test_detector_is_independent_of_controller_shared_core():
    detector = make_detector({"variant": "prne_only"})()
    detector.nom, detector.nom_box = Nominal(), Nominal()
    detector.nom_box.set_payload(2.95, (0.01, -0.01, 0.02))
    detector.box_mass, detector.box_com = 2.95, (0.01, -0.01, 0.02)
    controller_shared = object()
    detector.ctrl = type("Controller", (), {"MODE": "torque",
                                              "shared_estimator": controller_shared})()
    detector.reset()
    assert detector is not controller_shared
    assert detector.normal_model is None


def test_ridge_variant_accepts_explicit_normal_only_parameters():
    parameters = {"mean": np.zeros(77), "scale": np.ones(77),
                  "weights": np.zeros((77, 2)), "threshold": 3.0}
    detector = make_detector({"variant": "prne_plus_ridge",
                              "ridge": {"servo": parameters}})()
    detector.nom, detector.nom_box = Nominal(), Nominal()
    detector.box_mass, detector.box_com = 0.0, (0.0, 0.0, 0.0)
    detector.ctrl = type("Controller", (), {"MODE": "servo"})()
    detector.reset()
    assert detector.normal_model is not None
    assert detector.normal_model.threshold == 3.0


class PRNETests(unittest.TestCase):
    def test_math_contract(self):
        test_math_contract()

    def test_reverse_transpose(self):
        test_transpose_does_not_call_basis_action()

    def test_detector_isolation(self):
        test_detector_is_independent_of_controller_shared_core()

    def test_explicit_ridge_parameters(self):
        test_ridge_variant_accepts_explicit_normal_only_parameters()


def main():
    tests = [test_math_contract, test_transpose_does_not_call_basis_action,
             test_detector_is_independent_of_controller_shared_core,
             test_ridge_variant_accepts_explicit_normal_only_parameters]
    for test in tests:
        test()
    output = verify_math()
    destination = ROOT / "results/v2_protocol/R2_PRNE_MATH.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(f"{len(tests)} tests passed; wrote {destination}")


if __name__ == "__main__":
    main()
