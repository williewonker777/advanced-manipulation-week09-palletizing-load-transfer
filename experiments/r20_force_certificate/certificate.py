"""Diagnose and exactly certify stock multidirectional force arithmetic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sim import config as C
from sim.runner import COLLISION_HOLD, COLLISION_RAMP


TWINS = ("none", "first", "second", "both")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def pulse_at(t, impact):
    if impact is None:
        return np.zeros(3)
    onset, vector = impact
    age = float(t) - float(onset)
    weight = min(age / COLLISION_RAMP, 1.0,
                 (2 * COLLISION_RAMP + COLLISION_HOLD - age) / COLLISION_RAMP)
    return max(weight, 0.0) * np.asarray(vector, float)


def exact_trace(case, twin):
    first, second = case["impacts"]
    use_first = first if twin in ("first", "both") else None
    use_second = second if twin in ("second", "both") else None
    t = C.DT * np.arange(6000)
    native = np.stack([pulse_at(sample, use_first) for sample in t])
    extra = np.stack([pulse_at(sample, use_second) for sample in t])
    combined = native + extra
    # This is the stock runner's exact order: one 3-vector norm per loop tick.
    scalar = np.fromiter((np.linalg.norm(row) for row in native),
                         dtype=float, count=len(native))
    return {"t": t, "native": native, "extra": extra,
            "combined": combined, "scalar": scalar}


def exact_equal(observed, expected):
    return all(np.array_equal(np.asarray(observed[key]), expected[key])
               for key in ("native", "extra", "combined", "scalar"))


def ulp_distance_positive(a, b):
    left = np.asarray(a, dtype=np.float64).view(np.uint64)
    right = np.asarray(b, dtype=np.float64).view(np.uint64)
    return np.where(left >= right, left - right, right - left)


def mismatch_rows(trace):
    batched = np.linalg.norm(trace["native"], axis=1)
    indices = np.flatnonzero(batched != trace["scalar"])
    rows = []
    ulps = ulp_distance_positive(batched[indices], trace["scalar"][indices])
    for index, distance in zip(indices, ulps):
        rows.append({
            "tick": int(index), "time_s": float(trace["t"][index]),
            "signed_xyz_N": trace["native"][index].tolist(),
            "stock_scalar_N": float(trace["scalar"][index]),
            "batched_scalar_N": float(batched[index]),
            "absolute_difference_N": float(abs(trace["scalar"][index] - batched[index])),
            "ulp_difference": int(distance),
        })
    return rows


def negative_controls(trace):
    expected = trace
    active = np.argwhere(trace["native"] != 0.)
    if not len(active):
        raise ValueError("active multidirectional force required")
    tick, axis = map(int, active[len(active) // 2])
    sign = {key: value.copy() for key, value in trace.items() if key != "t"}
    sign["native"][tick, axis] *= -1
    sign["combined"][tick, axis] *= -1

    tick_shift = {key: value.copy() for key, value in trace.items() if key != "t"}
    first_active = int(np.flatnonzero(np.any(trace["native"] != 0., axis=1))[0])
    for key in ("native", "combined", "scalar"):
        tick_shift[key][[first_active, first_active + 1]] = \
            tick_shift[key][[first_active + 1, first_active]]

    magnitude = {key: value.copy() for key, value in trace.items() if key != "t"}
    magnitude["scalar"][tick] = np.nextafter(magnitude["scalar"][tick], np.inf)
    result = {
        "changed_sign_rejected": not exact_equal(sign, expected),
        "changed_tick_rejected": not exact_equal(tick_shift, expected),
        "changed_force_rejected": not exact_equal(magnitude, expected),
    }
    if not all(result.values()):
        raise ValueError("fail-closed negative control accepted")
    return result


def certify_case(case, challenge_row):
    evidence = {}
    for arm, record in challenge_row["arms"].items():
        evidence[arm] = {
            "signed_and_scalar_exact": record["signed_and_scalar_exact"],
            "stock_scalar_all_twins_exact": all(
                record["scalar_forces"][twin]["stock_scalar_bit_exact"] for twin in TWINS),
            "failed_r18_gates": [key for key, value in record["r18_gates"].items()
                                   if not value],
        }
    if not all(row["signed_and_scalar_exact"] and row["stock_scalar_all_twins_exact"]
               for row in evidence.values()):
        raise ValueError("existing actual-log certificate is incomplete")
    traces = {twin: exact_trace(case, twin) for twin in TWINS}
    negatives = negative_controls(traces["both"])
    mismatches = {twin: mismatch_rows(trace) for twin, trace in traces.items()}
    diagnostics_match = {}
    for arm, record in challenge_row["arms"].items():
        diagnostics_match[arm] = {}
        for twin in TWINS:
            rows = mismatches[twin]
            existing = record["scalar_forces"][twin]
            maximum = max((row["absolute_difference_N"] for row in rows), default=0.0)
            diagnostics_match[arm][twin] = bool(
                len(rows) == existing["different_batched_samples"]
                and maximum == existing["max_batched_difference_N"])
    diagnostic_exact = all(value for arm in diagnostics_match.values()
                           for value in arm.values())
    has_reproduced_mismatch = bool(mismatches["first"] and mismatches["both"])
    return {
        "id": case["id"], "existing_actual_log_evidence": evidence,
        "stock_exact_reconstruction": True,
        "actual_log_and_reconstruction_diagnostics_match": diagnostics_match,
        "signed_force_and_stock_scalar_are_individually_exact": True,
        "batched_native_norm_is_the_force_certificate_mismatch": bool(
            diagnostic_exact and has_reproduced_mismatch),
        "batched_norm_mismatches": mismatches,
        "negative_controls": negatives,
        "certificate_passed": bool(
            diagnostic_exact and has_reproduced_mismatch and all(negatives.values())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--certificate-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("new output path required")
    try:
        challenge, challenge_plan, cert_plan = map(
            read, (args.challenge, args.plan, args.certificate_plan))
        if (cert_plan.get("schema") != "r20-force-certificate-plan.v1"
                or challenge.get("schema") != "r20-bridge-challenge-result.v1"
                or challenge.get("normal_only") is not False
                or challenge.get("input_sha256", {}).get("plan") != sha(args.plan)):
            raise ValueError("incompatible or unauthenticated inputs")
        cases = {case["id"]: case for case in challenge_plan["fresh"]}
        rows = {row["id"]: row for row in challenge["rows"]}
        ids = cert_plan["case_ids"]
        certificates = [certify_case(cases[name], rows[name]) for name in ids]
        if not all(row["certificate_passed"] for row in certificates):
            raise ValueError("force certificate failed")

        summary = {}
        for arm in challenge["arms"]:
            original = corrected = 0
            for row in challenge["rows"]:
                protocol = row["original_protocol"][arm]
                gates = row["arms"][arm]["r18_gates"]
                strict = protocol["exactly_one_per_slot"]
                original += bool(strict and all(gates.values()))
                amended = all(value for key, value in gates.items()
                              if key != "signed_profiles_exact")
                if row["id"] not in ids:
                    amended = amended and gates["signed_profiles_exact"]
                corrected += bool(strict and amended)
            summary[arm] = {
                "original_strict": original,
                "supplemental_conditional_strict": corrected,
                "denominator": len(challenge["rows"]),
            }
        result = {
            "schema": "r20-force-certificate-result.v1",
            "diagnostic_only": True,
            "does_not_change_original_r18_gates": True,
            "input_sha256": {"challenge": sha(args.challenge), "plan": sha(args.plan),
                             "certificate_plan": sha(args.certificate_plan)},
            "certificates": certificates, "conditional_summary": summary,
        }
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"cases": ids, "conditional_summary": summary}))


if __name__ == "__main__":
    main()
