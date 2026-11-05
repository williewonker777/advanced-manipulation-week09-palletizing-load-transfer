"""Materialize a self-contained R4 detector from frozen calibrations."""

from pathlib import Path
import argparse
import hashlib
import json
import pprint
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.production_audit import audit_text


PROTOCOL = ROOT / "results/v2_protocol/PROTOCOL_R4.json"
R1 = ROOT / "r1_detector.py"
KERNEL = ROOT / "experiments/r4_sequential.py"


def _canonical_specs():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    return {spec["id"]: spec for spec in protocol["candidate_grid"]}


def _validated(parameters):
    if not isinstance(parameters, dict) or set(parameters) != {"servo", "torque"}:
        raise ValueError("R4 parameters require exactly servo and torque entries")
    canonical = _canonical_specs()
    result = {}
    for mode in ("servo", "torque"):
        item = parameters[mode]
        if item is None:
            result[mode] = None
            continue
        if not isinstance(item, dict) or set(item) != {"spec", "calibration"}:
            raise ValueError("R4 mode entry requires spec and calibration")
        spec = item["spec"]
        if spec.get("id") not in canonical or spec != canonical[spec["id"]]:
            raise ValueError("R4 materialization requires an exact registered spec")
        calibration = item["calibration"]
        expected = 5 if spec["family"] == "CUSUM_PHASE" else 1
        if (not isinstance(calibration, dict)
                or set(calibration) != {"location", "scale", "threshold"}
                or any(not isinstance(calibration[name], list)
                       or len(calibration[name]) != expected
                       for name in ("location", "scale", "threshold"))):
            raise ValueError("R4 materialization calibration shape is invalid")
        numbers = [float(value) for name in ("location", "scale", "threshold")
                   for value in calibration[name]]
        if (any(not (value == value and abs(value) != float("inf")) for value in numbers)
                or any(float(value) <= 0.0 for value in calibration["scale"])
                or any(float(value) < 0.0 for value in calibration["threshold"])):
            raise ValueError("final materialized calibration must be finite and valid")
        result[mode] = item
    return result


def _wrapper(parameters):
    literal = pprint.pformat(parameters, width=100, sort_dicts=True)
    return f'''\n\n# R4 fixed normal-only calibration. No runtime fitting or file access.\nR4_PARAMETERS = {literal}\n\n\nclass R4Detector(Detector):\n    """R1 detector OR a fixed sequential normal-residual monitor."""\n\n    def reset(self):\n        super().reset()\n        mode = getattr(getattr(self, "ctrl", None), "MODE", None)\n        parameters = R4_PARAMETERS.get(mode)\n        self._r4_monitor = (None if parameters is None else\n                            SequentialMonitor(parameters["spec"], parameters["calibration"]))\n        self._r4_baseline_flag = False\n        self._r4_last_extra = False\n        self._r4_last_score = 0.0\n\n    def update(self, t, th, thd, tau_m, tau_s, grip):\n        residual, baseline = super().update(t, th, thd, tau_m, tau_s, grip)\n        self._r4_baseline_flag = bool(baseline)\n        monitor = self._r4_monitor\n        if monitor is None or self.normal_model is None:\n            self._r4_last_score = 0.0\n            self._r4_last_extra = False\n            return residual, bool(baseline)\n        active = outside_table_height_band(self._shared, grip)\n        family = monitor.spec["family"]\n        if active:\n            error = self.normal_model.last_error\n            score = float(np.linalg.norm(error))\n        elif family == "CUSUM_PHASE":\n            features = normal_features(t, self.box_mass, self.box_com)\n            normalized = np.clip((features - self.normal_model.mean) /\n                                 self.normal_model.scale, -8.0, 8.0)\n            prediction = normalized @ self.normal_model.weights\n            score = float(np.linalg.norm(np.asarray(self._shared.last_wrench[:2], float) -\n                                         prediction))\n        else:\n            score = 0.0\n        extra = monitor.update(t, score, active)\n        self._r4_last_score = score\n        self._r4_last_extra = bool(extra)\n        return residual, bool(baseline or extra)\n\n\nDetector = R4Detector\n'''


def materialize(parameters, destination):
    parameters = _validated(parameters)
    destination = Path(destination)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination = destination.resolve()
    if not destination.is_relative_to(ROOT) or destination.is_symlink():
        raise ValueError("R4 materialized detector must be a regular workcopy path")
    # Preserve the complete frozen R1 implementation as the exact prefix, then
    # append the small NumPy-only statistic and adapter.
    output = (R1.read_text(encoding="utf-8")
              + "\n\n" + KERNEL.read_text(encoding="utf-8")
              + _wrapper(parameters))
    audit = audit_text(output, destination.relative_to(ROOT).as_posix())
    if audit["errors"]:
        raise ValueError(audit["errors"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text(encoding="utf-8") != output:
        raise RuntimeError("refuse to replace an R4 detector with different bytes")
    destination.write_text(output, encoding="utf-8")
    return {"path": destination.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "parameters": parameters, "audit": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    parameters = json.loads(args.parameters.read_text(encoding="utf-8"))
    print(json.dumps(materialize(parameters, args.destination), indent=2))


if __name__ == "__main__":
    main()
