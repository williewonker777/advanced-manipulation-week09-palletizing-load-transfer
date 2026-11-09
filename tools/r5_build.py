"""Build four self-contained frozen R5 detector candidates without simulation."""

from pathlib import Path
import argparse
import hashlib
import json
import pprint
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.production_audit import audit_text


PROTOCOL = ROOT / "results/v2_protocol/PROTOCOL_R5.json"
CORE = ROOT / "experiments/r5_core.py"
BASE_R1 = ROOT / "r1_detector.py"
GENERATED = ROOT / "experiments/r5_generated"
BUILD_RECORD = ROOT / "results/v2_protocol/R5_BUILD.json"
CONFIG_IDS = ("SCALE_PHASE", "SCALE_STATE", "AR20_FULL", "AR50_FULL")
PROTOCOL_SHA256 = "cfa8d71637c0546df0bd6618d2b13403278c2190df79437ba7a221cea764f6f8"
CORE_EXTERNAL_IMPORT = "from r1_detector import normal_features, MODEL_PARAMETERS\n"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _protocol():
    if _sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("PROTOCOL_R5 bytes differ from the registered protocol")
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    return value, {item["id"]: item for item in value["candidate_grid"]}


def _registered_payloads(config_id):
    _, candidates = _protocol()
    if config_id not in candidates or config_id not in CONFIG_IDS:
        raise ValueError("unknown registered R5 configuration")
    candidate = candidates[config_id]
    payloads = {}
    records = []
    for mode in ("servo", "torque"):
        registration = candidate["models"][mode]
        path = ROOT / registration["path"]
        digest = _sha(path)
        if digest != registration["sha256"]:
            raise ValueError("registered R5 model bytes changed: " + path.name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("mode") != mode or payload.get("spec") != payload.get("model", {}).get("spec")
                or payload.get("threshold") != registration["threshold"]
                or payload.get("model", {}).get("mode") != mode):
            raise ValueError("registered R5 model payload is inconsistent")
        payloads[mode] = payload
        records.append({
            "config": config_id, "mode": mode,
            "path": path.relative_to(ROOT).as_posix(), "sha256": digest,
            "threshold": payload["threshold"],
        })
    return payloads, records


def _compact_and_validate(payloads, config_id=None):
    if not isinstance(payloads, dict) or set(payloads) != {"servo", "torque"}:
        raise ValueError("payloads require exactly servo and torque")
    spec_ids = {payloads[mode].get("spec", {}).get("id") for mode in payloads}
    if len(spec_ids) != 1:
        raise ValueError("servo and torque payload specs differ")
    _, candidates = _protocol()
    matches = [item["id"] for item in candidates.values()
               if item["normal_spec_id"] in spec_ids]
    inferred = matches[0] if len(matches) == 1 else None
    if config_id is None:
        config_id = inferred
    if config_id not in CONFIG_IDS or config_id != inferred:
        raise ValueError("payload is not one registered runtime configuration")
    registered, _ = _registered_payloads(config_id)
    compact = {}
    for mode in ("servo", "torque"):
        supplied = payloads[mode]
        expected = registered[mode]
        if (supplied.get("mode") != expected["mode"]
                or supplied.get("threshold") != expected["threshold"]
                or supplied.get("model") != expected["model"]):
            raise ValueError("runtime payload differs from frozen model/threshold")
        compact[mode] = {
            "mode": expected["mode"], "model": expected["model"],
            "threshold": expected["threshold"],
        }
    return config_id, compact


def _wrapper(payloads, config_id):
    literal = pprint.pformat(payloads, width=110, sort_dicts=True)
    return f'''

# Frozen R5 normal-only payloads. No runtime fitting, adaptation, or file access.
R5_CONFIG_ID = {config_id!r}
R5_PAYLOADS = {literal}

_R1Detector = Detector
_R1NormalResidualModel = NormalResidualModel


class _R5TrackedNormalResidualModel(_R1NormalResidualModel):
    """R1 model with a passive record of the exact active argument."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_active = False

    def update(self, t, mass, com, wrench, active):
        self.last_active = bool(active)
        return super().update(t, mass, com, wrench, active)


class Detector(_R1Detector):
    """Unchanged R1 detector OR one private fixed R5 monitor."""

    def reset(self):
        super().reset()
        mode = getattr(getattr(self, "ctrl", None), "MODE", None)
        original_normal_enabled = self.normal_model is not None
        if original_normal_enabled and mode in MODEL_PARAMETERS:
            self.normal_model = _R5TrackedNormalResidualModel(**MODEL_PARAMETERS[mode])
        payload = R5_PAYLOADS.get(mode)
        self._r5_monitor = (FixedMonitor(payload)
                            if original_normal_enabled and payload is not None else None)
        self.r5_base_detect = False
        self.r5_extra = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, baseline = super().update(t, th, thd, tau_m, tau_s, grip)
        self.r5_base_detect = bool(baseline)
        monitor = self._r5_monitor
        if monitor is None:
            self.r5_extra = False
            return residual, bool(baseline)
        packet = self._shared
        self.r5_extra = monitor.update(
            t, self.box_mass, self.box_com, th, packet.r_joint,
            packet.last_wrench, self.normal_model.last_active, grip)
        return residual, bool(baseline or self.r5_extra)
'''


def _render(payloads, config_id=None):
    config_id, compact = _compact_and_validate(payloads, config_id)
    core = CORE.read_text(encoding="utf-8")
    if core.count(CORE_EXTERNAL_IMPORT) != 1:
        raise ValueError("r5_core must contain exactly one standalone R1 import")
    core = core.replace(CORE_EXTERNAL_IMPORT, "")
    output = BASE_R1.read_text(encoding="utf-8") + "\n\n" + core + _wrapper(compact, config_id)
    audit = audit_text(output, "experiments/r5_generated/" + config_id + ".py")
    if audit["errors"]:
        raise ValueError(audit["errors"])
    return output, config_id, compact, audit


def materialize(payloads, target, config_id=None):
    """Write one registered self-contained candidate and return its digest."""
    output, config_id, compact, audit = _render(payloads, config_id)
    target = Path(target)
    if not target.is_absolute():
        target = ROOT / target
    target = target.resolve()
    if not target.is_relative_to(ROOT) or target.is_symlink():
        raise ValueError("R5 generated target must be a regular workcopy path")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(output, encoding="utf-8")
    return {
        "config": config_id, "path": target.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(output.encode()).hexdigest(),
        "payload_modes": sorted(compact), "audit": audit["status"],
    }


def _build_record(generated_records, model_records):
    return {
        "schema": "R5_BUILD.v1",
        "protocol": {"path": PROTOCOL.relative_to(ROOT).as_posix(), "sha256": _sha(PROTOCOL)},
        "core": {"path": CORE.relative_to(ROOT).as_posix(), "sha256": _sha(CORE)},
        "builder": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                    "sha256": _sha(Path(__file__).resolve())},
        "base_R1": {"path": BASE_R1.relative_to(ROOT).as_posix(), "sha256": _sha(BASE_R1)},
        "models": model_records,
        "generated": generated_records,
        "runtime_file_reads": 0,
        "weights_or_threshold_updates": 0,
    }


def build_all(check=False):
    generated_records = []
    model_records = []
    expected_outputs = {}
    for config_id in CONFIG_IDS:
        payloads, records = _registered_payloads(config_id)
        model_records.extend(records)
        output, _, compact, audit = _render(payloads, config_id)
        target = GENERATED / (config_id + ".py")
        expected_outputs[target] = output
        generated_records.append({
            "config": config_id, "path": target.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(output.encode()).hexdigest(),
            "payload_modes": sorted(compact), "audit": audit["status"],
        })
    record = _build_record(generated_records, model_records)
    record_text = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    if check:
        differences = []
        for path, output in expected_outputs.items():
            if not path.is_file() or path.read_text(encoding="utf-8") != output:
                differences.append(path.relative_to(ROOT).as_posix())
        if not BUILD_RECORD.is_file() or BUILD_RECORD.read_text(encoding="utf-8") != record_text:
            differences.append(BUILD_RECORD.relative_to(ROOT).as_posix())
        if differences:
            raise RuntimeError("R5 build differs: " + ", ".join(differences))
        return record
    GENERATED.mkdir(parents=True, exist_ok=True)
    for path, output in expected_outputs.items():
        path.write_text(output, encoding="utf-8")
    BUILD_RECORD.write_text(record_text, encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        record = build_all(check=args.check)
    except (ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
