"""Append the frozen R23 post-detachment observer to the R22 LIGHT standalone."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import tempfile

from experiments.r22_export import build as r22


ROOT = Path(__file__).resolve().parents[2]
PINS = {
    "experiments/r23_detached/state.py": "b6175b317a05b231455ebd54e77705289ff56ddb491cba33872143e2be5539f7",
    "experiments/r23_detached/candidate.py": "49d283f94225b911476a3bc20c567ddab83afe8552a2817887c029e5f38c4912",
    "experiments/r23_detached/plan.json": "c51bdaa5cc9949ca6571311d32ca2538146bb62e4f7cb509685c65d6489de085",
}


R23 = r'''
_R23Light = Detector


class _R23OnsetGate:
    def __init__(self):
        self.quiet = 0
        self.armed = self.accepted = False

    def feed(self, xy, above, incumbent, bound):
        if xy is None:
            self.__init__(); return False
        if self.accepted:
            if above: return True
            self.accepted = self.armed = False; self.quiet = 0
        if incumbent:
            self.quiet = 0; self.armed = False
        elif _r22_math.hypot(*xy) < .8 * bound and not above:
            self.quiet += 1
            if self.quiet >= 3: self.armed = True
        else:
            self.quiet = 0
        if above and self.armed:
            self.accepted = True; self.armed = False
        return self.accepted


class _R23ReleasedForce:
    def __init__(self, mass, empty, loaded, require_onset):
        self.enabled = _r22_math.isfinite(mass) and 0 <= mass <= 8
        self.empty, self.loaded = empty, loaded
        self.require_onset = require_onset
        self.calibrator = self._new() if self.enabled else None
        self.bias = self.prev_grip = self.observer = None
        self.evidence = _R22Evidence(); self.gate = _R23OnsetGate(); self.vector = None

    def _new(self):
        return _R22NormalForce(self.empty, self.loaded)

    def feed(self, t, encoder, jts, grip, incumbent):
        grip = bool(grip)
        detach = self.prev_grip is True and not grip
        self.prev_grip = grip; self.vector = None
        if not self.enabled: return False
        if self.calibrator is not None:
            if t < .5 and not grip:
                self.calibrator.update(t, encoder, jts, False, project=False)
                if self.calibrator.bias is not None:
                    self.bias = self.calibrator.bias.copy(); self.calibrator = None
            else:
                self.calibrator = None
            return False
        if grip:
            self.observer = None; return False
        if detach and self.bias is not None:
            self.observer = self._new(); self.observer.bias = self.bias.copy()
            self.evidence = _R22Evidence(); self.gate = _R23OnsetGate()
        if self.observer is None: return False
        sample = self.observer.update(t, encoder, jts, False)
        self.vector = None if sample is None else sample.force_n[:2]
        bound = _R22_TANGENT_BOUNDS[0]
        above = self.evidence.feed(self.vector, 0, bound)
        return self.gate.feed(self.vector, above, incumbent, bound) if self.require_onset else above


class Detector(_R23Light):
    SOURCE = "r10+r22-light+r23-post-detachment-research"
    require_onset = %s

    def reset(self):
        super().reset()
        self.detached = (_R23ReleasedForce(self.box_mass, self.nom, self.nom_box, self.require_onset)
                         if self.r22_enabled else None)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        if not self.r22_enabled: return residual, bool(incumbent)
        extra = self.detached.feed(t, th, tau_s, grip, incumbent)
        return residual, bool(incumbent or extra)
'''


def sha(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    return hashlib.sha256(value).hexdigest()


def verify_inputs():
    r22.verify_inputs()
    if {name: sha(ROOT / name) for name in PINS} != PINS:
        raise ValueError("frozen R23 export inputs differ")


def build(output, variant="onset"):
    output = Path(output)
    if variant not in ("onset", "free"):
        raise ValueError("variant must be onset or free")
    if output.exists() or output.is_symlink():
        raise ValueError("new regular output path required")
    verify_inputs()
    with tempfile.TemporaryDirectory(prefix="r23-base-standalone-", dir=output.parent) as tmp:
        base = Path(tmp) / "r22.py"
        r22_receipt = r22.build(base, "light")
        source = base.read_text() + "\n" + (R23 % repr(variant == "onset"))
    ast.parse(source)
    output.write_text(source)
    verify_inputs()
    return {"schema": "r23-standalone-build.v1", "variant": variant,
            "base_r22_standalone_sha256": r22_receipt["standalone_sha256"],
            "frozen_r22_sources_sha256": r22_receipt["frozen_sources_sha256"],
            "frozen_r23_sources_sha256": PINS,
            "standalone_sha256": sha(source.encode()), "lines": len(source.splitlines())}
