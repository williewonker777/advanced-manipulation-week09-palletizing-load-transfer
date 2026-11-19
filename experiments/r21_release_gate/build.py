"""Build the optional R20 detector as one default-name submission file."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
R10 = ROOT / "my_detector.py"
BRANCH = ROOT / "experiments/r20_horizontal/candidate_detector.py"
R10_SHA256 = "09f9a8573814a31b335b36ec5a78e0dedf262daa9039954568abd6bedd8dbc67"
BRANCH_SHA256 = "1770c36e002404a660c9cfc1ef45ddbb6cb1f53a3ec973c82b760e60881e7025"
R21_CANDIDATE = ROOT / "experiments/r21_pick_observer/candidate.py"
R21_MODEL = ROOT / "experiments/r21_pick_observer/model.json"
R21_CANDIDATE_SHA256 = "1f03ed4918fb279783813c8f1f769ee3a4476b1d973a765b2c5d81ce81e9e3e8"
R21_MODEL_SHA256 = "499b2577f659dff86bcc665fffc385adddadfe21824ef0e06c7d848caa49c6bf"
YAW_SOURCES = {
    "state.py": ROOT / "experiments/r21_yaw_impulse/state.py",
    "detector.py": ROOT / "experiments/r21_yaw_impulse/detector.py",
    "frozen.py": ROOT / "experiments/r21_yaw_impulse/frozen.py",
    "coarse.py": ROOT / "experiments/r21_yaw_impulse/coarse.py",
    "normal_bins.json": ROOT / "experiments/r21_yaw_impulse/normal_bins.json",
}
YAW_SHA256 = {
    "state.py": "f04d46e93c55e04aa869050fe43c895951f3a970e3729b01e8d94744532f3a9f",
    "detector.py": "69e1e884f393a17751546165554f994c19b0cb127830233fba1f2329481c87cb",
    "frozen.py": "71a75f131ea6d3aae94f9acf40c42abe92d9123000d86e685f8bb5f4a00a4342",
    "coarse.py": "f3a8a4202e638346dfa6a7994b056cbf509be4f335fb09c76d6657d47abcafe9",
    "normal_bins.json": "5f22f640eb8824f764ffeec7ecbef7ba23df02a874547b93b8ddc72925e25c15",
}
YAW_LOCK_MANIFEST_SHA256 = "a21d8536d407b3e470b4213c3dd2533e4082135ef6766928f26b8b4b518cc4dc"

EXTENSION = r'''
# Optional R20 early-transfer horizontal detector.  The complete preceding
# source is the unchanged R10 detector; no sibling module is needed at runtime.
_R20ReleaseBase = Detector
_R20_PHASES = ((1.35, 1.75, 18.3753578739),)
_R20_CLEAR_SAMPLES = 8


def _r20_shared_contract(shared):
    """Recognize the R10 controller-owned estimator by its documented protocol."""
    return bool(
        getattr(shared, "API", None) == "icpbl.shared-estimator.v1"
        and getattr(shared, "SOURCE", None) == "jts"
        and callable(getattr(shared, "reset", None))
        and callable(getattr(shared, "update", None))
        and np.asarray(getattr(shared, "r_joint", ())).shape == (6,)
        and np.asarray(getattr(shared, "bias", ())).shape == (6,)
        and hasattr(shared, "mass_prev")
        and getattr(shared, "nom", None) is not None
        and getattr(shared, "nom_box", None) is not None
        and isinstance(getattr(shared, "box_com", None), tuple)
    )


class Detector(_R20ReleaseBase):
    """R10 OR R20 only for a compatible controller-owned torque estimator."""

    SOURCE = "r10+r20-horizontal-optional"

    @staticmethod
    def horizontal_threshold(t):
        for start, stop, value in _R20_PHASES:
            if start <= t < stop:
                return value
        return None

    def reset(self):
        ctrl = getattr(self, "ctrl", None)
        shared = getattr(ctrl, "shared_estimator", None)
        registered = bool(
            type(ctrl).__name__ == "Controller"
            and type(ctrl).__module__ in ("my_torque", "torque_controller",
                                          "my_servo", "servo_controller")
        )
        compatible = registered and _r20_shared_contract(shared)
        # An arbitrary API string must not make base R10 delegate into a partial
        # object. Known servo/torque shared cores retain their exact R10 path.
        inherited = self.USE_SHARED
        if shared is not None and not compatible:
            self.USE_SHARED = False
        try:
            super().reset()
        finally:
            self.USE_SHARED = inherited
        self.horizontal_enabled = bool(
            compatible and getattr(ctrl, "MODE", None) == "torque"
            and self._shared is shared
        )
        self.horizontal_count = 0
        self.horizontal_clear = 0
        self.horizontal_detected = False
        self.horizontal_score = 0.0

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, baseline = super().update(t, th, thd, tau_m, tau_s, grip)
        threshold = self.horizontal_threshold(float(t)) if self.horizontal_enabled else None
        if threshold is None:
            over = False
            self.horizontal_score = 0.0
        else:
            horizontal = np.asarray(self._shared.last_residual_wrench[:2], float)
            self.horizontal_score = float(np.max(np.abs(horizontal)) / threshold)
            over = bool(self.horizontal_score > 1.0)
        self.horizontal_count = self.horizontal_count + 1 if over else 0
        self.horizontal_clear = 0 if over else self.horizontal_clear + 1
        if self.horizontal_count >= 1:
            self.horizontal_detected = True
        elif self.horizontal_clear >= _R20_CLEAR_SAMPLES:
            self.horizontal_detected = False
        return residual, bool(baseline or self.horizontal_detected)
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def r21_extension(document):
    literal = repr(document)
    return r"""
_R21ReleaseBase = Detector
_R21_MODEL = %s


def _r21_features(mass, com):
    m = float(mass)
    c = np.asarray(com, float)
    if c.shape != (3,) or not np.isfinite(c).all() or not np.isfinite(m):
        raise ValueError("known finite payload required")
    return np.r_[1., m, m*m, c, m*c]


class Detector(_R21ReleaseBase):
    # R20 plus the embedded normal-only R21 observer inside its fitted domain.

    SOURCE = "r10+r20-horizontal+r21-pick-observer-optional"

    def reset(self):
        super().reset()
        doc = _R21_MODEL
        if doc.get("schema") != "r21-pick-normal-model.v1" or not doc.get("normal_only"):
            raise ValueError("frozen embedded normal-only model required")
        self.pick_start, self.pick_stop = map(float, doc["window_s"])
        self.pick_dt = float(doc["dt_s"])
        self.pick_scale = np.asarray(doc["scale"], float)
        self.pick_coef = np.asarray(doc["coefficients"], float)
        self.pick_threshold = float(doc["threshold_N"])
        self.pick_persistence = int(doc["persistence_samples"])
        masses = np.asarray(doc["masses"], float)
        coms = np.asarray(doc["coms"], float)
        self.pick_feature = _r21_features(self.box_mass, self.box_com)
        center = np.asarray(self.box_com, float)
        self.pick_supported = bool(
            self.horizontal_enabled
            and masses.min() <= self.box_mass <= masses.max()
            and np.all(center >= coms.min(0))
            and np.all(center <= coms.max(0))
        )
        self.pick_history = []
        self.pick_count = self.pick_clear = 0
        self.pick_detected = False
        self.pick_error = np.zeros(2)
        self.pick_smoothed = np.zeros(2)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, baseline = super().update(t, th, thd, tau_m, tau_s, grip)
        active = self.pick_supported and self.pick_start <= float(t) < self.pick_stop
        if active:
            k = int(round((float(t) - self.pick_start) / self.pick_dt))
            k = max(0, min(k, len(self.pick_coef) - 1))
            predicted = np.einsum(
                "f,fa->a", self.pick_feature / self.pick_scale, self.pick_coef[k])
            self.pick_error = (
                np.asarray(self._shared.last_residual_wrench[:2], float) - predicted)
            self.pick_history.append(self.pick_error.copy())
            self.pick_history = self.pick_history[-4:]
            if len(self.pick_history) == 4:
                self.pick_smoothed = np.mean(self.pick_history, axis=0)
                over = bool(np.max(np.abs(self.pick_smoothed)) > self.pick_threshold)
            else:
                over = False
        else:
            self.pick_history = []
            self.pick_error.fill(0.)
            self.pick_smoothed.fill(0.)
            over = False
        self.pick_count = self.pick_count + 1 if over else 0
        self.pick_clear = 0 if over else self.pick_clear + 1
        if self.pick_count >= self.pick_persistence:
            self.pick_detected = True
        elif self.pick_clear >= 8:
            self.pick_detected = False
        return residual, bool(baseline or self.pick_detected)
""" % literal


def yaw_extension(document):
    table = repr(document["threshold_Nm"])
    return r'''
from collections import deque as _r21_deque

_R21YawBase = Detector
_R21_YAW_BOUNDS = np.asarray(%s, dtype=float)
_R21_YAW_GLOBAL = (3.23779270273884, 5.111942835466052)
_R21_YAW_AVERAGE = 12
_R21_YAW_SEPARATION = 32


class _R21YawBalance:
    def __init__(self):
        self.dt = float(C.DT)
        self.elastic = np.asarray(C.ELASTIC, bool)
        stiffness = C.K[self.elastic]
        damping = C.D[self.elastic]
        self.decay = damping / (damping + stiffness * self.dt)
        self.in_gain = self.dt / (damping + stiffness * self.dt)
        self.phi = None
        self.prev = None
        self.mass_row = None
        self.grip = None
        self.integral = 0.
        self.terms = _r21_deque(maxlen=_R21_YAW_AVERAGE)
        self.history = _r21_deque(maxlen=_R21_YAW_SEPARATION + 1)

    def feed(self, th, tau, mass_row, grip):
        measured = np.asarray(tau)[self.elastic]
        q = np.asarray(th, dtype=float).copy()
        if self.phi is None:
            self.phi = -measured / C.K[self.elastic]
        else:
            self.phi = self.decay * self.phi - self.in_gain * measured
        q[self.elastic] += self.phi
        row = np.asarray(mass_row, dtype=float)
        if self.prev is None or self.grip != bool(grip):
            self.terms.clear()
            self.history.clear()
            self.integral = 0.
            value = None
        else:
            momentum = float((self.mass_row + row) @ (q - self.prev)) / (2 * self.dt)
            self.integral += float(tau[0]) * self.dt
            self.terms.append(momentum - self.integral)
            value = None
            if len(self.terms) == _R21_YAW_AVERAGE:
                self.history.append(sum(self.terms) / _R21_YAW_AVERAGE)
                if len(self.history) > _R21_YAW_SEPARATION:
                    value = ((self.history[-1] - self.history[0]) /
                             (_R21_YAW_SEPARATION * self.dt))
        self.mass_row = row.copy()
        self.grip = bool(grip)
        self.prev = q
        return value


class Detector(_R21YawBase):
    # Optional coarse normal-only yaw balance for the recognized R10 torque core.
    SOURCE = "r10+r20-horizontal+r21-yaw-optional"

    def reset(self):
        super().reset()
        self.yaw_enabled = bool(self.horizontal_enabled)
        self.yaw = _R21YawBalance()
        self.yaw_value = None
        self.yaw_count = self.yaw_quiet = 0
        self.yaw_active = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, detected = super().update(t, th, thd, tau_m, tau_s, grip)
        if not self.yaw_enabled:
            self.yaw_value = None
            self.yaw_count = self.yaw_quiet = 0
            self.yaw_active = False
            return residual, bool(detected)
        packet = self._shared
        self.yaw_value = self.yaw.feed(
            th, tau_s - packet.bias, packet.mass_prev[0], grip)
        index = int(round(1000 * float(t))) // 100
        bounds = (_R21_YAW_BOUNDS[:, index] if 0 <= index < 60
                  else np.asarray(_R21_YAW_GLOBAL))
        if self.yaw_value is None or t < .08:
            self.yaw_count = self.yaw_quiet = 0
            self.yaw_active = False
        else:
            above = abs(self.yaw_value) > bounds[int(bool(grip))]
            self.yaw_count = self.yaw_count + 1 if above else 0
            self.yaw_quiet = 0 if above else self.yaw_quiet + 1
            if self.yaw_count >= 3:
                self.yaw_active = True
            elif self.yaw_quiet >= 8:
                self.yaw_active = False
        return residual, bool(detected or self.yaw_active)
''' % table


def build(output, variant="r20"):
    if output.exists() or output.is_symlink():
        raise ValueError("new regular output path required")
    if variant not in ("r20", "r21", "yaw"):
        raise ValueError("variant must be r20, r21 or yaw")
    base, branch = R10.read_bytes(), BRANCH.read_bytes()
    if sha(base) != R10_SHA256 or sha(branch) != BRANCH_SHA256:
        raise ValueError("frozen R10/R20 input differs")
    source = base.decode() + "\n" + EXTENSION
    payload = {
        "schema": "r21-r20-standalone-build.v1",
        "r10_detector_sha256": sha(base),
        "r20_branch_sha256": sha(branch),
    }
    if variant == "r21":
        candidate, model = R21_CANDIDATE.read_bytes(), R21_MODEL.read_bytes()
        if sha(candidate) != R21_CANDIDATE_SHA256 or sha(model) != R21_MODEL_SHA256:
            raise ValueError("frozen R21 input differs")
        document = json.loads(model)
        source += "\n" + r21_extension(document)
        payload.update(
            schema="r21-observer-standalone-build.v1",
            r21_candidate_sha256=sha(candidate),
            r21_model_sha256=sha(model),
        )
    elif variant == "yaw":
        yaw_blobs = {name: path.read_bytes() for name, path in YAW_SOURCES.items()}
        if {name: sha(blob) for name, blob in yaw_blobs.items()} != YAW_SHA256:
            raise ValueError("frozen yaw source differs")
        document = json.loads(yaw_blobs["normal_bins.json"])
        table = document.get("threshold_Nm")
        if (document.get("schema") != "r21-yaw-normal-bins.v1"
                or not isinstance(table, list)
                or len(table) != 2
                or any(not isinstance(row, list) or len(row) != 60
                       for row in table)):
            raise ValueError("invalid frozen yaw bounds")
        source += "\n" + yaw_extension(document)
        payload.update(
            schema="r21-yaw-standalone-build.v1",
            yaw_sources_sha256=YAW_SHA256,
            yaw_lock_manifest_sha256=YAW_LOCK_MANIFEST_SHA256,
        )
    ast.parse(source)
    output.write_text(source)
    payload.update(standalone_sha256=sha(source.encode()), lines=len(source.splitlines()))
    return payload


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--variant", choices=("r20", "r21", "yaw"), default="r20")
    args = parser.parse_args(argv)
    result = build(args.out, variant=args.variant)
    if args.receipt:
        if args.receipt.exists() or args.receipt.is_symlink():
            parser.error("new receipt path required")
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
