"""Build the final Week09 hand-in: R1 controllers + one dispatching detector file.

Detector selection at ``reset()``:
  * supplied torque controller (MODE='torque' + shared-estimator contract) -> R31 held identity
  * supplied servo controller  (MODE='servo'  + shared-estimator contract) -> R32 servo ownership
  * any other controller                                                  -> unchanged R10

The torque linkage is reused byte-for-byte from ``tools/build_combined_submission.py``
(the verified R10-servo/R31-torque combination, detector SHA-256 ``f1c8c3be...``).
Only the servo linkage and the dispatch class are new; no detection rule, threshold
or model is changed.  Usage::

    python -B tools/build_final_submission.py --out <new empty directory>
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import build_combined_submission as base  # noqa: E402

CODEX_COMBINED_DETECTOR = 'f1c8c3be5ebfe4fdcda61152b25daef01ba4ea836ee61745faa264679e787e56'
R32_ENTRY = 'experiments.r32_servo_ownership.candidate'
SERVO_PINS = {
    'experiments/r32_servo_ownership/candidate.py': '6c5cfca6b6f8d387a6b2c22a1ca41bf20c3f6df6f238add21f01a8f2b4c88d7f',
    'experiments/r31_servo_release/candidate.py': '2c1af9274f310dd84a9a8909987be7fe41e28fd89faf477be07087a7c2738303',
    'experiments/r15_delayed_wrench/observer.py': 'eff8f0b76d8c0358bb15b020be0db6e0b55d0e666a93ee2f46a1de6190ba2483',
}
ZIP_NAME = 'week09-final-r32-servo-r31-torque.zip'
OLD_HEAD = '# Week09 combined hand-in: unchanged R10 body below; R31 linkage and mode adapter follow.\n'
NEW_HEAD = ('# Week09 final hand-in: unchanged R10 body below; R31 torque linkage, R32 servo linkage\n'
            '# and the mode adapter follow.\n')

DISPATCH = '''
class Detector:
    """Course adapter: R31 with the supplied torque controller, R32 with the supplied
    servo controller, unchanged R10 for any other controller.

    The simulator supplies the controller and nominal models before reset.
    Reset selects one method; each physical sample updates that method once.
    """
    SOURCE = _submission_r10.Detector.SOURCE
    API = _submission_r10.Detector.API

    def __init__(self, nom=None, nom_box=None, box_mass=0.0, box_com=(0.0, 0.0, 0.0)):
        self._impl = None
        self.box_mass = float(box_mass)
        self.box_com = tuple(box_com)
        if nom is not None and nom_box is not None:
            self.nom, self.nom_box = nom, nom_box
            self.reset()

    def reset(self):
        ctrl = vars(self).get('ctrl')
        shared = getattr(ctrl, 'shared_estimator', None)
        contract = (getattr(shared, 'API', None) == self.API
                    and getattr(shared, 'SOURCE', None) == self.SOURCE
                    and callable(getattr(shared, 'update', None))
                    and callable(getattr(shared, 'reset', None)))
        mode = getattr(ctrl, 'MODE', None)
        if contract and mode == 'torque':
            cls, method = _submission_torque_class(), 'r31-torque'
        elif contract and mode == 'servo':
            cls, method = _submission_servo_class(), 'r32-servo'
        else:
            cls, method = _submission_r10.Detector, 'r10'
        engine = cls()
        engine.nom, engine.nom_box, engine.ctrl = self.nom, self.nom_box, ctrl
        engine.box_mass, engine.box_com = self.box_mass, self.box_com
        engine.reset()
        self._impl = engine
        self.selected_method = method
        # No routing, wrapper allocation or second update inside the 1 ms loop.
        self.update = engine.update

    def update(self, t, th, thd, tau_m, tau_s, grip):
        raise RuntimeError('attach course models and call reset() before the first sample')

    def __getattr__(self, name):
        impl = vars(self).get('_impl')
        if impl is None:
            raise AttributeError(name)
        return getattr(impl, name)
'''


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def graph(files, entries):
    """Same traversal as ``base.graph`` for an explicit entry list."""
    done, visiting, order = set(), set(), []
    exports = defaultdict(set)
    for entry in entries:
        exports[entry].add('Detector')

    def visit(name):
        if name in done:
            return
        if name in visiting:
            raise ValueError('dependency cycle')
        visiting.add(name)
        tree = ast.parse(files[base.path_for(name)])
        for node in ast.walk(tree):
            if base.inside(node):
                for local, source, item in base.module_edges(node, files):
                    if item is None:
                        attrs = {v.attr for v in ast.walk(tree) if isinstance(v, ast.Attribute)
                                 and isinstance(v.value, ast.Name) and v.value.id == local}
                        if not attrs:
                            raise ValueError('unscoped module import: ' + source)
                        exports[source].update(attrs)
                    else:
                        exports[source].add(item)
                    visit(source)
        visiting.remove(name)
        done.add(name)
        order.append(name)

    for entry in entries:
        visit(entry)
    return order, exports


def materialize():
    blobs, codex_record = base.materialize()
    codex_detector = blobs['my_detector.py'].decode()
    if digest(codex_detector.encode()) != CODEX_COMBINED_DETECTOR:
        raise ValueError('R10/R31 combination differs from the verified build')
    if not codex_detector.endswith(base.DISPATCH) or not codex_detector.startswith(OLD_HEAD):
        raise ValueError('unexpected R10/R31 detector layout')
    _, files, _ = base.frozen.load()
    files = dict(files)
    for name, pin in SERVO_PINS.items():
        data = (ROOT / name).read_bytes()
        if digest(data) != pin:
            raise ValueError('changed servo source: ' + name)
        if name in files and files[name] != data:
            raise ValueError('servo source differs from the R31 closure copy: ' + name)
        files[name] = data
    order, exports = graph(files, [R32_ENTRY])
    if {base.path_for(m) for m in order} != set(SERVO_PINS):
        raise ValueError('unexpected servo closure: ' + repr(order))
    sections = [base.section(m, files, exports[m]) for m in order]
    servo_body = '\n'.join(sections) + '\nreturn ' + base.symbol(R32_ENTRY) + '.Detector\n'
    servo_glue = ('\n\n# --- Fixed, authored R32 servo inference (loaded only with the supplied servo '
                  'controller). ---\n'
                  '@_submission_cache(maxsize=1)\ndef _submission_servo_class():\n'
                  + textwrap.indent(servo_body, '    ') + '\n')
    detector = NEW_HEAD + codex_detector[len(OLD_HEAD):-len(base.DISPATCH)] + servo_glue + DISPATCH
    compile(detector, 'my_detector.py', 'exec')
    out = dict(blobs)
    out['my_detector.py'] = detector.encode()
    record = dict(
        schema='week09-final-combination.v1',
        selected={'servo_controller': 'R1 (my_servo.py)', 'torque_controller': 'R1 (my_torque.py)',
                  'servo_detector': 'R32 servo ownership', 'torque_detector': 'R31 held identity',
                  'fallback_detector': 'R10'},
        controllers=base.RELEASE,
        codex_r10_r31_detector_sha256=CODEX_COMBINED_DETECTOR,
        codex_r31_record={k: codex_record[k] for k in ('r31_entry', 'r31_source', 'r31_closure',
                                                       'modules', 'normal_tables')},
        r32_entry=R32_ENTRY + ':Detector', servo_sources=SERVO_PINS, servo_modules=order,
        detector_sha256=digest(detector.encode()), new_detection_rules=False)
    return out, record


def build(out: Path):
    if out.exists() and (out.is_symlink() or not out.is_dir() or any(out.iterdir())):
        raise ValueError('choose an empty or new directory')
    blobs, receipt = materialize()
    out.mkdir(parents=True, exist_ok=True)
    for name, b in blobs.items():
        (out / name).write_bytes(b)
    z = out / ZIP_NAME
    base.write_zip(z, blobs)
    with zipfile.ZipFile(z) as a:
        if a.testzip() or sorted(a.namelist()) != sorted(blobs) or any(a.read(n) != v for n, v in blobs.items()):
            raise ValueError('archive differs from built sources')
    receipt.update(zip=z.name, zip_sha256=digest(z.read_bytes()),
                   members={n: digest(v) for n, v in blobs.items()})
    (out / 'build.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    rec = build(p.parse_args().out)
    print(rec['zip_sha256'], rec['zip'], rec['members'])
