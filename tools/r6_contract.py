"""Check exact submitted filenames and public controller/detector interfaces."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
from types import SimpleNamespace

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_class(path, name, label):
    spec = importlib.util.spec_from_file_location(label, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def check(root, expected_detector):
    root = Path(root).resolve()
    expected_detector = Path(expected_detector).resolve()
    sys.path.insert(0, str(root))
    from sim.nominal import Nominal
    from sim.drive import Drive
    from sim import trajectory as trajectory
    from tools.check_integrity import check_integrity

    protected = check_integrity(root)
    expected_hash = sha(expected_detector)
    for name in ('my_detector.py', 'collision_detector.py'):
        if (root / name).is_symlink() or sha(root / name) != expected_hash:
            raise ValueError('submission detector alias differs: ' + name)
    for mode in ('servo', 'torque'):
        for name in ('my_' + mode + '.py', mode + '_controller.py'):
            if (root / name).is_symlink() or sha(root / name) != sha(root / ('r1_' + mode + '.py')):
                raise ValueError('controller is not the exact fixed R1 source: ' + name)

    def models(instance):
        instance.nom = Nominal()
        instance.nom_box = Nominal()
        instance.box_mass = 3.25
        instance.box_com = (0.02, -0.01, 0.03)
        instance.nom_box.set_payload(instance.box_mass, instance.box_com)

    controllers = {}
    for mode in ('servo', 'torque'):
        cls = load_class(root / (mode + '_controller.py'), 'Controller', 'contract_ctrl_' + mode)
        ctrl = cls()
        models(ctrl)
        if mode == 'servo':
            ctrl.drive = Drive()
        ctrl.reset()
        controllers[mode] = ctrl

    detector_class = load_class(root / 'collision_detector.py', 'Detector', 'unrelated_submission_detector')
    cases = [('servo', controllers['servo']), ('torque', controllers['torque']),
             ('None', SimpleNamespace(MODE=None)), ('unknown', SimpleNamespace(MODE='unknown')),
             ('substring', SimpleNamespace(MODE='ser')), ('missing_MODE', object()),
             ('controller_None', None), ('controller_absent', None)]
    candidate = expected_hash != sha(root / 'r1_detector.py')
    results = []
    for label, controller in cases:
        detector = detector_class()
        models(detector)
        if label != 'controller_absent':
            detector.ctrl = controller
        detector.reset()
        enabled = getattr(detector, '_r5_monitor', None) is not None
        if enabled != (candidate and label == 'servo'):
            raise ValueError('additional channel dispatch differs: ' + label)
        for index in range(3):
            t = index * .001
            q, qd, qdd = trajectory.reference(t)
            torque = detector.nom.gravity(q)
            theta = q.copy()
            elastic = detector.nom.elastic
            theta[elastic] += torque[elastic] / detector.nom.K[elastic]
            residual, alarm = detector.update(t, theta, np.zeros(6), torque, torque, False)
            if np.asarray(residual).shape != (6,) or not np.isfinite(residual).all():
                raise ValueError('nonfinite or invalid residual: ' + label)
            if not isinstance(alarm, (bool, np.bool_)):
                raise ValueError('invalid alarm type: ' + label)
            if label in controllers:
                command = np.asarray(controller.compute(t, theta, np.zeros(6), torque, False, q, qd, qdd))
                shape = (4, 6) if label == 'servo' else (6,)
                if command.shape != shape or not np.isfinite(command).all():
                    raise ValueError('invalid default controller command: ' + label)
        results.append({'case': label, 'additional_channel_enabled': enabled,
                        'finite_updates': 3, 'status': 'PASS'})
    if check_integrity(root) != protected:
        raise ValueError('protected input changed during interface verification')
    return {'schema': 'R6_SUBMISSION_CONTRACT.v1', 'status': 'PASS',
            'detector_sha256': expected_hash, 'default_names_exact': True,
            'controller_sha256': {mode: sha(root / ('my_' + mode + '.py')) for mode in controllers},
            'cases': results, 'protected': protected,
            'scope': 'Reset/public API/finite synthetic updates; no Plant or physical simulation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-detector', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.root, args.expected_detector)
    with args.out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('submission contract PASS:', len(result['cases']), 'interface cases')


if __name__ == '__main__':
    main()
