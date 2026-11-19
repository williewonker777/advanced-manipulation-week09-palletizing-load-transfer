"""Reproduce the passive R13 grasp-height audit on an authenticated stock kit.

The estimator sees only the encoder, joint-torque sensor, grip state, known
payload and public nominal model through 3.55 s.  Free-box state is retained in
memory until every prediction for a mode has been made, then used only for the
offline bottom-face score.  The output contains derived calibration and score
records, never raw sensor captures.

This portable audit deliberately uses ``sim.runner.run_episode``.  That public
interface accepts ``pick_offset`` but not the private research fixture's drive
perturbations.  Consequently the 42 episodes preserve the frozen masses, COMs,
offsets and calibration/evaluation split, but are a stock-plant reproduction,
not numeric parity with the private perturbed study.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('zero', 'xy', 'up', 'down')


def digest_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def digest(path):
    return digest_bytes(path.read_bytes())


def snapshot_files(paths):
    """Read each input once so reported and executed bytes have one identity."""
    blobs = {key: path.read_bytes() for key, path in paths.items()}
    return blobs, {key: digest_bytes(blob) for key, blob in blobs.items()}


def install_snapshot(blob, pin, destination):
    destination.write_bytes(blob)
    if digest(destination) != pin:
        raise ValueError('assembly checksum mismatch: ' + destination.name)


def validate_paths(kit_arg, out_arg, parser=None):
    """Require an original regular kit and a genuinely new external output."""
    fail = parser.error if parser is not None else lambda message: (_ for _ in ()).throw(ValueError(message))
    if kit_arg.is_symlink() or not kit_arg.is_dir():
        fail('regular ORIGINAL kit directory required')
    try:
        kit = kit_arg.resolve(strict=True)
    except OSError:
        fail('regular ORIGINAL kit directory required')
    if any(not (kit / name).exists() for name in ('sim', 'model', 'evaluate.py')):
        fail('incomplete ORIGINAL kit')
    if out_arg.is_symlink() or out_arg.exists():
        fail('NEW output path must not exist or be a symlink')
    parent = out_arg.parent.resolve()
    if not parent.is_dir() or out_arg.resolve().is_relative_to(kit):
        fail('NEW output must have an existing parent outside ORIGINAL kit')
    return kit, out_arg.resolve()


def offset(group, variant):
    if variant == 'zero':
        return None
    x, y, z = map(float, group['shift'])
    return [x, y, {'xy': 0.0, 'up': z, 'down': -z}[variant]]


def validate_plan(plan):
    if (plan.get('schema') != 'r13-carried-datum.v1' or len(plan.get('fit', ())) != 5
            or len(plan.get('test', ())) != 4):
        raise ValueError('unexpected frozen R13 plan shape')
    ids = []
    for split in ('fit', 'test'):
        for case in plan[split]:
            ids.append((split, case.get('id')))
            mass = float(case['mass'])
            com = np.asarray(case['com'], dtype=float)
            if not np.isfinite(mass) or mass <= 0 or com.shape != (3,) or not np.isfinite(com).all():
                raise ValueError('invalid payload in frozen plan')
            if split == 'test':
                shift = np.asarray(case['shift'], dtype=float)
                if shift.shape != (3,) or not np.isfinite(shift).all():
                    raise ValueError('invalid pickup displacement in frozen plan')
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate case id in frozen plan')


def capture(cls):
    """Record exactly the public samples accepted by the frozen estimator."""
    samples = {name: [] for name in ('t', 'encoder', 'jts', 'grip')}

    class Record(cls):
        def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
            if t <= 3.55:
                for name, value in zip(samples, (float(t), th.copy(), tau_s.copy(), bool(grip))):
                    samples[name].append(value)
            return super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)

    return Record(), samples


def flange(nom, q):
    import mujoco
    nom.d.qpos[:] = q
    mujoco.mj_kinematics(nom.m, nom.d)
    body = nom.m.body('link_6').id
    return nom.d.xpos[body].copy(), nom.d.xmat[body].reshape(3, 3).copy()


def score_capsule(log):
    """Retain four score-only states without keeping an entire simulator log."""
    from sim import config as c

    address = log['plant'].m.jnt_qposadr[log['plant'].m.joint('box_free').id]
    indices = np.array([round(time_s / c.DT) - 1 for time_s in (2.05, 2.48, 3.04, 3.46)])
    return {'q': log['q'][indices].copy(),
            'box_qpos': log['qpos'][indices, address:address + 7].copy()}


def observed_geometry(capsule, com):
    """Offline truth: actual box bottom projected to the taught placement pose."""
    import mujoco
    from sim import config as c, trajectory
    from sim.nominal import Nominal

    nominal = Nominal()
    _, placed = flange(nominal, trajectory.reference(trajectory.T_RELEASE)[0])
    half = np.asarray(c.BOX_SIZE) * 0.5
    taught_box = np.asarray(c.BOX_POS)
    taught_bottom = placed[2] @ taught_box - abs(placed[2]) @ half
    rows = []
    for q, box in zip(capsule['q'], capsule['box_qpos']):
        position, rotation = flange(nominal, q)
        quaternion = box[3:] / np.linalg.norm(box[3:])
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quaternion)
        relative_rotation = rotation.T @ matrix.reshape(3, 3)
        local_box = rotation.T @ (box[:3] - position)
        local_com = local_box + relative_rotation @ np.asarray(com)
        bottom = placed[2] @ local_box - abs(placed[2] @ relative_rotation) @ half
        rows.append(np.r_[(local_com - taught_box - com) * 1000.0,
                          (bottom - taught_bottom) * 1000.0])
    rows = np.asarray(rows)
    median = np.median(rows, axis=0)
    return {
        'local_com_shift_mm': median[:3].tolist(),
        'bottom_shift_at_placement_mm': float(median[3]),
        'sample_span_mm': np.ptp(rows, axis=0).tolist(),
    }


def compact_metrics(metrics):
    fields = ('rms_mm', 'max_mm', 'diverged', 'saturation_pct', 'torque_jitter',
              'false_alarms', 'detect_delay_ms', 'alarm_before_collision',
              'tool_contact_delay_ms')
    return {name: metrics[name] for name in fields if name in metrics}


def public_features(theory, samples, case):
    arrays = {name: np.asarray(value) for name, value in samples.items()}
    return theory.public_features(arrays['t'], arrays['encoder'], arrays['jts'], arrays['grip'],
                                  float(case['mass']), np.asarray(case['com']))


def serial_model(model):
    return {name: np.asarray(value).tolist() for name, value in model.items()}


def scores(rows):
    vertical = [row for row in rows if row['variant'] in ('up', 'down')]
    controls = [row for row in rows if row['variant'] in ('zero', 'xy')]
    rmse = lambda key: float(np.mean([row[key] ** 2 for row in vertical]) ** 0.5)
    result = {
        'height_sign_correct': int(sum(np.sign(row['extra_height_mm']) ==
                                       np.sign(row['true_extra_mm']) for row in vertical)),
        'height_confident': int(sum(row['accept'] for row in vertical)),
        'accepted_height_sign_correct': int(sum(row['accept'] and
                np.sign(row['extra_height_mm']) == np.sign(row['true_extra_mm'])
                for row in vertical)),
        'vertical_count': len(vertical),
        'vertical_rmse_mm': rmse('error_mm'),
        'paired_height_rmse_mm': rmse('paired_error_mm'),
        'true_required_zero_action_rmse_mm': float(
            np.mean([row['true_extra_mm'] ** 2 for row in vertical]) ** 0.5),
        'control_max_abs_error_mm': max(abs(row['error_mm']) for row in controls),
        'control_count': len(controls),
        'accepted_not_covered': int(sum(row['accept'] and not row['covered'] for row in rows)),
    }
    result['qualified'] = bool(
        result['accepted_height_sign_correct'] == len(vertical)
        and result['height_confident'] >= 6
        and result['vertical_rmse_mm'] <= 2.0
        and result['control_max_abs_error_mm'] <= 2.0
        and result['accepted_not_covered'] == 0)
    return result


def run_mode(mode, plan):
    """Run one 21-episode mode inside an isolated, source-pinned overlay."""
    from sim import config as c
    from sim.runner import run_episode
    import collision_detector
    import grasp_estimator as theory

    validate_plan(plan)
    controller = importlib.import_module(mode + '_controller').Controller
    if getattr(controller, 'MODE', None) != mode:
        raise ValueError('controller mode mismatch')

    calibration_features = []
    calibration_runs = []
    for case in plan['fit']:
        obj, samples = capture(controller)
        metrics, _ = run_episode(obj, float(case['mass']), tuple(case['com']),
                                 collision_detector.Detector())
        if metrics['diverged'] or len(samples['t']) not in (3550, 3551):
            raise ValueError('incomplete/diverged calibration episode')
        calibration_features.append(public_features(theory, samples, case))
        calibration_runs.append({'id': case['id'], 'mass': case['mass'], 'com': case['com'],
                                 'run': compact_metrics(metrics)})
    model, guard, loo = theory.calibration(calibration_features, c.JTS_NOISE)

    # Complete every inference before invoking the first function that reads
    # free-box pose.  The retained logs are score-only and never enter theory.
    rows = []
    score_logs = []
    for case in plan['test']:
        for variant in VARIANTS:
            obj, samples = capture(controller)
            metrics, log = run_episode(obj, float(case['mass']), tuple(case['com']),
                                       collision_detector.Detector(),
                                       pick_offset=offset(case, variant))
            if metrics['diverged'] or len(samples['t']) not in (3550, 3551):
                raise ValueError('incomplete/diverged evaluation episode')
            feature = public_features(theory, samples, case)
            prediction = theory.estimate(feature, model, guard)
            rows.append({'id': case['id'], 'variant': variant,
                         **prediction, 'run': compact_metrics(metrics)})
            score_logs.append((score_capsule(log), np.asarray(case['com'])))

    truth = [observed_geometry(capsule, com) for capsule, com in score_logs]
    bases = {}
    for row, actual in zip(rows, truth):
        group, variant = row['id'], row['variant']
        measured = actual['bottom_shift_at_placement_mm']
        if variant == 'zero':
            bases[group] = (measured, row['extra_height_mm'])
        true_extra = measured - bases[group][0]
        prediction = row['extra_height_mm']
        row.update(
            true_extra_mm=true_extra,
            true_bottom_shift_at_placement_mm=measured,
            true_local_com_shift_mm=actual['local_com_shift_mm'],
            error_mm=prediction - true_extra,
            paired_error_mm=prediction - bases[group][1] - true_extra,
            covered=bool(row['interval_mm'][0] <= true_extra <= row['interval_mm'][1]),
            max_observed_pose_span_mm=max(actual['sample_span_mm']),
        )
    return {
        'calibration': {'mode': mode, 'runs': calibration_runs, 'model': serial_model(model),
                        'normal_guard_mm': float(guard), 'loo_mm': loo},
        'evaluation': {'mode': mode, 'rows': rows, 'metrics': scores(rows)},
    }


def assemble_inputs():
    paths = {name: ROOT / ('my_' + name + '.py') for name in ('servo', 'torque', 'detector')}
    paths.update({
        'estimator': ROOT / 'tools/r13_grasp_estimator.py',
        'plan': ROOT / 'experiments/r13_carried/plan.json',
        'audit': Path(__file__),
        'kit_manifest': ROOT / 'tools/kit_manifest.sha256',
    })
    if not all(path.is_file() and not path.is_symlink() for path in paths.values()):
        raise ValueError('regular public candidate, estimator, plan and manifest files required')
    return snapshot_files(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', type=Path, required=True, help='licensed ORIGINAL icpbl_sim kit')
    parser.add_argument('--out', type=Path, required=True, help='NEW output directory')
    args = parser.parse_args(argv)
    kit, out = validate_paths(args.kit, args.out, parser)
    blobs, pins = assemble_inputs()
    plan = json.loads(blobs['plan'])
    validate_plan(plan)

    sys.path.insert(0, str(ROOT))
    from tools.validate_submission import prepare_overlay

    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    environment.pop('PYTHONPATH', None)
    results = {}
    for mode in ('servo', 'torque'):
        with tempfile.TemporaryDirectory(prefix='grasp-audit-', dir=out.parent) as folder:
            overlay = Path(folder)
            prepare_overlay(kit, overlay, 'my', mode)
            if digest(overlay / 'tools/kit_manifest.sha256') != pins['kit_manifest']:
                raise ValueError('kit manifest differs from frozen audit input')
            destinations = {
                mode: overlay / (mode + '_controller.py'),
                'detector': overlay / 'collision_detector.py',
                'estimator': overlay / 'grasp_estimator.py',
                'plan': overlay / 'plan.json',
                'audit': overlay / 'audit.py',
            }
            for key, destination in destinations.items():
                install_snapshot(blobs[key], pins[key], destination)
            code = ('import audit,json; from pathlib import Path; '
                    'json.dump(audit.run_mode(' + repr(mode) +
                    ',json.loads(Path("plan.json").read_text())),sys.stdout,allow_nan=False)')
            # The executed audit is the immutable overlay copy, never this working tree.
            code = 'import sys;' + code
            process = subprocess.run([sys.executable, '-c', code], cwd=overlay,
                                     env=environment, text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
            if process.returncode:
                raise RuntimeError(mode + ' audit failed:\n' + process.stderr)
            results[mode] = json.loads(process.stdout)

    ignored = [{'split': split, 'id': case['id'], 'research_perturb': case.get('perturb', {})}
               for split in ('fit', 'test') for case in plan[split]
               if case.get('perturb', {})]
    scope = {
        'episodes': 42,
        'calibration_episodes': 10,
        'evaluation_episodes': 32,
        'inference_deadline_s': 3.55,
        'runner': 'authenticated stock sim.runner.run_episode',
        'pickup_offsets': 'authored plan shifts reproduced through public pick_offset',
        'research_parity': False,
        'parity_limit': ('Stock runner fixes JTS seed and exposes no stiffness/viscous/Coulomb '
                         'perturbation arguments. Research perturb dictionaries were not applied; '
                         'numeric equality with the private 42-run captures is neither expected nor claimed.'),
        'ignored_research_perturbations': ignored,
        'causal_split': ('calibration uses only five zero-offset public-signal runs per mode; all '
                         '16 predictions are completed before any free-box pose is scored'),
        'retained_data': 'derived calibration/evaluation JSON only; no raw captures',
    }
    calibration = {'schema': 'r13-grasp-calibration.v1', 'pins': pins, 'scope': scope,
                   'modes': {mode: results[mode]['calibration'] for mode in results}}
    evaluation = {'schema': 'r13-grasp-evaluation.v1', 'pins': pins, 'scope': scope,
                  'modes': {mode: results[mode]['evaluation'] for mode in results}}
    summary = {'schema': 'r13-grasp-audit.v1', 'pins': pins, 'scope': scope,
               'metrics': {mode: results[mode]['evaluation']['metrics'] for mode in results}}

    out.mkdir()
    for name, document in (('calibration.json', calibration), ('evaluation.json', evaluation),
                           ('summary.json', summary)):
        with (out / name).open('x') as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print('Completed 42 stock-runner episodes; wrote passive grasp audit:', out)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
