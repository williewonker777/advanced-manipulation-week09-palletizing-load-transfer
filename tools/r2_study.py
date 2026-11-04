"""Serial preregistered R2 studies; no remote access or simulator patching."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import numpy as np
from sim.runner import run_episode
from sim import trajectory as T
from tools import v2_study as S, sweep, v2_resources as resources
from tools.v2_learned_core import normal_features
from tools.check_integrity import check_integrity
import r1_detector

D = ROOT/'results/v2_protocol'
PROTOCOL = D/'PROTOCOL_R2.json'
PROTOCOL_COMMIT = '00c3a6675919cc5b8f97e1e5d5890d9e32aef291'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specs():
    out = [dict(id='R1', family='R1'), dict(id='base', family='base'),
           dict(id='v1', family='v1')]
    for fraction, feedback in itertools.product((.5, .85), (0., .25)):
        out.append(dict(id=f'C1_F{fraction:g}_fb{feedback:g}', family='C1',
                        fraction=fraction, feedback_gain=feedback))
    for geometry, scale in itertools.product(('euclidean', 'se3'), (.0006894392593624343, .0026785554844834247)):
        out.append(dict(id=f'C2_{geometry}_{"small" if scale < .001 else "stroke"}', family='C2',
                        geometry=geometry, error_scale=scale))
    for gamma in (1., 2., 4.):
        out.append(dict(id=f'C3_NRIC_g{gamma:g}', family='C3', law='NRIC', gamma=gamma))
    for bandwidth in (4.5, 9., 18.):
        out.append(dict(id=f'C3_DOB_w{bandwidth:g}', family='C3', law='DOB', bandwidth=bandwidth))
    for law, rate, damping in itertools.product(('P-APD', 'P-IDC'), (4.5, 9.), (.5, 1.)):
        out.append(dict(id=f'C4_{law}_w{rate:g}_d{damping:g}', family='C4', law=law,
                        rate=rate, damping_scale=damping))
    for variant in ('prne_only', 'prne_plus_ridge'):
        out.append(dict(id='C5_'+variant, family='C5', variant=variant))
    out.append(dict(id='C5_ridge_only_matched', family='C5', variant='ridge_only_matched', diagnostic_only=True))
    for variant in ('seventh_joint', 'S_curve_joint', 'SO3_R3_seventh'):
        out.append(dict(id='C6_'+variant, family='C6', variant=variant))
    return out


def candidate_classes(mode, spec, ridge=None):
    family = spec['family']
    if family in ('R1', 'C5'):
        ctrl = importlib.import_module('r1_'+mode).Controller
    elif family in ('base', 'v1'):
        ctrl = importlib.import_module(mode+'_controller' if family == 'base' else 'v1_'+mode).Controller
    else:
        name = 'experiments.r2_contact' if family in ('C1', 'C2', 'C6') else 'experiments.r2_robust'
        ctrl = importlib.import_module(name).make_controller(mode, spec)
    if family == 'C5':
        if spec['variant'] == 'ridge_only_matched':
            validate_ridge(ridge)
            if not ridge.get('matched_models'):
                raise ValueError('Matched attribution model is missing')
            detector = importlib.import_module('experiments.r2_matched_ridge').make_detector(ridge['matched_models'])
            return ctrl, detector
        if spec['variant'] == 'prne_plus_ridge':
            validate_ridge(ridge)
        params = dict(spec)
        if ridge is not None:
            params['ridge'] = ridge['models']
        detector = importlib.import_module('experiments.r2_prne').make_detector(params)
    else:
        detector = importlib.import_module('collision_detector' if family == 'base' else
                                           'v1_detector' if family == 'v1' else 'r1_detector').Detector
    return ctrl, detector


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def source_hashes():
    names = ['r1_servo.py', 'r1_torque.py', 'r1_detector.py',
             'experiments/r2_contact.py', 'experiments/r2_prne.py', 'experiments/r2_robust.py',
             'experiments/r2_matched_ridge.py',
             'tools/r2_study.py', 'tools/v2_study.py', 'tools/v2_resources.py',
             'servo_controller.py', 'torque_controller.py', 'collision_detector.py',
             'v1_servo.py', 'v1_torque.py', 'v1_detector.py']
    return {n: sha(ROOT/n) for n in names if (ROOT/n).exists()}


def validate_ridge(ridge):
    if ridge is None:
        raise ValueError('pRNE+ridge requires a newly frozen normal-only model')
    if ridge.get('protocol_sha256') != sha(PROTOCOL) or ridge.get('seed') != 369601:
        raise ValueError('Wrong normal-model protocol/training seed')
    sources = ridge.get('normal_source_sha256', {})
    required = ('experiments/r2_prne.py', 'r1_servo.py', 'r1_torque.py')
    if any(sources.get(n) != sha(ROOT/n) for n in required):
        raise ValueError('Normal model observer/controller source mismatch')


def require_private(path):
    path = Path(path).resolve()
    configured = os.environ.get('R2_PRIVATE_EVIDENCE_ROOT')
    if not configured:
        raise ValueError('Set R2_PRIVATE_EVIDENCE_ROOT to an external ignored evidence directory')
    allowed = Path(configured).resolve()
    if not path.is_relative_to(allowed) or path.is_relative_to(ROOT):
        raise ValueError('Raw evidence must stay under the configured private root outside the work/public repo')
    return path


def case_row(spec, mode, condition, index=0, collision=None, pick_offset=None, **tags):
    return dict(variant=spec['id'], specification=spec, architecture=mode,
                architecture_label='A' if mode == 'servo' else 'B', case_id=index,
                mass=float(condition['mass']), com=list(condition['com']), collision=collision,
                pick_offset=pick_offset, kind='reference' if collision is None and pick_offset is None
                else 'offset' if collision is None else 'collision', official=False, **tags)


def run_case(row, private_dir, ridge=None, record_training=False):
    """Recorded target comes ONLY from detector input-derived packet, not log."""
    ctrl_base, det_base = candidate_classes(row['architecture'], row['specification'], ridge)
    TC, TD = S.timed_classes(ctrl_base, det_base)

    class Recorded(TD):
        def reset(self):
            super().reset()
            self.observed = []

        def update(self, t, th, thd, tau_m, tau_s, grip):
            answer = super().update(t, th, thd, tau_m, tau_s, grip)
            # Outside S.TimedDetector's measured inference interval.
            core = getattr(self, '_shared', None)
            core = self if core is None else core
            active = (r1_detector.outside_table_height_band(core, grip)
                      if hasattr(core, '_point_jacobian') and getattr(core, 'q_prev', None) is not None
                      else False)
            wrench = core.last_wrench.copy() if hasattr(core, 'last_wrench') else np.zeros(6)
            self.observed.append((t, th.copy(), thd.copy(), tau_s.copy(), bool(grip),
                                  core.r_joint.copy() if hasattr(core, 'r_joint') else answer[0].copy(),
                                  wrench, active))
            return answer

    controller, detector = TC(), Recorded()
    before = resources.wait_for_capacity()
    result, log = run_episode(controller, box_mass=row['mass'], box_com=row['com'],
                              detector=detector, collision=row['collision'], pick_offset=row['pick_offset'])
    out = dict(row)
    out.update(sweep.evaluate_control_metrics(result, log))
    out.update(g4=S.g4_window_metrics(log, row['mass']), timing=S.timing_summary(controller, detector),
               false_alarms=int(result['false_alarms']), false_alarm_times=result['false_alarm_times'],
               detect_delay_ms=result.get('detect_delay_ms'), alarm_before_collision=bool(result.get('alarm_before_collision', False)),
               tool_contact_delay_ms=result.get('tool_contact_delay_ms'), tool_contact_peak_n=float(np.max(log['f_tool'])),
               r_peak_nm=np.asarray(result['r_peak']).tolist(), host_before=before, host_after=resources.snapshot(),
               _detect=log['detect'].copy())
    observed = detector.observed
    t, th, thd, jts, grip, residual, wrench, active = map(np.asarray, zip(*observed))
    transport = (t >= 1.85) & (t < 3.55)
    packet = getattr(detector, '_shared', None)
    packet = detector if packet is None else packet
    wrench_available = hasattr(packet, 'last_wrench')
    out['normal_residual'] = dict(joint_RMS_Nm=float(np.sqrt(np.mean(residual[transport]**2))),
        joint_norm_p99_Nm=float(np.percentile(np.linalg.norm(residual[transport], axis=1), 99)),
        horizontal_wrench_available=wrench_available,
        horizontal_wrench_RMS_N=float(np.sqrt(np.mean(wrench[transport, :2]**2))) if wrench_available else None,
        horizontal_wrench_norm_p99_N=float(np.percentile(np.linalg.norm(wrench[transport, :2], axis=1), 99)) if wrench_available else None)
    private_dir = require_private(private_dir)
    private_dir.mkdir(parents=True, exist_ok=True)
    name = f'{row["stage"]}_{row["variant"]}_{row["architecture"]}_{row["case_id"]:04}.npz'
    raw = private_dir/name
    arrays = {k: v for k, v in log.items() if isinstance(v, np.ndarray)}
    arrays.update(obs_t=t, encoder=th, encoder_velocity=thd, jts=jts, grip=grip,
                  joint_residual=residual, estimated_wrench=wrench, active=active)
    np.savez_compressed(raw, **arrays)
    if record_training:
        # Physically separate permitted self-supervised data from posthoc
        # oracle logs. The trainer receives ONLY this directory and manifest.
        training = private_dir/'allowed_training'
        training.mkdir(exist_ok=True)
        np.savez_compressed(training/name, t=t,
                            x=np.array([normal_features(at, row['mass'], row['com']) for at in t]),
                            y=wrench[:, :2], active=active)
        out['allowed_training_basename'] = name
        out['allowed_training_sha256'] = sha(training/name)
    out['private_raw_sha256'] = sha(raw)
    out['private_raw_basename'] = name
    return out


def write_group(output, metadata, rows):
    output.parent.mkdir(parents=True, exist_ok=True)
    S._atomic_json(output, json_safe(dict(metadata=metadata, results=rows)))


def metadata(stage, hashes):
    return dict(stage=stage, created_utc=datetime.now(timezone.utc).isoformat(),
                protocol_sha256=sha(PROTOCOL), protocol_commit=PROTOCOL_COMMIT,
                source_sha256=hashes, official=False, remote_operations=0, owned_simulation_workers=1)


def validation(protocol, chosen, private, ridge=None):
    hashes = source_hashes()
    for spec in chosen:
        target = D/f'R2_VALIDATION_{spec["id"]}.json'
        if target.exists():
            raise RuntimeError('Refuse to overwrite completed candidate validation: '+target.name)
        rows = []
        for mode, (index, condition) in itertools.product(('servo', 'torque'), enumerate(protocol['validation'])):
            row = run_case(case_row(spec, mode, condition, index, stage='validation'), private, ridge)
            rows.append(row)
            print(spec['id'], mode, index, 'rms/max', round(row['rms_mm'], 3), round(row['max_mm'], 3),
                  'support', round(row['g4']['table_mean_over_mg'], 4), 'FA', row['false_alarms'], flush=True)
        for row in rows:
            row.pop('_detect')
        if hashes != source_hashes():
            raise RuntimeError('Source changed during candidate evaluation; raw attempt retained, no valid result written')
        write_group(target, metadata('validation', hashes), rows)


def collect_prne(protocol, private):
    hashes = source_hashes()
    spec = next(s for s in specs() if s['id'] == 'C5_prne_only')
    target = D/'R2_PRNE_NORMAL.json'
    if target.exists():
        raise RuntimeError('normal data already collected')
    records, rows = [], []
    for mode, split in itertools.product(('servo', 'torque'), ('train', 'validation')):
        for index, condition in enumerate(protocol[split]):
            row = run_case(case_row(spec, mode, condition, index, stage='normal_'+split), private, record_training=True)
            raw = private/'allowed_training'/row['allowed_training_basename']
            row.pop('_detect')
            records.append(dict(file=raw.name, mode=mode, split=split, condition=condition,
                                source='allowed encoder/JTS-derived PRNE last_wrench; no oracle targets', sha256=sha(raw)))
            rows.append(row)
            print('prne normal', mode, split, index, 'FA', row['false_alarms'], flush=True)
    manifest = dict(protocol_sha256=sha(PROTOCOL), seed=protocol['seeds']['training'],
                    target_definition='C5 pRNE observer wrench from allowed signals ONLY; no Plant/log target',
                    records=records, source_sha256=hashes, official_conditions_used=False, holdout_opened=False)
    if hashes != source_hashes():
        raise RuntimeError('Source changed during normal collection')
    (private/'allowed_training'/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    write_group(target, metadata('normal_only', hashes), rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=('validation', 'normal'), required=True)
    ap.add_argument('--spec', help='comma separated exact candidate IDs; default all (normal ignores)')
    ap.add_argument('--private-dir', type=Path, required=True)
    ap.add_argument('--ridge', type=Path)
    args = ap.parse_args()
    args.private_dir = require_private(args.private_dir)
    check_integrity(ROOT)
    protocol = json.loads(PROTOCOL.read_text())
    ridge = json.loads(args.ridge.read_text()) if args.ridge else None
    available = specs()
    chosen = available if not args.spec else [s for s in available if s['id'] in args.spec.split(',')]
    if args.spec and len(chosen) != len(args.spec.split(',')):
        raise ValueError('Unknown or repeated candidate ID')
    if args.stage == 'normal':
        collect_prne(protocol, args.private_dir)
    else:
        validation(protocol, chosen, args.private_dir, ridge)
    check_integrity(ROOT)


if __name__ == '__main__':
    main()
