"""Serial R3 studies over unchanged kit APIs and self-contained controllers."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from sim.runner import run_episode
from tools import r3_resources as resources, sweep, v2_study as V
from tools.check_integrity import check_integrity
from tools.r3_build import compiled_path
import r1_detector

D = ROOT/'results/v2_protocol'
PROTOCOL = D/'PROTOCOL_R3.json'
REGISTERED_COMMIT = '90f5d0fa140f602a34c26a59e5c9e57032ac6958'
REGISTERED_PROTOCOL_SHA256 = 'dcb76b495d8c2e73a5047f253a7a43a9fc46b642653e4054c5d947109f8f6c39'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write(path, value):
    V._atomic_json(Path(path), json_safe(value))


def require_private(path):
    root = Path(os.environ.get('R2_PRIVATE_EVIDENCE_ROOT',
                               '/missing-private-root')).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root) or path.is_relative_to(ROOT):
        raise ValueError('private evidence must stay outside work/public under the configured local root')
    return path


def protocol():
    actual = PROTOCOL.read_bytes()
    if hashlib.sha256(actual).hexdigest() != REGISTERED_PROTOCOL_SHA256:
        raise RuntimeError('registered R3 protocol bytes changed')
    # Historical registration is recorded at REGISTERED_COMMIT.  Portable
    # evidence trees use the immutable byte pin, not an unavailable work-branch
    # Git object.  The real sealed-stage freeze separately verifies Git bytes.
    return json.loads(actual)


def specifications():
    specs = list(protocol()['candidate_grid'])
    for path in sorted(D.glob('PROTOCOL_R3_REDESIGN_*.json')):
        redo = json.loads(path.read_text())
        if redo.get('redesign_number') != 1 or not redo.get('reason'):
            raise ValueError('redesign needs its single-use number and contemporaneous reason')
        specs.append(redo['spec'])
    return specs


def baseline_spec(mode):
    return dict(id='R1', family='R1', architecture=mode)


def candidate_classes(mode, spec):
    if spec['architecture'] != mode:
        raise ValueError('cross-architecture candidate is forbidden')
    if spec['id'] == 'R1':
        controller = importlib.import_module('r1_'+mode).Controller
    else:
        expected = next((s for s in specifications() if s['id'] == spec['id']), None)
        if expected != spec:
            raise ValueError('unregistered or altered candidate spec')
        path = compiled_path(spec)
        if not path.is_file():
            raise FileNotFoundError('materialize the self-contained candidate before studies')
        module = importlib.import_module('experiments.r3_generated.'+path.stem)
        if module.R3_DEFAULT_PARAMETERS != spec:
            raise RuntimeError('materialized source parameters differ from registration')
        controller = module.Controller
    return controller, r1_detector.Detector


def source_hashes(mode, spec):
    names = ['r1_'+mode+'.py', 'r1_detector.py', 'tools/r3_resources.py',
             'tools/r3_build.py', 'tools/r3_study.py',
             'tools/v2_study.py', 'tools/sweep.py', 'tools/kit_manifest.sha256',
             'sim/config.py', 'sim/trajectory.py', 'sim/runner.py', 'sim/nominal.py',
             'results/v2_protocol/PROTOCOL_R3.json']
    if spec['id'] != 'R1':
        names += ['experiments/r3_'+mode+'.py', compiled_path(spec).relative_to(ROOT).as_posix()]
    return {n: sha(ROOT/n) for n in names}


def runtime_hashes(mode, spec, harness=None):
    hashes = source_hashes(mode, spec)
    if harness is not None:
        hashes['private_analysis_harness_sha256'] = sha(harness.__file__)
    return hashes


def case_row(spec, condition, index, *, stage, collision=None, pick_offset=None, **tags):
    return dict(variant=spec['id'], specification=spec, architecture=spec['architecture'],
                case_id=index, mass=float(condition['mass']), com=list(condition['com']),
                collision=collision, pick_offset=pick_offset, official=False, stage=stage,
                kind='collision' if collision is not None else 'offset' if pick_offset is not None else 'reference',
                **tags)


def run_case(row, private_dir, *, harness=None, perturb=None):
    private = require_private(private_dir)
    private.mkdir(parents=True, exist_ok=True)
    name = f"{row['stage']}_{row['variant']}_{row['architecture']}_{row['case_id']:04}"
    npz, sidecar = private/(name+'.npz'), private/(name+'.json')
    hashes = runtime_hashes(row['architecture'], row['specification'], harness)
    if npz.exists() or sidecar.exists():
        if not (npz.is_file() and sidecar.is_file()):
            raise RuntimeError('incomplete prior case retained; do not silently rerun it')
        cached = json.loads(sidecar.read_text())
        if cached['case'] != json_safe(row) or cached['source_sha256'] != hashes or cached['raw_sha256'] != sha(npz):
            raise RuntimeError('cached case/source bytes differ; no repeated experiment')
        out = cached['result']
        with np.load(npz) as archive:
            out['_detect'] = archive['detect'].copy()
        return out
    controller_class, detector_class = candidate_classes(row['architecture'], row['specification'])
    TC, TD = V.timed_classes(controller_class, detector_class)

    class Guarded(TC):
        def compute(self, t, *args):
            if int(round(float(t)*1000)) % 250 == 0:
                resources.wait_for_capacity()
            return super().compute(t, *args)

    class Observed(TD):
        def reset(self):
            super().reset()
            self.observations = []

        def update(self, t, th, thd, tau_m, tau_s, grip):
            answer = super().update(t, th, thd, tau_m, tau_s, grip)
            packet = self._shared
            self.observations.append((t, th.copy(), thd.copy(), tau_m.copy(), tau_s.copy(), bool(grip),
                                      packet.r_joint.copy(), packet.last_wrench.copy(),
                                      r1_detector.outside_table_height_band(packet, grip)))
            return answer

    ctrl, det = Guarded(), Observed()
    before = resources.wait_for_capacity()
    kwargs = dict(box_mass=row['mass'], box_com=row['com'], detector=det,
                  collision=row['collision'], pick_offset=row['pick_offset'])
    result, log = (run_episode(ctrl, **kwargs) if harness is None else
                   harness.run_perturbed(ctrl, **kwargs, perturb=perturb))
    out = dict(row)
    out.update(sweep.evaluate_control_metrics(result, log))
    out.update(g4=V.g4_window_metrics(log, row['mass']), timing=V.timing_summary(ctrl, det),
               false_alarms=int(result['false_alarms']), false_alarm_times=result['false_alarm_times'],
               detect_delay_ms=result.get('detect_delay_ms'),
               alarm_before_collision=bool(result.get('alarm_before_collision', False)),
               tool_contact_delay_ms=result.get('tool_contact_delay_ms'),
               tool_contact_peak_n=float(np.max(log['f_tool'])),
               host_before=before, host_after=resources.snapshot())
    t, th, thd, tau_m, jts, grip, residual, wrench, active = map(np.asarray, zip(*det.observations))
    mask = (t >= 1.85) & (t < 3.55)
    out['normal_residual'] = dict(joint_RMS_Nm=float(np.sqrt(np.mean(residual[mask]**2))),
                                  horizontal_wrench_RMS_N=float(np.sqrt(np.mean(wrench[mask,:2]**2))))
    step = np.abs(np.diff(log['tau'], axis=0))
    sample, joint = np.unravel_index(int(np.argmax(step)), step.shape)
    out['full_cycle_step'] = dict(max_nm=float(step[sample,joint]), time_s=float(log['t'][sample+1]),
                                  joint_1based=int(joint+1))
    arrays = {k:v for k,v in log.items() if isinstance(v, np.ndarray)}
    arrays.update(obs_t=t, encoder=th, encoder_velocity=thd, motor_torque=tau_m,
                  jts=jts, grip=grip, joint_residual=residual, estimated_wrench=wrench, active=active)
    if runtime_hashes(row['architecture'], row['specification'], harness) != hashes:
        raise RuntimeError('science source changed during an episode; attempt retained as invalid')
    np.savez_compressed(npz, **arrays)
    out.update(private_raw_basename=npz.name, private_raw_sha256=sha(npz))
    write(sidecar, dict(case=row, source_sha256=hashes, raw_sha256=sha(npz), result=out))
    out['_detect'] = np.asarray(log['detect']).copy()
    return out


def reference_key(row):
    return (row['mass'], tuple(row['com']),
            json.dumps(row.get('perturb'), sort_keys=True, separators=(',', ':')))


def classify_group(rows):
    refs = {reference_key(r): r['_detect'] for r in rows if r['kind'] == 'reference'}
    for row in rows:
        if row['collision'] is not None:
            key = reference_key(row)
            if key not in refs:
                raise RuntimeError('collision lacks a matched no-collision reference')
            row.update(sweep.classify_collision(row, refs[key], row['collision'][0]))
        row.pop('_detect', None)
    return rows


def run_group(stage, spec, cases, private_dir, *, output=None, harness=None):
    target = Path(output) if output else D/f"R3_{stage.upper()}_{spec['id']}_{spec['architecture']}.json"
    if target.exists():
        raise RuntimeError('completed stage exists; never overwrite evidence')
    check_integrity(ROOT)
    hashes = runtime_hashes(spec['architecture'], spec, harness)
    rows = []
    for row in cases:
        rows.append(run_case(row, private_dir, harness=harness, perturb=row.get('perturb')))
        print(stage, spec['id'], spec['architecture'], len(rows), '/', len(cases),
              'RMS/max', round(rows[-1]['rms_mm'],3), round(rows[-1]['max_mm'],3),
              'FA', rows[-1]['false_alarms'], flush=True)
    classify_group(rows)
    if hashes != runtime_hashes(spec['architecture'], spec, harness):
        raise RuntimeError('source changed during stage')
    result = dict(metadata=dict(stage=stage, created_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=sha(PROTOCOL), protocol_commit=REGISTERED_COMMIT,
                  source_sha256=hashes, official=False, owned_simulation_workers=1,
                  completed=True), results=rows)
    write(target, result)
    check_integrity(ROOT)
    return result


def validation_cases(spec, redesign=False):
    key = 'redesign_validation' if redesign else 'validation'
    return [case_row(spec, c, i, stage=key) for i,c in enumerate(protocol()[key])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('validation','redesign_validation','diagnostic'), required=True)
    parser.add_argument('--architecture', choices=('servo','torque'), required=True)
    parser.add_argument('--spec', default='R1')
    parser.add_argument('--private-dir', type=Path, required=True)
    parser.add_argument('--mass', type=float)
    args = parser.parse_args()
    spec = baseline_spec(args.architecture) if args.spec == 'R1' else next(s for s in specifications() if s['id'] == args.spec)
    if args.stage == 'diagnostic':
        if args.mass is None or spec['id'] != 'R1':
            parser.error('diagnostic requires an explicit R1 mass')
        cases = [case_row(spec, dict(mass=args.mass,com=[.013,-.009,.019]),0,stage='diagnostic')]
        target = D/f'R3_DIAGNOSTIC_R1_{args.architecture}_{args.mass:g}.json'
    else:
        cases = validation_cases(spec, args.stage == 'redesign_validation')
        target = None
    run_group(args.stage, spec, cases, args.private_dir, output=target)


if __name__ == '__main__':
    main()
