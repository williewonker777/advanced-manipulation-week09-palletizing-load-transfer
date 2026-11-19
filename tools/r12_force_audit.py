"""Check a public-signal virtual force estimate against actual simulation *after* a run."""
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


def capture(cls):
    """Passive recorder. Unavailable legacy diagnostics stay missing, not zero."""
    samples = []

    class Record(cls):
        def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
            inp = (float(t), th.copy(), tau_s.copy(), bool(grip))
            out = super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
            sh = getattr(self, 'shared_estimator', None)
            legacy = (1. - sh.lambda_support if getattr(sh, 'lambda_support_valid', False)
                      else np.nan)
            samples.append((*inp, legacy))
            return out
    return Record(), samples


def targets(truth, indices, weights, lag):
    """Return the force supported by the estimator's stencil, and force now."""
    half = len(weights) // 2
    if min(indices) - lag - half < 0 or max(indices) >= len(truth):
        raise ValueError('force targets outside recorded episode')
    groups = indices[:, None] - lag + np.arange(-half, half+1)
    return truth[groups] @ weights, truth[indices]


def errors(prediction, truth):
    """Never silently drop unavailable estimates or treat them as force."""
    p, t = np.asarray(prediction), np.asarray(truth)
    if p.shape != t.shape or p.size == 0 or not np.isfinite(t).all():
        raise ValueError('positive matched finite truth required')
    ok = np.isfinite(p)
    result = {'available': int(ok.sum()), 'required': int(len(p))}
    if not ok.all():
        return {**result, 'bias': None, 'rmse': None, 'mae': None}
    delta = p - t
    return {**result, 'bias':float(delta.mean()), 'rmse':float(np.mean(delta**2)**.5),
            'mae':float(abs(delta).mean())}


def assess(samples, log, mass, com, physical, observer):
    from sim import config as c, trajectory
    from sim.nominal import Nominal
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('positive known payload required')
    if len(samples) != 6000 or len(log['t']) != 6000:
        raise ValueError('incomplete episode')
    # A conservative footprint check excludes contact with the wrong table.
    measured = physical(log, mass, trajectory.T_RELEASE, c.DT)
    loaded = Nominal()
    loaded.set_payload(mass, com)
    est = observer.NormalForce(Nominal(), loaded, dt=c.DT, viscous_spring=c.D, jts_hz=c.JTS_CUTOFF)
    now = np.full(len(samples), np.nan)
    prior = np.full(len(samples), np.nan)
    times = log['t']
    for i, (t, th, tau_s, grip, proxy) in enumerate(samples):
        if abs(t - i*c.DT) > 1e-10 or t != times[i]:
            raise ValueError('sensor and evaluator clocks differ')
        packet = est.update(t, th, tau_s, grip, project=3.70 <= t < trajectory.T_RELEASE)
        if packet is not None:
            if abs(packet.effective_time - (t - observer.LAG*c.DT)) > 1e-9:
                raise ValueError('unexpected observer latency')
            now[i] = packet.vertical_n/(mass*9.81)
        prior[i] = proxy
    indices = np.flatnonzero((times >= 3.77) & (times < trajectory.T_RELEASE))
    last = indices[-20:]
    truth = np.asarray(log['f_table'])/(mass*9.81)
    delayed, instant = targets(truth, indices, observer.KERNEL, observer.LAG)
    # Elastic JTS/encoder live at nodal state time; the rigid-axis torque
    # is the preceding interval's average. Report the adjacent alignment too.
    nodal, _ = targets(truth, indices, observer.KERNEL, observer.STATE_LAG)
    end = errors(now[last], truth[last])
    full = np.isfinite(now[indices]).all()
    result = {'id_window_s':[3.77,3.90], 'n':len(indices),
              'matched_sensor':errors(now[indices], delayed),
              'matched_legacy_at_availability':errors(prior[indices], delayed),
              'matched_legacy_at_effective_time':errors(prior[indices-observer.LAG], delayed),
              'matched_zero':errors(np.zeros(len(indices)), delayed),
              'adjacent_state_clock_sensor':errors(now[indices], nodal),
              'current_sensor':errors(now[indices], instant),
              'current_legacy':errors(prior[indices], instant),
              'current_last20':end, 'true_support':measured['support'],
              'current_last20_estimated_mean':float(now[last].mean()) if np.isfinite(now[last]).all() else None,
              'current_false_confidence_samples':(int(np.sum((instant < .25) & (now[indices] >= .5)))
                                                  if full else None),
              'lag_samples':observer.LAG, 'state_lag_samples':observer.STATE_LAG}
    traces = {'time':times[indices], 'true_current_mg':instant, 'true_weak_mg':delayed,
              'estimate_mg':now[indices], 'legacy_mg':prior[indices]}
    return result, traces


def run_kit(mode, design):
    import r12_normal_force as observer
    from geometry_eval import physical, FootprintUnverified
    from collision_detector import Detector
    from sim.runner import run_episode
    cls = importlib.import_module(mode + '_controller').Controller
    if cls.MODE != mode:
        raise ValueError('controller mode mismatch')
    rows = []
    for group in design['cases']:
        obj, samples = capture(cls)
        mass, com = group['mass'], tuple(group['com'])
        meta, log = run_episode(obj, mass, com, Detector(), pick_offset=group.get('offset'))
        if meta['diverged']:
            rows.append({'id':group['id'], 'status':'diverged'})
        else:
            try:
                score, _ = assess(samples, log, mass, com, physical, observer)
            except FootprintUnverified:
                rows.append({'id':group['id'], 'status':'placing footprint unverified'})
            else:
                rows.append({'id':group['id'], 'status':'measured', 'score':score})
    return rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_files(paths):
    """Bind reported digests and executed module bytes to one read per input."""
    blobs = {key:path.read_bytes() for key,path in paths.items()}
    return blobs, {key:hashlib.sha256(blob).hexdigest() for key,blob in blobs.items()}


def install_snapshot(blob, pin, destination):
    destination.write_bytes(blob)
    if digest(destination) != pin:
        raise ValueError('assembly checksum mismatch: '+destination.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', type=Path, required=True)
    parser.add_argument('--candidate-dir', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    kit, src, out = [p.resolve() for p in (a.kit, a.candidate_dir, a.out)]
    if a.out.is_symlink() or out.exists() or not out.parent.is_dir() or out.is_relative_to(kit):
        parser.error('new result outside the kit required')
    files = {n:src/('my_'+n+'.py') for n in ('servo','torque','detector')}
    if not all(f.is_file() and not f.is_symlink() for f in files.values()):
        parser.error('three self-contained regular candidate modules required')
    sys.path.insert(0, str(ROOT))
    from tools.validate_submission import prepare_overlay
    design = ROOT/'results/r11/placement_plan.json'
    sensor = ROOT/'tools/r12_normal_force.py'
    geometry = ROOT/'tools/r11_placement.py'
    blobs, pins = snapshot_files({**files, 'design':design, 'sensor':sensor, 'geometry':geometry,
                                  'scoring':Path(__file__), 'kit_manifest':ROOT/'tools/kit_manifest.sha256'})
    result = {'schema':'r12-virtual-load-audit.v2', 'solver_step_lag_ms':18, 'state_time_lag_ms':17,
              'scope':'Three unforced original-kit cases per mode; net vertical external force, not a contact classifier.',
              'pins':pins, 'modes':{}}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    for mode in ('servo', 'torque'):
        with tempfile.TemporaryDirectory(prefix='load-audit-') as td:
            dest = Path(td)
            prepare_overlay(kit, dest, 'my', mode, archived_r9=True)
            if digest(dest/'tools/kit_manifest.sha256') != pins['kit_manifest']:
                raise ValueError('kit manifest differs from frozen audit input')
            for key, name in ((mode,mode+'_controller.py'),('detector','collision_detector.py'),
                              ('design','design.json'),('sensor','r12_normal_force.py'),
                              ('geometry','geometry_eval.py'),('scoring','audit.py')):
                install_snapshot(blobs[key], pins[key], dest/name)
            code = ('import audit,json,sys; from pathlib import Path; '
                    'json.dump(audit.run_kit(sys.argv[1],json.loads(Path("design.json").read_text())),'
                    'sys.stdout,allow_nan=False)')
            proc = subprocess.run([sys.executable,'-c',code,mode],cwd=dest,env=env,
                                  text=True,stdout=subprocess.PIPE,check=True)
            result['modes'][mode] = json.loads(proc.stdout)
    with out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print('Wrote causal virtual-force audit:',out)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
