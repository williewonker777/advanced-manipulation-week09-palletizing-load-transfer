"""Paired finite-band motion/torque oscillation test with an unchanged local kit.

Run from the release tree with --kit <original> --out <new.json>. Copy and
verify each mode in a temporary overlay before running; never import the kit in
place. Outputs are summaries, not simulation state or inputs to the controller.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / 'results/r10/ringdown_plan.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def band_metrics(values, dt, band):
    """One-sided Hann-windowed band RMS, with affine trend removed per joint."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[1] != 6 or len(a) < 8 or not np.isfinite(a).all():
        raise ValueError('need finite N by 6 log')
    if not 0 < band[0] < band[1] < .5 / dt:
        raise ValueError('invalid band')
    n = len(a)
    x = np.arange(n, dtype=float) - (n - 1) / 2
    a = a - a.mean(axis=0) - x[:, None] * ((x @ a) / (x @ x))
    w = np.hanning(n)
    power = np.abs(np.fft.rfft(w[:, None] * a, axis=0)) ** 2
    freq = np.fft.rfftfreq(n, dt)
    mask = (freq >= band[0]) & (freq <= band[1])
    if not mask.any():
        raise ValueError('band not resolved by window')
    sub = power[mask]
    rms = np.sqrt(2 * sub.sum(axis=0) / (n * np.sum(w*w)))
    peak = freq[mask][np.argmax(sub, axis=0)]
    return {'rms': rms.tolist(), 'peak_hz': [
        float(f) if r > 1e-10 else None for r, f in zip(rms, peak)]}


def validate(log, plan):
    n = plan['cycles']
    for key, shape in [('t', (n,)), ('q', (n, 6)), ('q_d', (n, 6)),
                       ('tau', (n, 6)), ('f_ext', (n,))]:
        a = log[key]
        if np.shape(a) != shape or not np.all(np.isfinite(a)):
            raise ValueError('incomplete/invalid trace: ' + key)
    if not np.array_equal(log['t'], np.arange(n) * plan['dt']):
        raise ValueError('unexpected sample grid')


def normal(log, limits, p):
    validate(log, p)
    n = p['window_samples']
    if np.any(log['f_ext']) or not np.all(log['q_d'][-2*n:] == log['q_d'][-1]):
        raise ValueError('unforced stationary reference required at end')
    ans = {}
    for name, data, threshold in (
        ('angle_deg', np.rad2deg(log['q'] - log['q_d']), p['angle_rms_limit_deg']),
        ('torque_pct_rated', log['tau'] * 100 / limits, p['torque_rms_limit_pct_rated'])):
        windows = [band_metrics(x, p['dt'], p['band_hz'])
                   for x in (data[-2*n:-n], data[-n:])]
        ans[name] = {'windows': windows, 'quiet_both_windows': bool(
            max(max(r['rms']) for r in windows) <= threshold)}
    return ans


def compare(ref, trial, pulse, limits, p):
    validate(ref, p)
    validate(trial, p)
    t, f = trial['t'], trial['f_ext']
    if np.any(ref['f_ext']) or np.any(f < 0.) or not np.array_equal(ref['q_d'], trial['q_d']):
        raise ValueError('mismatched references or invalid force')
    active = np.flatnonzero(f > 1e-10)
    if not len(active) or not np.isclose(np.max(f), np.linalg.norm(pulse['force'])):
        raise ValueError('requested pulse missing or wrong magnitude')
    if (not 0 <= t[active[0]] - pulse['t'] <= 1.01*p['dt']
            or not np.all(np.diff(active) == 1)):
        raise ValueError('unexpected pulse onset or support')
    for name in ('q', 'tau'):
        measured, baseline = trial[name][:active[0]], ref[name][:active[0]]
        if measured.dtype != baseline.dtype or measured.tobytes() != baseline.tobytes():
            raise ValueError('unmatched pre-pulse ' + name)
    start = int(active[-1] + 1)
    n, late = p['window_samples'], start + p['late_start_samples_after_force_off']
    if late < start+n or late+n > len(t) or np.any(f[start:] > 1e-10):
        raise ValueError('post-force windows not available')
    delta = np.rad2deg(trial['q'] - ref['q'])
    peak_deg = np.max(abs(delta[active[0]:late+n]), axis=0)
    tail_deg = np.max(abs(delta[late+n-50:late+n]), axis=0)
    out = dict(force_off_s=float(t[start]), peak_response_deg=peak_deg.tolist(),
               tail_abs_deg=tail_deg.tolist(), excited=float(peak_deg.max()) >= p['min_response_deg'],
               returned=bool(np.all(tail_deg <= p['return_error_deg'])))
    for name, data, limit in (
        ('angle_deg', delta, p['angle_rms_limit_deg']),
        ('torque_pct_rated', (trial['tau']-ref['tau']) * 100 / limits, p['torque_rms_limit_pct_rated'])):
        early_spectrum = band_metrics(data[start:start+n], p['dt'], p['band_hz'])
        late_spectrum = band_metrics(data[late:late+n], p['dt'], p['band_hz'])
        early_rms, late_rms = [np.asarray(v['rms']) for v in (early_spectrum, late_spectrum)]
        tolerable = np.minimum(limit, np.maximum(limit / 10, early_rms*p['late_to_early_ratio']))
        out[name] = {'early': early_spectrum, 'late': late_spectrum,
                     'failed_joints': (np.flatnonzero(late_rms > tolerable) + 1).tolist()}
    out['screen'] = ('inconclusive' if not out['excited'] else
                     'flagged' if not out['returned'] or out['angle_deg']['failed_joints']
                     or out['torque_pct_rated']['failed_joints'] else 'within_screen')
    return out


def run_assembled(plan, mode):
    # Executed *only* in a checked private assembly, using the submission names.
    import importlib
    from collision_detector import Detector
    from sim import config
    from sim.runner import run_episode
    controller = importlib.import_module(mode + '_controller').Controller
    if controller.MODE != mode:
        raise ValueError('wrong controller assembly')
    res = []
    for load in plan['payloads']:
        def run(pulse):
            metrics, log = run_episode(controller(), load['mass'], tuple(load['com']),
                                       detector=Detector(), collision=pulse)
            if metrics['diverged']:
                raise ValueError('diverged in ' + mode + '/' + load['name'] + ': ' + str(pulse))
            return log, {'saturation_pct': metrics['saturation_pct'], 'diverged': metrics['diverged']}
        base, base_metrics = run(None)
        row = {'payload': load['name'], 'normal': normal(base, config.TAU_MAX, plan),
               'normal_health': base_metrics, 'pulses': []}
        for pulse in plan['pulses']:
            trial, health = run((pulse['t'], tuple(pulse['force'])))
            row['pulses'].append({'name': pulse['name'], 'health': health,
                                  **compare(base, trial, pulse, config.TAU_MAX, plan)})
        res.append(row)
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--archived-r9', action='store_true',
                        help='repeat the original vibration study with the frozen FAST2 detector')
    args = parser.parse_args()
    kit, out = args.kit.resolve(), args.out.resolve()
    if out.exists() or out.is_relative_to(kit) or not out.parent.is_dir():
        parser.error('choose a new result file in an existing directory outside the input kit')
    sys.path.insert(0, str(ROOT))
    from tools.validate_submission import prepare_overlay
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    spec = json.loads(DESIGN.read_text())
    result = {'schema': 'ringdown-result.v1', 'plan_sha256': digest(DESIGN),
              'analysis_sha256': digest(Path(__file__)), 'modes': {}}
    for mode in spec['modes']:
        with tempfile.TemporaryDirectory(prefix='ringdown-') as d:
            root = Path(d)
            ctrl, det = prepare_overlay(kit, root, 'my', mode, archived_r9=args.archived_r9)
            shutil.copyfile(__file__, root / 'ringdown.py')
            shutil.copyfile(DESIGN, root / 'design.json')
            worker = ('import json, sys, ringdown as r; from pathlib import Path; '
                      "json.dump(r.run_assembled(json.loads(Path('design.json').read_text()), "
                      "sys.argv[1]), sys.stdout, allow_nan=False)")
            completed = subprocess.run([sys.executable, '-c', worker, mode], cwd=root,
                                       check=True, env=env, stdout=subprocess.PIPE, text=True)
            result['modes'][mode] = {'controller_sha256': ctrl, 'detector_sha256': det,
                                     'kit_manifest_sha256': digest(root / 'tools/kit_manifest.sha256'),
                                     'rows': json.loads(completed.stdout)}
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('completed paired ringdown assessment:', out)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
