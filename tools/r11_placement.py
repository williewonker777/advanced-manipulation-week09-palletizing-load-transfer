"""Measure sustained box support and free-box clearance from an unchanged local kit.

Run in a private checked overlay; structural/contact ground truth is used only
after an episode, never as an input or training target for a submission.
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
PLAN = ROOT / 'results/r11/placement_plan.json'


class FootprintUnverified(ValueError):
    """A force from the combined table log cannot be attributed confidently."""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def face_clearance(positions, quat, half_size, table_top):
    """Lowest face corners above a horizontal plane, meters; wxyz world pose."""
    positions, quat, half = map(np.asarray, (positions, quat, half_size))
    if (positions.ndim != 2 or positions.shape[1] != 3 or quat.shape != (len(positions), 4)
            or half.shape != (3,) or not np.all(half > 0)
            or not np.isfinite(positions).all() or not np.isfinite(quat).all()
            or not np.isfinite(table_top) or not np.allclose(np.linalg.norm(quat, axis=1), 1., rtol=0, atol=1e-7)):
        raise ValueError('expected finite box positions, unit wxyz quaternions and positive half-sizes')
    w, x, y, z = quat.T
    r20, r21, r22 = 2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)
    center = positions[:, 2] - np.abs(r22)*half[2] - table_top
    heights = center[:, None] + (r20*half[0])[:, None]*[-1., -1., 1., 1.] + (r21*half[1])[:, None]*[-1., 1., -1., 1.]
    return np.sort(heights, axis=1), np.rad2deg(np.arccos(np.clip(np.abs(r22), 0., 1.)))


def support(t, f, mass, release, dt=.001):
    """Use every sample of the fixed final 20 ms; peak uses [-300,+200) ms."""
    t, f = map(np.asarray, (t, f))
    if (mass <= 0 or not np.isfinite(mass) or dt <= 0 or not np.isfinite(dt)
            or t.ndim != 1 or len(t) < 20 or f.shape != t.shape or not np.isfinite(f).all()
            or np.any(f < -1e-9) or not np.allclose(t, np.arange(len(t))*dt, rtol=0., atol=1e-10)):
        raise ValueError('positive mass, nonnegative force and a complete regular time grid required')
    k, final, pre, post = (round(value/dt) for value in (release, .020, .300, .200))
    if not np.isclose(k*dt, release, rtol=0, atol=1e-10) or k < pre or k + post > len(t):
        raise ValueError('release and assessment windows must lie on the grid')
    mg = 9.81 * mass
    z = f/mg
    window = z[k-final:k]
    return {'mean_mg': float(window.mean()), 'min_mg': float(window.min()),
            'peak_mg': float(z[k-pre:k+post].max()), 'samples': len(window),
            'mean_ge_050': bool(window.mean() >= .50),
            'min_ge_025': bool(window.min() >= .25),
            'peak_le_110': bool(z[k-pre:k+post].max() <= 1.10)}


def physical(log, mass, release, dt):
    """Measure force, plus independent geometric evidence, after the run."""
    import mujoco
    from sim import config
    from sim.nominal import Nominal
    t = np.asarray(log['t'])
    if len(t) != 6000 or mass <= 0:
        raise ValueError('expected a complete positive-payload episode')
    contact = support(t, log['f_table'], mass, release, dt)
    plant = log['plant']
    model = plant.m
    body = model.body('box_free').id
    geom_ids = np.flatnonzero(model.geom_bodyid == body)
    if len(geom_ids) != 1:
        raise ValueError('one known box geometry required')
    g = geom_ids[0]
    tab = model.geom('table_place').id
    if not (np.array_equal(model.geom_pos[g], np.zeros(3))
            and np.allclose(model.geom_quat[g], [1.,0.,0.,0.])
            and np.allclose(plant.d.geom_xmat[tab].reshape(3,3)[:,2], [0.,0.,1.])):
        raise ValueError('expected centered box and level placing table')
    top = plant.d.geom_xpos[tab,2] + model.geom_size[tab,2]
    adr = model.jnt_qposadr[model.joint('box_free').id]
    pose = np.asarray(log['qpos'])[:, adr:adr+7]
    clearance, tilt = face_clearance(pose[:, :3], pose[:, 3:], model.geom_size[g], top)
    k = round(release/dt)
    # Each saved qpos is after integration, whereas the corresponding force is
    # from the dynamics step. Do not interpret these as simultaneous force-gap
    # observations; the last 20 pre-release step-start poses are k-21..k-2.
    lo = clearance[:, 0] * 1000
    paired = slice(k-21, k-1)
    pxy = pose[paired, :2] - plant.d.geom_xpos[tab,:2]
    radius = np.linalg.norm(model.geom_size[g])
    if not np.all(np.abs(pxy) + radius <= model.geom_size[tab,:2]):
        raise FootprintUnverified('box footprint not conservatively inside placement table')
    # Which object is rising? Forward-kinematic box geometry at the taught or
    # actual link pose assumes a rigid nominal grasp; the freejoint does not.
    nominal = Nominal()
    flange = nominal.m.body('link_6').id
    def rigid_height(q):
        nominal.d.qpos[:] = q
        mujoco.mj_kinematics(nominal.m, nominal.d)
        axis = nominal.d.xmat[flange].reshape(3,3)[2]
        return 1000*(nominal.d.xpos[flange,2] + axis @ config.BOX_POS
                     - np.abs(axis) @ model.geom_size[g] - top)
    ref = [rigid_height(q) for q in log['q_d'][k-20:k]]
    link = [rigid_height(q) for q in log['q'][paired]]
    return {'support': contact, 'last20_geometry': {
        'min_corner_range_mm': [float(lo[paired].min()), float(lo[paired].max())],
        'mean_lowest_corner_mm': float(lo[paired].mean()),
        'mean_corner_spread_mm': float(1000*(clearance[paired,3]-clearance[paired,0]).mean()),
        'mean_face_tilt_deg': float(tilt[paired].mean()),
        'all_four_corners_below_plane_samples': int(np.count_nonzero(clearance[paired,-1] <= 0.)),
        'all_four_corners_above_plane_samples': int(np.count_nonzero(lo[paired] > 0.)),
        'taught_rigid_corner_speed_mm_s': float((ref[-1]-ref[0])/(19*dt)),
        'actual_link_rigid_corner_speed_mm_s': float((link[-1]-link[0])/(19*dt)),
        'free_minus_rigid_mean_mm': float((lo[paired]-link).mean()),
        'first_to_last_lowest_corner_speed_mm_s': float((lo[k-2]-lo[k-21])/(19*dt))}}


def score(baseline, candidate):
    b_status, c_status = (r.get('status', 'ok') for r in (baseline, candidate))
    if c_status == 'diverged':
        return {'mean_gain_mg': None, 'all_four_g4': False, 'reason': 'candidate diverged'}
    if b_status != 'ok' or c_status != 'ok':
        return {'mean_gain_mg': None, 'all_four_g4': None,
                'reason': f'no verified pair: reference={b_status}, candidate={c_status}'}
    b, c = baseline['support'], candidate['support']
    gain = c['mean_mg'] - b['mean_mg']
    return {'mean_gain_mg': gain, 'all_four_g4': bool(c['mean_ge_050'] and c['min_ge_025']
                                            and c['peak_le_110'] and gain >= .25)}


def measure(mode, plan):
    import importlib
    from collision_detector import Detector
    from sim import config as C, trajectory as T
    from sim.runner import run_episode
    ctrl = importlib.import_module(mode + '_controller').Controller
    if ctrl.MODE != mode:
        raise ValueError('wrong control mode')
    rows = []
    for case in plan['cases']:
        mass = case['mass']
        metric, log = run_episode(ctrl(), mass, tuple(case['com']), Detector(),
                                  pick_offset=case.get('offset'))
        row = {'id': case['id'], 'mass': mass, 'status': 'ok', 'rms_mm': metric['rms_mm'],
               'peak_mm': metric['max_mm'], 'saturation_pct': metric['saturation_pct'],
               'false_alarms': metric['false_alarms']}
        if metric['diverged']:
            row['status'] = 'diverged'
        else:
            try:
                row.update(physical(log, mass, T.T_RELEASE, C.DT))
            except FootprintUnverified:
                row['status'] = 'footprint_unverified'
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', required=True, type=Path)
    parser.add_argument('--candidate-dir', type=Path, default=ROOT,
                        help='directory of three self-contained my_*.py files (default: this release)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    kit, candidate, out = (p.resolve() for p in (args.kit, args.candidate_dir, args.out))
    if out.exists() or not out.parent.is_dir() or out.is_relative_to(kit):
        parser.error('choose a new result file outside the kit')
    files = [candidate/f'my_{name}.py' for name in ('servo','torque','detector')]
    if not all(f.is_file() and not f.is_symlink() for f in files):
        parser.error('candidate-dir must contain three regular, self-contained my_*.py files')
    sys.path.insert(0, str(ROOT))
    from tools.validate_submission import prepare_overlay
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    p = json.loads(PLAN.read_text())
    result = {'schema':'r11-placement.v1', 'plan_sha256':digest(PLAN), 'analysis_sha256':digest(Path(__file__)),
              'kit_manifest_sha256':digest(ROOT/'tools/kit_manifest.sha256'), 'modes':{}}
    for mode in ('servo','torque'):
        cases = {}
        for name in ('archived_r1', 'candidate'):
            with tempfile.TemporaryDirectory(prefix='placement-') as folder:
                overlay = Path(folder)
                ctrl, det = prepare_overlay(kit, overlay, 'my', mode, archived_r9=True)
                if name == 'candidate':
                    for src, dest in [(candidate/f'my_{mode}.py', overlay/f'{mode}_controller.py'),
                                      (candidate/'my_detector.py', overlay/'collision_detector.py')]:
                        shutil.copyfile(src, dest)
                        if digest(dest) != digest(src):
                            raise ValueError('incomplete candidate assembly')
                    ctrl, det = digest(overlay/f'{mode}_controller.py'), digest(overlay/'collision_detector.py')
                shutil.copyfile(__file__, overlay/'placement_eval.py')
                shutil.copyfile(PLAN, overlay/'design.json')
                cmd = ('import json,sys,placement_eval as ev;from pathlib import Path;'
                       'json.dump(ev.measure(sys.argv[1],json.loads(Path("design.json").read_text())),sys.stdout,allow_nan=False)')
                run = subprocess.run([sys.executable, '-c', cmd, mode], cwd=overlay, env=env, check=True,
                                     stdout=subprocess.PIPE, text=True)
                cases[name] = {'controller_sha256':ctrl, 'detector_sha256':det, 'rows':json.loads(run.stdout)}
        pairs = list(zip(cases['archived_r1']['rows'], cases['candidate']['rows'], strict=True))
        if [r['id'] for r, _ in pairs] != [r['id'] for r in p['cases']] or any(x['id'] != y['id'] for x,y in pairs):
            raise ValueError('unmatched placement conditions')
        cases['comparison'] = [dict(id=a['id'], **score(a,b)) for a,b in pairs]
        result['modes'][mode] = cases
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print('Paired placement assessment written:', out)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
