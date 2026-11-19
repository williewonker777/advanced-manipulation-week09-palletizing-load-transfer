"""Original runner instructions with strictly local material changes; no production access."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import types

import numpy as np

import my_detector
from my_torque import Controller
import sim.runner as stock
from experiments.r18_paired_impulses import runner, scoring
from experiments.r18_single_pass import paired_replay
from experiments.r19_close_pairs import assay
from experiments.r20_force_certificate import certificate
from experiments.r20_horizontal.candidate_detector import Detector as R20
from experiments.r21_pick_observer.candidate import Detector as Pick
from experiments.r21_yaw_impulse.coarse import Detector as Yaw

ARMS = {'r10': my_detector.Detector, 'r20': R20, 'pick': Pick, 'yaw': Yaw}
KEYS = {'native': 'f_ext_xyz_native', 'extra': 'f_ext_xyz_second',
        'combined': 'f_ext_xyz_committed', 'scalar': 'f_ext'}


def local_episode(case, mode, det, impacts):
    parent = runner.instrumented(impacts[1])
    friction, tc = (case['modified'][n] if mode == 'changed' else 1. for n in ('friction', 'time_constant'))

    class Surface(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            ids = [self.m.geom(n).id for n in ('table_pick', 'table_place')]
            if self.has_box:
                ids.extend(np.flatnonzero(self.m.geom_bodyid == self.m.body('box_free').id))
            # Same stock settled initial condition on each side of a comparison.
            # Starting at t=0, only contact properties of both material partners change.
            old = self.m.geom_friction[ids, 0].copy()
            sol = self.m.geom_solref[ids, 0].copy()
            if not (np.all(old > 0) and np.all(sol > 0)):
                raise ValueError('positive stock sliding friction and solver time required')
            if mode == 'changed':
                self.m.geom_friction[ids, 0] = old * friction
                self.m.geom_solref[ids, 0] = sol * tc
            self.material_changed = bool(not np.array_equal(old, self.m.geom_friction[ids, 0])
                                         or not np.array_equal(sol, self.m.geom_solref[ids, 0]))

    fn = types.FunctionType(stock.run_episode.__code__, dict(stock.run_episode.__globals__, Plant=Surface),
                            stock.run_episode.__name__, stock.run_episode.__defaults__, stock.run_episode.__closure__)
    fn.__kwdefaults__ = stock.run_episode.__kwdefaults__
    metrics, log = fn(Controller(), case['mass'], tuple(case['com']), detector=det(),
                      pick_offset=case['offset'], collision=impacts[0])
    pl = log['plant']
    if pl.material_changed != (mode == 'changed'):
        raise ValueError('material change not actually applied')
    for key, name in [('native', 'r18_native_force'), ('extra', 'r18_second_force'), ('combined', 'r18_committed_force')]:
        log[KEYS[key]] = np.asarray(getattr(pl, name))
    log['f_ext_combined_norm'] = np.linalg.norm(log[KEYS['combined']], axis=1)
    log['r18_diverged'] = metrics['diverged']
    if not scoring.finite_grid(log):
        raise ValueError('incomplete simulated episode')
    return log


def signal_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def surface_case(case, mode):
    first, second = case['impacts']
    twins = {'none': (None, None), 'first': (first, None), 'second': (None, second), 'both': (first, second)}
    expected = {tw: certificate.exact_trace(case, tw) for tw in twins}
    arms = {}
    physical_hash = {}
    for arm, detector in ARMS.items():
        logs = {tw: local_episode(case, mode, detector, spec) for tw, spec in twins.items()}
        hashes = {tw: {k: signal_hash(l[k]) for k in paired_replay.PHYSICAL} for tw, l in logs.items()}
        if arm == 'r10':
            physical_hash = hashes
        equal = hashes == physical_hash
        forces = all(np.array_equal(logs[tw]['t'], expected[tw]['t']) and certificate.exact_equal(
            {k: logs[tw][v] for k, v in KEYS.items()}, expected[tw]) for tw in twins)
        rec, raw = assay.characterize(case, logs, quiet_ms=20, window_ms=320)
        gates = rec['r18_gates']
        original = assay.one_per_pulse(rec) and all(gates.values())
        certified = assay.one_per_pulse(rec) and forces and all(v for k, v in gates.items()
                                                                if k != 'signed_profiles_exact')
        arms[arm] = {'record': rec, 'physical_and_residual_hash_separate': equal, 'force_exact': forces,
                     'strict_original': original, 'strict_exact_force': certified,
                     'score_sha256': paired_replay.signature(raw)}
    for arm, row in arms.items():
        row['retained_vs'] = {ref: assay.all_comparisons([assay.compare(row['record'], arms[ref]['record'])])
                              for ref in ('r10', 'r20')}
    return {'surface': mode, 'arms': arms, 'physics_sha256': physical_hash, 'episodes': len(twins) * len(ARMS)}


def run_case(case):
    variants = {mode: surface_case(case, mode) for mode in ('stock', 'changed')}
    a, b = [variants[k]['physics_sha256']['none'] for k in ('stock', 'changed')]
    channels = {k: a[k] != b[k] for k in ('q', 'tau', 'f_table')}
    if not any(channels.values()):
        raise ValueError('material intervention had no observed effect on motion or load')
    print(case['id'], {mode: {k: v['strict_exact_force'] for k, v in row['arms'].items()}
                       for mode, row in variants.items()}, flush=True)
    return {'id': case['id'], 'case': case, 'material_effect_observed': channels, 'variants': variants}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, choices=(1, 2), default=1)
    a = p.parse_args()
    plan = Path(__file__).with_name('plan.json')
    if a.out.exists():
        p.error('new output required')
    definition = json.loads(plan.read_bytes())
    if definition['arms'] != list(ARMS):
        raise ValueError('changed registered arms')
    with ProcessPoolExecutor(a.jobs) as pool:
        rows = list(pool.map(run_case, definition['cases']))
    doc = {'schema': 'r21-surface-pair-assay.v1', 'plan_sha256': signal_hash(np.frombuffer(plan.read_bytes(), np.uint8)),
           'unmodified_stock_runner_sha256': runner.source_sha256(), 'official_table': False, 'promoted': False,
           'episodes': sum(v['episodes'] for r in rows for v in r['variants'].values()), 'rows': rows}
    with a.out.open('x') as f:
        json.dump(doc, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
