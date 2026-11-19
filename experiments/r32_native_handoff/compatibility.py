"""Run the legacy notice client and the separate level client on identical disclosed physics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r22_census import check
from experiments.r26_holdout import evaluator as pair
from experiments.r31_event_replay.runner import select
from experiments.r31_held_identity.candidate import Detector as Legacy
from .candidate import Detector, NoticeCompatible

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {'experiments/r31_held_identity/candidate.py':
               '1e33b4698ef3678afe72cc4e286fcab5d27b2232e7bf087fdef7500ccc674764',
           'experiments/r32_native_handoff/candidate.py':
               'd3949f6452b6d3573a9b61a63b77900d0ce78e8603dba2912a7b2e8c034dd837'}


def assert_sources():
    for path, digest in SOURCES.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError('registered source changed: ' + path)


def measure(job):
    kind, case, name, inputs = job
    assert_sources()
    observations = {}
    for arm, cls in [('legacy', Legacy), ('level', Detector), ('compatible', NoticeCompatible)]:
        class Trace(cls):
            def reset(self):
                super().reset()
                self.samples = []
                Trace.latest = self

            def update(self, *args):
                samples = self.samples
                residual, bit = super().update(*args)
                decision = getattr(self, 'decision', None)
                if decision is not None:
                    samples.append((decision.event_output, decision.measured_load, decision.linked_native,
                                    decision.alarm_level, decision.event_started))
                return residual, bit

        if kind == 'd4':
            metrics, log, signed = check.old.execute(Trace, case)
            if not np.array_equal(signed, check.old.expected_signed(log['t'], case['collision'])):
                raise ValueError('signed force not applied')
            forces = {'external': signed}
        else:
            mech, impacts = inputs
            metrics, log = pair.run_episode(case, mech, Trace, impacts)
            if not pair.exact_force(log, impacts)['passed']:
                raise ValueError('signed force components not applied')
            forces = {k: log[k] for k in ['f_ext_xyz_native', 'f_ext_xyz_second', 'f_ext_xyz_committed']}
        if metrics['diverged'] or len(log['t']) != 6000:
            raise ValueError('incomplete course episode')
        returned = np.asarray(log['detect'], bool)
        channels = np.asarray(Trace.latest.samples, bool)
        if len(channels) and (channels.shape != (6000,5)
                or not np.array_equal(channels[:,3], np.any(channels[:,:3], axis=1))
                or not np.array_equal(channels[:,4], channels[:,0] & ~np.r_[False,channels[:-1,0]])):
            raise ValueError('bad causal decomposition')
        observations[arm] = dict(physical={k:check.old.array_sha(log[k]) for k in check.old.PHYSICAL},
            forces={k:check.old.array_sha(z) for k,z in forces.items()}, residual=check.old.array_sha(log['r']),
            returned=returned, channels=channels)
    o = observations
    if not all(o[n][key] == o['legacy'][key] for n in o for key in ['physical','forces','residual']):
        raise ValueError('client modified residual, actual forces, or physics')
    ch = o['level']['channels']
    if not (np.array_equal(ch, o['compatible']['channels'])
            and np.array_equal(ch[:,3],o['level']['returned'])
            and np.array_equal(ch[:,0],o['legacy']['returned'])
            and np.array_equal(ch[:,0],o['compatible']['returned'])):
        raise ValueError('actual client API broke notice/level contract')
    spans = check.old.intervals
    return dict(kind=kind,id=name,matched_samples=6000,residual_sha256=o['legacy']['residual'],
        physics_sha256=o['legacy']['physical'], signed_sha256=o['legacy']['forces'],
        signed_intervals={k:spans(np.any(z!=0,axis=1)) for k,z in forces.items()},
        returned_by_mode={'historical_r31':spans(ch[:,0]),'notice_compatible':spans(o['compatible']['returned']),
                          'sustained_level':spans(o['level']['returned'])},
        measured_load=spans(ch[:,1]),linked_native=spans(ch[:,2]),native_without_measured_load=spans(ch[:,2] & ~ch[:,1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--jobs',type=int,choices=(1,2,3,4),default=2)
    a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():p.error('fresh file required')
    assert_sources()
    cases,_=select(check.build_cases('my_torque'),(251,266,267,307,316))
    jobs=[('d4',c,str(c['case_id']),None) for c in cases]
    plan=json.loads(pair.PLAN.read_bytes());c=next(c for c in plan['paired'] if c['id']=='p26_close_place_cross')
    first, second=c['impacts']
    for n,impacts in [('none',(None,None)),('first',(first,None)),('second',(None,second)),
                      ('both',(first,second)),('long',(c['long'],None))]:
        jobs.append(('r26',c,c['id']+'/'+n,(plan['variants']['stock'],impacts)))
    with ProcessPoolExecutor(a.jobs) as pool: rows=list(pool.map(measure,jobs))
    assert_sources()
    report=dict(schema='r32-client-compatibility.v1',classification='Previously disclosed matched simulation. No timing, real-hardware or independent generalization claim.',
        sources=SOURCES,conditions=len(rows),episodes=3*len(rows),matched_controller_ticks_per_arm=sum(x['matched_samples'] for x in rows),
        exact_residual_signed_force_and_physics=True,actual_notice_equal_at_every_tick=True,actual_level_equal_to_separate_or=True,rows=rows)
    with a.out.open('x') as fp:json.dump(report,fp,indent=2,sort_keys=True,allow_nan=False);fp.write('\n')
    print(report['conditions'],report['episodes'],report['matched_controller_ticks_per_arm'])


if __name__ == '__main__':
    main()
