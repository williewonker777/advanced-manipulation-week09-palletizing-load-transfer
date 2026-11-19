"""Measure both execution schedules of the same rolling-recovery decision."""
import argparse
import hashlib
import json
from pathlib import Path

from . import benchmark, recovery, reference_recovery


def study(plan, height, early, model):
    # The measurement engine takes the two actual factories and the per-tick
    # observable list from its module globals. Scope overrides to this run.
    previous=(benchmark.Reference, benchmark.Detector, benchmark.OBSERVABLES)
    try:
        benchmark.Reference=reference_recovery.Detector
        benchmark.Detector=recovery.Detector
        benchmark.OBSERVABLES=(*previous[2],'recovery_instantaneous','recovery_detected')
        out=benchmark.study(plan,height,early,model)
    finally:
        benchmark.Reference, benchmark.Detector, benchmark.OBSERVABLES=previous
    out['schema']='r18-recovery-scheduling-timing.v1'
    out['scope']='old and single-pass compute the identical R18 4-of-8 observer; controller R16D'
    return out


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('plan','height','early','model','out'):cli.add_argument('--'+k,type=Path,required=True)
    a=cli.parse_args()
    if a.out.exists():cli.error('new output required')
    read=lambda p:json.loads(p.read_bytes())
    result=study(read(a.plan),read(a.height),read(a.early),a.model)
    result['pins']={k:hashlib.sha256(getattr(a,k).read_bytes()).hexdigest() for k in ('plan','height','early','model')}
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
