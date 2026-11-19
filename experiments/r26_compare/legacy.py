"""Compare registered R26 against actual R25 on preexisting legacy simulations."""
import argparse
import json
from pathlib import Path
from types import FunctionType

import mujoco

from experiments.r25_fragment import legacy as earlier
from experiments.r26_compare.study import Baseline,construct


def fork(fn,**changes):
    out=FunctionType(fn.__code__,dict(fn.__globals__,**changes),fn.__name__,fn.__defaults__,fn.__closure__)
    out.__kwdefaults__=fn.__kwdefaults__
    return out


def case(job):
    return fork(earlier.case,Joint=Baseline,constructor=construct)(job)


def weak(job):
    return fork(earlier.weak,Joint=Baseline,constructor=construct)(job)


def pair(job):
    return fork(earlier.pair,Joint=Baseline,constructor=construct)(job)


def run(panel,arm,workers):
    if mujoco.__version__!='3.17.0':raise ValueError('course MuJoCo 3.17.0 required')
    j=fork(earlier.run,case=case,weak=weak,pair=pair)(panel,arm,workers)
    j.update(schema='r26-historical-real-R25-nonloss.v1',baseline='experiments.r25_composed.detector:Detector',runtime=mujoco.__version__)
    return j


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',choices=('legacy66','weak','pairs'),required=True)
    p.add_argument('--arm',required=True);p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():p.error('fresh path required')
    j=run(a.panel,a.arm,a.workers)
    with a.out.open('x')as f:json.dump(j,f,allow_nan=False);f.write('\n')
    print(j['episodes'],j['regressions'])
