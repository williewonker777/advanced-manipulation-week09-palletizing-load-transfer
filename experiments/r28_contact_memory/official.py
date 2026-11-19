"""Authentic course grader. Each row is a separately simulated arm on the R24 controller."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile

import mujoco
from experiments.r27_landing_integration import official as course

ARMS = {
    'c1': 'experiments.r26_delivery_pair.candidate',
    'v3': 'experiments.r27_landing_integration.unloaded_aware',
    'contact-reset': 'experiments.r28_finite_contact.candidate',
    'observation-memory': 'experiments.r28_contact_memory.candidate',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if os.path.lexists(args.out) or mujoco.__version__!='3.17.0' or course.sha(course.ROOT/'evaluate.py')!=course.EVALUATOR:
        raise ValueError('requires fresh output and authentic course grader/physics')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='disable')
    env.pop('PYTHONPATH',None)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temp:
        jobs=[(name,det,Path(temp),env)for name,det in ARMS.items()]
        with ThreadPoolExecutor(2) as pool:
            records=dict(pool.map(course.score,jobs))
    control=records['c1']['cells']['control']
    result=dict(schema='r28-g4-course-four-arm.v1',classification='Known original grading conditions on research controller; not original servo demo or independent qualification.',
                episodes=116,grader_sha256=course.EVALUATOR,physics=mujoco.__version__,records=records,
                same_control_cells=all(item['cells']['control']==control for item in records.values()))
    if any(item['episodes']!=29 for item in records.values()):raise ValueError('incomplete run')
    with args.out.open('x')as handle:json.dump(result,handle,indent=2,sort_keys=True);handle.write('\n')


if __name__=='__main__':
    main()
