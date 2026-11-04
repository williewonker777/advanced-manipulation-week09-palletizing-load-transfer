"""Fresh nonofficial validation/development after train-support fallback design."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from tools import v2_study as S


def cases(design, stage, version, mode, label="L2", candidate="v2_detector_guarded"):
    conditions=design['validation'] if stage=='validation' else design['development']['conditions']
    rows=[]
    def add(condition, collision=None, direction='none', force=0.):
        row=S._base_case('domain_'+stage,'v1',mode,len(rows),condition['mass'],condition['com'],collision=collision,
                         kind='reference' if collision is None else 'collision',phase=stage,direction=direction,force_n=force)
        row['variant']=version
        if version==label:row['detector']=candidate
        rows.append(row)
    for condition in conditions:add(condition)
    if stage=='development':
        d=design[stage]
        for condition, t, f, direction in itertools.product(conditions,d['times'],d['forces'],d['directions']):
            add(condition,(t,S._force(direction,f)),direction,f)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--protocol',type=Path,default=ROOT/'results/v2_protocol/PROTOCOL_OOD.json')
    ap.add_argument('--candidate',default='v2_detector_guarded')
    ap.add_argument('--label',default='L2')
    ap.add_argument('--prefix',default='S4_L2')
    args=ap.parse_args()
    path=args.protocol;design=json.loads(path.read_text())
    hashes={n:S.sha256_file(ROOT/n) for n in ['v1_servo.py','v1_torque.py','v1_detector.py',args.candidate+'.py','models/v2_residual.json']}
    output=[]
    start=time.perf_counter()
    for stage in ['validation','development']:
        for version in ['v1',args.label]:
            for mode in ['servo','torque']:
                group=[]
                for case in cases(design,stage,version,mode,args.label,args.candidate):
                    group.append(S.run_study_case(case))
                    if len(group)%10==0:print(stage,version,mode,len(group),flush=True)
                S.classify_group(group);output.extend(group)
        target=ROOT/f'results/v2_protocol/{args.prefix}_{stage.upper()}.json'
        S._atomic_json(target,dict(metadata=dict(official=False,protocol_sha256=S.sha256_file(path),source_sha256=hashes,
                      stage=stage,elapsed_s=time.perf_counter()-start),results=[r for r in output if r['stage']=='domain_'+stage]))
    assert hashes=={n:S.sha256_file(ROOT/n) for n in hashes}


if __name__=='__main__':main()
