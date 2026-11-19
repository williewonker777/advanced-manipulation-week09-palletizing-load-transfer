"""Retest the two exposed R22 sensor misses against an otherwise matched no-offset motion."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.r22_challenge import evaluator as old
from experiments.r22_light.candidate import Detector as Light
from experiments.r23_tool_contact.candidate import Detector as Approach


def run():
    plan=json.loads(old.PLAN.read_bytes())
    case=next(c for c in plan['negative'] if c['id']=='n_tool_xyz')
    rows=[]
    for variant in plan['variants'].values():
        obs={};physics={}
        for label, offset in [('offset',case['offset']),('normal',None)]:
            for arm,cls in [('light',Light),('approach',Approach)]:
                _,log=old.run_episode(dict(case,offset=offset),variant,cls,(None,None))
                if not old.exact_force(log,(None,None))['passed']:raise ValueError('unexpected external force')
                idx=np.flatnonzero(log['f_tool']>1.)
                key=label+':'+arm
                obs[key]={'tool':old.tool_record(log),'positive_duration_ms':len(idx),
                          'onset':int(idx[0]) if len(idx) else None,'rises':old.rises(log).astype(int).tolist()}
                h=old.physical_hash(log)
                if arm=='light':physics[label]=h
                elif physics[label]!=h:raise ValueError('detector changed physical motion')
        for arm in ('light','approach'):
            normal=obs['normal:'+arm];hit=obs['offset:'+arm];t=hit['onset']
            hit['genuine_contact_rises']=[] if t is None else [x for x in hit['rises'] if t<=x<t+50
                and not any(abs(y-x)<=20 for y in normal['rises'])]
        rows.append({'mechanics':variant['kind'],'offset_m':case['offset'],'physical_and_zero_external_force_certified':True,'records':obs})
    return {'schema':'r23-measured-offset-replay.v1','classification':'separate scheduled pregrasp research witness; previously exposed cases, not R23 FREE or a new holdout',
            'episodes':12,'case_id':case['id'],'mass_kg':case['mass'],'candidate_sha256':old.sha(Path(__file__).resolve().parents[1]/'r23_tool_contact/candidate.py'),
            'model_sha256':old.sha(Path(__file__).resolve().parents[1]/'r23_tool_contact/model.json'),'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    data=run()
    with a.out.open('x') as fp:json.dump(data,fp,indent=2,allow_nan=False);fp.write('\n')
