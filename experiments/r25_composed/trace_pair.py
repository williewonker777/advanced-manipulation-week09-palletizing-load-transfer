"""Export measured force and alarm intervals for the disclosed fragmented-force case, not the course demo."""
import argparse
import json
from pathlib import Path

from experiments.r24_holdout import evaluator as lab
from experiments.r24_post.diagnose import intervals
from experiments.r25_fragment.study import ANCHOR,constructor


def run(arm):
    plan=json.loads(lab.PLAN.read_bytes())
    c=next(c for c in plan['paired'] if c['id']=='p24_transfer_near_cross')
    v=plan['variants']['actuator_delay_gain']
    schedule={'none':(None,None),'first':(c['impacts'][0],None),
              'second':(None,c['impacts'][1]),'both':tuple(c['impacts'])}
    result={};phy={}
    for a in [ANCHOR,arm]:
        result[a]={};logs={}
        for name,forces in schedule.items():
            _,log=lab.run_episode(c,v,constructor(a),forces)
            if not lab.certified.exact_force(log,forces)['passed']:raise ValueError('different measured input')
            bi=intervals(log['detect'])
            applied=intervals((log['f_ext_xyz_committed']!=0).any(axis=1))
            if len(applied)!=sum(x is not None for x in forces):
                raise ValueError('applied-force interval count differs from measured twin')
            result[a][name]={'alarms_ms':bi,'actual_force_half_open_ms':applied,
                             'physics_sha256':lab.certified.physical_hash(log)}
            if a==ANCHOR:phy[name]=result[a][name]['physics_sha256']
            elif phy[name]!=result[a][name]['physics_sha256']:raise ValueError('changed physical trace')
            if name in ('second','both') and a==ANCHOR:
                result[a][name]['force_xy_samples_1ms_N']=log['f_ext_xyz_committed'][1480:1851,:2].tolist()
            logs[name]=log
        checks=[lab.certified.prefix(logs['none'],logs[tw],lab.certified.actual_start(logs[tw]['t'],f)[0],
                                      lab.certified.actual_start(logs[tw]['t'],f)[0])['passed']
                for tw,f in [('first',c['impacts'][0]),('second',c['impacts'][1])]]
        if not all(checks):raise ValueError('different pre-impact data')
    return {'schema':'r25-fragmentation-observed-twins.v1','scope':'Disclosed research run on the original physics; not the stock course demo.',
            'source_study':'p24_transfer_near_cross','candidate':arm,'mechanics':v['kind'],'case':c,'episodes':8,
            'force_xy_first_sample_index':1480,'records':result}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();doc=run(a.arm)
    with a.out.open('x')as f:json.dump(doc,f,indent=1,allow_nan=False);f.write('\n')
    print({k:{t:x['alarms_ms'] for t,x in v.items()} for k,v in doc['records'].items()})
