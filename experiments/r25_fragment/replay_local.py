"""Read-only preliminary comparison of signed continuations on newly measured R24 twins."""
import argparse
import json
from pathlib import Path
import numpy as np
import my_detector
from experiments.r25_fragment.event import SignedEvent
from experiments.r15_delayed_wrench.observer import LAG
from experiments.r24_post.diagnose import intervals


def run(data):
    out={}
    for arm,pred,station in [('signed',False,False),('predict',True,False),('stationary',False,True)]:
        out[arm]={}
        for tw,tr in data['twins'].items():
            e=SignedEvent(pred,station);rows=[dict(zip(data['fields'],x)) for x in tr['samples']]
            merged=[];extra=[]
            for v in rows:
                ext=e.feed(v['normal_model'],np.asarray(v['normal_xy_error'],dtype=float),np.asarray(v['inverse_force'],dtype=float),
                           v['grip'],8,LAG,my_detector.MODEL_PARAMETERS['torque']['threshold'])
                merged.append(ext or v['alarm']);extra.append(ext and not v['alarm'])
            i0=rows[0]['tick']
            offset=lambda xs:[[a+i0,b+i0] for a,b in intervals(xs)]
            out[arm][tw]={'merged_intervals':offset(merged),'extended_intervals':offset(extra)}
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);q=p.parse_args()
    result=run(json.loads(q.input.read_bytes()))
    with q.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print({a:{k:r['merged_intervals'] for k,r in v.items()} for a,v in result.items()})
