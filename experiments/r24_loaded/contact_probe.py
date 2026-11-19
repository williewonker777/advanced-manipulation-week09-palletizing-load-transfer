"""Compare base yaw to table reaction after the simulation; truth is never a detector input."""
import argparse
import json
from pathlib import Path
import types

import mujoco
import numpy as np

from experiments.r23_holdout import evaluator as held
from experiments.r24_loaded.probe import Probe


class MechanismProbe(Probe):
    def update(self,t,th,thd,tau_m,tau_s,grip):
        out = super().update(t,th,thd,tau_m,tau_s,grip)
        if 3.58 <= t < 4.15:
            s=self._shared
            jac,point,_=s._point_jacobian(s.q_prev)
            support=s.last_table_wrench
            self.rows[-1].update(yaw_Nm=self.yaw_value,bound_Nm=float(self.bounds[int(bool(grip))]),
                support_yaw_Nm=float(jac[:,0]@support),support_wrench=support.tolist(),bottom_xyz_m=point.tolist(),
                raw_yaw_R10_Nm=float(jac[:,0]@s.last_wrench))
        return out


def contact_plant(second,variant):
    parent=held.local_plant(second,variant)
    class PhysicalRecorder(parent):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.truth=[]; self.box_body=self.m.body('box_free').id if self.has_box else -1
            self.tables={self.m.geom(n).id for n in ('table_pick','table_place')}
        def step(self,tau):
            out=super().step(tau)
            if 3579 <= self.k <= 4150:
                table=np.zeros(6)
                for c in range(self.d.ncon):
                    contact=self.d.contact[c]
                    a,b=contact.geom
                    if a in self.tables and self.m.geom_bodyid[b]==self.box_body:sign=1.
                    elif b in self.tables and self.m.geom_bodyid[a]==self.box_body:sign=-1.
                    else:continue
                    f=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,c,f)
                    axis=contact.frame.reshape(3,3).T
                    fw=sign*(axis@f[:3]);tw=sign*(axis@f[3:])
                    table[:3]+=fw;table[3:]+=np.cross(contact.pos,fw)+tw
                self.truth.append((self.k,table))
            return out
    return PhysicalRecorder


def one(case,variant,impact):
    d=MechanismProbe()
    original=held.run_episode
    runner=types.FunctionType(original.__code__, dict(original.__globals__,local_plant=contact_plant))
    _,log=runner(case,variant,lambda:d,(impact,None))
    c=held.certified; assert c.exact_force(log,(impact,None))['passed']
    owner=log['plant'];truth=dict(owner.truth)
    for row in d.rows:
        ix=round(row['t']*1000)
        row.update(table_N=float(log['f_table'][ix]),external_N=float(log['f_ext'][ix]),table_world=truth[ix+1].tolist(),
                   tool_N=float(log['f_tool'][ix]))
    return {'rises':c.rises(log).astype(int).tolist(),'physical_sha256':c.physical_hash(log),'rows':d.rows}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--panel',choices=('mechanism','marked'),default='mechanism');a=p.parse_args()
    plan=json.loads(held.PLAN.read_bytes());case=next(c for c in plan['negative'] if c['id']=='n23_detached_heavy_interp')
    out={}
    if a.panel=='mechanism':
        for name in ('stock','actuator_skew'):
            out[name]={str(k):one(case,plan['variants'][name],impact) for k,impact in
                       [('normal',None),('nearby_x',(3.837,(14.,0.,0.))),('nearby_y',(3.847,(0.,-14.,0.)))]}
    else:
        raise ValueError('unsupported expansion')
    with a.out.open('x') as f:json.dump(out,f,allow_nan=False)
    for name,sets in out.items():
        for k,v in sets.items():
            ix=next((i for i,r in enumerate(v['rows']) if r['t']==3.861),None); rows=v['rows'];r=rows[ix]
            print(name,k,v['rises'],[(w,round(r[w],3)) for w in ('yaw_Nm','bound_Nm','support_yaw_Nm','raw_yaw_R10_Nm')],
                  'truth Fz minimum',round(min(y['table_world'][2] for y in rows),4))


if __name__=='__main__':
    main()
