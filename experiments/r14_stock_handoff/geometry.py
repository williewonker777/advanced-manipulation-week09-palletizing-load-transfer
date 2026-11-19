"""Manufactured fixed-pose FK check only; no episode or scoring truth."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from sim import config as C
from sim.nominal import Nominal
from experiments.r14_stock_handoff.controller import Controller


def state(nom,q):
    nom.d.qpos[:]=q;mujoco.mj_forward(nom.m,nom.d);b=nom.m.body('link_6').id
    p=nom.d.xpos[b].copy();r=nom.d.xmat[b].reshape(3,3).copy();half=.5*np.asarray(C.BOX_SIZE);centre=p+r@np.asarray(C.BOX_POS)
    corners=np.array([centre+r@np.array([sx*half[0],sy*half[1],sz*half[2]]) for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
    return p,r,corners

def main(out):
    c=Controller.__new__(Controller);c.nom=Nominal();c.dt=float(C.DT);c.ik_max_position_error_m=0.;c.ik_max_orientation_error_rad=0.;rows=[]
    for t in (3.79,3.90):
        q=c._base_reference(t)[0];p0,r0,k0=state(c.nom,q)
        for dz in (-.008,.008):
            q1=c._full_pose(q,dz);p1,r1,k1=state(c.nom,q1);shift=k1-k0
            rows.append({'time_s':t,'command_world_z_mm':1000*dz,'flange_shift_mm':(1000*(p1-p0)).tolist(),'max_flange_rotation_matrix_error':float(np.max(np.abs(r1-r0))),'corner_shift_mm':(1000*shift).tolist(),'max_corner_translation_error_um':float(1e6*np.max(np.abs(shift-np.array([0,0,dz]))))})
    result={'schema':'r14-stock-full-pose-geometry.v1','rows':rows,'max_ik_position_error_m':c.ik_max_position_error_m,'max_ik_orientation_error_rad':c.ik_max_orientation_error_rad}
    with Path(out).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');a=p.parse_args();main(a.out)
