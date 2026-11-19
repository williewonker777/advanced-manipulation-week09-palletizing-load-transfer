"""Nominal-only excitation of a grasp-point offset while the box is carried.

No fitting to plant position, command or contact outcomes. A nonzero Jacobian
in the nominal model is not a payload calibration or an accuracy guarantee.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sim import config as C, trajectory as T
from sim.nominal import Nominal


def preview():
    mass=6.
    com=np.array([.015,-.02,.012])
    dz=.001
    t=np.arange(1.85,3.51,.005)
    refs=[T.reference(s) for s in t]
    scales=np.asarray(C.JTS_NOISE)
    result={}
    for name, dynamic in [('gravity_only',False),('nominal_carry_dynamics',True)]:
        cols=[]
        for direction in np.eye(3):
            a,b=Nominal(),Nominal()
            a.set_payload(mass,com+dz*direction);b.set_payload(mass,com-dz*direction)
            rows=[]
            for q,v,acc in refs:
                if not dynamic:v=acc=np.zeros(6)
                rows.append((a.inverse_dynamics(q,v,acc)-b.inverse_dynamics(q,v,acc))*.5)
            cols.append(rows)
        # Columns are joint torque change per millimetre of body COM displacement.
        j=np.moveaxis(np.asarray(cols),0,-1)
        xyz=j.reshape((-1,3))
        weighted=(j/scales[None,:,None]).reshape((-1,3))
        # Remove any constant joint bias (and the two tangential columns) before
        # claiming the vertical translation has a distinguishable signal.
        centered=(j-j.mean(axis=0))/scales[None,:,None]
        mat=centered.reshape((-1,3))
        target=mat[:,2]
        fit=np.linalg.lstsq(mat[:,:2],target,rcond=None)[0]
        remaining=target-mat[:,:2]@fit
        result[name]={'samples':len(t),'rms_torque_nm_per_mm_xyz':np.sqrt(np.mean(xyz**2,axis=0)).tolist(),
            'max_joint_change_for_4mm_z_nm':(4*np.max(abs(j[:,:,2]),axis=0)).tolist(),
            'noise_scaled_rank_singular_values':np.linalg.svd(weighted,compute_uv=False).tolist(),
            'z_rms_after_xy_and_joint_bias_per_4mm_in_sensor_noise_units':float(4*np.mean(remaining**2)**.5),
            'note':'Finite nominal sensitivity at taught motion; noise scaling is per-channel, not independent repeated evidence.'}
    return {'schema':'r12-carried-offset-preview.v1','mass_kg':mass,'nominal_com_m':com.tolist(),
            'sample_time_s':[float(t[0]),float(t[-1])],'probe_mm':1,
            'scope':'Known mass and COM; pure translational change of effective payload COM. Does not identify grasp rotation, structural errors, actual offset, or contact force.',
            'conditions':result}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('new output required')
    with a.out.open('x') as f:json.dump(preview(),f,indent=2,allow_nan=False);f.write('\n')
