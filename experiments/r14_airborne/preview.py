"""Compare nominal height information at the original and the TCP-preserving tilted motion."""
import argparse
import json
from pathlib import Path
import numpy as np
from sim import config as c, trajectory as t
from sim.nominal import Nominal
from experiments.r11_datum.controller import frame
from experiments.r14_airborne.probe import START, STOP, correction, angle


def preview():
    mass=6.; com=np.array([.015,-.02,.012]); nom=Nominal()
    delta=np.eye(3)*.001
    tk=np.arange(round(START/.001),round(STOP/.001)+1)*.001
    ref=np.array([t.reference(k) for k in tk])
    change=np.array([correction(k) for k in tk])
    active=ref+change
    tcp=[];rot_error=[]
    for time,base,test in zip(tk,ref[:,0],active[:,0]):
        x,r=frame(nom,base); px,pr=frame(nom,test)
        tcp.append(np.linalg.norm(px-x))
        rel=r.T@pr
        rot_error.append(abs(np.arctan2(rel[0,2],rel[0,0])-angle(time)))
    result={'schema':'r14-airborne-preview.v1','max_reference_tcp_change_m':float(max(tcp)),
            'max_orientation_error_rad':float(max(rot_error)),'max_commanded_pitch_deg':float(np.rad2deg(max(abs(angle(s)) for s in tk))),
            'max_joint_command_displacement_deg':np.rad2deg(np.max(abs(change[:,0]),axis=0)).tolist(),
            'max_fraction_of_published_joint_speed':float(np.max(abs(active[:,1])/c.QD_MAX))}
    for name,arr in [('passive',ref),('probe',active)]:
        samples=arr[::5]
        def dyn(d):
            nom.set_payload(mass,com+d)
            return np.vstack([nom.inverse_dynamics(*q) for q in samples])
        J=np.stack([.5*(dyn(d)-dyn(-d)) for d in delta],axis=-1)
        X=((J-J.mean(axis=0))/c.JTS_NOISE[None,:,None]).reshape((-1,3))
        z=X[:,2]; xy=X[:,:2]
        rem=z-xy@np.linalg.lstsq(xy,z,rcond=None)[0]
        result[name]={'rms_4mm_height_after_xy_and_bias_sensor_noise_units':float(4*np.mean(rem**2)**.5),
                      'max_joint_signal_for_4mm_nm':np.max(abs(4*J[:,:,2]),axis=0).tolist()}
    result['geometric_gate']=result['max_reference_tcp_change_m']<=1e-8 and result['max_orientation_error_rad']<=1e-8 and result['max_fraction_of_published_joint_speed']<=1
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);args=parser.parse_args()
    data=preview()
    with args.out.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
    print(json.dumps(data))
