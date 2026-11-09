"""Separate tool-contact and artificial-event scoring without changing the kit."""
import numpy as np


def event_result(t,detect,case,f_tool,reference=None):
    t=np.asarray(t);detect=np.asarray(detect,dtype=bool);n=len(t)
    if n!=6000 or detect.shape!=(n,) or np.asarray(f_tool).shape!=(n,):
        raise ValueError('a complete fixed-grid episode is required')
    rise=detect&~np.r_[False,detect[:-1]]
    true=np.zeros(n,bool);delay=None;pre=False;outcome='reference';det_index=None
    collision=case.get('collision')
    if collision is not None:
        tc=collision[0];artificial=(t>=tc)&(t<tc+.32);true|=artificial
        k0=int(np.searchsorted(t,tc));pre=bool(k0 and detect[k0-1])
        found=np.flatnonzero(rise&artificial)
        if not pre and len(found):
            det_index=int(found[0]);delay=float((t[det_index]-tc)*1000)
        if pre:outcome='masked'
        elif delay is None:outcome='missed'
        elif reference is not None and np.asarray(reference)[max(0,det_index-20):det_index+21].any():
            outcome='indistinguishable'
        else:outcome='detected'
    tool_hit=np.asarray(f_tool)>1.;tool_detected=False;tool_pre=False;tool_delay=None
    if tool_hit.any():
        first,last=np.flatnonzero(tool_hit)[[0,-1]]
        tool=(np.arange(n)>=first)&(np.arange(n)<min(n,last+300));true|=tool
        tool_pre=bool(first and detect[first-1]);found=np.flatnonzero(rise&tool)
        tool_detected=bool(len(found) and not tool_pre)
        if tool_detected:tool_delay=float((t[found[0]]-t[first])*1000)
    false_indices=np.flatnonzero(rise&~true)
    return dict(outcome=outcome,delay_ms=delay,prealarm=pre,false_alarms=int(len(false_indices)),
        false_alarm_times=t[false_indices].tolist(),tool_present=bool(tool_hit.any()),
        tool_detected=tool_detected,tool_prealarm=tool_pre,tool_delay_ms=tool_delay)


def case_pair_key(case,mode):
    """All hidden fixture settings are provenance only, never model inputs."""
    return (mode,float(case['mass']),tuple(case['com']),
        case['sensor_seeds'][mode],float(case['stiffness_scale']),
        float(case['viscous_scale']),float(case['coulomb_scale']))


def assert_paired_case(candidate,reference):
    if candidate!=reference:raise ValueError('source ID or physical/sensor condition mismatch')
