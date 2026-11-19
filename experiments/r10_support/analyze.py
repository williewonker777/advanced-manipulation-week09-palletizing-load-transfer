"""Derive held-negative thresholds and support timing from a trace ledger."""
from __future__ import annotations
import argparse,json
import numpy as np
from sim import trajectory as T

DT=.001
G4_WINDOW=.020
REQUIRED_TRANSFER=.075


def first_persistent(t, values, threshold, samples=4):
    mask=np.asarray(values)>=threshold
    for i in range(len(mask)-samples+1):
        if bool(np.all(mask[i:i+samples])):
            return float(t[i])
    return None


def first_true(t, mask, samples=3):
    mask=np.asarray(mask,bool)
    for i in range(len(mask)-samples+1):
        if bool(np.all(mask[i:i+samples])):
            return float(t[i])
    return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('ledger')
    args=parser.parse_args()
    rows=[json.loads(line) for line in open(args.ledger,encoding='utf-8') if line.strip()]
    held_gravity=[]; held_delta=[]
    for row in rows:
        if row['case']['role']!='threshold_negative': continue
        tr=row['trace']; time=np.array([x['t'] for x in tr]); table=np.array([x['table_over_mg'] for x in tr]); gravity=np.array([x['gravity_support'] for x in tr]); dynamic=np.array([x['dynamic_support'] for x in tr])
        contact=np.where(table>=.001)[0]
        stop=float(time[contact[0]]) if len(contact) else T.T_RELEASE
        held=(time<stop)&(time<T.T_RELEASE)&(table<.001)
        held_gravity.extend(gravity[held].tolist())
        delta=np.r_[np.zeros(5),dynamic[5:]-dynamic[:-5]]
        valid=held&np.r_[np.zeros(5,dtype=bool),held[:-5]]
        held_delta.extend(delta[valid].tolist())
    # Fixed 1%-payload guard is added beyond every held-negative sample. No
    # positive/support/impact trace contributes to either threshold.
    gravity_threshold=float(max(held_gravity)+.01)
    dynamic_delta_threshold=float(max(held_delta)+.01)
    result=[]
    for row in rows:
        tr=row['trace']; t=np.array([x['t'] for x in tr]); table=np.array([x['table_over_mg'] for x in tr]); gravity=np.array([x['gravity_support'] for x in tr]); dynamic=np.array([x['dynamic_support'] for x in tr]); delta=np.r_[np.zeros(5),dynamic[5:]-dynamic[:-5]]
        observer=np.array([x['observer_support'] if x['observer_valid'] else -9. for x in tr])
        truth_onset=first_persistent(t,table,.01,4)
        truth_quarter=first_persistent(t,table,.25,4)
        before_release=t<T.T_RELEASE
        gravity_time=first_persistent(t,np.where(before_release,gravity,-np.inf),gravity_threshold,3)
        dynamic_time=first_persistent(t,np.where(before_release,delta,-np.inf),dynamic_delta_threshold,3)
        observer_time=first_persistent(t,np.where(before_release,observer,-np.inf),.05,4)
        fixed=(t>=T.T_RELEASE-G4_WINDOW)&(t<T.T_RELEASE)
        g4_mean=float(table[fixed].mean()); g4_min=float(table[fixed].min())
        g4_equivalent=None
        count=int(round(G4_WINDOW/DT))
        for i in range(len(t)-count+1):
            if t[i]<3.7 or t[i+count-1]>=T.T_RELEASE: continue
            window=table[i:i+count]
            if float(window.mean())>=.5 and float(window.min())>=.25:
                g4_equivalent=float(t[i+count-1]+DT); break
        result.append(dict(mode=row['mode'],case=row['case']['name'],role=row['case']['role'],
            truth_onset_s=truth_onset,truth_quarter_s=truth_quarter,
            ideal_quarter_margin_ms=None if truth_quarter is None else 1e3*(T.T_RELEASE-truth_quarter),
            ideal_gap_to_75ms_ms=None if truth_quarter is None else 1e3*(truth_quarter-(T.T_RELEASE-REQUIRED_TRANSFER)),
            gravity_detect_s=gravity_time,dynamic_change_detect_s=dynamic_time,
            existing_observer_detect_s=observer_time,
            observer_lag_from_quarter_ms=None if truth_quarter is None or observer_time is None else 1e3*(observer_time-truth_quarter),
            g4_fixed_mean=g4_mean,g4_fixed_min=g4_min,g4_fixed_pass=bool(g4_mean>=.5 and g4_min>=.25),
            first_g4_equivalent_window_end_s=g4_equivalent))
    print(json.dumps(dict(schema='R10_SUPPORT_TIMING.v1',release_s=T.T_RELEASE,
        required_transfer_ms=1e3*REQUIRED_TRANSFER,g4_fixed_window_s=[T.T_RELEASE-G4_WINDOW,T.T_RELEASE],
        threshold_provenance=dict(roles=['threshold_negative'],truth_requirement='table_over_mg < 0.001',
            positive_samples_used=False,gravity_level=float(max(held_gravity)),gravity_threshold=gravity_threshold,
            dynamic_5ms_delta=float(max(held_delta)),dynamic_5ms_delta_threshold=dynamic_delta_threshold,
            persistence_samples=3),rows=result),indent=2))

if __name__=='__main__': main()
