"""Strict four-twin scorer for a predeclared two-impact research case."""
from __future__ import annotations
import numpy as np
from tools.r14_paired_events import match_false_events
from .runner import expected_signed

TWINS=('none','first','second','both')
CAUSAL_KEYS=('q','q_d','tau','e_mm','f_table','f_tool','qpos','r','detect','f_ext',
             'f_ext_xyz_committed','f_ext_xyz_native','f_ext_xyz_second','f_ext_combined_norm')

def rise_indices(trace):
    detect=np.asarray(trace['detect'],bool);return np.flatnonzero(detect & ~np.r_[False,detect[:-1]])

def event_rises(trace,onset,window_s):
    t=np.asarray(trace['t']);r=rise_indices(trace);return r[(t[r]>=onset)&(t[r]<onset+window_s)]

def false_times(trace,impacts,window_s):
    t=np.asarray(trace['t']);r=rise_indices(trace);true=np.zeros(len(t),bool)
    for impact in impacts:
        if impact is not None:
            onset=impact[0];true|=(t>=onset)&(t<onset+window_s)
    return t[r[~true[r]]].tolist()

def prefix_equal(left,right,onset):
    t=np.asarray(left['t']);other=np.asarray(right['t'])
    if t.shape!=(6000,) or not np.array_equal(t,other):raise ValueError('complete matched traces required')
    end=int(np.searchsorted(t,float(onset),side='right'))
    fields={key:bool(np.array_equal(np.asarray(left[key])[:end],np.asarray(right[key])[:end])) for key in CAUSAL_KEYS}
    return {'passed':all(fields.values()),'samples':end,'fields':fields}

def profile(trace,first,second):
    t=np.asarray(trace['t']);committed=np.asarray(trace['f_ext_xyz_committed']);native=np.asarray(trace['f_ext_xyz_native']);extra=np.asarray(trace['f_ext_xyz_second'])
    expected_native=expected_signed(t,[first]);expected_extra=expected_signed(t,[second]);expected=expected_native+expected_extra
    combined=np.linalg.norm(committed,axis=1)
    gates={'committed_signed_exact':bool(np.array_equal(committed,expected)),
           'native_first_exact':bool(np.array_equal(native,expected_native)),
           'hook_second_exact':bool(np.array_equal(extra,expected_extra)),
           'stock_native_magnitude_exact':bool(np.array_equal(np.asarray(trace['f_ext']),np.linalg.norm(native,axis=1))),
           'combined_magnitude_exact':bool(np.array_equal(np.asarray(trace['f_ext_combined_norm']),combined))}
    return {'passed':all(gates.values()),'gates':gates,'peak_committed_xyz_N':committed[np.argmax(combined)].tolist(),
            'native_active_samples':int(np.any(native!=0.,axis=1).sum()),'second_active_samples':int(np.any(extra!=0.,axis=1).sum()),
            'combined_active_samples':int(np.any(committed!=0.,axis=1).sum())}

def quiet(trace,onset,samples):
    t=np.asarray(trace['t']);k=int(np.searchsorted(t,onset));return not np.asarray(trace['detect'],bool)[max(0,k-samples):k].any()

def valid_genuine(candidate,references,rises,guard_samples,onset):
    t=np.asarray(candidate['t']);onset_index=int(np.searchsorted(t,float(onset)))
    valid=[]
    for index in rises:
        # At the programmed onset the trapezoid is still exactly zero.  A rise
        # there cannot be credited as a physical response.
        if index<=onset_index: continue
        lo,hi=max(0,index-guard_samples),min(len(t),index+guard_samples+1)
        if all(not np.asarray(reference['detect'],bool)[lo:hi].any() for reference in references):
            valid.append(int(index))
    return np.asarray(valid,dtype=int)

def finite_grid(trace):
    t=np.asarray(trace['t'])
    arrays=[np.asarray(trace[key]) for key in CAUSAL_KEYS]
    return bool(t.shape==(6000,) and np.array_equal(t,.001*np.arange(6000)) and
                all(np.isfinite(value).all() for value in arrays) and trace.get('r18_diverged') is False)

def direction_preserved(trace,impact):
    t=np.asarray(trace['t']);index=int(np.searchsorted(t,float(impact[0])+.030))
    return bool(np.dot(np.asarray(trace['f_ext_xyz_committed'])[index],np.asarray(impact[1],float))>0.)

def assess(case,traces,quiet_ms,event_window_ms):
    if set(traces)!=set(TWINS):raise ValueError('none/first/second/both twins required')
    first,second=case['impacts'];t=np.asarray(traces['both']['t']);dt=float(t[1]-t[0]);quiet_n=max(1,int(round(quiet_ms/1000./dt)));guard_n=max(1,int(round(.020/dt)));window=event_window_ms/1000.
    assignments={'none':(None,None),'first':(first,None),'second':(None,second),'both':(first,second)}
    profiles={name:profile(traces[name],*assignments[name]) for name in TWINS}
    causal={'none_vs_first_pre_first':prefix_equal(traces['none'],traces['first'],first[0]),
            'none_vs_both_pre_first':prefix_equal(traces['none'],traces['both'],first[0]),
            'first_vs_both_pre_second':prefix_equal(traces['first'],traces['both'],second[0]),
            'second_vs_none_pre_second':prefix_equal(traces['second'],traces['none'],second[0])}
    raw={'isolated_first':event_rises(traces['first'],first[0],window),
         'isolated_second':event_rises(traces['second'],second[0],window),
         'both_first':event_rises(traces['both'],first[0],min(window,second[0]-first[0])),
         'both_second':event_rises(traces['both'],second[0],window)}
    valid={'isolated_first':valid_genuine(traces['first'],[traces['none']],raw['isolated_first'],guard_n,first[0]),
           'isolated_second':valid_genuine(traces['second'],[traces['none']],raw['isolated_second'],guard_n,second[0]),
           'both_first':valid_genuine(traces['both'],[traces['none'],traces['second']],raw['both_first'],guard_n,first[0]),
           'both_second':valid_genuine(traces['both'],[traces['first']],raw['both_second'],guard_n,second[0])}
    false={name:false_times(traces[name],assignments[name],window) for name in TWINS}
    false_match={name:match_false_events(false['both'],false[name]) for name in ('none','first','second')}
    normal_window=bool(event_rises(traces['none'],first[0],window).size or event_rises(traces['none'],second[0],window).size)
    direction_ok=(direction_preserved(traces['first'],first) and direction_preserved(traces['second'],second)
                  and direction_preserved(traces['both'],first) and direction_preserved(traces['both'],second))
    distinct=(not np.array_equal(np.asarray(first[1]),np.asarray(second[1])) and profiles['both']['second_active_samples']>0)
    genuine_gates={key+'_genuine':len(valid[key])>0 and len(valid[key])==len(raw[key]) for key in raw}
    gates={'complete_finite_strict_grid':all(finite_grid(traces[n]) for n in TWINS),
           'signed_profiles_exact':all(x['passed'] for x in profiles.values()),
           'force_free_prefixes':all(not np.any(np.asarray(traces[n]['f_ext_xyz_committed'])[:int(np.searchsorted(t,impact[0]))]) for n,impact in (('first',first),('second',second),('both',first))),
           'all_required_prefixes_exact':all(x['passed'] for x in causal.values()),
           **genuine_gates,
           'quiet_before_first':quiet(traces['first'],first[0],quiet_n) and quiet(traces['both'],first[0],quiet_n),
           'quiet_before_second':quiet(traces['second'],second[0],quiet_n) and quiet(traces['both'],second[0],quiet_n),
           'distinct_non_noop_impacts':distinct,'no_programmed_normal_event_overlap':not normal_window,
           'no_sign_inversion':direction_ok,'no_added_unmatched_rise':all(x['passed'] for x in false_match.values())}
    onsets={'isolated_first':first[0],'isolated_second':second[0],'both_first':first[0],'both_second':second[0]}
    delays={key:[float(1000*(t[k]-onsets[key])) for k in valid[key]] for key in valid}
    return {'schema':'r18-paired-impulse-assessment.v3','id':case['id'],'profiles':profiles,'causality':causal,
            'rises_s':{n:t[rise_indices(traces[n])].tolist() for n in TWINS},
            'raw_event_times_s':{key:t[value].tolist() for key,value in raw.items()},
            'valid_genuine_times_s':{key:t[value].tolist() for key,value in valid.items()},
            'absolute_delays_ms':delays,'event_counts':{key:len(value) for key,value in valid.items()},
            'false_times_s':false,'false_event_match':false_match,'quiet_samples_required':quiet_n,
            'gates':gates,'passed':all(gates.values()),
            'retention_note':'multi-arm driver must compare these absolute delays to R10/R15/D with +5 ms bound'}
