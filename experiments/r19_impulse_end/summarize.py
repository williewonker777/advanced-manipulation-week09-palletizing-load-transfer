"""Derive the bounded exposed-case verdict for the frozen impulse boundary."""
import argparse,json,hashlib
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
 if a.out.exists():p.error('fresh output required')
 result=read(a.run/'RESULT.json');assert result['complete'] and result['passed'] and len(result['rows'])==4
 rows={r['id']:r for r in result['rows']};ids=('pair_return_opposed','diag_known_late_fz','diag_known_late_fx','gap80_weak');assert tuple(rows)==ids
 for r in rows.values():
  assert all(r['gates'].values());assert all(all(v.values()) for v in r['physics'].values());assert all(all(v.values()) for v in r['comparisons'].values() for v in v.values())
 event={k:rows[k]['arms']['event'] for k in ids}
 summary={'schema':'r19-impulse-end-summary.v1','decision':'passes_four_previously_exposed_rows_not_independent_not_promoted','episodes':80,
  'exact_R18_onsets':{'pair_return_opposed_both_second_ms':event['pair_return_opposed']['slots']['both_second']['credited_delays_ms'],'anchor_fz_both_ms':[event['diag_known_late_fz']['slots']['both_first']['credited_delays_ms'],event['diag_known_late_fz']['slots']['both_second']['credited_delays_ms']],'anchor_fx_both_ms':[event['diag_known_late_fx']['slots']['both_first']['credited_delays_ms'],event['diag_known_late_fx']['slots']['both_second']['credited_delays_ms']],'gap80_both_ms':[event['gap80_weak']['slots']['both_first']['credited_delays_ms'],event['gap80_weak']['slots']['both_second']['credited_delays_ms']]},
  'both_intervals_ms':{k:event[k]['levels']['both']['on_intervals_ms'] for k in ids},
  'protected':{'all_force_prefix_and_physics_gates':True,'exact_D_R18_physics_and_R18_residual':True,'no_added_false':True,'no_lost_tool':True,'no_stuck_tail':True,'quiet_at_each_credited_onset':True},
  'scope':{'previously_exposed_rows_only':True,'new_independent_stimuli_read':False,'promotion_eligible':False,'submission_changed':False},
  'limitations':'The boundary passed 80 replays over four already-exposed rows. This is a development result only; independent normals, strong/alternating physical pulses, other axes/gaps and timing remain unopened by this lane.',
  'evidence':{f.name:sha(f) for f in a.run.glob('*.json')}}
 a.out.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(summary['decision'],summary['both_intervals_ms'])
if __name__=='__main__':main()
