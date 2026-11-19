"""Relocatable replay. The scientific runner body is byte-identical to the preregistered one."""
from __future__ import annotations
import argparse,hashlib,importlib,json,os,types
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PLAN=HERE/'plan.json'
PROTOCOL=HERE/'registration.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def resolve(entry):
    module, name=entry.rsplit(':',1)
    return getattr(importlib.import_module(module),name)


def verify():
    import mujoco
    p=read(PROTOCOL)
    if (sha(PROTOCOL)!='08071dd20f19e341939b86da25c8f8864f3682072488709da85d8413b96e2f9a'
            or sha(PLAN)!=p['plan_sha256'] or p['runner_sha256']!='6c566a1dd260c02e4686fbfddd533a9187d5db7c87d07255796513baa17b9b16'
            or {n:sha(ROOT/n)for n in p['sources']}!=p['sources'] or mujoco.__version__!='3.17.0'):
        raise ValueError('registered three-arm source, plan, runtime or original kit differs')
    start=b'def '+b'plant_type('
    data=Path(__file__).read_bytes()
    if hashlib.sha256(data[data.index(start):]).hexdigest()!='ab2bd9ac3d7a9031b45ef130c1423db787b6e38c539edfeec2b9459d3eed57ab':
        raise ValueError('frozen physics and scoring body differs')
    return p

def plant_type(second,m):
 from experiments.r18_paired_impulses import runner as paired
 parent=paired.instrumented(second);vals=[float(m[k]) for k in ('table_friction_scale','table_solref_time_scale','suction_solref_time_scale','dof_damping_scale')];dead=float(m['torque_deadband_nm'])
 if any(not .9<=x<=1.1 for x in vals) or not 0<=dead<=.03:raise ValueError('mechanics bound')
 friction,solref,suction,damping=vals
 class Plant(parent):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);geoms=[self.m.geom(n).id for n in ('table_pick','table_place')]
   if self.has_box:geoms.extend(np.flatnonzero(self.m.geom_bodyid==self.m.body('box_free').id).tolist())
   self.m.geom_friction[geoms,0]*=friction;self.m.geom_solref[geoms,0]*=solref
   if self.m.neq:self.m.eq_solref[:,0]*=suction
   self.m.dof_damping[:]*=damping
  def step(self,tau):
   value=np.asarray(tau,float);return super().step(np.where(np.abs(value)<dead,0.,value))
 return Plant
def run(case,detector,controller,impacts):
 from experiments.r22_challenge import evaluator as certified
 from sim import runner as stock
 plant=plant_type(impacts[1],case['mechanics']);fn=types.FunctionType(stock.run_episode.__code__,dict(stock.run_episode.__globals__,Plant=plant),stock.run_episode.__name__,stock.run_episode.__defaults__,stock.run_episode.__closure__);fn.__kwdefaults__=stock.run_episode.__kwdefaults__;ctrl=controller();metrics,log=fn(ctrl,box_mass=case['mass'],box_com=tuple(case['com']),pick_offset=tuple(case['offset']),detector=detector(),collision=impacts[0]);owner=log['plant'];log['f_ext_xyz_native']=np.asarray(owner.r18_native_force);log['f_ext_xyz_second']=np.asarray(owner.r18_second_force);log['f_ext_xyz_committed']=np.asarray(owner.r18_committed_force);fields=certified.PHYSICAL+('r','detect')
 if metrics['diverged'] or any(np.asarray(log[k]).shape[0]!=6000 for k in fields) or any(not np.isfinite(np.asarray(log[k])).all() for k in fields) or not certified.exact_force(log,impacts)['passed']:raise ValueError('invalid R29 episode')
 return metrics,log,ctrl
def record(metrics,log,ctrl,mass,impacts,starts,window):
 from experiments.r22_challenge import evaluator as certified
 from experiments.r14_stock_handoff.qualification import compact
 bits=np.asarray(log['detect'],bool);edge=certified.rises(log);g=compact(metrics,log,ctrl,mass);g.pop('ctrl_p99_us',None)
 if g.get('registration'):g['registration'].pop('stream_timing_us',None)
 return {'rises':edge.astype(int).tolist(),'rise_times_s':np.asarray(log['t'])[edge].tolist(),'active_samples':int(bits.sum()),'false_times_s':certified.false_record(log,starts,window),'tool':certified.tool_record(log),'force_exact':certified.exact_force(log,impacts),'physical_sha256':certified.physical_hash(log),'stuck_tail':bool(bits[-1]),'g4':g,'ending_table_mean_N':float(np.mean(np.asarray(log['f_table'])[-20:])),'ending_table_max_N':float(np.max(np.asarray(log['f_table'])[-20:]))}
def event(log,refs,start,end,plan):
 from experiments.r23_holdout import evaluator as paired
 row=paired.event_record(log,refs,start,end,plan['scoring']['quiet_ms'],plan['scoring']['reference_twin_ms']);bits=np.asarray(log['detect'],bool);force=np.linalg.norm(np.asarray(log['f_ext_xyz_committed']),axis=1);row['window_active_samples']=int(np.count_nonzero(bits[start:end]));row['input_active_samples']=int(np.count_nonzero(force[start:end]>0));row['input_present_but_no_attributable_rise']=bool(row['input_active_samples'] and not row['attributable_rise_indices']);return row
def match(candidate,reference,tol=2):return all(any(abs(x-y)<=tol/1000 for y in reference) for x in candidate)
def edges(old,new,r=2):
 a=np.asarray(old,int);b=np.asarray(new,int);return {'added':[int(x) for x in b if not np.any(np.abs(a-x)<=r)],'lost':[int(x) for x in a if not np.any(np.abs(b-x)<=r)]}
def same_physics(records,twins):return all(len({json.dumps(records[a][tw]['physical_sha256'],sort_keys=True) for a in records})==1 for tw in twins)
def same_g4(records,twins):return all(len({json.dumps(records[a][tw]['g4'],sort_keys=True) for a in records})==1 and len({(records[a][tw]['ending_table_mean_N'],records[a][tw]['ending_table_max_N']) for a in records})==1 for tw in twins)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);ap.add_argument('--validate-only',action='store_true');a=ap.parse_args()
 if a.validate_only==(a.out is not None):ap.error('one mode')
 if a.out is not None and os.path.lexists(a.out):ap.error('fresh')
 protocol=verify();plan=read(PLAN)
 if a.validate_only:print(json.dumps({'validated':True,'seal_plan':sha(PLAN),'runner':sha(Path(__file__)),'candidate':protocol['candidate_sha256'],'episodes':90},sort_keys=True));return
 arms={n:resolve(e) for n,e in plan['arms'].items() if n!='controller'};controller=resolve(plan['arms']['controller']);rows=[];episodes=0;sc=plan['scoring']
 for case in plan['negative']:
  rec={};logs={}
  for name,det in arms.items():m,l,c=run(case,det,controller,(None,None));episodes+=1;logs[name]=l;rec[name]={'none':record(m,l,c,case['mass'],(None,None),[],sc['event_window_ms'])}
  base=rec['c1']['none'];memory=rec['memory']['none'];attrs={k:edges(rec[k]['none']['rises'],memory['rises']) for k in ('c1','v3')};preserve_tool=all(not rec[k]['none']['tool']['detected'] or memory['tool']['detected'] and memory['tool']['delay_ms']<=rec[k]['none']['tool']['delay_ms']+5 for k in ('c1','v3'))
  gates={'same_physics':same_physics(rec,('none',)),'same_g4_support':same_g4(rec,('none',)),'forces_exact':all(rec[k]['none']['force_exact']['passed'] for k in rec),'no_added_false':all(match(memory['false_times_s'],rec[k]['none']['false_times_s']) for k in ('c1','v3')),'no_added_edge':all(not attrs[k]['added'] for k in attrs),'no_lost_edge':all(not attrs[k]['lost'] for k in attrs),'preserve_tool_plus5ms':preserve_tool,'no_new_pretool':not memory['tool']['prealarm'] or any(rec[k]['none']['tool']['prealarm'] for k in ('c1','v3')),'duration_le_plus5':all(memory['active_samples']<=rec[k]['none']['active_samples']+5 for k in ('c1','v3')),'no_new_stuck':not memory['stuck_tail'] or any(rec[k]['none']['stuck_tail'] for k in ('c1','v3'))}
  rows.append({'id':case['id'],'kind':'negative','case':case,'records':rec,'edge_attribution':attrs,'gates':gates,'passed':all(gates.values()),'gains':[],'losses':[]})
 for case in plan['single']:
  logs={k:{} for k in arms};rec={}
  for name,det in arms.items():
   rec[name]={}
   for tw,imp in (('none',(None,None)),('force',(case['impact'],None))):m,l,c=run(case,det,controller,imp);episodes+=1;logs[name][tw]=l;rec[name][tw]=record(m,l,c,case['mass'],imp,[],sc['event_window_ms'])
  from experiments.r22_challenge import evaluator as certified
  start,boundary=certified.actual_start(logs['c1']['force']['t'],case['impact']);end=min(6000,start+sc['event_window_ms'])
  for name in arms:rec[name]['event']=event(logs[name]['force'],[logs[name]['none']],start,end,plan);rec[name]['prefix']=certified.prefix(logs[name]['none'],logs[name]['force'],start,start);rec[name]['force']['false_times_s']=certified.false_record(logs[name]['force'],[start],sc['event_window_ms'])
  mem=rec['memory'];attrs={k:{tw:edges(rec[k][tw]['rises'],mem[tw]['rises']) for tw in ('none','force')} for k in ('c1','v3')};preserve={k:(not rec[k]['event']['strict'] or mem['event']['strict'] and mem['event']['delays_ms'][0]<=rec[k]['event']['delays_ms'][0]+5) for k in ('c1','v3')};place=np.flatnonzero((np.asarray(logs['memory']['force']['t'])>=sc['post_place_search_start_s'])&(np.asarray(logs['memory']['force']['f_table'])>sc['table_contact_threshold_N']));late_ok=case['phase']!='late_box_support' or bool(len(place) and int(place[0])<start)
  gates={'same_physics':same_physics(rec,('none','force')),'same_g4_support':same_g4(rec,('none','force')),'force_prefix_exact':all(rec[k]['force']['force_exact']['passed'] and rec[k]['prefix']['passed'] for k in rec),'late_support_valid':late_ok,'preserve_c1_strict_plus5ms':preserve['c1'],'preserve_v3_strict_plus5ms':preserve['v3'],'preserve_quiet':all(not rec[k]['event']['quiet_before'] or mem['event']['quiet_before'] for k in ('c1','v3')),'duration_le_plus5':all(mem['event']['window_active_samples']<=rec[k]['event']['window_active_samples']+5 for k in ('c1','v3')),'no_added_duplicate':all(mem['event']['duplicate_rises']<=rec[k]['event']['duplicate_rises'] for k in ('c1','v3')),'no_added_none_edge':all(not attrs[k]['none']['added'] for k in attrs),'no_added_false':all(match(mem['force']['false_times_s'],rec[k]['force']['false_times_s']) for k in ('c1','v3')),'no_new_stuck':not mem['force']['stuck_tail'] or any(rec[k]['force']['stuck_tail'] for k in ('c1','v3'))}
  gains=[k for k in ('c1','v3') if not rec[k]['event']['strict'] and mem['event']['strict'] and mem['event']['quiet_before'] and mem['event']['duplicate_rises']==0 and mem['event']['input_active_samples']>0 and mem['prefix']['passed']];losses=[k for k in ('c1','v3') if rec[k]['event']['strict'] and not preserve[k]]
  rows.append({'id':case['id'],'kind':'single','case':case,'actual_start':start,'boundary':boundary,'first_place_contact_tick':None if not len(place) else int(place[0]),'records':rec,'edge_attribution':attrs,'gates':gates,'passed':all(gates.values()),'gains':gains,'losses':losses})
 for case in plan['paired']:
  assignment={'none':(None,None),'first':(case['impacts'][0],None),'second':(None,case['impacts'][1]),'both':tuple(case['impacts'])};logs={k:{} for k in arms};rec={}
  for name,det in arms.items():
   rec[name]={}
   for tw,imp in assignment.items():m,l,c=run(case,det,controller,imp);episodes+=1;logs[name][tw]=l;rec[name][tw]=record(m,l,c,case['mass'],imp,[],sc['event_window_ms'])
  from experiments.r22_challenge import evaluator as certified
  starts=[certified.actual_start(logs['c1']['both']['t'],x)[0] for x in case['impacts']];first,second=starts;windows={'isolated_first':('first',first,min(second,first+320),['none']),'isolated_second':('second',second,min(6000,second+320),['none']),'both_first':('both',first,min(second,first+320),['none','second']),'both_second':('both',second,min(6000,second+320),['first'])}
  for name in arms:
   rec[name]['events']={slot:event(logs[name][tw],[logs[name][x] for x in refs],begin,end,plan) for slot,(tw,begin,end,refs) in windows.items()};rec[name]['prefixes']={'first':certified.prefix(logs[name]['none'],logs[name]['first'],first,first),'second':certified.prefix(logs[name]['none'],logs[name]['second'],second,second),'both_first':certified.prefix(logs[name]['none'],logs[name]['both'],first,first),'both_second':certified.prefix(logs[name]['first'],logs[name]['both'],second,second)}
   for tw,imp in assignment.items():rec[name][tw]['false_times_s']=certified.false_record(logs[name][tw],[first if imp[0] else None,second if imp[1] else None],320)
  mem=rec['memory'];attrs={k:{tw:edges(rec[k][tw]['rises'],mem[tw]['rises']) for tw in assignment} for k in ('c1','v3')};preserve={k:{slot:(not ev['strict'] or mem['events'][slot]['strict'] and mem['events'][slot]['delays_ms'][0]<=ev['delays_ms'][0]+5) for slot,ev in rec[k]['events'].items()} for k in ('c1','v3')}
  gates={'same_physics':same_physics(rec,tuple(assignment)),'same_g4_support':same_g4(rec,tuple(assignment)),'force_prefix_exact':all(rec[k][tw]['force_exact']['passed'] for k in rec for tw in assignment) and all(x['passed'] for k in rec for x in rec[k]['prefixes'].values()),'preserve_strict_plus5ms':all(all(v.values()) for v in preserve.values()),'preserve_quiet':all(not rec[k]['events'][slot]['quiet_before'] or mem['events'][slot]['quiet_before'] for k in ('c1','v3') for slot in windows),'duration_le_plus5':all(mem['events'][slot]['window_active_samples']<=rec[k]['events'][slot]['window_active_samples']+5 for k in ('c1','v3') for slot in windows),'no_added_duplicate':all(mem['events'][slot]['duplicate_rises']<=rec[k]['events'][slot]['duplicate_rises'] for k in ('c1','v3') for slot in windows),'no_added_none_edge':all(not attrs[k]['none']['added'] for k in attrs),'no_added_false':all(match(mem[tw]['false_times_s'],rec[k][tw]['false_times_s']) for k in ('c1','v3') for tw in assignment),'no_new_stuck':all(not mem[tw]['stuck_tail'] or any(rec[k][tw]['stuck_tail'] for k in ('c1','v3')) for tw in assignment)}
  gains=[{'vs':k,'slot':slot} for k in ('c1','v3') for slot in windows if not rec[k]['events'][slot]['strict'] and mem['events'][slot]['strict'] and mem['events'][slot]['quiet_before'] and mem['events'][slot]['duplicate_rises']==0 and mem['events'][slot]['input_active_samples']>0];losses=[{'vs':k,'slot':slot} for k in preserve for slot,v in preserve[k].items() if not v]
  rows.append({'id':case['id'],'kind':'paired','case':case,'starts':starts,'records':rec,'edge_attribution':attrs,'gates':gates,'passed':all(gates.values()),'gains':gains,'losses':losses})
 if episodes!=plan['episode_accounting']['total_actual']:raise ValueError(f'episodes {episodes}')
 forced=[x for x in rows if x['kind']!='negative'];absolute={}
 for name in arms:
  events=[]
  for row in forced:events.extend([row['records'][name]['event']] if row['kind']=='single' else row['records'][name]['events'].values())
  traces=[]
  for row in rows:
   if row['kind']=='negative':traces.append(row['records'][name]['none'])
   elif row['kind']=='single':traces.extend((row['records'][name]['none'],row['records'][name]['force']))
   else:traces.extend(row['records'][name][tw] for tw in ('none','first','second','both'))
  absolute[name]={'strict_events':sum(x['strict'] for x in events),'event_slots':len(events),'no_rise_misses':sum(x['input_present_but_no_attributable_rise'] for x in events),'duplicates':sum(x['duplicate_rises'] for x in events),'event_duration_samples':sum(x['window_active_samples'] for x in events),'false_rises':sum(len(x['false_times_s']) for x in traces),'stuck_traces':sum(x['stuck_tail'] for x in traces),'tool_present_traces':sum(x['tool']['present'] for x in traces),'tool_detected_traces':sum(x['tool']['detected'] for x in traces)}
 gains=[{'id':x['id'],'kind':x['kind'],'gains':x['gains']} for x in rows if x['gains']];losses=[{'id':x['id'],'kind':x['kind'],'losses':x['losses']} for x in rows if x['losses']];failures=[{'id':x['id'],'kind':x['kind'],'gates':[k for k,v in x['gates'].items() if not v]} for x in rows if not x['passed']];old_limits=[]
 for row in forced:
  for name in ('c1','v3'):
   evs={'event':row['records'][name]['event']} if row['kind']=='single' else row['records'][name]['events']
   old_limits.extend({'id':row['id'],'arm':name,'slot':slot,'event':ev} for slot,ev in evs.items() if not ev['strict'])
 doc={'schema':'r29-g4-prospect-result.v1','classification':plan['classification'],'plan_sha256':sha(PLAN),'runner_sha256':sha(Path(__file__)),'protocol_sha256':sha(PROTOCOL),'candidate_sha256':protocol['candidate_sha256'],'episodes':episodes,'rows':rows,'absolute_counts':absolute,'novel_gains':gains,'regression_losses':losses,'failed_rows':failures,'inherited_old_limits':old_limits,'primary_protection_passed':not failures and not losses,'claim_status':'prospective_lab_only_no_promotion'}
 a.out.mkdir();(a.out/'result.json').write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n');print(json.dumps({'episodes':episodes,'absolute':absolute,'gains':gains,'losses':losses,'failed_rows':failures,'old_limits':len(old_limits),'primary_protection_passed':doc['primary_protection_passed']},indent=2,sort_keys=True))
if __name__=='__main__':main()
