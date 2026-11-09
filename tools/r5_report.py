"""Generate deterministic public R5 documents from hash-pinned JSON results."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, re, subprocess, sys, tempfile, textwrap

ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'results/v2_protocol'
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
CURATION=D/'R5_REPORT_CURATION.json'
OUTPUTS=['README.md','docs/R5_RESULTS.md','report/PRESENTATION_R5.md','results/presentation.pdf']
REQUIRED=['results/v2_protocol/R5_REFERENCE_METRICS.json','results/v2_protocol/PROTOCOL_R5.json',
 'results/v2_protocol/S4_OFFICIAL.json',
 'results/v2_protocol/PROTOCOL_R5_HOLDOUT_RESEAL.json','results/v2_protocol/R5_BUILD.json',
 'results/v2_protocol/R5_S2_DESIGN.json','results/v2_protocol/R5_S2_UPPER.json','results/v2_protocol/R5_S2_NORMAL_AUDIT.json',
 'results/v2_protocol/R5_S2_CAUSE.json','results/v2_protocol/R5_BASE_validation_R1_servo.json',
 'results/v2_protocol/R5_BASE_validation_R1_torque.json',
 'results/v2_protocol/R5_NORMAL_AUDIT.json','results/v2_protocol/R5_VALIDATION_REVIEW.json',
 'results/v2_protocol/R5_REDESIGN_DECISION.json','results/v2_protocol/R5_DEVELOPMENT_REVIEW.json','results/v2_protocol/R5_D4_RESULTS.json',
 'results/v2_protocol/R5_PERTURB_RESULTS.json','results/v2_protocol/R5_PREFREEZE_DECISION.json',
 'results/v2_protocol/R5_ADOPTION.json','results/v2_protocol/R5_FINAL_CONTRACT_REVIEW.json',
 'results/v2_protocol/R5_FINAL_ADOPTION.json','tools/r5_report.py','tools/test_r5_report.py',
 'tools/r4_report_links.py']
REQUIRED += [f'models/r5/{mode}_{variant}.json' for mode in ('servo','torque') for variant in ('SCALE_PHASE','SCALE_STATE','AR20_FULL','AR50_FULL')]
BASELINE_SOURCES=['results/FINAL_OFFICIAL.json','results/v2_protocol/R2_COMMON_REFERENCE.json','results/v2_protocol/R2_D4_base.json',
 'results/v2_protocol/S5_D4.json','results/v2_protocol/R2_PERTURBATIONS.json','results/v2_protocol/R3_TIMING_REVIEW.json',
 'results/v2_protocol/R4_BASE_revalidation_servo.json','results/v2_protocol/R4_BASE_revalidation_torque.json','docs/R4_README.md']
REQUIRED += BASELINE_SOURCES
OPTIONAL=['results/v2_protocol/R5_FREEZE.json','results/v2_protocol/R5_HOLDOUT_REVIEW.json','results/v2_protocol/R5_OFFICIAL_REGRESSION.json']
MEDIA=['artifacts/media/09_r5_servo_positive_comparison.gif','artifacts/media/09_r5_servo_positive_comparison.mp4',
 'artifacts/media/10_r5_servo_negative_comparison.gif','artifacts/media/10_r5_servo_negative_comparison.mp4','artifacts/media/r5_manifest.json']
OPTIONAL += MEDIA
FORBIDDEN=[re.compile(x,re.I) for x in ['/ho'+'me/','TA'+'SK-','Cl'+'aude','Co'+'dex','G'+'PT','O'+'MX','local '+'staging','미'+'게시']]
MODES={'servo':'A 서보','torque':'B 토크'}
VARIANTS=['SCALE_PHASE','SCALE_STATE','AR20_FULL','AR50_FULL']
FAILURES={'S2_PERMANENT_DIAGNOSTIC':'S2에서 영구 진단으로 고정','INSUFFICIENT_ADDITIONAL_DETECTION':'추가 검출 기준 미달',
 'NORMAL_AUDIT_FAILED':'정상 전용 검사 실패','PER_CASE_FA_INCREASE':'조건별 오감지 증가','VALIDATION_NOT_ELIGIBLE':'사전 검증 비적격',
 'no_lost_R1':'R1 검출 손실','no_tool_loss':'도구 접촉 검출 손실','no_case_FA_increase':'조건별 오감지 증가',
 'minimum_additional_detected':'추가 검출 기준 미달','reference_FA_zero':'무충돌 오감지 0 기준 미달',
 'FINAL_SUBMISSION_COMPATIBILITY_FAILED':'최종 제출 호환성 검사 실패'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,text):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as f:f.write(text);tmp=Path(f.name)
 tmp.replace(path)
def safe(text,label):
 for pattern in FORBIDDEN:
  if pattern.search(text):raise ValueError(f'{label}: prohibited public wording {pattern.pattern}')

def build_curation(root=ROOT):
 root=Path(root);sources={};pending=[]
 for name in REQUIRED+OPTIONAL:
  p=root/name
  if p.is_file():sources[name]={'sha256':sha(p),'size_bytes':p.stat().st_size}
  elif name in REQUIRED:pending.append(name)
 return {'schema':'R5_REPORT_CURATION.v1','sources':sources,'pending_required':pending,
  'outputs':OUTPUTS,'official_strings_preserve_source_precision':True,
  'private_raw_or_acquisition_inputs':False,'created_utc':'2000-01-01T00:00:00+00:00'}

def validate_curation(root=ROOT,curation=CURATION,complete=True):
 root=Path(root);doc=read(curation)
 if doc.get('schema')!='R5_REPORT_CURATION.v1':raise ValueError('bad curation schema')
 if complete and doc.get('pending_required'):raise ValueError('report inputs pending: '+str(doc['pending_required']))
 for name,meta in doc.get('sources',{}).items():
  p=root/name
  if not p.is_file() or sha(p)!=meta.get('sha256') or p.stat().st_size!=meta.get('size_bytes'):
   raise ValueError('curated input changed: '+name)
 if complete and set(REQUIRED)-set(doc.get('sources',{})):raise ValueError('curation lacks required inputs')
 existing_optional={name for name in OPTIONAL if (root/name).is_file()}
 if existing_optional-set(doc.get('sources',{})):raise ValueError('curation lacks existing optional inputs')
 adoption=root/'results/v2_protocol/R5_ADOPTION.json'
 if complete and adoption.is_file() and not read(adoption).get('sealed_stages_skipped') and 'results/v2_protocol/R5_FREEZE.json' not in doc['sources']:
  raise ValueError('sealed adoption lacks frozen source pin')
 return doc

def entry(doc,variant,mode):
 rows=[x for x in doc['entries'] if x['variant']==variant and x['mode']==mode]
 if len(rows)!=1:raise ValueError('missing/duplicate result row')
 return rows[0]
def phase_text(metrics):
 p=metrics['phases'];return ' · '.join(f"{k} {v['R1_detected']}→{v['candidate_detected']}/{v['rows']}" for k,v in [('집기',p['pick']),('운반',p['transport']),('놓기',p['place'])])
def robustness_phase(row):
 out=[];labels={'pick':'집기','transport':'운반','place':'놓기'}
 for phase in ('pick','transport','place'):
  d=[x for x in row.get('details',[]) if x.get('case',{}).get('kind')=='collision' and x.get('case',{}).get('phase')==phase]
  old=sum(x['R1']['outcome']=='detected' for x in d);new=sum(x['candidate']['outcome']=='detected' for x in d)
  out.append(f"{labels[phase]} {old}→{new}/{len(d)}")
 return ' · '.join(out)
def fmt(x,n=3):return f'{float(x):.{n}f}'
def failure_text(*rows):
 values=[]
 for row in rows:
  values.extend(row.get('failures',[]))
  if row.get('robustness_gates_passed') is False:
   values.extend(k for k,v in row.get('criteria',{}).items() if not v)
 unknown=set(values)-set(FAILURES)
 if unknown:raise ValueError('unmapped failure codes: '+str(sorted(unknown)))
 return ', '.join(dict.fromkeys(FAILURES[x] for x in values)) or '없음'

def context(root=ROOT):
 root=Path(root);curated=validate_curation(root,root/'results/v2_protocol/R5_REPORT_CURATION.json')
 report_jsons=[n for n in REQUIRED+OPTIONAL if (n.startswith('results/v2_protocol/R5_')
               or n.endswith('PROTOCOL_R5.json') or n.endswith('S4_OFFICIAL.json'))]
 docs={Path(n).stem:read(root/n) for n in report_jsons if (root/n).is_file()}
 ref=docs['R5_REFERENCE_METRICS'];p=docs['PROTOCOL_R5'];normal=docs['R5_NORMAL_AUDIT'];s2audit=docs['R5_S2_NORMAL_AUDIT']
 for name,meta in ref['sources'].items():
  if name not in curated['sources'] or sha(root/name)!=meta['sha256']:raise ValueError('baseline catalog source differs: '+name)
 val=docs['R5_VALIDATION_REVIEW'];dev=docs['R5_DEVELOPMENT_REVIEW'];d4=docs['R5_D4_RESULTS'];pert=docs['R5_PERTURB_RESULTS']
 earlier=docs['R5_ADOPTION'];adoption=docs['R5_FINAL_ADOPTION'];contract=docs['R5_FINAL_CONTRACT_REVIEW']
 if adoption['supersedes']['sha256']!=sha(root/adoption['supersedes']['path']):raise ValueError('final adoption supersedes hash mismatch')
 if adoption['contract_review']['sha256']!=sha(root/adoption['contract_review']['path']):raise ValueError('final contract review hash mismatch')
 if adoption['effective_variants']!={'servo':'R1','torque':'R1'} or contract['status']!='FAIL':raise ValueError('unexpected final authority')
 effective=adoption['effective_variants'];rows=[]
 for mode in ('servo','torque'):
  for spec in p['candidate_grid']:
   variant=spec['id'];v=entry(val,variant,mode);de=entry(dev,variant,mode);dr=entry(d4,variant,mode);pr=entry(pert,variant,mode)
   na=next(x for x in normal['entries'] if x['variant']==variant and x['mode']==mode)
   model=spec['models'][mode];payload=read(root/model['path']);selected=effective[mode]==variant
   if payload['float_parameters']!=model['float_parameters'] or payload['threshold']!=model['threshold']:
    raise ValueError('model summary differs from registered protocol')
   audit=next(x for x in s2audit['entries'] if x['mode']==mode and x['id']==payload['spec']['id'])['statistics']
   failures='없음' if selected else failure_text(v,de,dr,pr)
   if earlier['effective_variants'][mode]==variant and effective[mode]!=variant:
    failures=', '.join(x for x in (failures,FAILURES['FINAL_SUBMISSION_COMPATIBILITY_FAILED']) if x!='없음')
   rows.append({'mode':MODES[mode],'mode_id':mode,'variant':variant,'params':model['float_parameters'],'threshold':model['threshold'],
    'normal':f"fit {fmt(payload['fit_statistics']['rmse'])}/{fmt(payload['fit_statistics']['max'])}; calibration {fmt(payload['calibration_statistics']['rmse'])}/{fmt(payload['calibration_statistics']['max'])}; audit {fmt(audit['rmse'])}/{fmt(audit['max'])} N; 추가 표본/시작 {na['extra_samples']}/{na['extra_rises']}",
    'validation':f"+{v['metrics']['additional_detected']}, 손실 {len(v['metrics']['lost_R1_case_ids'])}, FA증가 {len(v['metrics']['FA_increase_case_ids'])}",
    'development':f"+{de['metrics']['additional_detected']}, 손실 {len(de['metrics']['lost_R1_case_ids'])}, FA증가 {len(de['metrics']['FA_increase_case_ids'])}",
    'D4':f"+{dr['additional_detected']}, 손실 {dr['lost_R1']}, FA증가행 {dr['outside_FA_increased_cases']}",
    'perturb':f"+{pr['additional_detected']}, 손실 {pr['lost_R1']}, FA {pr['R1_FA']}→{pr['candidate_FA']}",
    'timing':f"det {fmt(v['timing']['detector_wall_p99_worst_us'])}/{fmt(v['timing']['detector_wall_max_worst_us'])}; 합산 {fmt(v['metrics']['combined_p99_worst_us'])}/{fmt(v['metrics']['combined_max_worst_us'])} µs",
    'status':'채택' if selected else '탈락','failures':failures,
    'val_phase':phase_text(v['metrics']),'dev_phase':phase_text(de['metrics']),'d4_phase':robustness_phase(dr)})
 roles=docs['R5_S2_DESIGN']['roles'];labels={'fit':'정상 fit','calibration':'정상 calibration','audit':'정상 audit'}
 splits=', '.join(f"{labels[k]} {len(roles[k])}행" for k in ('fit','calibration','audit'))
 redesign=docs['R5_REDESIGN_DECISION']
 if redesign['redesigns_used']!=0:raise ValueError('unexpected R5 redesign count')
 base_timing={m:timing_summary(docs[f'R5_BASE_validation_R1_{m}']) for m in ('servo','torque')}
 return {'root':root,'docs':docs,'ref':ref,'protocol':p,'rows':rows,'effective':effective,'adoption':adoption,'earlier_adoption':earlier,'splits':splits,'redesign':redesign,'base_timing':base_timing}

def timing_summary(doc):
 rows=doc['rows']
 if len(rows)!=38:raise ValueError('fresh validation baseline must contain 38 rows')
 def worst(part,key):return max(float(x['timing_summary'][part]['wall'][key]) for x in rows)
 return {'detector_p99':worst('detector','p99_us'),'detector_max':worst('detector','max_us'),
  'combined_p99':worst('combined','p99_us'),'combined_max':worst('combined','max_us'),
  'outliers':sum(len(x['timing_summary']['outliers']) for x in rows)}

def media_block(root=ROOT):
 text='''| R1 A 서보 | R1 B 토크 | 기준/v1/R1/R2 A 네 열 | 기준/v1/R1/R2 B 네 열 |\n|---|---|---|---|\n| [![R1 A](artifacts/media/05_v2_servo_transport_collision.gif)](artifacts/media/05_v2_servo_transport_collision.mp4) | [![R1 B](artifacts/media/06_v2_torque_transport_collision.gif)](artifacts/media/06_v2_torque_transport_collision.mp4) | [![과거 A 네 열](artifacts/media/07_r2_servo_comparison.gif)](artifacts/media/07_r2_servo_comparison.mp4) | [![과거 B 네 열](artifacts/media/08_r2_torque_comparison.gif)](artifacts/media/08_r2_torque_comparison.mp4) |'''
 if (Path(root)/MEDIA[0]).is_file():
  if not all((Path(root)/name).is_file() for name in MEDIA):raise ValueError('incomplete R5 media set')
  text+='''\n\n| R5 A +x 정방향 | R5 A -x 역방향 |\n|---|---|\n| [![R5 A +x 정방향](artifacts/media/09_r5_servo_positive_comparison.gif)](artifacts/media/09_r5_servo_positive_comparison.mp4) | [![R5 A -x 역방향](artifacts/media/10_r5_servo_negative_comparison.gif)](artifacts/media/10_r5_servo_negative_comparison.mp4) |\n\n09/10은 저장된 사전 검증 재생의 비공식 설명 영상이며 독립 효능 근거가 아니다.'''
 return text
def md_table(columns,rows):
 line='| '+' | '.join(columns)+' |\n|'+'|'.join(['---']*len(columns))+'|\n'
 return line+'\n'.join('| '+' | '.join(str(r.get(c,'')) for c in columns)+' |' for r in rows)
def conclusion(c):
 e=c['effective'];chosen=[MODES[m]+' '+v for m,v in e.items() if v!='R1']
 return ('채택: '+', '.join(chosen)) if chosen else '새 후보는 채택되지 않았고 R1을 유지했다.'
def candidate_table(c,compact=False):
 cols=['방식','후보','정상 전용 검사','사전 검증','개발','D4','현재 34조건 섭동','시간','판정','실패 관문']
 rows=[]
 for x in c['rows']:
  rows.append({'방식':x['mode'],'후보':x['variant'],'정상 전용 검사':x['normal'],'사전 검증':x['validation'],'개발':x['development'],'D4':x['D4'],'현재 34조건 섭동':x['perturb'],'시간':x['timing'],'판정':x['status'],'실패 관문':x['failures']})
 return md_table(cols,rows)
def perturb_subset_rows(c):
 rows=[];doc=c['docs']['R5_PERTURB_RESULTS']
 for mode in ('servo','torque'):
  for variant in VARIANTS:
   details=entry(doc,variant,mode)['details']
   def summary(values):
    return f"{len(values)}행; FA {sum(x['R1']['false_alarms'] for x in values)}→{sum(x['candidate']['false_alarms'] for x in values)}; gain {sum(x['additional_detected'] for x in values)}, loss {sum(x['lost_R1'] for x in values)}"
   physical=[x for x in details if x['source_row_index']<25]
   fresh=[x for x in details if x['source_row_index']>=25]
   reference=[x for x in details if x['case']['kind']=='reference']
   rows.append({'방식':MODES[mode],'후보':variant,'기존 물리 25':summary(physical),
                '새 잡음 9':summary(fresh),'무충돌 28':summary(reference),'전체 34':summary(details)})
 return rows
def t_rows(c):
 r=c['ref'];versions={'기준':'baseline','v1':'v1','R1':'R1'}
 def ab(fn):return 'A '+fn('servo')+'; B '+fn('torque')
 rows=[]
 for label,key in versions.items():pass
 common=r['common_8kg_analysis'];pert=r['perturbation'];d4=r['D4']
 vals={}
 for label,key in versions.items():
  vals.setdefault('T1',{})[label]=ab(lambda m:f"{common[key][m]['support_mean_percent']:.4f}")
  vals.setdefault('T2',{})[label]=ab(lambda m:f"{float(common[key][m]['place_step_nm_exact']):.3f}")
  vals.setdefault('T3',{})[label]=ab(lambda m:f"{pert[key][m]['all34_max_tcp_mm']:.3f} mm / {pert[key][m]['all34_max_place_force_n']:.3f} N / FA {pert[key][m]['all34_false_alarm_rises']}")
  vals.setdefault('T4',{})[label]=ab(lambda m:f"sat {d4[key][m]['reference_saturated_rows']}/189, peak {d4[key][m]['reference_max_place_peak_over_mg']:.4f} mg")
  vals.setdefault('T5',{})[label]=ab(lambda m:f"{float(common[key][m]['rms_mm_exact']):.3f}/{float(common[key][m]['max_mm_exact']):.3f}")
  vals.setdefault('T6',{})[label]=ab(lambda m:f"{d4[key][m]['detected']}/{d4[key][m]['missed']}/{d4[key][m]['masked']}")
 # Detector-only R5 preserves T1-T5. T6 follows adopted D4 detector, or R1.
 for t in ('T1','T2','T4','T5'):vals[t]['R5']=vals[t]['R1']
 t3=[]
 for mode,prefix in (('servo','A'),('torque','B')):
  old=pert['R1'][mode];physical=f"{old['all34_max_tcp_mm']:.3f} mm / {old['all34_max_place_force_n']:.3f} N"
  if c['effective'][mode]=='R1':physical+=f" / FA {old['all34_false_alarm_rises']} (과거 보존·재실행 아님)"
  else:physical+=' / FA는 현재 34조건 별도 집계'
  t3.append(prefix+' '+physical)
 vals['T3']['R5']='; '.join(t3)
 for mode in ('servo','torque'):
  variant=c['effective'][mode]
  if variant=='R1':v=d4['R1'][mode];text=f"{v['detected']}/{v['missed']}/{v['masked']}"
  else:
   row=entry(c['docs']['R5_D4_RESULTS'],variant,mode);details=[x for x in row['details'] if x['case']['kind']=='collision'];det=sum(x['candidate']['outcome']=='detected' for x in details);masked=sum(x['candidate']['outcome'] in ('alarm_before_collision','indistinguishable') for x in details);text=f"{det}/{len(details)-det-masked}/{masked}"
  vals.setdefault('T6',{}).setdefault('R5_parts',{})[mode]=text
 vals['T6']['R5']='A '+vals['T6']['R5_parts']['servo']+'; B '+vals['T6']['R5_parts']['torque']
 names={'T1':'놓기 전 마지막 20 ms 평균 지지율 [%mg]','T2':'놓기 토크 계단 [N·m]','T3':'분석 섭동 최악 TCP/놓기힘/FA','T4':'D4 reference 포화/놓기 peak','T5':'8 kg TCP RMS/최대 [mm]','T6':'D4 검출/놓침/선행 경보'}
 return [{'항목':k+': '+names[k],**{v:vals[k][v] for v in ('기준','v1','R1','R5')}} for k in names]
def status_ko(value):
 value=str(value);return {'PASS':'통과','FAIL':'탈락','pass':'통과','fail':'탈락','passed':'통과','failed':'탈락'}.get(value,value)

def official_rows(c,candidate=False):
 off=c['docs'].get('R5_OFFICIAL_REGRESSION') if candidate else None;rows=[]
 for mode in ('servo','torque'):
  current=off and off.get('records') and off['records'].get(mode,{}).get('control_and_collisions',{}).get('control')
  for mass in ('0.0','4.0','8.0'):
   if current:cells=[str(x) for x in current[mass]]
   else:
    x=c['docs']['S4_OFFICIAL']['combinations'][mode+'_v2_v2']['control'][mass]
    cells=[str(x['rms_mm']),str(x['max_mm']),str(x['saturation_pct']),str(x['pick_nm']),str(x['place_nm']),f"{x['table_n']:g}",*[str(v) for v in x['false_alarms']],str(x['status'])]
   if len(cells)!=10:raise ValueError('unexpected official evaluator cell width')
   rows.append({'방식':MODES[mode],'질량':mass+' kg','TCP RMS/최대 [mm]':cells[0]+' / '+cells[1],
    '포화 [%]':cells[2],'집기/놓기 계단 [N·m]':cells[3]+' / '+cells[4],'놓기힘 [N]':cells[5],
    'FA 집기/놓기/기타':cells[6]+' / '+cells[7]+' / '+cells[8],'상태':status_ko(cells[9])})
 return rows
def official_label(c):
 return '최종 R1 보존값·fallback 뒤 새 실행 0회'
def collision_rows(c,candidate=False):
 off=c['docs'].get('R5_OFFICIAL_REGRESSION') if candidate else None;rows=[]
 for mode in ('servo','torque'):
  current=off and off.get('records') and off['records'].get(mode,{}).get('control_and_collisions',{}).get('collisions')
  if current:
   values=[x for x in current if float(x['mass'])==8.0]
   for x in values:rows.append({'방식':MODES[mode],'시각':f"{x['time']:g} s",'10/15 N':x['cells'][0]+' / '+x['cells'][1]})
  else:
   for x in c['ref']['official_8kg_collision_10_15N']['R1'][mode]:
    rows.append({'방식':MODES[mode],'시각':x['time_s_exact']+' s','10/15 N':x['force10_exact']+' / '+x['force15_exact']})
 return rows
def holdout_rows(c):
 review=c['docs'].get('R5_HOLDOUT_REVIEW')
 if not review:return [{'방식':MODES[m],'소스':'R1','충돌 검출':'새 실행 0','정상 FA':'새 실행 0','판정':'sealed 단계 생략'} for m in ('servo','torque')]
 rows=[]
 for x in review['entries']:
  if x['variant']=='R1':
   rows.append({'방식':MODES[x['mode']],'소스':'R1 비교 38조건·후보 미실행','충돌 검출':'R1 비교값','정상 FA':'R1 비교값','판정':status_ko(x['status'])});continue
  details=x['details'];coll=[d for d in details if d['kind']=='collision'];refs=[d for d in details if d['kind']=='reference']
  metrics=x['metrics']
  rows.append({'방식':MODES[x['mode']],'소스':x['variant'],'충돌 검출':f"{metrics['R1_detected_collisions']}→{metrics['candidate_detected_collisions']}/{metrics['collision_rows']}",
   '정상 FA':f"전체 {metrics['R1_false_alarms_total']}→{metrics['candidate_false_alarms_total']}; reference {sum(d['candidate_event']['false_alarms'] for d in refs)}",
   '판정':status_ko(x['status'])})
 return rows

def timing_rows(c):
 rows=[]
 for mode in ('servo','torque'):
  base=c['base_timing'][mode];v=entry(c['docs']['R5_VALIDATION_REVIEW'],'SCALE_STATE',mode)
  candidate={'detector_p99':v['timing']['detector_wall_p99_worst_us'],'detector_max':v['timing']['detector_wall_max_worst_us'],
   'combined_p99':v['metrics']['combined_p99_worst_us'],'combined_max':v['metrics']['combined_max_worst_us'],
   'outliers':len(v['timing'].get('all_outliers',[]))}
  for source,data in (('R1 비교 38조건',base),('STATE 후보 38조건',candidate)):
   rows.append({'방식':MODES[mode],'소스':source,'감지 p99/최대 [µs]':fmt(data['detector_p99'])+' / '+fmt(data['detector_max']),
    '합산 p99/최대 [µs]':fmt(data['combined_p99'])+' / '+fmt(data['combined_max']),'1 ms 초과':data['outliers']})
 return rows

def holdout_timing_rows(c):
 review=c['docs'].get('R5_HOLDOUT_REVIEW');rows=[]
 if not review:return rows
 for x in review['entries']:
  if x['variant']=='R1':continue
  t=x['timing'];rows.append({'방식':MODES[x['mode']],'후보':x['variant'],
   '감지 p99/최대 [µs]':fmt(t['detector_wall_p99_worst_us'])+' / '+fmt(t['detector_wall_max_worst_us']),
   '합산 p99/최대 [µs]':fmt(t['combined_wall_p99_worst_us'])+' / '+fmt(t['combined_wall_max_worst_us']),
   '1 ms 초과':len(t['all_outliers']),'p99 관문':status_ko(x['status']),'최대 1 ms 목표':'통과' if x['max_time_goal_met'] else '미달'})
 return rows

def holdout_max_text(c):
 rows=holdout_timing_rows(c)
 if not rows:return 'sealed 단계에서 새 계산 시간을 측정하지 않았다.'
 review=c['docs']['R5_HOLDOUT_REVIEW'];maximum=max(x['timing']['combined_wall_max_worst_us'] for x in review['entries'] if x['variant']!='R1')
 return f'holdout 합산 최대는 {maximum/1000:.3f} ms로 최대 1 ms 목표에 미달했다.'

def candidate_official_comparison(c):
 final8=next(x for x in official_rows(c) if x['방식']==MODES['servo'] and x['질량']=='8.0 kg')
 cand8=next(x for x in official_rows(c,True) if x['방식']==MODES['servo'] and x['질량']=='8.0 kg')
 final_collision=next(x for x in collision_rows(c) if x['방식']==MODES['servo'] and x['시각'] in ('3.0 s','3 s'))
 cand_collision=next(x for x in collision_rows(c,True) if x['방식']==MODES['servo'] and x['시각'] in ('3.0 s','3 s'))
 return [
  {'조건':'8 kg TCP RMS/최대 [mm]','R1 보존':final8['TCP RMS/최대 [mm]'],'A STATE 후보':cand8['TCP RMS/최대 [mm]'],'판정':'후보 탈락'},
  {'조건':'8 kg 놓기 계단 [N·m]','R1 보존':final8['집기/놓기 계단 [N·m]'].split(' / ')[1],'A STATE 후보':cand8['집기/놓기 계단 [N·m]'].split(' / ')[1],'판정':'후보 탈락'},
  {'조건':'운반 3.0 s 충돌 10/15 N','R1 보존':final_collision['10/15 N'],'A STATE 후보':cand_collision['10/15 N'],'판정':'후보 탈락'}]

def cause_rows(c):
 cause=c['docs']['R5_S2_CAUSE']['modes'];upper=c['docs']['R5_S2_UPPER'];rows=[]
 for mode in ('servo','torque'):
  cats=cause[mode]['categories'];state=next(x for x in upper['entries'] if x['mode']==mode and x['spec']['id']=='scale_active_state_log');phases=state['phases']
  rows.append({'방식':MODES[mode],'R1 검출':cats['detected'],'힘 중 gate 꺼짐':cats['learning_gate_off_during_force'],
   '활성 중 임계 미만':cats['active_below_learning_threshold'],'STATE 추가':state['additional_detected'],
   '추가 구간':f"집기 {phases['pick']['gains']} / 운반 {phases['transport']['gains']} / 놓기 {phases['place']['gains']}"})
 return rows
def seed_status(c):
 return ('새 seed 조건은 동결 뒤 1회 생성됐다.' if c['docs'].get('R5_HOLDOUT_REVIEW')
         else '모든 후보가 사전 탈락해 새 seed 조건은 생성하지 않았다.')
def redesign_text(c):
 return '재설계 0회. ' + ' '.join(f"{x['family']}: {x['reason']}" for x in c['redesign']['families'])
def integrity_text(c):
 if any(value!='R1' for value in c['effective'].values()):
  return '현재 제출 감지기 alias는 R5에서 미리 동결한 assembly로 바뀌는 허용 예외다. 과거 R2/R3/R4 동결 목록은 alias를 제외한 52/100/198파일이 그대로이며, R5 동결 119파일과 runtime 62·원본 29·보호 20·환경 21 검사를 적용한다.'
 return ('최종 R1 fallback이면 과거 R2/R3/R4 동결 53/101/199파일 전체가 그대로다. '
  'R5 동결 목록은 119파일 중 118파일이 그대로이고, 제출 감지기 alias 한 파일만 미리 동결한 R1 fallback 바이트로 바뀌었다. '
  'runtime 62·원본 29·보호 20·환경 21 검사도 적용한다.')
def compatibility_text(c):
 review=c['docs']['R5_FINAL_CONTRACT_REVIEW']
 if review['status']!='FAIL' or review['tests_run']!=526 or review['errors']!=8:raise ValueError('unexpected final contract review')
 return ('A STATE는 validation·development·D4·holdout과 공식 조건에서는 양의 측정값을 냈지만, '
  '최종 제출 호환성 526개 검사에서 8개 오류가 발생해 탈락했다. 한 원소 모드 집합이 문자열로 생성되어 '
  '제어기 정보가 없는 reset에서 TypeError가 났다. 동결 뒤 수리·재평가하지 않고 미리 동결한 R1로 fallback했다.')

def render_readme(c):
 return f'''# Week 9 Palletizing Load Transfer — R5 정상 전용 학습 감지기\n\n**최종 결론: 새 후보는 채택하지 않고 A/B 모두 R1을 유지한다.**\n\n기준=초기 제공본, v1=첫 개선본, R1=고정 비교 제어·감지기, R5=R1 제어기를 유지하고 정상 데이터만으로 보정한 추가 감지 채널이다. A=서보, B=토크, FA=오감지다.\n\n{media_block(c['root'])}\n\nA STATE 잠정 판정 때 만든 09/10 설명 영상 두 쌍은 최종 호환성 탈락 뒤 보관 전용으로 옮겼고 공개 결과에는 넣지 않았다.\n\n## 결론\n\n{conclusion(c)} {compatibility_text(c)} 최종 세 제출 파일은 R1과 바이트가 같다. 상세 수치와 모든 후보 행은 [R5 결과](docs/R5_RESULTS.md)에 있다. {integrity_text(c)}\n\n## 기준/v1/R1/R5 — T1~T6\n\n{md_table(['항목','기준','v1','R1','R5'],t_rows(c))}\n\nT1/T2/T4/T5는 감지기만 바꾼 R5에서 R1 물리값을 그대로 보존한다. T3의 기준/v1/R1은 과거 전체 34조건 TCP·놓기힘·FA 보존값이며 R5에서 다시 실행한 값이 아니다. 현재 34조건의 짝지은 후보 FA는 결과 문서에 별도로 둔다.\n\n## 후보 8개\n\n{candidate_table(c)}\n\n### 새 사전 검증 38조건 계산 시간\n\n{md_table(['방식','소스','감지 p99/최대 [µs]','합산 p99/최대 [µs]','1 ms 초과'],timing_rows(c))}\n\n보존된 R3 전수 최대값(A 11.703 ms, B 14.163 ms)은 범위가 달라 위 짝지은 38조건 표의 대체값으로 쓰지 않았다.\n\n## 정상 전용 학습과 공개 범위\n\n역할 수는 {c['splits']}이다. fit은 가중치 학습, calibration은 정상 임계값 고정, audit은 학습 뒤 무충돌 확인이다. 충돌 라벨·내부 물리값·별도 평가 자료는 학습이나 임계값 선택에 쓰지 않았다. SCALE_STATE는 방식별 263개 부동소수 파라미터를 가진다. 원인 자료에서 R1은 A 54/B 52건을 검출했고, 힘 구간에서 gate 꺼짐은 A 20/B 23건, 활성 중 임계 미만은 A 16/B 15건이었다. STATE의 경험적 추가 검출은 각 +6(집기 2·운반 4·놓기 0)이지만 형식적 상한이나 단일 물리 원인으로 해석하지 않는다. 비공개 취득·sealed 실행 드라이버와 raw 신호는 배포하지 않고, 안전한 runtime/build/정상 재학습/scoring/result 파일만 제공한다.\n\n## 공식 회귀 — {official_label(c)}\n\n{md_table(['방식','질량','TCP RMS/최대 [mm]','포화 [%]','집기/놓기 계단 [N·m]','놓기힘 [N]','FA 집기/놓기/기타','상태'],official_rows(c))}\n\n{md_table(['방식','시각','10/15 N'],collision_rows(c))}\n\n탈락한 A STATE의 공식 8 kg 측정은 TCP 1.17/3.91 mm, 놓기 계단 14.6 N·m였고 운반 충돌 10/15 N은 49/37이었다. 이 값은 후보 기록일 뿐 최종 제출값이 아니다.\n\n원본 evaluator 문자열의 자릿수를 늘리지 않았다. 공식 평가는 마지막 회귀 확인이며 학습·선정 자료가 아니다.\n\n## Holdout\n\n{md_table(['방식','소스','충돌 검출','정상 FA','판정'],holdout_rows(c))}\n\n{md_table(['방식','후보','감지 p99/최대 [µs]','합산 p99/최대 [µs]','1 ms 초과','p99 관문','최대 1 ms 목표'],holdout_timing_rows(c)) if holdout_timing_rows(c) else ''}\n\n{holdout_max_text(c)}\n\n## 평가기 점검·R1 놓침 측정\n\n평가기 자체가 R1 놓침을 만든다는 증거는 없었다. 공식 8 kg의 놓침 6건은 충돌 시점별 활성 구간, 신호 크기와 기존 경보 상태를 분리해 기록했다. 보존 상세는 [R4 README의 점검·측정 절](docs/R4_README.md#평가기-점검과-미검출-원인)에 있다.\n\n## 남은 미달과 요구사항\n\nG4는 A 평균 0.2098 %mg·최소 0, B 평균 11.1507 %mg·최소 0.3802 %mg로 평균 50 %mg·최소 25 %mg 관문에 미달한다. 최대 1 ms 목표, 약한 집기·놓기 충돌, 보존 격자의 서보 포화도 남는다. 기존 가이드 33항목은 24 충족/4 부분/5 판정 불가이며 입력·형식·무결성 8항목은 통과 판정을 유지한다. [요구사항 판정](docs/REQUIREMENTS_AUDIT.md), [원인 분석](docs/R5_CAUSE_ANALYSIS.md), [문헌](docs/R5_LITERATURE.md), [방법](docs/R5_METHODS.md), [재현](docs/R5_REPRODUCE.md)을 함께 본다. 원자료: [protocol](results/v2_protocol/PROTOCOL_R5.json), [초기 sealed 판정](results/v2_protocol/R5_ADOPTION.json), [최종 판정](results/v2_protocol/R5_FINAL_ADOPTION.json), [호환성 검사](results/v2_protocol/R5_FINAL_CONTRACT_REVIEW.json).\n\n## 별도 평가 seed 조치\n\n폐기 seed로 동결 전 조건 설계가 최소 12회 생성됐지만 물리 실행·학습·후보 선정은 0회였다. 해당 seed와 예약 번호를 폐기하고 새 seed를 봉인했다. {seed_status(c)}\n'''

def render_results(c):
 phase=[]
 for x in c['rows']:phase.append({'방식':x['mode'],'후보':x['variant'],'사전 검증':x['val_phase'],'개발':x['dev_phase'],'D4':x['d4_phase']})
 normal=[]
 for x in c['rows']:normal.append({'방식':x['mode'],'후보':x['variant'],'파라미터':x['params'],'임계값':repr(x['threshold']),'정상 검사':x['normal'],'시간':x['timing']})
 sealed=c['docs'].get('R5_HOLDOUT_REVIEW');official=c['docs'].get('R5_OFFICIAL_REGRESSION')
 sealed_text=('모든 후보가 사전 탈락해 sealed holdout·공식 평가를 생략했다.' if c['adoption'].get('sealed_stages_skipped') else
  '동결 뒤 holdout 1회와 공식 회귀 1회를 열었다. 결과는 방식별 독립 fallback에 반영했다.')
 return f'''# R5 결과\n\nA=서보, B=토크, FA=오감지다.\n\n## 결론\n\n{conclusion(c)} {compatibility_text(c)} 최종 세 제출 파일은 R1과 바이트가 같다. A STATE 잠정 판정 때 만든 비교 영상 두 쌍은 최종 탈락 뒤 보관 전용으로 옮겼고 공개 결과에는 넣지 않았다.\n\n## 후보별 판정 — 8행 보존\n\n{candidate_table(c)}\n\n## 원인 분류와 경험적 여지\n\n{md_table(['방식','R1 검출','힘 중 gate 꺼짐','활성 중 임계 미만','STATE 추가','추가 구간'],cause_rows(c))}\n\n허용 신호에서 관측한 유한 자료 분류다. 정상 잔차 임계값의 경험적 여지만 보며, 형식적 상한·교환가능성 보장·단일 물리 원인을 주장하지 않는다.\n\n## 구간별 검출\n\n{md_table(['방식','후보','사전 검증','개발','D4'],phase)}\n\n## 정상 전용 보정·모델 크기·시간\n\n{md_table(['방식','후보','파라미터','임계값','정상 검사','시간'],normal)}\n\n시간은 새 사전 검증 38조건에서 감지기 단독 p99/최대와 제어기+감지기 합산 p99/최대를 나란히 썼다.

{md_table(['방식','소스','감지 p99/최대 [µs]','합산 p99/최대 [µs]','1 ms 초과'],timing_rows(c))}\n\n보존된 R3 전수 R1은 A p99 327.671 µs·최대 11.703 ms, B p99 350.440 µs·최대 14.163 ms이며 범위가 달라 대체하지 않는다.\n\n## 기준/v1/R1/R5 T1–T6\n\n{md_table(['항목','기준','v1','R1','R5'],t_rows(c))}\n\nT3 과거 34조건과 현재 34조건은 다르다. 현재 자료는 노출된 물리 25행과 새 잡음 9행이고, 과거 old 34는 R2 보존 조건이다. old 28 무충돌 R1 FA는 A 6/B 9, 과거 전체 34는 A 6/B 10이다. 두 범위를 합쳐 개선으로 주장하지 않는다.\n\n## 현재 섭동 자료의 범위별 집계\n\n{md_table(['방식','후보','기존 물리 25','새 잡음 9','무충돌 28','전체 34'],perturb_subset_rows(c))}\n\n현재 자료 안에서 짝지은 R1→후보만 관문에 사용했다. 과거 old 34와 현재 34의 직접 증감은 효능 근거로 쓰지 않는다.\n\n## 재설계\n\n{redesign_text(c)} [당시 결정](../results/v2_protocol/R5_REDESIGN_DECISION.json)을 그대로 보존한다.\n\n## 동결·holdout·공식\n\n{sealed_text}\n\n{md_table(['방식','소스','충돌 검출','정상 FA','판정'],holdout_rows(c))}\n\n{md_table(['방식','후보','감지 p99/최대 [µs]','합산 p99/최대 [µs]','1 ms 초과','p99 관문','최대 1 ms 목표'],holdout_timing_rows(c)) if holdout_timing_rows(c) else ''}\n\n{holdout_max_text(c)}\n\n공식 표 상태: {official_label(c)}. 원래 폐기 seed 조건은 동결 전에 최소 12회 메모리에서 만들어졌으나 물리 실행·학습·선정은 0회였다. 새 seed는 재봉인했다. {seed_status(c)}\n\n## 공식 셀\n\n{md_table(['방식','질량','TCP RMS/최대 [mm]','포화 [%]','집기/놓기 계단 [N·m]','놓기힘 [N]','FA 집기/놓기/기타','상태'],official_rows(c))}\n\n{md_table(['방식','시각','10/15 N'],collision_rows(c))}\n\n### 탈락한 sealed 후보의 공식 측정 보존\n\n{md_table(['방식','질량','TCP RMS/최대 [mm]','포화 [%]','집기/놓기 계단 [N·m]','놓기힘 [N]','FA 집기/놓기/기타','상태'],[x for x in official_rows(c,True) if x['방식']==MODES['servo']])}\n\n{md_table(['방식','시각','10/15 N'],[x for x in collision_rows(c,True) if x['방식']==MODES['servo']])}\n\nA STATE가 유효 controller mode를 받은 공식 조건에서 기록한 값은 바뀌지 않았지만 최종 제출 호환성 실패로 채택 근거가 아니다. fallback 뒤 공식 평가는 다시 실행하지 않았다.\n\n## G4와 남은 미달\n\nA 지지 평균/최소 0.2098/0.0000 %mg, B 11.1507/0.3802 %mg로 G4 평균 50·최소 25 %mg 관문을 통과하지 못했다. 감지기 개선은 하중 이양 제어 성능을 바꾸지 않는다. 최대 1 ms 목표와 일부 약한 충돌 놓침도 남는다.\n\n## 무결성\n\n{integrity_text(c)}\n\n## 재현·공개 범위\n\n정상-only 재학습 도구는 fit과 calibration 역할을 분리하고 NumPy 추론 모델을 만든다. 공개 결과는 해시로 고정한다. 비공개 취득·sealed 실행 드라이버와 raw 신호는 포함하지 않는다. [이전 R4 결과](R4_RESULTS.md), [원인](R5_CAUSE_ANALYSIS.md), [문헌](R5_LITERATURE.md), [방법](R5_METHODS.md), [재현](R5_REPRODUCE.md), [요구사항](REQUIREMENTS_AUDIT.md)을 함께 본다. 원자료는 [protocol](../results/v2_protocol/PROTOCOL_R5.json), [초기 sealed 판정](../results/v2_protocol/R5_ADOPTION.json), [최종 판정](../results/v2_protocol/R5_FINAL_ADOPTION.json), [호환성 검사](../results/v2_protocol/R5_FINAL_CONTRACT_REVIEW.json), [D4](../results/v2_protocol/R5_D4_RESULTS.json), [섭동](../results/v2_protocol/R5_PERTURB_RESULTS.json)이다.\n'''

def slides(c):
 compact=[]
 for x in c['rows']:
  compact.append({'방식':x['mode'],'후보':x['variant'],'검증':x['validation'].split(',')[0],
   '개발':x['development'].split(',')[0],'D4':x['D4'].split(',')[0],
   '섭동 FA':x['perturb'].split(',')[-1].strip(),'판정':x['status']})
 normal={'servo':[],'torque':[]}
 for x in c['rows']:
  parts=x['normal'].split('; ')
  stats='\n'.join(('학습 '+parts[0].removeprefix('fit '),'보정 '+parts[1].removeprefix('calibration '),'감사 '+parts[2].removeprefix('audit ')))
  normal[x['mode_id']].append({'후보':x['variant'],'파라미터/임계값':f"{x['params']} / {x['threshold']:.3f}",'잔차 RMS/최대 [N]':stats})
 t=t_rows(c)
 return [
 {'title':'R5 결론','bullets':[conclusion(c),'R1 제어기는 고정하고 정상-only 감지 채널만 비교','A=서보, B=토크, FA=오감지']},
 {'title':'최종 제출 호환성 판정','bullets':[compatibility_text(c),'최종 A/B 모두 R1, 최종 세 제출 파일은 R1과 바이트 동일','양의 sealed 측정값은 탈락 후보 기록으로만 보존','잠정 비교 영상 두 쌍은 보관 전용이며 공개 결과에 없음']},
 {'title':'R1 놓침 원인과 경험적 여지','table':(['방식','R1 검출','힘 중 gate 꺼짐','활성 중 임계 미만','STATE 추가','추가 구간'],cause_rows(c)),
  'note':'허용 신호에서 관측한 분류이며 단일 물리 원인이나 형식적 상한을 뜻하지 않는다.'},
 {'title':'후보 8개 판정','table':(['방식','후보','검증','개발','D4','섭동 FA','판정'],compact)},
 {'title':'정상 전용 모델 통계 — A 서보','table':(['후보','파라미터/임계값','잔차 RMS/최대 [N]'],normal['servo']),
  'prewrapped':True,'note':'학습·보정·감사를 분리했고 충돌 라벨은 학습·임계값 선택에 쓰지 않았다.'},
 {'title':'정상 전용 모델 통계 — B 토크','table':(['후보','파라미터/임계값','잔차 RMS/최대 [N]'],normal['torque']),
  'prewrapped':True,'note':'학습·보정·감사를 분리했고 충돌 라벨은 학습·임계값 선택에 쓰지 않았다.'},
 {'title':'새 검증 38조건 계산 시간','table':(['방식','소스','감지 p99/최대 [µs]','합산 p99/최대 [µs]','1 ms 초과'],timing_rows(c)),
  'note':'보존 R3 전수 최대 A 11.703 ms/B 14.163 ms는 범위가 달라 대체값이 아니다.'},
 {'title':'T1–T2','table':(['항목','기준','v1','R1','R5'],t[:2])},
 {'title':'T3 A 서보','table':(['항목','기준','v1','R1','R5'],[{'항목':t[2]['항목'],**{k:t[2][k].split('; B ')[0].replace('A ','',1) for k in ('기준','v1','R1','R5')}}]),
  'note':'기준/v1/R1은 과거 34조건 보존값. 현재 후보 FA는 짝지은 현재 34조건에서 별도 판정했다.'},
 {'title':'T3 B 토크','table':(['항목','기준','v1','R1','R5'],[{'항목':t[2]['항목'],**{k:('B '+t[2][k].split('; B ',1)[1] if '; B ' in t[2][k] else t[2][k]).replace('B ','',1) for k in ('기준','v1','R1','R5')}}]),
  'note':'기준/v1/R1은 과거 34조건 보존값. 현재 후보 FA는 짝지은 현재 34조건에서 별도 판정했다.'},
 {'title':'T4–T6','table':(['항목','기준','v1','R1','R5'],t[3:])},
 {'title':'D4 구간별 검출','table':(['방식','후보','D4'],[{'방식':x['mode'],'후보':x['variant'],'D4':x['d4_phase']} for x in c['rows']])},
 {'title':'동결·holdout·공식','table':(['방식','소스','충돌 검출','정상 FA','판정'],holdout_rows(c)),
  'note':'폐기 seed 설계 최소 12회: 물리 실행·학습·선정 0회. '+seed_status(c)+' '+official_label(c)},
 {'title':'holdout 계산 시간','table':(['방식','후보','감지 p99/최대 [µs]','합산 p99/최대 [µs]','1 ms 초과','p99 관문','최대 1 ms 목표'],[{**x,'후보':'상태 정규화' if x['후보']=='SCALE_STATE' else x['후보']} for x in holdout_timing_rows(c)]) if holdout_timing_rows(c) else (['상태'],[{'상태':'sealed 단계 생략'}]),
  'note':'p99 ≤ 800 µs는 선택 관문이고 최대 1 ms는 별도 목표다. '+holdout_max_text(c)+' 모든 outlier를 보존했다.'},
 {'title':'공식 회귀 — '+official_label(c),'table':(['방식','질량','TCP RMS/최대 [mm]','포화 [%]','집기/놓기 계단 [N·m]','놓기힘 [N]','FA 집기/놓기/기타','상태'],official_rows(c))},
 {'title':'공식 8 kg 충돌 10/15 N','table':(['방식','시각','10/15 N'],collision_rows(c)),
  'note':'원본 evaluator 문자열 자릿수를 그대로 보존했다.'},
 {'title':'탈락한 A STATE 공식 비교','table':(['조건','R1 보존','A STATE 후보','판정'],candidate_official_comparison(c)),
  'note':'운반 3.0 s 10/15 N은 R1 62/45, 후보 49/37이다. 후보 측정은 보존하지만 최종 제출값이 아니며 fallback 뒤 공식 평가는 다시 실행하지 않았다.'},
 {'title':'남은 미달','bullets':['G4: A 0.2098/0.0000 %mg, B 11.1507/0.3802 %mg','최대 1 ms 목표와 약한 집기·놓기 충돌 놓침이 남음','감지기 결과로 제어·실기 안전 향상을 주장하지 않음']},
 ]
def render_notes(c):
 out=['# R5 발표 자료 원고','']
 for i,s in enumerate(slides(c),1):
  out += [f"## {i}. {s['title']}",'']
  if 'bullets' in s:out += ['- '+x for x in s['bullets']]+['']
  else:out += [md_table(s['table'][0],s['table'][1]),'']
  if s.get('note'):out += ['- '+s['note'],'']
 return '\n'.join(out)
def configure_plot():
 import matplotlib;matplotlib.use('Agg');from matplotlib import font_manager
 names={f.name for f in font_manager.fontManager.ttflist};font=next((x for x in ('Noto Sans CJK KR','Noto Sans CJK JP','NanumGothic','DejaVu Sans') if x in names),'DejaVu Sans')
 matplotlib.rcParams.update({'font.family':font,'axes.unicode_minus':False})
def wrap(x,n=34):return '\n'.join(textwrap.wrap(str(x),n,break_long_words=True,replace_whitespace=False))
def make_pdf(path,c):
 configure_plot();import matplotlib.pyplot as plt;from matplotlib.backends.backend_pdf import PdfPages
 meta={'Title':'R5 learned detector results','Creator':'r5_report.py','CreationDate':datetime(2000,1,1,tzinfo=timezone.utc),'ModDate':datetime(2000,1,1,tzinfo=timezone.utc)}
 pages=slides(c)
 with PdfPages(path,metadata=meta) as pdf:
  for i,s in enumerate(pages,1):
   fig=plt.figure(figsize=(13.333,7.5),facecolor='#f6f8fb');fig.text(.04,.92,s['title'],fontsize=23,weight='bold',color='#183153')
   if 'bullets' in s:
    y=.82
    for b in s['bullets']:
     for line in wrap('• '+b,72).splitlines():fig.text(.055,y,line,fontsize=14,color='#263238');y-=.05
     y-=.02
   else:
    cols,rows=s['table'];ax=fig.add_axes([.02,.09,.96,.76]);ax.axis('off');width=s.get('wrap_width',max(10,70//len(cols)))
    data=[[str(r.get(k,'')) if s.get('prewrapped') else wrap(r.get(k,''),width) for k in cols] for r in rows]
    tab=ax.table(cellText=data,colLabels=cols,loc='center',cellLoc='left');tab.auto_set_font_size(False);tab.set_fontsize(10)
    h=.74/max(1,len(rows)+1)
    for (rr,_),cell in tab.get_celld().items():cell.set_height(min(.16,h));cell.set_edgecolor('#b0bec5');cell.set_facecolor('#dbe7f3' if rr==0 else '#ffffff')
   if s.get('note'):fig.text(.04,.055,wrap(s['note'],115),fontsize=10,color='#455a64')
   fig.text(.04,.025,f"R5 검증 | {i}/{len(pages)}",fontsize=9,color='#607d8b');pdf.savefig(fig);plt.close(fig)
def render_all(c):return render_readme(c),render_results(c),render_notes(c)
def generate(root=ROOT):
 c=context(root);texts=render_all(c)
 for label,text in zip(OUTPUTS[:3],texts):safe(text,label);write(Path(root)/label,text)
 out=Path(root)/OUTPUTS[3];out.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(suffix='.pdf',dir=out.parent,delete=False) as f:tmp=Path(f.name)
 try:make_pdf(tmp,c);tmp.replace(out)
 finally:tmp.unlink(missing_ok=True)
 return c
def pdf_text(path):
 with tempfile.NamedTemporaryFile(suffix='.txt') as f:
  subprocess.run(['pdftotext',str(path),f.name],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  return Path(f.name).read_text(encoding='utf-8')
def check_links(root,paths=None):
 from tools.r4_report_links import check_public_links
 result=check_public_links(root,paths)
 if not result['passed']:raise ValueError('broken public links: '+str(result['missing']))
 return result
def public_markdown_paths(root):
 from tools.r4_report_links import check_public_links
 return check_public_links(root)['documents']
def check_one(root):
 root=Path(root);c=context(root);expected=render_all(c)
 for p,text in zip(OUTPUTS[:3],expected):
  actual=root/p
  if not actual.is_file() or actual.read_text(encoding='utf-8')!=text:raise ValueError('stale report output: '+p)
  safe(text,p)
 check_links(root)
 pdf=root/OUTPUTS[3]
 if not pdf.is_file() or not pdf.read_bytes().startswith(b'%PDF'):raise ValueError('missing PDF')
 extracted=pdf_text(pdf)
 for token in ('R5 결론','T1','T6','남은 미달'):
  if token not in extracted:raise ValueError('PDF text missing '+token)
 with tempfile.NamedTemporaryFile(suffix='.pdf') as f:
  make_pdf(Path(f.name),c)
  if Path(f.name).read_bytes()!=pdf.read_bytes():raise ValueError('PDF is stale or nondeterministic')
 return True
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--build-curation',action='store_true');ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--projection',type=Path,action='append',default=[]);a=ap.parse_args();root=a.root.resolve();cur=root/'results/v2_protocol/R5_REPORT_CURATION.json'
 if a.build_curation:write(cur,json.dumps(build_curation(root),ensure_ascii=False,indent=2)+'\n');print(cur);return
 if a.check:
  check_one(root)
  for p in a.projection:check_one(p.resolve())
  print('R5 report check passed');return
 generate(root);print('R5 report generated')
if __name__=='__main__':main()
