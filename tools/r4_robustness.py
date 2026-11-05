"""R4 cached-D4 and perturbation orchestration with explicit provenance gates.

Importing this module is side-effect free. Each action refuses to overwrite
records and requires the committed R4 equivalence and development review.
"""
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from sim.runner import run_episode
from tools import r4_study as S
from tools import d4_robustness
from tools.r3_extended import _load_perturb_harness

D = S.D
OLD_D4_RAW = Path(os.environ.get('R4_D4_RAW', '/missing-r4-d4-raw'))
OLD_PERTURB_RAW = Path(os.environ.get('R4_PERTURB_RAW', '/missing-r4-perturb-raw'))
OLD_HARNESS = Path(os.environ.get('R4_PERTURB_HARNESS', '/missing-r4-perturb-harness'))
HARNESS_SHA256 = 'ed769acd63fa85f5b7b19569bb1d28d009fbac4c76552defd1423f5e5c55b0f7'
HISTORICAL_COMMIT = '5a86c02f29e8c930b8bd5c3bb4ffadafcd49655e'
CURRENT_R3_COMMIT = '8b57c79693d76d7a33e820adcd9fd7af9ec19bb8'
D4_TOOL_DIFF_SHA256 = '66ff7382fedf8b7e05a0fe5f3a855b35fed8fce1dfdc5dc03fc5a28361bcef45'
D4_SCOPE_SHA256 = 'bc2afd090ca927881083467c3f92fafb695db1d5304b07371ff8827258a44c3c'
R3_S6_SHA256 = '8ff78e27a17feff6cea43dea5a479f4cae58a2edeacdf48d3cb3e8485b6f835d'
HISTORICAL_D4_ORCHESTRATION = {
    'tools/r3_resources.py': ('b9169410337675b1981ac4cc6568397f97e4c666e35dba3b883ff16db0b68529',
                              '6b9e62532b94eb092207e16370c4711916d872259b59e593362e72e383dfd22e'),
    'tools/r3_study.py': ('60697198f672ebebdcd44d8f2a2246aec955d802e3c4022a204b62a5840b3b9a',
                          'bcb5400ead4b992e91baf82f2162e8de8eee1c38d8300070f52f071a6775fc68'),
}


def _digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_bench_ready():
    bundle = S.require_equivalence()
    path = D/'R4_DEVELOPMENT_REVIEW.json'
    if not path.is_file(): raise RuntimeError('R4 development review must complete before robustness')
    review = S.read(path); expected = 2*len(S.protocol()['candidate_grid'])
    expected_keys = {(mode, spec['id']) for mode in ('servo','torque')
                     for spec in S.protocol()['candidate_grid']}
    actual_keys = {(row.get('architecture'),row.get('variant'))
                   for row in review.get('results', [])}
    if (review.get('schema') != 'R4_STAGE_REVIEW.v1' or review.get('stage') != 'development'
            or review.get('protocol_sha256') != S.sha(S.P)
            or review.get('diagnostics_cannot_revive') is not True
            or len(review.get('results', [])) != expected
            or actual_keys != expected_keys
            or any(not (r.get('adoption_eligible') or r.get('permanently_disqualified'))
                   for r in review['results'])):
        raise RuntimeError('R4 development review is incomplete or malformed')
    return bundle, review


def _git_bytes(commit, name):
    return subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)


def _verify_d4_historical_noninterference():
    """Pin and bound the only two post-D4 orchestration-only source changes."""
    records = {}
    for name, (old_digest, new_digest) in HISTORICAL_D4_ORCHESTRATION.items():
        old = _git_bytes(HISTORICAL_COMMIT, name); new = _git_bytes(CURRENT_R3_COMMIT, name)
        if hashlib.sha256(old).hexdigest() != old_digest or hashlib.sha256(new).hexdigest() != new_digest:
            raise RuntimeError('pinned D4 historical Git object differs: '+name)
        if _digest(ROOT/name) != new_digest: raise RuntimeError('current R3 tool differs: '+name)
        records[name] = {'historical_commit': HISTORICAL_COMMIT, 'historical_sha256': old_digest,
            'current_R3_commit': CURRENT_R3_COMMIT, 'current_sha256': new_digest}
    diff = subprocess.check_output(['git','diff','--no-ext-diff','--unified=80',HISTORICAL_COMMIT,
        CURRENT_R3_COMMIT,'--',*HISTORICAL_D4_ORCHESTRATION], cwd=ROOT)
    if hashlib.sha256(diff).hexdigest() != D4_TOOL_DIFF_SHA256:
        raise RuntimeError('historical D4 orchestration diff is not the reviewed diff')
    scope, s6 = D/'R3_D4_SCOPE_PARITY.json', D/'R3_S6_VALIDATION.json'
    if _digest(scope) != D4_SCOPE_SHA256 or _digest(s6) != R3_S6_SHA256:
        raise RuntimeError('R3 D4 scope/S6 evidence bytes changed')
    scope_doc, s6_doc = S.read(scope), S.read(s6)
    if (any(not a.get('historical_R1_reported_field_parity') or a.get('mismatches')
            for a in scope_doc['architectures'].values())
            or s6_doc['scientific_evidence']['source_commit'] != CURRENT_R3_COMMIT
            or s6_doc['scientific_evidence']['source_current_HEAD_Git_verified'] is not True
            or s6_doc['scientific_evidence']['final_three_files_R1_byte_identical'] is not True):
        raise RuntimeError('R3 D4 scope/S6 evidence does not support bounded reuse')
    return {'exceptions': records, 'exact_diff_sha256': D4_TOOL_DIFF_SHA256,
        'comparison_reason': ('Exact Git diff adds/refactors host capacity detection and replaces protocol Git lookup '
          'with the same pinned protocol-byte hash; run_case, plant/controller/detector calls, metrics and '
          'classification are unchanged. This is a bounded instrumentation/guard exception, not an assertion '
          'that all historical source bytes equal current bytes.'),
        'R3_D4_SCOPE_PARITY_sha256': D4_SCOPE_SHA256, 'R3_S6_VALIDATION_sha256': R3_S6_SHA256}


def _verify_source_hashes(document, allow_historical_d4=False):
    expected = document.get('metadata', {}).get('source_sha256')
    if not isinstance(expected, dict) or not expected: raise RuntimeError('R3 source hashes missing')
    checks = {}
    for name, recorded in expected.items():
        if name == 'private_analysis_harness_sha256':
            if recorded != HARNESS_SHA256: raise RuntimeError('another private harness was recorded')
            checks[name] = {'recorded': recorded, 'status': 'separately_pinned_private'}; continue
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not (ROOT/relative).is_file():
            raise RuntimeError('unsafe or missing R3 source: '+name)
        current = _digest(ROOT/relative)
        if current == recorded:
            checks[name] = {'recorded': recorded, 'current': current, 'status': 'current_exact'}; continue
        allowed = HISTORICAL_D4_ORCHESTRATION.get(name) if allow_historical_d4 else None
        if allowed != (recorded, current): raise RuntimeError('R3 source differs from current: '+name)
        checks[name] = {'recorded': recorded, 'current': current,
                        'status': 'pinned_historical_orchestration_only'}
    return checks


def _validate_document(document, mode, stage, count, raw_dir, historical=False, raw_limit=None):
    meta, rows = document.get('metadata', {}), document.get('results')
    if (meta.get('stage') != stage or meta.get('completed') is not True or meta.get('official') is not False
            or not isinstance(rows,list) or len(rows)!=count
            or [r.get('case_id') for r in rows] != list(range(count))
            or any(r.get('architecture') != mode or r.get('official') is not False for r in rows)):
        raise RuntimeError('R3 cache shape/provenance differs')
    checks = _verify_source_hashes(document, historical); manifest=[]
    required={'obs_t','estimated_wrench','active','detect',*S.protocol()['detector_gates']['physical_arrays']}
    for row in rows[:count if raw_limit is None else raw_limit]:
        arrays=S.raw(row,raw_dir); missing=required-set(arrays)
        if missing: raise RuntimeError('cached raw arrays missing: '+','.join(sorted(missing)))
        manifest.append({'case_id':row['case_id'],'basename':row['private_raw_basename'],
                         'sha256':row['private_raw_sha256']})
    return checks,manifest


def _baseline(stage,mode,rows,source_file,checks,manifest,fresh=0,history=None):
    evidence={'document':source_file.name,'document_sha256':S.sha(source_file),'source_checks':checks,
              'raw_sha256':manifest,'classification':'exposed finite cache; never independent efficacy'}
    if history: evidence['historical_orchestration_noninterference']=history
    return {'schema':'R4_BASE_STREAMS.v1','stage':stage,'architecture':mode,
        'protocol_sha256':S.sha(S.P),'registration':S.REGISTRATION,'source_sha256':S.science_hashes(),
        'official':False,'fresh_simulation':bool(fresh),'fresh_rows':fresh,'cached_rows':len(rows)-fresh,
        'exposed_cached_source_evidence':evidence,'results':rows}


def _cal(mode,spec):
    return S.read(D/'R4_CALIBRATION.json')['parameters'][mode+':'+spec['id']]['calibration']


def replay_d4(raw_dir=OLD_D4_RAW):
    bundle,_=require_bench_ready(); history=_verify_d4_historical_noninterference(); outputs=[]
    scope = S.read(D/'R3_D4_SCOPE_PARITY.json')
    design = d4_robustness.build_cases('unused', 'unused')
    for mode in ('servo','torque'):
        source_file=D/f'R3_D4_R1_{mode}.json'; source=S.read(source_file)
        if scope['artifacts'].get(source_file.name) != S.sha(source_file):
            raise RuntimeError('D4 document differs from the scope-parity artifact')
        checks,manifest=_validate_document(source,mode,'d4',385,raw_dir,True)
        if Counter(r['kind'] for r in source['results']) != {'reference':189,'collision':180,'offset':16}:
            raise RuntimeError('D4 189/180/16 inventory changed')
        condition_fields = ('case_id','mass','com','collision','pick_offset','phase','direction','force_n')
        if any(any(row.get(k) != S.R.json_safe(expected.get(k)) for k in condition_fields)
               for row, expected in zip(source['results'], design)):
            raise RuntimeError('D4 cached conditions differ from the fixed 385-row design')
        base=_baseline('d4',mode,copy.deepcopy(source['results']),source_file,checks,manifest,history=history)
        S.write_new(D/f'R4_BASE_d4_{mode}.json',base)
        for spec in S.protocol()['candidate_grid']:
            result,_=S.replay_document(base,raw_dir,spec,_cal(mode,spec))
            result['exposed_cached_source_evidence']=base['exposed_cached_source_evidence']
            S.write_new(D/f'R4_D4_{spec["id"]}_{mode}.json',result); outputs.append([mode,spec['id'],385])
    if S.runtime_bundle()!=bundle: raise RuntimeError('runtime bundle drift during D4')
    return outputs


def _verify_harness(path):
    path=S.R.require_private(path); record=S.read(D/'R3_PERTURB_HARNESS.json')
    if (record.get('instrumented_harness_sha256') != HARNESS_SHA256
            or record.get('physical_matrix_AST_identical') is not True
            or record.get('PerturbedPlant_AST_identical') is not True
            or record.get('sim_and_model_modules_modified_or_monkeypatched') is not False
            or _digest(path)!=HARNESS_SHA256): raise RuntimeError('private harness not pinned')
    return path,_load_perturb_harness(path)


def perturb_parity(private,harness_path=OLD_HARNESS):
    bundle,_=require_bench_ready(); private=S.R.require_private(private); private.mkdir(parents=True,exist_ok=True)
    target=D/'R4_PERTURB_PARITY.json'
    if target.exists(): raise RuntimeError('perturb parity already exists')
    harness_path,harness=_verify_harness(harness_path); c=S.protocol()['development']['conditions'][0]
    keys=S.protocol()['detector_gates']['physical_arrays']; records=[]
    for mode in ('servo','torque'):
        controller=importlib.import_module('r1_'+mode).Controller
        common=dict(box_mass=c['mass'],box_com=c['com'],collision=None,pick_offset=None)
        detector=importlib.import_module('r1_detector').Detector
        before=S.R.resources.wait_for_capacity(); a,la=run_episode(controller(),detector=detector(),**common)
        b,lb=harness.run_perturbed(controller(),detector=detector(),**common,
            perturb={'seed':7,'stiffness_scale':1.,'viscous_scale':1.,'coulomb_scale':1.})
        bit={k:bool(np.array_equal(la[k],lb[k])) for k in keys}
        def same(left,right):
            return bool(np.array_equal(left,right)) if isinstance(left,np.ndarray) or isinstance(right,np.ndarray) else left==right
        equal={k:same(a.get(k),b.get(k)) for k in sorted(set(a)|set(b)) if k!='ctrl_time_us'}; files={}
        for label,log in (('original',la),('harness',lb)):
            path=private/f'r4_perturb_parity_{mode}_{label}.npz'
            if path.exists(): raise RuntimeError('retain prior parity attempt')
            np.savez_compressed(path,**{k:v for k,v in log.items() if isinstance(v,np.ndarray)})
            files[label]={'basename':path.name,'sha256':S.sha(path)}
        records.append({'architecture':mode,'mass':c['mass'],'com':c['com'],'physical_arrays_bit_equal':bit,
            'result_equal_except_timing':equal,'passed':all(bit.values()) and all(equal.values()),
            'private_raw':files,'host_before':before})
    out={'schema':'R4_PERTURB_PARITY.v1','created_utc':datetime.now(timezone.utc).isoformat(),
         'protocol_sha256':S.sha(S.P),'runtime_bundle_sha256':bundle,'harness_sha256':HARNESS_SHA256,
         'harness_record_sha256':S.sha(D/'R3_PERTURB_HARNESS.json'),'source_sha256':S.science_hashes(),
         'official':False,'new_nonofficial_development_condition':True,'results':records,
         'passed':all(r['passed'] for r in records),'no_sim_or_model_patch':True}
    S.write_new(target,out)
    if not out['passed']: raise RuntimeError('neutral harness parity failed')
    if S.runtime_bundle()!=bundle or _digest(harness_path)!=HARNESS_SHA256: raise RuntimeError('runtime/harness drift')
    return out


def noise_design():
    rows=[]; p=S.protocol()['perturbations']
    for seed in p['noise_seeds']:
        for mass in p['new_noise_masses_kg']:
            rows.append({'mass':float(mass),'com':[.02,-.01,.03],'collision':None,'pick_offset':None,
              'perturb':{'seed':int(seed),'stiffness_scale':1.,'viscous_scale':1.,'coulomb_scale':1.},
              'perturb_family':'new_noise','seed':int(seed),'stiffness_scale':1.,'viscous_scale':1.,'coulomb_scale':1.})
    if len(rows)!=9: raise RuntimeError('noise design must have nine rows')
    return rows


def collect_noise(private,harness_path=OLD_HARNESS):
    bundle,_=require_bench_ready(); parity=S.read(D/'R4_PERTURB_PARITY.json')
    if parity.get('passed') is not True or parity.get('runtime_bundle_sha256')!=bundle or parity.get('harness_sha256')!=HARNESS_SHA256:
        raise RuntimeError('current harness parity required')
    harness_path,harness=_verify_harness(harness_path)
    out=S.collect('noise',private,explicit=noise_design(),harness=harness)
    if S.runtime_bundle()!=bundle or _digest(harness_path)!=HARNESS_SHA256: raise RuntimeError('runtime/harness drift')
    return out


def _physical_rows(document):
    rows=document['results'][:25]; families=Counter(r.get('perturb_family') for r in rows)
    if (families != {'noise':9,'friction':4,'stiffness':4,'corner':2,'collision':6}
            or sum(r['collision'] is None for r in rows)!=19 or sum(r['collision'] is not None for r in rows)!=6):
        raise RuntimeError('first-25 physical perturbation inventory changed')
    return copy.deepcopy(rows)


def _merged_rows(cached,fresh):
    merged=[]
    for source_kind,rows in (('cached_physical',cached),('fresh_noise',fresh)):
        for row in rows:
            item=copy.deepcopy(row); item.update(source_case_id=row['case_id'],source_stage=row['stage'],
                source_kind=source_kind,case_id=len(merged),stage='r4_perturb',official=False); merged.append(item)
    if len(merged)!=34 or [r['case_id'] for r in merged]!=list(range(34)): raise RuntimeError('merge incomplete')
    return merged


def replay_perturbations(fresh_private,cached_private=OLD_PERTURB_RAW):
    bundle,_=require_bench_ready(); parity=S.read(D/'R4_PERTURB_PARITY.json')
    if parity.get('passed') is not True or parity.get('runtime_bundle_sha256')!=bundle: raise RuntimeError('current parity required')
    outputs=[]; aggregate=S.read(D/'R3_PERTURBATIONS.json')
    if (aggregate.get('harness_sha256')!=HARNESS_SHA256
            or aggregate.get('official') is not False):
        raise RuntimeError('R3 perturbation aggregate provenance differs')
    for mode in ('servo','torque'):
        source_file=D/f'R3_PERTURB_R1_{mode}.json'; source=S.read(source_file)
        binding=[row for row in aggregate.get('groups',[]) if row.get('result_file')==source_file.name]
        if len(binding)!=1 or binding[0].get('result_sha256')!=S.sha(source_file) or binding[0].get('rows')!=34:
            raise RuntimeError('R3 perturbation document differs from aggregate binding')
        checks,old_manifest=_validate_document(source,mode,'perturb',34,cached_private,raw_limit=25)
        cached={'stage':'perturb','architecture':mode,'results':_physical_rows(source)}
        fresh_file=D/f'R4_BASE_noise_{mode}.json'; fresh=S.read(fresh_file)
        if (fresh.get('schema')!='R4_BASE_STREAMS.v1' or fresh.get('stage')!='noise' or fresh.get('architecture')!=mode
                or fresh.get('protocol_sha256')!=S.sha(S.P) or fresh.get('source_sha256')!=S.science_hashes()
                or len(fresh.get('results',[]))!=9): raise RuntimeError('fresh noise provenance differs')
        new_manifest=[]
        for design,row in zip(noise_design(),fresh['results']):
            if row.get('kind') != 'reference' or any(row.get(k)!=v for k,v in design.items()):
                raise RuntimeError('fresh noise design differs')
            S.raw(row,fresh_private); new_manifest.append({'case_id':row['case_id'],'basename':row['private_raw_basename'],'sha256':row['private_raw_sha256']})
        base=_baseline('perturb',mode,_merged_rows(cached['results'],fresh['results']),source_file,checks,old_manifest+new_manifest,fresh=9)
        base['fresh_noise_source_evidence']={'document':fresh_file.name,'document_sha256':S.sha(fresh_file),'seeds':S.protocol()['perturbations']['noise_seeds'],'rows':9}
        S.write_new(D/f'R4_BASE_perturb_{mode}.json',base)
        for spec in S.protocol()['candidate_grid']:
            old,_=S.replay_document(cached,cached_private,spec,_cal(mode,spec)); new,_=S.replay_document(fresh,fresh_private,spec,_cal(mode,spec))
            old.update(baseline_document_stage='perturb',results=_merged_rows(old['results'],new['results']),cached_physical_rows=25,
                       fresh_noise_rows=9,perturbation_parity_sha256=S.sha(D/'R4_PERTURB_PARITY.json'))
            S.write_new(D/f'R4_PERTURB_{spec["id"]}_{mode}.json',old); outputs.append([mode,spec['id'],34])
    if S.runtime_bundle()!=bundle: raise RuntimeError('runtime bundle drift during perturb replay')
    return outputs


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('action',choices=('d4','perturb-parity','perturb-collect','perturb-replay'))
    ap.add_argument('--private-dir',type=Path); ap.add_argument('--d4-raw',type=Path,default=OLD_D4_RAW)
    ap.add_argument('--perturb-raw',type=Path,default=OLD_PERTURB_RAW); ap.add_argument('--harness',type=Path,default=OLD_HARNESS); a=ap.parse_args(argv)
    if a.action=='d4': result=replay_d4(a.d4_raw)
    elif a.action=='perturb-parity':
        if a.private_dir is None: ap.error('perturb-parity requires --private-dir')
        result=perturb_parity(a.private_dir,a.harness)
    elif a.action=='perturb-collect':
        if a.private_dir is None: ap.error('perturb-collect requires --private-dir')
        result=collect_noise(a.private_dir,a.harness)
    else:
        if a.private_dir is None: ap.error('perturb-replay requires --private-dir')
        result=replay_perturbations(a.private_dir,a.perturb_raw)
    print(json.dumps(S.R.json_safe(result),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
