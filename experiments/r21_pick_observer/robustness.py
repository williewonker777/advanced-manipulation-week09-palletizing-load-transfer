"""Run frozen R21 on authenticated JTS-seed-only local fixtures."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from experiments.r18_seed_repeatability.kit import prepare
ROOT=Path(__file__).resolve().parents[2];MODEL=Path(__file__).with_name('model.json');CAND=Path(__file__).with_name('candidate.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(plan_path,out,work):
 raw=Path(plan_path).read_bytes();p=json.loads(raw);work=Path(work)
 if p['status']!='FROZEN_BEFORE_NOISE_OUTCOME' or sha(CAND)!=p['candidate_sha256'] or sha(MODEL)!=p['model_sha256'] or work.exists():raise ValueError('fresh frozen inputs required')
 work.mkdir();rows=[];pins=[]
 for seed in p['seeds']:
  kit=work/f'seed-{seed}';manifest=prepare(ROOT,kit,seed);result=work/f'seed-{seed}.json'
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='disable')
  subprocess.run([sys.executable,'-B','-m','experiments.r21_pick_observer.seed_worker','--seed',str(seed),'--masses',','.join(map(str,p['masses'])),'--com',','.join(map(str,p['com'])),'--time',str(p['collision']['time']),'--force',','.join(map(str,p['collision']['force'])),'--out',str(result)],cwd=kit,env=env,check=True)
  data=json.loads(result.read_bytes());rows.extend(data['rows']);pins.append(manifest)
 doc={'schema':'r21-seed-only-result.v1','plan_sha256':hashlib.sha256(raw).hexdigest(),'fixture':'only sim/config.py JTS_SEED differs from authenticated protected source','episodes':4*len(rows),'manifests':pins,'rows':rows,'passed':all(r['passed'] for r in rows)}
 Path(out).write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n');print(json.dumps({'episodes':doc['episodes'],'passed':doc['passed'],'delays':[(r['seed'],r['mass'],r['r20_delay'],r['r21_delay']) for r in rows]}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);p.add_argument('--work',required=True);a=p.parse_args();main(a.plan,a.out,a.work)
