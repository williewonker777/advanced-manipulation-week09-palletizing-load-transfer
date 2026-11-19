"""Normal-calibrated short-window horizontal COM solve with DC/stiffness nuisance."""
import numpy as np
from sim import config as C

def fit_normal(features):
 masses=np.array([float(f['mass']) for f in features]);x=np.c_[np.ones(len(masses)),masses-masses.mean()];ys=np.stack([f['y'] for f in features]);b=np.linalg.lstsq(x,ys.reshape(len(x),-1),rcond=None)[0].reshape(2,*ys.shape[1:]);err=ys-np.einsum('nk,ktj->ntj',x,b);sigma=np.maximum(np.asarray(C.JTS_NOISE)*.22,np.median(abs(err),axis=(0,1))*1.4826);return {'bias':b,'center':masses.mean(),'range':np.array([masses.min(),masses.max()]),'sigma':sigma}
def _solutions(f,m,deleted=True):
 bias=np.asarray(m['bias']);sigma=np.asarray(m['sigma']);y=(f['y']-bias[0]-(float(f['mass'])-m['center'])*bias[1])/sigma;n,j=y.shape;eye=np.broadcast_to(np.eye(j),(n,j,j));x=np.concatenate((f['jac'],f['stiffness'][:,:,None],eye),axis=-1)/sigma[None,:,None];sc=np.r_[np.full(3,15.),.065,np.maximum(sigma*3,.06)];fold=np.arange(n)*5//n
 def solve(keep):
  z=x[keep].reshape(-1,x.shape[-1]);t=y[keep].ravel();c=np.linalg.solve(z.T@z+np.diag(1/sc**2),z.T@t);return np.asarray(f['prior_com_mm'])+c[:3]
 full=solve(np.ones(n,bool));deletions=np.array([solve(fold!=i) for i in range(5)]) if deleted else None;return full,deletions
def _model(rows):
 d=fit_normal(rows);obs=np.array([_solutions(f,d,False)[0][:2] for f in rows]);mass=np.array([float(f['mass']) for f in rows]);a=np.c_[np.ones(len(rows)),mass-d['center']];return d,np.linalg.lstsq(a,obs,rcond=None)[0]
def expected(m,d,b):return b[0]+(float(m)-d['center'])*b[1]
def calibrate(rows):
 d,b=_model(rows);loo=[]
 for i,r in enumerate(rows):
  x,y=_model([q for k,q in enumerate(rows) if k!=i]);loo.append(_solutions(r,x,False)[0][:2]-expected(r['mass'],x,y))
 return {'schema':'r15-early-com-calibration.v1','dynamics':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in d.items()},'baseline_xy':b.tolist(),'normal_guard_mm':float(max(np.linalg.norm(x) for x in loo)),'loo_vectors_mm':np.asarray(loo).tolist()}
def decide(f,m):
 if m.get('schema')!='r15-early-com-calibration.v1':raise ValueError('calibration schema')
 d=m['dynamics'];absolute,deleted=_solutions(f,d);normal=expected(f['mass'],d,np.asarray(m['baseline_xy']));xy=absolute[:2]-normal;length=float(np.linalg.norm(xy));radius=.75+float(m['normal_guard_mm'])+float(np.max(np.linalg.norm(deleted[:,:2]-absolute[:2],axis=1)));domain=d['range'][0]<=float(f['mass'])<=d['range'][1] and np.linalg.norm(absolute-np.asarray(f['prior_com_mm']))<=20;accept=bool(domain and radius<=2.5 and length>radius);action=xy*min(1,15/max(length,1e-12)) if accept else np.zeros(2);return {'relative_xy_mm':xy.tolist(),'absolute_com_offset_mm':absolute.tolist(),'expected_normal_xy_mm':normal.tolist(),'deleted_relative_xy_mm':(deleted[:,:2]-normal).tolist(),'halfwidth_radius_mm':float(radius),'accept':accept,'model_action_xy_mm':action.tolist(),'completed_at_s':float(f['completed_at_s'])}
