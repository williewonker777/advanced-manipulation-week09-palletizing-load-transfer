"""Experimental pre-contact vertical grasp registration from public signals only.

Each inference uses one episode, through t=3.55 s, and a calibration from
known, undisplaced pickups. No force ground truth, box state, collision or
fixture parameters are accepted. Millimetres unless otherwise stated.
"""
import numpy as np


DECISION_S=3.55
DT=.001
CENTRES=np.r_[np.arange(1550,1691,8),np.arange(1840,3511,8)]
HALF=20
U=np.linspace(-1.,1.,2*HALF+1)
WEIGHT=(1.-U**2)**2
WEIGHT/=WEIGHT.sum()


def flange(nom,q):
    import mujoco
    nom.d.qpos[:]=q
    mujoco.mj_kinematics(nom.m,nom.d)
    ix=nom.m.body('link_6').id
    return nom.d.xpos[ix].copy(),nom.d.xmat[ix].reshape(3,3).copy()


def filtered_input(t,theta,sensed,grip,dt,stiffness,damping,elastic,hz):
    t,theta,sensed,grip=map(np.asarray,(t,theta,sensed,grip))
    if (not np.isclose(dt,DT,atol=1e-12,rtol=0) or t.ndim!=1 or len(t)<3550 or
            len(t)>3551 or theta.shape!=(len(t),6) or sensed.shape!=theta.shape or
            grip.shape!=t.shape or not np.array_equal(t,dt*np.arange(len(t))) or
            not all(np.isfinite(a).all() for a in (theta,sensed,grip))):
        raise ValueError('expected regular finite public data from 0 to the 3.55 s deadline')
    if not np.isin(grip,[0,1]).all() or np.any(~grip[CENTRES].astype(bool)):
        raise ValueError('a carried payload is required throughout the free-flight windows')
    edges=np.flatnonzero(np.diff(grip.astype(int),prepend=0)==1)
    if len(edges)!=1 or not 0<edges[0]<CENTRES[0]-HALF-2:
        raise ValueError('one prior grasp required')
    # Reconstruct the signal available at a nodal pose (all look-ahead <=
    # 22 samples *relative to the past fit centre*, never past the decision).
    inverse=1./np.expm1(2*np.pi*hz*dt)
    sensor=sensed[1:]+inverse*(sensed[1:]-sensed[:-1])
    tau=sensor[:-1].copy()
    tau[:,~elastic]=.5*(sensor[:-1,~elastic]+sensor[1:,~elastic])
    tconst=np.asarray(damping)[elastic]/np.asarray(stiffness)[elastic]
    if np.any(tconst<=0):raise ValueError('positive elastic time constants required')
    memory=np.exp(-dt/tconst)
    spring=tau[:,elastic]/np.asarray(stiffness)[elastic]
    disp=np.zeros_like(tau)
    d=spring[0].copy();disp[0,elastic]=d
    ramp=1.-(tconst/dt)*(1.-memory)
    for i in range(1,len(spring)):
        prev=spring[i-1]
        d=memory*d+(1.-memory)*prev+ramp*(spring[i]-prev)
        disp[i,elastic]=d
    return theta[:len(tau)],disp,tau,int(edges[0])


def kinematics(table):
    ix=CENTRES[:,None]+np.arange(-HALF,HALF+1)
    values=table[ix]
    avg=lambda a:np.einsum('s,nsj->nj',WEIGHT,a)
    return (avg(values),avg(table[ix+1]-table[ix-1])/(2.*DT),
            avg((table[ix+1]-values)-(values-table[ix-1]))/(DT**2))


def public_features(t,theta,jts,grip,mass,com):
    from sim import config as c, trajectory
    from sim.nominal import Nominal
    com=np.asarray(com,dtype=float)
    if not np.isfinite(mass) or mass<=0 or com.shape!=(3,) or not np.isfinite(com).all():
        raise ValueError('known positive finite payload and 3-D COM required')
    nom=Nominal()
    enc,disp,tau,latch=filtered_input(t,theta,jts,grip,c.DT,nom.K,c.D,nom.elastic,c.JTS_CUTOFF)
    pick,rot=flange(nom,trajectory.reference(trajectory.T_GRASP)[0])
    meas,actual=flange(nom,enc[latch]-disp[latch])
    _,placed=flange(nom,trajectory.reference(trajectory.T_RELEASE)[0])
    box=np.asarray(c.BOX_POS);half=.5*np.asarray(c.BOX_SIZE)
    rel=actual.T@rot
    centre=actual.T@(pick+rot@box-meas)
    nominal_offset=centre+rel@com-box-com
    prior_bottom=placed[2]@(centre-box) - (abs(placed[2]@rel)-abs(placed[2]))@half
    q0,v0,a0=kinematics(enc-disp)
    dq,dv,da=kinematics(disp)
    def physics(com_shift,rho=0.):
        nom.set_payload(mass,com+nominal_offset+com_shift)
        return np.array([nom.inverse_dynamics(q,v,a) for q,v,a in
                         zip(q0-rho*dq,v0-rho*dv,a0-rho*da)])
    base=physics(np.zeros(3))
    delta=np.eye(3)*.001
    jac=np.stack([.5*(physics(d)-physics(-d)) for d in delta],axis=-1)
    scale_col=(physics(np.zeros(3),.05)-physics(np.zeros(3),-.05))/.10
    tau_mean=kinematics(tau)[0]
    return {'mass':np.array(mass),'y':tau_mean-base,'jac':jac,'stiffness':scale_col,
            'prior_bottom_mm':np.array(1000*prior_bottom),
            'prior_com_mm':1000*nominal_offset,'vertical':placed[2].copy()}


def fit_normal(features,noise):
    masses=np.array([float(s['mass']) for s in features])
    if len(features)<3 or len(np.unique(masses))!=len(features):
        raise ValueError('three or more distinct calibrated masses required')
    x=np.c_[np.ones(len(masses)),masses-masses.mean()]
    ys=np.stack([s['y'] for s in features])
    b=np.linalg.lstsq(x,ys.reshape(len(x),-1),rcond=None)[0].reshape((2,*ys.shape[1:]))
    error=ys-np.einsum('nk,ktj->ntj',x,b)
    sigma=np.maximum(np.asarray(noise)*.22,np.median(abs(error),axis=(0,1))*1.4826)
    return {'bias':b,'center':masses.mean(),'range':np.array([masses.min(),masses.max()]),'sigma':sigma}


def _one(features,model,keep):
    mass=float(features['mass'])
    y=features['y'][keep]-(model['bias'][0]+(mass-model['center'])*model['bias'][1])[keep]
    jac=features['jac'][keep];stiff=features['stiffness'][keep]
    n,j=y.shape
    # Unknown fixed zero/drift plus one stiffness nuisance coordinate. Priors below
    # are in physical units, not chosen on the evaluation offsets.
    t=(DT*CENTRES[keep]-2.55)
    eye=np.broadcast_to(np.eye(j),(n,j,j))
    design=np.concatenate((jac,stiff[:,:,None],eye,eye*t[:,None,None]),axis=-1)
    X=(design/model['sigma'][None,:,None]).reshape(n*j,-1)
    target=(y/model['sigma']).ravel()
    scales=np.r_[np.full(3,15.),.065,np.maximum(model['sigma']*3,.06),
                 np.maximum(model['sigma']*.8,.02)]
    reg=np.diag(1./scales)
    coef=np.linalg.lstsq(np.vstack((X,reg)),np.r_[target,np.zeros(len(scales))],rcond=None)[0]
    if not np.isfinite(coef).all():raise ValueError('nonfinite fit')
    unit=X[:,:3]
    # Report the measured vertical information left after completely free nuisance
    # terms rather than suggesting a prior proves observability.
    u=features['vertical']
    orth=np.linalg.svd(u[None,:],full_matrices=True)[2][1:].T
    nuisance=np.c_[unit@orth,X[:,3:]]
    target=unit@u
    residual=target-nuisance@np.linalg.lstsq(nuisance,target,rcond=None)[0]
    return coef[:3],float(coef[3]),float(np.linalg.norm(residual)/np.sqrt(n*j))


def estimate(features,model,normal_guard):
    """Pre-contact estimate of the signed extra vertical displacement of the box."""
    full=np.ones(len(CENTRES),bool)
    com,rho,info=_one(features,model,full);v=features['vertical']
    point=float(v@com)
    fold=(CENTRES//80)%5
    deletion=[float(v@_one(features,model,fold!=i)[0]) for i in range(5)]
    half=.75+normal_guard+max(abs(point-x) for x in deletion)
    plausible=(float(model['range'][0])<=float(features['mass'])<=float(model['range'][1])
               and np.linalg.norm(com)<=15. and info>1e-4)
    return {'extra_height_mm':point,'absolute_bottom_mm':float(features['prior_bottom_mm'])+point,
            'absolute_com_mm':(features['prior_com_mm']+com).tolist(),
            'prior_bottom_mm':float(features['prior_bottom_mm']),'stiffness_nuisance':rho,
            'block_delete_height_mm':deletion,'normal_guard_mm':float(normal_guard),
            'half_width_mm':float(half),'interval_mm':[point-half,point+half],
            'independent_height_information_per_mm':info,
            'accept':bool(plausible and half<=2.5 and abs(point)>half),
            'fitted_displacement_within_domain':bool(plausible)}


def calibration(features,noise):
    model=fit_normal(features,noise)
    loo=[]
    for i,item in enumerate(features):
        smaller=fit_normal([x for j,x in enumerate(features) if j!=i],noise)
        c,_,_=_one(item,smaller,np.ones(len(CENTRES),bool));loo.append(float(item['vertical']@c))
    return model,max(abs(v) for v in loo),loo
