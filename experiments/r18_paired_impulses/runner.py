"""Closure-local Plant instrumentation for signed paired impulses on the stock runner."""
from __future__ import annotations
import hashlib,types
from pathlib import Path
import numpy as np
import sim.runner as stock
from sim.plant import Plant as StockPlant


def source_sha256():
    return hashlib.sha256(Path(stock.__file__).read_bytes()).hexdigest()


def pulse(t,impact):
    if impact is None:return np.zeros(3)
    onset,vector=impact;age=float(t)-float(onset)
    weight=min(age/stock.COLLISION_RAMP,1.,(2*stock.COLLISION_RAMP+stock.COLLISION_HOLD-age)/stock.COLLISION_RAMP)
    return max(weight,0.)*np.asarray(vector,float)


def instrumented(second):
    """Create a per-call Plant class; no module or shared simulator state is patched."""
    class InstrumentedPlant(StockPlant):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.r18_push_count=0;self.r18_committed_force=[];self.r18_native_force=[];self.r18_second_force=[]
        def push(self,native_force):
            t=self.r18_push_count*stock.C.DT;extra=pulse(t,second);combined=np.asarray(native_force,float)+extra
            super().push(combined)
            body=self.m.body('link_6').id
            self.r18_native_force.append(np.asarray(native_force,float).copy());self.r18_second_force.append(extra.copy())
            self.r18_committed_force.append(self.d.xfrc_applied[body,:3].copy());self.r18_push_count+=1
    return InstrumentedPlant


def cloned_run_episode(second=None):
    namespace=dict(stock.run_episode.__globals__,Plant=instrumented(second))
    function=types.FunctionType(stock.run_episode.__code__,namespace,stock.run_episode.__name__,
                                stock.run_episode.__defaults__,stock.run_episode.__closure__)
    function.__kwdefaults__=stock.run_episode.__kwdefaults__
    return function


def run_episode(*args,impacts=None,**kwargs):
    """Execute stock bytecode; paired mode uses native first plus hook-only second.

    ``impacts`` is ``(first, second)``; either entry may be ``None`` for twins.
    The native stock ``f_ext`` remains the first-pulse report.  Independent hook
    records expose the actual signed wrench committed to the flange.
    """
    if impacts is None:
        second=None
    else:
        if kwargs.get('collision') is not None:raise ValueError('collision and impacts are mutually exclusive')
        if not isinstance(impacts,(tuple,list)) or len(impacts)!=2:raise ValueError('impacts must be (first, second)')
        kwargs['collision'],second=impacts
    metrics,log=cloned_run_episode(second)(*args,**kwargs)
    plant=log['plant'];committed=np.asarray(plant.r18_committed_force);native=np.asarray(plant.r18_native_force);extra=np.asarray(plant.r18_second_force)
    if committed.shape!=(len(log['t']),3):raise ValueError('incomplete committed flange-wrench record')
    log['f_ext_xyz_committed']=committed;log['f_ext_xyz_native']=native;log['f_ext_xyz_second']=extra
    log['f_ext_combined_norm']=np.linalg.norm(committed,axis=1)
    return metrics,log


def expected_signed(t,impacts):
    t=np.asarray(t,float);out=np.zeros((len(t),3))
    for impact in impacts:
        if impact is not None:
            out+=np.stack([pulse(sample,impact) for sample in t])
    return out
