"""Export and parity-check the frozen R12 hybrid/counterfactual simulation demo."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCES = {
    "servo": ROOT / "r1_servo.py",
    "torque": ROOT / "r1_torque.py",
    "detector": ROOT / "my_detector.py",
    "hybrid": ROOT / "experiments/r12_hybrid/controller.py",
    "counter": ROOT / "experiments/r12_counterfactual/detector.py",
    "coefficients": ROOT / "models/r12/counterfactual_coefficients.json",
}
EXPECTED = {
    "servo": "40706968fb72ed8d7a864e44e828787ae8db0365c2927e5166e4a5b2d83bba8b",
    "torque": "8f66f8f03cde65fcc6d8ef2f11643691cd8fe2e96dad6eaeaa17a0c7aa3db371",
    "detector": "09f9a8573814a31b335b36ec5a78e0dedf262daa9039954568abd6bedd8dbc67",
    "hybrid": "d9fc2c3bc618e42fbed914b03257d442f44da39f7ac5b0033abafb333d493dc7",
    "counter": "62e40ff4d81b6d14643cb87bbebdfafb6f9d62835416ef47fe7215a5fff9aec9",
    "coefficients": "183df664558017be3e9d3faa9c5883cd6ca55c593c45a1e995afb0dcefee886c",
}
PRIVATE_FIXTURE_SHA = "8f637a5d67d6bfd2f9021093b08e5d8b463bd58c71fa4e7da3f752087ed72f2d"
RESEARCH_MODEL = ROOT / "experiments/r12_counterfactual/model.npz"
RESEARCH_MODEL_SHA = "5684d187ebf9a531f0b976fd5dc448c96ba56a8d94803bed4778f69c5f2be50b"
CASES = [
    {"id":"ce0","mass":4.85,"com":[.017,.027,-.015],"perturb":{"seed":945201,"stiffness_scale":1.05,"viscous_scale":.95,"coulomb_scale":.96},"force":[0,17,0]},
    {"id":"ce1","mass":6.35,"com":[-.025,.012,.021],"perturb":{"seed":945202,"stiffness_scale":.93,"viscous_scale":1.07,"coulomb_scale":1.04},"force":[0,0,-35]},
    {"id":"ce2","mass":7.75,"com":[.014,-.026,.028],"perturb":{"seed":945203,"stiffness_scale":1.08,"viscous_scale":.92,"coulomb_scale":1.06},"force":[13,-11,16]},
]
ARRAYS = ("q","q_d","tau","e_mm","f_table","f_tool","qpos","r","detect")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def has_symlink_component(path):
    path=Path(path).absolute()
    return any(os.path.lexists(item) and item.is_symlink() for item in (path,*path.parents))


HYBRID_APPENDIX = r'''

# R12 coupled demo: frozen physically tested hybrid controller.
_R12BaseController = Controller
class _R12Parameters:
    contact_threshold=.05
    placement_start=3.780
    blend_duration=.020
    release_return=.040
    robot_support_fraction=.45
    normal_position_scale=.20
def _r12_smoothstep(s):
    s=float(np.clip(s,0,1));return 10*s**3-15*s**4+6*s**5
def _r12_scale_vertical(torque,jz,scale):
    jz=np.asarray(jz,float);torque=np.asarray(torque,float);projection=jz*(float(jz@torque)/(float(jz@jz)+1e-12));return torque+(float(scale)-1.)*projection
class Controller(_R12BaseController):
    parameters=_R12Parameters()
    def reset(self):
        super().reset();self.hybrid_active=False;self.hybrid_start=None;self.hybrid_alpha=0.;self.commanded_load=0.;self.normal_position_scale=1.;self.minimum_normal_position_scale_seen=1.;self.max_alpha=0.;self.pre_release_load=None;self.release_load=None
    def _alpha(self,t):
        p=self.parameters
        if not self.hybrid_active and p.placement_start<=t<T.T_RELEASE:
            valid=self.shared_estimator.lambda_support_valid;support=float(np.clip(1-self.shared_estimator.lambda_support,0,1)) if valid else 0.
            if support>=p.contact_threshold:self.hybrid_active=True;self.hybrid_start=float(t)
        if not self.hybrid_active:return 0.
        if t<T.T_RELEASE:return _r12_smoothstep((t-self.hybrid_start)/p.blend_duration)
        return 1-_r12_smoothstep((t-T.T_RELEASE)/p.release_return)
    def compute(self,t,th,thd,tau_s,grip,q_d,qd_d,qdd_d):
        q_ref,qd_ref,qdd_ref=_R12BaseController._reference(self,t);base=self._estimate_load(t,th,thd,tau_s,grip)
        if self.UNLOAD_START<t<T.T_RELEASE:
            s=np.clip((t-self.UNLOAD_START)/(T.T_RELEASE-self.UNLOAD_START),0,1);base=min(base,1-self.UNLOAD_AMOUNT*_r12_smoothstep(s))
        alpha=self._alpha(t);p=self.parameters;load=(1-alpha)*base+alpha*p.robot_support_fraction
        if t>=T.T_RELEASE:load=alpha*p.robot_support_fraction
        self.hybrid_alpha=float(alpha);self.max_alpha=max(self.max_alpha,float(alpha));self.commanded_load=float(load);self.normal_position_scale=float(1-alpha*(1-p.normal_position_scale));self.minimum_normal_position_scale_seen=min(self.minimum_normal_position_scale_seen,self.normal_position_scale)
        if T.T_RELEASE-.0015<=t<T.T_RELEASE:self.pre_release_load=float(load)
        if t>=T.T_RELEASE and self.release_load is None:self.release_load=float(load)
        tau_empty=self.nom.inverse_dynamics(q_ref,qd_ref,qdd_ref);tau_full=self.nom_box.inverse_dynamics(q_ref,qd_ref,qdd_ref);tau_link=tau_empty+load*(tau_full-tau_empty);theta=q_ref+self.inv_k*tau_link;self.v_motor += .2*(thd-self.v_motor)
        if self.tau_link_prev is None:theta_dot,theta_ddot=qd_ref.copy(),qdd_ref.copy()
        else:
            raw=(tau_link-self.tau_link_prev)/self.dt;old=self.tau_dot.copy();self.tau_dot += .04*(raw-self.tau_dot);theta_dot=np.clip(qd_ref+self.inv_k*self.tau_dot,qd_ref-.35,qd_ref+.35);theta_ddot=np.clip(qdd_ref+self.inv_k*(self.tau_dot-old)/self.dt,-25.,25.)
        self.tau_link_prev=tau_link.copy();friction=C.VISCOUS*theta_dot+C.COULOMB*np.tanh(theta_dot/C.FRICTION_EPS);ff=self.nom.B*theta_ddot+tau_link+friction;position=self.nom.B*(self.wn**2*(theta-th));damping=self.nom.B*(2*self.wn*(theta_dot-self.v_motor));jz=self.nom.jacobian(q_ref)[2];position=_r12_scale_vertical(position,jz,self.normal_position_scale)
        return ff+position+damping+self.k_tau*(tau_s-tau_link)
'''


def detector_appendix(wrench, fast):
    return r'''

# R12 coupled demo: embedded normal-only counterfactual and independent replay.
_R12BaseDetector = Detector
_R12_WRENCH = np.asarray(''' + wrench + r''',dtype=np.float64)
_R12_FAST = np.asarray(''' + fast + r''',dtype=np.float64)
class Detector(_R12BaseDetector):
    def reset(self):
        super().reset();self.cf_alarm=0;self.cf_fast=0;self.cf_clear=0;self.cf_detected=False
    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,original=super().update(t,th,thd,tau_m,tau_s,grip)
        if getattr(getattr(self,"ctrl",None),"MODE",None)!="torque":return residual,original
        p=self._shared;k=min(round(float(t)*1000),len(_R12_WRENCH)-1);basis=np.array([1.,float(self.box_mass)]);dw=np.einsum("fc,c->f",_R12_WRENCH[k],basis);df=np.einsum("fc,c->f",_R12_FAST[k],basis);wrench=p.last_wrench-dw;res=p.last_residual_wrench-dw;innovation=p.last_fast_innovation-df;over,_=p._directional_alarm(res,wrench,p.last_table_allowed);fast_over,_=p._fast_directional_alarm(innovation)
        if grip:fast_over=False
        self.cf_alarm=self.cf_alarm+1 if over else 0;self.cf_fast=self.cf_fast+1 if fast_over else 0
        if over or fast_over:self.cf_clear=0
        else:self.cf_clear+=1
        if self.cf_alarm>=p.persistence or self.cf_fast>=p.fast_persistence:self.cf_detected=True
        elif self.cf_clear>=p.release_persistence:self.cf_detected=False
        learned=bool(getattr(self.normal_model,"detected",False) or getattr(self,"r5_extra",False) or getattr(self,"r8_extra",False))
        return residual,bool(self.cf_detected or learned)
'''


def build_sources():
    blobs={key:path.read_bytes() for key,path in SOURCES.items()}
    actual={key:hashlib.sha256(blob).hexdigest() for key,blob in blobs.items()}
    if actual != EXPECTED: raise ValueError("pinned research source mismatch: "+repr(actual))
    document=json.loads(blobs["coefficients"])
    if document.get("schema")!="r12-counterfactual-coefficients.v1":raise ValueError("coefficient schema mismatch")
    literals={}
    for name in ("wrench_coef","fast_coef"):
        record=document["arrays"][name]
        array=np.asarray(record["values"],dtype=np.float64)
        if record["dtype"]!="float64" or list(array.shape)!=record["shape"] or not np.all(np.isfinite(array)) or hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()!=record["raw_c_order_sha256"]:
            raise ValueError("invalid coefficient array: "+name)
        literals[name]=json.dumps(record["values"],separators=(",",":"),allow_nan=False)
    return {
        "my_servo.py":blobs["servo"].decode(),
        "my_torque.py":blobs["torque"].decode()+HYBRID_APPENDIX,
        "my_detector.py":blobs["detector"].decode()+detector_appendix(literals["wrench_coef"],literals["fast_coef"]),
    }, actual


def write_sources(out_dir, sources):
    out_dir=Path(out_dir)
    if os.path.lexists(out_dir) or has_symlink_component(out_dir): raise ValueError("out-dir must be new and not a symlink")
    if not out_dir.parent.is_dir():
        raise ValueError("out-dir parent must be an existing real directory")
    out_dir.mkdir(parents=False)
    for name,text in sources.items():
        with (out_dir/name).open("x") as stream: stream.write(text)
    return {name:sha(out_dir/name) for name in sources}


def verify_sources(out_dir, expected):
    actual={name:sha(Path(out_dir)/name) for name in expected}
    if actual != expected: raise ValueError("exported source tamper detected")
    if sorted(p.name for p in Path(out_dir).iterdir()) != sorted(expected):
        raise ValueError("export directory must contain exactly three source modules")
    return actual


def _research(case, collision):
    import numpy as np
    from experiments.r12_hybrid.controller import Controller
    from experiments.r12_counterfactual.detector import bound
    from tools.r5_fixture import run_owned_episode
    _,log=run_owned_episode(Controller(),case["mass"],tuple(case["com"]),bound(RESEARCH_MODEL)(),collision=collision,perturb=case["perturb"])
    return {key:np.asarray(log[key]) for key in ARRAYS}


def _subprocess_output(overlay, worker, filename):
    import numpy as np
    env=os.environ.copy();env.pop("PYTHONPATH",None);env.update(PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    done=subprocess.run([sys.executable,"-c",worker],cwd=overlay,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if done.returncode: raise RuntimeError("standalone worker failed: "+done.stderr)
    with np.load(overlay/filename,allow_pickle=False) as z:
        return {name:z[name].copy() for name in z.files}


def _overlay_run(kit, exported, cases, fixture):
    from tools.validate_submission import prepare_overlay
    outputs={}
    with tempfile.TemporaryDirectory(prefix="r12-coupled-") as folder:
        overlay=Path(folder);prepare_overlay(kit,overlay,"r1","torque")
        shutil.copyfile(exported/"my_torque.py",overlay/"torque_controller.py")
        shutil.copyfile(exported/"my_detector.py",overlay/"collision_detector.py")
        shutil.copyfile(fixture,overlay/"r5_fixture.py")
        design=overlay/"r12_cases.json";design.write_text(json.dumps(cases))
        worker=r'''import json,numpy as np
from pathlib import Path
from torque_controller import Controller
from collision_detector import Detector
from r5_fixture import run_owned_episode
cases=json.loads(Path("r12_cases.json").read_text());out={}
for c in cases:
 for kind,collision in (("normal",None),("release",[3.9,c["force"]])):
  _,log=run_owned_episode(Controller(),c["mass"],tuple(c["com"]),Detector(),collision=collision,perturb=c["perturb"])
  for key in ("q","q_d","tau","e_mm","f_table","f_tool","qpos","r","detect"):out[c["id"]+"-"+kind+"-"+key]=log[key]
np.savez("r12_output.npz",**out)'''
        outputs.update(_subprocess_output(overlay,worker,"r12_output.npz"))

        servo_worker=r'''import numpy as np
from servo_controller import Controller
from collision_detector import Detector
from sim.runner import run_episode
out={}
for kind,collision in (("normal",None),("collision",[2.2,[0,0,35]])):
 _,log=run_episode(Controller(),4.85,(.017,.027,-.015),Detector(),collision=collision)
 for key in ("q","q_d","tau","e_mm","f_table","f_tool","qpos","r","detect"):out[kind+"-"+key]=log[key]
np.savez("servo_output.npz",**out)'''
        shutil.copyfile(SOURCES["servo"],overlay/"servo_controller.py")
        shutil.copyfile(SOURCES["detector"],overlay/"collision_detector.py")
        servo_base=_subprocess_output(overlay,servo_worker,"servo_output.npz")
        shutil.copyfile(exported/"my_servo.py",overlay/"servo_controller.py")
        shutil.copyfile(exported/"my_detector.py",overlay/"collision_detector.py")
        servo_export=_subprocess_output(overlay,servo_worker,"servo_output.npz")
        for name,value in servo_base.items(): outputs["servo-base-"+name]=value
        for name,value in servo_export.items(): outputs["servo-export-"+name]=value
    return outputs


def export(out_dir):
    sources,inputs=build_sources();hashes=write_sources(out_dir,sources);verify_sources(out_dir,hashes)
    sizes={name:(Path(out_dir)/name).stat().st_size for name in hashes}
    return {"schema":"r12-coupled-demo.v1","status":"EXPORTED","scope":"simulation demo; detector-qualified scientific result, not adoption","inputs":inputs,"exports":hashes,"sizes_bytes":sizes,"embedded_model":{"public_json_bytes":SOURCES["coefficients"].stat().st_size,"encoding":"transparent float64 numeric literals in my_detector.py","compression":"none","sha256":inputs["coefficients"]},"limitations":["counterfactual correction is torque-mode only; servo retains the frozen R10 detector","two of three pickup-offset hybrid cases failed physical G4","simulation evidence only; no hardware or universal claim"]}


def run(kit,out_dir,fixture):
    import numpy as np
    if sha(RESEARCH_MODEL)!=RESEARCH_MODEL_SHA:raise ValueError("private research model pin mismatch")
    result=export(out_dir)
    standalone=_overlay_run(kit,Path(out_dir),CASES,fixture);rows=[]
    for case in CASES:
        for kind,collision in (("normal",None),("release",[3.9,case["force"]])):
            research=_research(case,collision);checks={key:np.array_equal(research[key],standalone[f'{case["id"]}-{kind}-{key}']) for key in ARRAYS}
            rows.append({"case":case["id"],"kind":kind,"arrays":checks,"all_equal":all(checks.values()),"detect_cycles":int(np.count_nonzero(research["detect"]))})
    if not all(row["all_equal"] for row in rows): raise ValueError("standalone parity failure")
    servo=[]
    for kind in ("normal","collision"):
        checks={key:np.array_equal(standalone[f"servo-base-{kind}-{key}"],standalone[f"servo-export-{kind}-{key}"]) for key in ARRAYS}
        servo.append({"kind":kind,"arrays":checks,"all_equal":all(checks.values()),"finite":all(np.all(np.isfinite(standalone[f"servo-export-{kind}-{key}"])) for key in ARRAYS if key!="detect")})
    if not all(row["all_equal"] and row["finite"] for row in servo): raise ValueError("servo fallback parity failure")
    result.update(status="PASS",parity=rows,servo_fallback_parity=servo)
    result["inputs"]["private_fixture"]=PRIVATE_FIXTURE_SHA
    result["inputs"]["private_research_npz"]=RESEARCH_MODEL_SHA
    result["limitations"].insert(2,"no tool contact occurred in the frozen release pairs")
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--kit",type=Path);p.add_argument("--fixture",type=Path,help="private parity fixture; not part of the export");p.add_argument("--out-dir",type=Path,required=True);p.add_argument("--out",type=Path,help="new parity JSON (requires --kit and --fixture)");a=p.parse_args(argv)
    parity=any(x is not None for x in (a.kit,a.fixture,a.out))
    if parity and not all(x is not None for x in (a.kit,a.fixture,a.out)):p.error("--kit, --fixture, and --out are required together for private parity")
    if parity and (not a.kit.is_dir() or a.kit.is_symlink()):p.error("kit must be a real directory")
    if parity and (not a.fixture.is_file() or a.fixture.is_symlink() or sha(a.fixture)!=PRIVATE_FIXTURE_SHA):p.error("fixture must be the pinned private parity fixture")
    if os.path.lexists(a.out_dir) or has_symlink_component(a.out_dir):p.error("out-dir must be new and not a symlink")
    if not a.out_dir.parent.is_dir():p.error("out-dir parent must be an existing real directory")
    if parity and (os.path.lexists(a.out) or has_symlink_component(a.out) or not a.out.parent.is_dir()):p.error("out must be a new non-symlink file in an existing real directory")
    out_dir=(a.out_dir.parent.resolve()/a.out_dir.name)
    out=(a.out.parent.resolve()/a.out.name) if parity else None
    if parity and (out_dir.is_relative_to(a.kit.resolve()) or out.is_relative_to(a.kit.resolve())):p.error("outputs must be outside the supplied kit")
    try:
        result=run(a.kit.resolve(),out_dir,a.fixture.resolve()) if parity else export(out_dir)
        if parity:
            with out.open("x") as stream: stream.write(json.dumps(result,indent=2)+"\n")
    except (OSError,ValueError,RuntimeError) as e:p.error(str(e))
    print(json.dumps({"status":result["status"],"exports":result["exports"]},sort_keys=True));return 0
if __name__=="__main__":sys.dont_write_bytecode=True;raise SystemExit(main())
