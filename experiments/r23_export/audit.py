"""Audit a generated R23 detector using an authenticated original simulator copy."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

import my_detector
import my_servo
from experiments.r22_challenge import evaluator as challenge
from experiments.r23_detached.candidate import Detector as Onset, Free
from experiments.r23_export import build
from sim.runner import run_episode
from tools.compare_baseline import official_cells
from tools.r21_replay import authenticate_kit
from tools.validate_submission import prepare_overlay


ROOT = Path(__file__).resolve().parents[2]
PHYSICAL = ("t", "q", "q_d", "tau", "e_mm", "f_table", "f_tool", "qpos", "f_ext")
TARGET = {"id":"s_after_release_diag", "mass":4.83, "com":[-.021,.018,.007], "offset":None,
          "impact":[3.927,[-7.7,-7.1,0.0]]}
PAIR = {"id":"p_release_cross", "mass":7.11, "com":[-.023,-.012,.016], "offset":None,
        "impacts":[[3.751,[-8.,7.5,0.]],[3.916,[7.,9.,0.]]]}
STOCK = {"kind":"stock"}
PARITY_CASES = ("released_normal", "released_force", "cross_none", "cross_first",
                "cross_second", "cross_both", "servo_normal", "servo_force")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    spec = importlib.util.spec_from_file_location("collision_detector", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def exact(left, right):
    return {key: bool(np.array_equal(left[key], right[key])) for key in PHYSICAL + ("r", "detect")}


def metric_exact(left, right):
    keys=("detect_delay_ms","alarm_before_collision","false_alarm_times","tool_contact_delay_ms")
    return {key:left[key]==right[key] for key in keys}


def episode(detector, case, impacts):
    return challenge.run_episode(case, STOCK, detector, impacts)


def parity(module, variant):
    modular = Onset if variant == "onset" else Free
    definitions = [
        ("released_normal", TARGET, (None, None)),
        ("released_force", TARGET, (TARGET["impact"], None)),
        ("cross_none", PAIR, (None, None)),
        ("cross_first", PAIR, (PAIR["impacts"][0], None)),
        ("cross_second", PAIR, (None, PAIR["impacts"][1])),
        ("cross_both", PAIR, tuple(PAIR["impacts"])),
    ]
    rows=[]
    for case_id, case, impacts in definitions:
        mm, ml = episode(modular, case, impacts); sm, sl = episode(module.Detector, case, impacts)
        same = exact(ml, sl)
        metrics=metric_exact(mm,sm)
        rows.append({"case_id":case_id,"controller":"torque","samples":len(sl["t"]),
                     "exact":same,"metrics_exact":metrics,
                     "passed":len(sl["t"])==6000 and all(same.values()) and all(metrics.values())})
    for case in ({"case_id":"servo_normal","mass":8.,"com":[0.,0.,0.],"collision":None},
                 {"case_id":"servo_force","mass":8.,"com":[0.,0.,0.],"collision":(1.42,[0.,10.,0.])}):
        bm, bl=run_episode(my_servo.Controller(),box_mass=case["mass"],box_com=case["com"],
                           detector=my_detector.Detector(),collision=case["collision"])
        sm, sl=run_episode(my_servo.Controller(),box_mass=case["mass"],box_com=case["com"],
                           detector=module.Detector(),collision=case["collision"])
        same=exact(bl,sl)
        metrics=metric_exact(bm,sm)
        rows.append({"case_id":case["case_id"],"controller":"servo","samples":len(sl["t"]),
                     "exact":same,"metrics_exact":metrics,
                     "passed":len(sl["t"])==6000 and all(same.values()) and all(metrics.values())})
    return rows


def run_evaluator(overlay, mode, env):
    process=subprocess.run([os.sys.executable,"-B","evaluate.py","--ctrl",mode+"_controller",
                            "--detector","collision_detector","--out",mode+".txt"],
                           cwd=overlay,env=env,text=True,capture_output=True)
    if process.returncode: raise RuntimeError(process.stderr)
    if sum(line.strip().startswith("실행 중: 박스") for line in process.stdout.splitlines())!=29:
        raise ValueError("incomplete original evaluator run")
    control,collisions=official_cells((overlay/(mode+".txt")).read_text())
    return {"episodes":29,"control":{str(k):v for k,v in control.items()},
            "collisions":[{"mass":m,"time":t,"cells":collisions[m,t]} for m,t in sorted(collisions)]}


def cells(report): return {key:report[key] for key in ("control","collisions")}
def count(report): return sum(value.isdecimal() for row in report["collisions"] for value in row["cells"][:4])
def expected(mode): return json.loads((ROOT/"results/r10/official_regression.json").read_bytes())[mode]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--kit",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--variant",choices=("onset","free"),required=True)
    a=p.parse_args(argv)
    try:
        kit,out=a.kit.resolve(strict=True),a.out.resolve()
        if a.kit.is_symlink() or not kit.is_dir() or os.path.lexists(a.out) or out.parent.is_relative_to(kit) or out.parent.is_relative_to(ROOT):
            raise ValueError("regular original kit and new external output required")
        manifest=(ROOT/"tools/kit_manifest.sha256").read_bytes();original=authenticate_kit(kit,manifest)
        with tempfile.TemporaryDirectory(prefix="r23-export-",dir=out.parent) as tmp:
            overlay=Path(tmp);torque_pin,detector_pin=prepare_overlay(kit,overlay,"my","torque")
            receipt=build.build(overlay/"standalone.py",a.variant)
            shutil.copyfile(overlay/"standalone.py",overlay/"collision_detector.py")
            shutil.copyfile(ROOT/"my_servo.py",overlay/"servo_controller.py")
            if digest(overlay/"collision_detector.py")!=receipt["standalone_sha256"]:raise ValueError("generated detector differs")
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",MUJOCO_GL="disable")
            env.pop("PYTHONPATH",None);env.pop("PYTHONHOME",None)
            torque,servo=run_evaluator(overlay,"torque",env),run_evaluator(overlay,"servo",env)
            if cells(servo)!=expected("servo")["control_and_collisions"]:raise ValueError("servo fallback differs from R10")
            expected_control=expected("torque")["control_and_collisions"]["control"]
            if any(row[:5]+row[6:]!=expected_control[mass][:5]+expected_control[mass][6:] for mass,row in torque["control"].items()):
                raise ValueError("torque controller cells differ from R10")
            module=load(overlay/"collision_detector.py");paired=parity(module,a.variant)
            if not all(row["passed"] for row in paired):raise ValueError("modular/generated or servo parity differs")
            if authenticate_kit(kit,manifest)!=original or digest(ROOT/"my_detector.py")!=detector_pin:raise ValueError("protected input changed")
            payload={"schema":"r23-release-audit.v1","variant":a.variant,"status":"research_option",
                     "default_submission_promoted":False,"build":receipt,
                     "kit":{"status":"ok","files":len(original),"manifest_sha256":hashlib.sha256(manifest).hexdigest()},
                     "evaluator_sha256":digest(overlay/"evaluate.py"),"controllers":{"torque":torque_pin,"servo":expected("servo")["source_sha256"]},
                     "official":{"torque":torque,"servo":servo},"recognized_collision_cells":{"torque":count(torque),"servo":count(servo)},
                     "parity":paired,"gates":{"default_filename":True,"torque_controller_R10":True,"servo_R10_exact":True,
                     "modular_generated_6000_exact":True,"released_force_and_normal_exact":True,"cross_release_four_twins_exact":True,
                     "protected_kit_unchanged":True}}
        out.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    except (OSError,ValueError,RuntimeError,json.JSONDecodeError) as error:p.error(str(error))
    print(json.dumps({"variant":a.variant,"torque":count(torque),"servo":count(servo),"parity":len(paired),"promoted":False}))


if __name__=="__main__":main()
