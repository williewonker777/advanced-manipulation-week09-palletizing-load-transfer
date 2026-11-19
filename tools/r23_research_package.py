"""Create a deterministic, audited, opt-in R23 simulator research package."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.r23_export import build
from tools import r22_challenge_replay
from tools.r21_replay import authenticate_kit
from tools.r22_torque_package import PUBLIC_DEPENDENCIES as R22_PUBLIC


TORQUE=ROOT/"my_torque.py"
TORQUE_SHA256="8f66f8f03cde65fcc6d8ef2f11643691cd8fe2e96dad6eaeaa17a0c7aa3db371"
DISCLOSURE=ROOT/"results/r23/selection.json"
NAMES=("torque_controller.py","collision_detector.py","README.md","receipt.json")
PARITY_CASES=("released_normal","released_force","cross_none","cross_first",
              "cross_second","cross_both","servo_normal","servo_force")
R23_FILES=("experiments/r23_detached/__init__.py","experiments/r23_detached/state.py",
           "experiments/r23_detached/candidate.py","experiments/r23_detached/plan.json",
           "experiments/r23_export/__init__.py","experiments/r23_export/build.py",
           "experiments/r23_export/audit.py")
PUBLIC_DEPENDENCIES=tuple(dict.fromkeys(R22_PUBLIC+r22_challenge_replay.PUBLIC_SUPPORT+
    r22_challenge_replay.CHALLENGE+R23_FILES))


def sha(blob):return hashlib.sha256(blob).hexdigest()


def validate_paths(kit,output):
    if kit.is_symlink() or not kit.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("regular kit and fresh non-symlink output required")
    if not output.parent.is_dir() or output.parent.is_symlink():raise ValueError("regular output parent required")
    kit=kit.resolve(strict=True);parent=output.parent.resolve(strict=True)
    if parent.is_relative_to(ROOT.resolve()) or parent.is_relative_to(kit):
        raise ValueError("output must be external to repository and kit")
    return kit


def disclosure(variant):
    if not DISCLOSURE.exists():
        return {"status":"evaluation_pending","selected_for_independent_evaluation":False,
                "default_promoted":False,"independent_result":None}
    raw=DISCLOSURE.read_bytes();doc=json.loads(raw)
    if (doc.get("schema")!="r23-selection-and-independent-result.v1"
            or doc.get("default_promoted") is not False
            or doc.get("candidate_entrypoint") not in
              ("experiments.r23_detached.candidate:Detector","experiments.r23_detached.candidate:Free")
            or doc.get("candidate_sources")!={n:sha((ROOT/n).read_bytes()) for n in build.PINS}):
        raise ValueError("invalid or stale published R23 disclosure")
    selected=(doc["candidate_entrypoint"]==("experiments.r23_detached.candidate:"+
                                             ("Detector" if variant=="onset" else "Free")))
    if not selected:
        return {"status":"not_selected_or_independently_evaluated",
                "selected_for_independent_evaluation":False,"default_promoted":False,
                "independent_result":None,"published_disclosure_sha256":sha(raw)}
    result=doc.get("independent_result")
    required=("verdict","episodes","rows","failed_rows","strict_gain_vs_r22","strict_loss_vs_r22",
              "remaining_false","remaining_misses")
    if (doc.get("selected_for_independent_evaluation") is not True or not isinstance(result,dict)
            or result.get("verdict") not in ("PASS","FAIL")
            or any(type(result.get(k)) is not int or result[k]<0 for k in required[1:])):
        raise ValueError("selected R23 disclosure lacks bounded independent result")
    return {"status":doc["status"],"selected_for_independent_evaluation":doc["selected_for_independent_evaluation"],
            "default_promoted":False,"independent_result":result,
            "published_disclosure_sha256":sha(raw)}


def run_audit(kit,output,variant):
    manifest=(ROOT/"tools/kit_manifest.sha256").read_bytes();authenticate_kit(kit,manifest)
    with tempfile.TemporaryDirectory(prefix="r23-audit-runtime-",dir=output.parent) as tmp:
        runtime=Path(tmp)
        for relative in PUBLIC_DEPENDENCIES:
            source=ROOT/relative;rel=Path(relative)
            if rel.is_absolute() or ".." in rel.parts or source.is_symlink() or not source.is_file():
                raise ValueError("regular bounded public dependency required: "+relative)
            target=runtime/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        for tree in ("sim","model"):shutil.copytree(kit/tree,runtime/tree,symlinks=False)
        shutil.copyfile(kit/"evaluate.py",runtime/"evaluate.py");authenticate_kit(runtime,manifest)
        bootstrap=("import runpy,sys;root=sys.argv.pop(1);sys.path.insert(0,root);"
                   "runpy.run_module('experiments.r23_export.audit',run_name='__main__')")
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",MUJOCO_GL="disable")
        env.pop("PYTHONPATH",None);env.pop("PYTHONHOME",None)
        process=subprocess.run([sys.executable,"-I","-B","-c",bootstrap,str(runtime),"--kit",str(kit),
                                "--out",str(output),"--variant",variant],cwd=runtime,env=env,text=True,capture_output=True)
    if process.returncode:raise RuntimeError(process.stderr.strip() or "R23 audit failed")


def validate_audit(doc,variant,detector_sha):
    required=("default_filename","torque_controller_R10","servo_R10_exact","modular_generated_6000_exact",
              "released_force_and_normal_exact","cross_release_four_twins_exact","protected_kit_unchanged")
    parity=doc.get("parity")
    if (doc.get("schema")!="r23-release-audit.v1" or doc.get("variant")!=variant
            or doc.get("default_submission_promoted") is not False
            or doc.get("build",{}).get("standalone_sha256")!=detector_sha
            or doc.get("recognized_collision_cells")!={"torque":24,"servo":23}
            or not isinstance(parity,list) or [r.get("case_id") for r in parity]!=list(PARITY_CASES)
            or not all(r.get("samples")==6000 and r.get("passed") is True for r in parity)
            or not all(doc.get("gates",{}).get(k) is True for k in required)
            or not all(doc.get("official",{}).get(k,{}).get("episodes")==29 for k in ("torque","servo"))):
        raise ValueError("audit does not prove the frozen R23 standalone")


def readme(variant,decision):
    mode=("requires a low-force interval before accepting a new post-detachment onset" if variant=="onset"
          else "accepts magnitude-only post-detachment force evidence")
    result=decision.get("independent_result")
    if decision["status"]=="evaluation_pending":
        status="No independent R23 result is published; this artifact is evaluation-pending and must not be selected."
    elif result is None:
        status="This variant was not selected or independently evaluated; it must not borrow the selected variant's score."
    else:
        status=("Independent verdict: {verdict}; {failed_rows}/{rows} rows failed across {episodes} episodes; "
                "strict gains/losses versus R22 were {strict_gain_vs_r22}/{strict_loss_vs_r22}; "
                "remaining false events/misses were {remaining_false}/{remaining_misses}. ").format(**result)
        totals=result.get("strict_totals",{}).get("r23")
        if isinstance(totals,dict):
            combined=totals["single_total"]+totals["pair_slots"]
            status+=("R23 strict singles/pair slots were {strict_singles}/{single_total} and "
                     "{strict_pair_slots}/{pair_slots}; {strict_failed_slots} of their "+str(combined)+" combined slots failed; "
                     "multiple-rise slots were {slots_with_multiple_rises}; genuine tool rows detected were "
                     "{negative_actual_tool_detected}/{negatives_with_actual_tool}. ").format(**totals)
    return ("# Optional R23 post-detachment detector ("+variant+")\n\n"
            "Research/education simulator option only. It is not the original course demo and not the default submission, "
            "a hardware detector, or a safety/hard-real-time certificate. The R10 torque controller is unchanged. "
            "The generated detector retains R22 LIGHT and "+mode+". "+status+" "
            "The generator authenticated the original evaluator, checked 24/24 torque cells, exact R10 servo fallback, "
            "and exact 6,000-sample modular/standalone traces for a disclosed post-release normal/impact and all four "
            "twins of a disclosed cross-release pair. These finite exposed checks do not justify promotion.\n").encode()


def info(name):
    value=ZipInfo(name,(1980,1,1,0,0,0));value.compress_type=ZIP_DEFLATED;value.create_system=3;value.external_attr=0o100644<<16
    return value


def create_package(kit,output,variant="onset",*,audit_runner=run_audit):
    kit,output=Path(kit),Path(output);kit=validate_paths(kit,output)
    torque=TORQUE.read_bytes()
    if sha(torque)!=TORQUE_SHA256:raise ValueError("frozen R10 torque differs")
    decision=disclosure(variant)
    with tempfile.TemporaryDirectory(prefix="r23-package-",dir=output.parent) as tmp:
        temp=Path(tmp);detector_path=temp/"collision_detector.py";build_receipt=build.build(detector_path,variant)
        detector=detector_path.read_bytes();audit_path=temp/"audit.json";audit_runner(kit,audit_path,variant)
        if not audit_path.is_file() or audit_path.is_symlink():raise ValueError("audit did not create a regular receipt")
        audit_raw=audit_path.read_bytes();audit_doc=json.loads(audit_raw);validate_audit(audit_doc,variant,sha(detector))
        guide=readme(variant,decision)
        receipt={"schema":"r23-research-package.v1","variant":variant,"status":"research_option",
                 "stock_course_demo":False,"default_submission_promoted":False,"selection":decision,
                 "audit_sha256":sha(audit_raw),"build":build_receipt,
                 "kit_manifest_sha256":audit_doc["kit"]["manifest_sha256"],"evaluator_sha256":audit_doc["evaluator_sha256"],
                 "recognized_collision_cells":audit_doc["recognized_collision_cells"],
                 "parity_case_ids":[r["case_id"] for r in audit_doc["parity"]],"gates":audit_doc["gates"],
                 "files":{"torque_controller.py":sha(torque),"collision_detector.py":sha(detector),"README.md":sha(guide)}}
        receipt_raw=(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode();buffer=BytesIO()
        with ZipFile(buffer,"w",ZIP_DEFLATED,compresslevel=9,strict_timestamps=True) as archive:
            for name,blob in zip(NAMES,(torque,detector,guide,receipt_raw),strict=True):archive.writestr(info(name),blob,compresslevel=9)
        with output.open("xb") as stream:stream.write(buffer.getvalue())
    return {"archive_sha256":sha(output.read_bytes()),"audit_sha256":receipt["audit_sha256"],
            "standalone_sha256":build_receipt["standalone_sha256"],"entries":list(NAMES),
            "selection_status":decision["status"]}


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--kit",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--variant",choices=("onset","free"),default="onset");a=p.parse_args(argv)
    try:result=create_package(a.kit,a.out,a.variant)
    except (OSError,ValueError,RuntimeError,json.JSONDecodeError) as error:p.error(str(error))
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":main()
