import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile

from experiments.r23_export import audit, build
from tools import r23_research_package as package


class PackageTests(unittest.TestCase):
    def audit_doc(self,variant,receipt):
        return {"schema":"r23-release-audit.v1","variant":variant,"default_submission_promoted":False,
                "build":receipt,"kit":{"manifest_sha256":"kit"},"evaluator_sha256":"evaluator",
                "recognized_collision_cells":{"torque":24,"servo":23},
                "official":{k:{"episodes":29,"control":{"0.0":[]},"collisions":[[]]} for k in ("torque","servo")},
                "parity":[{"case_id":case,"samples":6000,"passed":True} for case in audit.PARITY_CASES],
                "gates":{k:True for k in ("default_filename","torque_controller_R10","servo_R10_exact",
                 "modular_generated_6000_exact","released_force_and_normal_exact","cross_release_four_twins_exact",
                 "protected_kit_unchanged")}}

    def audit(self,kit,out,variant):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"detector.py";receipt=build.build(path,variant)
        out.write_text(json.dumps(self.audit_doc(variant,receipt),sort_keys=True))

    def test_builder_is_deterministic_and_pins_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"a.py",Path(tmp)/"b.py"
            ra,rb=build.build(a,"onset"),build.build(b,"onset")
            self.assertEqual(a.read_bytes(),b.read_bytes());self.assertEqual(ra,rb)
            self.assertEqual(ra["frozen_r23_sources_sha256"],build.PINS)
            self.assertIn("class _R23OnsetGate",a.read_text())
            for forbidden in ("experiments.r23", "Path(__file__)", "open(", "read_text(", "read_bytes("):
                self.assertNotIn(forbidden,a.read_text())

    def test_audit_requires_exact_nonempty_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"detector.py";receipt=build.build(path,"onset");doc=self.audit_doc("onset",receipt)
            package.validate_audit(doc,"onset",receipt["standalone_sha256"])
            doc["parity"].pop()
            with self.assertRaises(ValueError):package.validate_audit(doc,"onset",receipt["standalone_sha256"])

    def test_bounded_runtime_uses_kit_and_excludes_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);public=root/"public";kit=root/"kit";out=root/"audit.json"
            (public/"tools").mkdir(parents=True);(public/"tools/kit_manifest.sha256").write_text("manifest")
            (public/"required.py").write_text("required");(public/"private-sentinel").write_text("private")
            (kit/"evaluate.py").parent.mkdir(parents=True);(kit/"evaluate.py").write_text("evaluate")
            for tree in ("sim","model"):(kit/tree).mkdir();(kit/tree/"marker").write_text(tree)
            def run(command,**kw):
                runtime=Path(kw["cwd"]);self.assertEqual((runtime/"required.py").read_text(),"required")
                self.assertFalse((runtime/"private-sentinel").exists());self.assertEqual(command[1:4],["-I","-B","-c"])
                self.assertNotIn("PYTHONPATH",kw["env"]);return SimpleNamespace(returncode=0,stderr="")
            with mock.patch.object(package,"ROOT",public),mock.patch.object(package,"PUBLIC_DEPENDENCIES",("tools/kit_manifest.sha256","required.py")), \
                 mock.patch.object(package,"authenticate_kit") as auth,mock.patch.object(package.subprocess,"run",side_effect=run):
                package.run_audit(kit,out,"onset")
            self.assertEqual(auth.call_count,2)

    def test_package_is_deterministic_four_entry_and_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);kit=root/"kit";kit.mkdir();missing=root/"no-selection.json"
            with mock.patch.object(package,"DISCLOSURE",missing):
                for variant in ("onset","free"):
                    a,b=root/(variant+"-a.zip"),root/(variant+"-b.zip")
                    package.create_package(kit,a,variant,audit_runner=self.audit)
                    package.create_package(kit,b,variant,audit_runner=self.audit)
                    self.assertEqual(a.read_bytes(),b.read_bytes())
                    with ZipFile(a) as archive:
                        self.assertEqual(archive.namelist(),list(package.NAMES));receipt=json.loads(archive.read("receipt.json"))
                        self.assertEqual(receipt["selection"]["status"],"evaluation_pending")
                        self.assertFalse(receipt["default_submission_promoted"]);self.assertEqual(receipt["variant"],variant)

    def test_disclosure_rejects_stale_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"selection.json";f.write_text(json.dumps({"schema":"r23-selection-and-independent-result.v1",
                "default_promoted":False,"candidate_entrypoint":"experiments.r23_detached.candidate:Detector",
                "candidate_sources":{},"status":"FAIL","selected_for_independent_evaluation":True}))
            with mock.patch.object(package,"DISCLOSURE",f),self.assertRaises(ValueError):package.disclosure("free")

    def test_selected_result_cannot_leak_into_other_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"selection.json";result={"verdict":"FAIL","episodes":12,"rows":4,"failed_rows":1,
                "strict_gain_vs_r22":2,"strict_loss_vs_r22":0,"remaining_false":1,"remaining_misses":1}
            f.write_text(json.dumps({"schema":"r23-selection-and-independent-result.v1","default_promoted":False,
                "candidate_entrypoint":"experiments.r23_detached.candidate:Free","candidate_sources":build.PINS,
                "status":"independent_fail","selected_for_independent_evaluation":True,"independent_result":result}))
            with mock.patch.object(package,"DISCLOSURE",f):
                selected=package.disclosure("free");other=package.disclosure("onset")
            self.assertEqual(selected["independent_result"],result)
            self.assertIsNone(other["independent_result"])
            self.assertEqual(other["status"],"not_selected_or_independently_evaluated")

    def test_output_and_readme_boundaries(self):
        text=package.readme("onset",{"status":"evaluation_pending"}).decode()
        for phrase in ("not the original course demo","not the default submission","must not be selected","do not justify promotion"):
            self.assertIn(phrase,text)
        result={"verdict":"FAIL","episodes":12,"rows":4,"failed_rows":1,
                "strict_gain_vs_r22":2,"strict_loss_vs_r22":0,"remaining_false":1,"remaining_misses":1}
        disclosed=package.readme("free",{"status":"independent_fail","independent_result":result}).decode()
        self.assertIn("1/4 rows failed",disclosed);self.assertIn("gains/losses versus R22 were 2/0",disclosed)
        with tempfile.TemporaryDirectory() as tmp:
            kit=Path(tmp)/"kit";kit.mkdir()
            with self.assertRaises(ValueError):package.create_package(kit,kit/"x.zip",audit_runner=self.audit)


if __name__=="__main__":unittest.main()
