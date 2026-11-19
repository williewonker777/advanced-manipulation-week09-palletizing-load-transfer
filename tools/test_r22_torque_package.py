import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
from zipfile import ZipFile

from experiments.r22_export import build
from tools import r22_torque_package as package


class PackageTest(unittest.TestCase):
    def audit_doc(self, variant, receipt):
        return {"schema":"r22-release-audit.v1", "variant":variant,
            "default_submission_promoted":False, "build":receipt,
            "kit":{"manifest_sha256":"kit-pin"}, "evaluator_sha256":"eval-pin",
            "recognized_collision_cells":{"torque":24,"servo":23},
            "official":{controller:{"episodes":29,"control":{"0.0":[]},"collisions":[[]]}
                        for controller in ("torque","servo")},
            "parity":[{"case_id":case_id,"samples":6000,"passed":True}
                      for case_id in package.PARITY_CASES],
            "gates":{name:True for name in ("default_filename","torque_official_24_of_24",
                "torque_controls_R10_exact_except_compute","servo_R10_exact",
                "modular_generated_6000_exact","protected_kit_unchanged")}}

    def audit(self, kit, output, variant):
        with tempfile.TemporaryDirectory() as tmp:
            detector = Path(tmp) / "detector.py"; receipt = build.build(detector, variant)
        output.write_text(json.dumps(self.audit_doc(variant, receipt), sort_keys=True))

    def test_audit_rejects_missing_or_duplicate_parity_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            detector = Path(tmp) / "detector.py"; receipt = build.build(detector, "light")
            detector_sha = receipt["standalone_sha256"]
            missing = self.audit_doc("light", receipt); missing["parity"].pop()
            with self.assertRaises(ValueError):
                package.validate_audit(missing, "light", detector_sha)
            duplicate = self.audit_doc("light", receipt)
            duplicate["parity"][-1] = dict(duplicate["parity"][0])
            with self.assertRaises(ValueError):
                package.validate_audit(duplicate, "light", detector_sha)

    def test_audit_runtime_comes_from_authenticated_kit_without_pythonpath(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);public=root/"public";kit=root/"kit";out=root/"audit.json"
            (public/"tools").mkdir(parents=True);(public/"tools/kit_manifest.sha256").write_text("manifest")
            (public/"required.py").write_text("required")
            (public/"private-sentinel.txt").write_text("must not copy")
            (kit/"evaluate.py").parent.mkdir(parents=True,exist_ok=True);(kit/"evaluate.py").write_text("evaluate")
            for tree in ("sim","model"):
                (kit/tree).mkdir(parents=True);(kit/tree/"marker").write_text(tree)
            before=sorted(path.relative_to(kit) for path in kit.rglob("*"))
            def run(command, **kwargs):
                runtime=Path(kwargs["cwd"])
                self.assertEqual((runtime/"sim/marker").read_text(),"sim")
                self.assertEqual((runtime/"model/marker").read_text(),"model")
                self.assertEqual((runtime/"required.py").read_text(),"required")
                self.assertFalse((runtime/"private-sentinel.txt").exists())
                self.assertEqual(command[1:4],["-I","-B","-c"])
                self.assertNotIn("PYTHONPATH",kwargs["env"])
                return SimpleNamespace(returncode=0,stderr="")
            with mock.patch.object(package,"ROOT",public), \
                    mock.patch.object(package,"PUBLIC_DEPENDENCIES",
                                      ("tools/kit_manifest.sha256","required.py")), \
                    mock.patch.object(package,"authenticate_kit") as authenticate, \
                    mock.patch.object(package.subprocess,"run",side_effect=run), \
                    mock.patch.dict("os.environ",{"PYTHONPATH":"poison"}):
                package.run_audit(kit,out,"light")
            self.assertEqual(authenticate.call_count,2)
            self.assertEqual(before,sorted(path.relative_to(kit) for path in kit.rglob("*")))

    def test_light_package_rejects_stale_independent_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);kit=root/"kit";kit.mkdir();holdout=root/"holdout.json"
            holdout.write_text("stale")
            with mock.patch.object(package,"LIGHT_HOLDOUT",holdout), self.assertRaises(ValueError):
                package.create_package(kit,root/"out.zip","light",audit_runner=self.audit)

    def test_both_variants_are_deterministic_four_entry_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);kit=root/"kit";kit.mkdir()
            for variant in ("core","augmented","guarded","light"):
                a,b=root/(variant+"1.zip"),root/(variant+"2.zip")
                package.create_package(kit,a,variant,audit_runner=self.audit)
                package.create_package(kit,b,variant,audit_runner=self.audit)
                self.assertEqual(a.read_bytes(),b.read_bytes())
                with ZipFile(a) as archive:
                    self.assertEqual(archive.namelist(),list(package.NAMES))
                    receipt=json.loads(archive.read("receipt.json"));self.assertEqual(receipt["variant"],variant)
                    self.assertFalse(receipt["default_submission_promoted"])
                    expected = {"light":package.INDEPENDENT_VERDICT,
                                "augmented":package.AUGMENTED_VERDICT}.get(
                                    variant,package.UNASSERTED_VERDICT)
                    self.assertEqual(receipt["independent_evaluation"], expected)

    def test_existing_symlink_and_raced_output_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);kit=root/"kit";kit.mkdir();out=root/"out.zip";out.write_bytes(b"keep")
            with self.assertRaises(ValueError):package.create_package(kit,out,"core",audit_runner=self.audit)
            self.assertEqual(out.read_bytes(),b"keep");out.unlink();link=root/"link.zip";link.symlink_to(root/"none")
            with self.assertRaises(ValueError):package.create_package(kit,link,"core",audit_runner=self.audit)
            def race(k,p,v):self.audit(k,p,v);out.write_bytes(b"raced")
            with self.assertRaises(FileExistsError):package.create_package(kit,out,"core",audit_runner=race)
            self.assertEqual(out.read_bytes(),b"raced")

    def test_readme_is_explicitly_non_default_and_non_hardware(self):
        for variant in ("core","augmented","guarded","light"):
            text=package.readme(variant).decode();self.assertIn("not the default",text);self.assertIn("not for hardware",text)
            self.assertIn("do not justify promotion", text)
            if variant in ("light","augmented"): self.assertIn("did not generalize", text)
            else: self.assertIn("asserts no locked blind result", text)

    def test_external_output_path_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            kit=Path(tmp)/"kit";kit.mkdir()
            with self.assertRaises(ValueError):package.create_package(kit,kit/"bad.zip","core",audit_runner=self.audit)
            with self.assertRaises(ValueError):package.create_package(kit,build.ROOT/"bad.zip","core",audit_runner=self.audit)


if __name__ == "__main__": unittest.main()
