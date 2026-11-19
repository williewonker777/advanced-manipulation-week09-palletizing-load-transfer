from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.r34_archive_replay import runner
from tools import r34_placement_replay as tool


class ReplayContractTests(unittest.TestCase):
    def test_scope_counts_and_only_placement_collisions(self):
        expected = {"smoke": (2, 4), "placement9": (9, 18), "full-place": (60, 120)}
        for scope, (collisions, inputs) in expected.items():
            cases = runner.selected_cases(scope)
            selected = [case for case in cases if case["kind"] == "collision"]
            self.assertEqual((len(selected), len(cases)), (collisions, inputs))
            self.assertTrue(all(case["phase"] == "place" for case in selected))

    def make_registry(self, root: Path, relative: str = "experiments/r34_public/candidate.py"):
        source = root / relative
        source.parent.mkdir(parents=True)
        source.write_text("class Detector:\n    pass\n")
        pin = hashlib.sha256(source.read_bytes()).hexdigest()
        registry = root / "registry.json"
        registry.write_text(json.dumps({
            "schema": "r34-placement-opt-in-registry.v1",
            "public_sources_only": True,
            "kit_included": False,
            "candidate": {"entry": relative[:-3].replace("/", ".") + ":Detector", "sha256": pin},
            "files": {relative: pin},
        }))
        return registry, relative

    def test_optional_registry_is_explicit_and_hash_authenticated(self):
        self.assertIsNone(runner.optional_registration(None))
        with TemporaryDirectory() as folder:
            root = Path(folder)
            registry, relative = self.make_registry(root)
            obj = runner.optional_registration(registry, root)
            self.assertEqual(obj["candidate"]["entry"], "experiments.r34_public.candidate:Detector")
            (root / relative).write_text("class Detector:\n    changed = True\n")
            with self.assertRaisesRegex(ValueError, "source differs"):
                runner.optional_registration(registry, root)

    def test_optional_registry_rejects_path_escape_and_private_names(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            registry, _ = self.make_registry(root)
            obj = json.loads(registry.read_text())
            for bad in ("../candidate.py", "experiments/r34_private/candidate.py"):
                candidate = root / "candidate.py"
                candidate.write_text("class Detector:\n    pass\n")
                pin = hashlib.sha256(candidate.read_bytes()).hexdigest()
                obj["candidate"] = {"entry": "experiments.r34_public.candidate:Detector", "sha256": pin}
                obj["files"] = {bad: pin, "experiments/r34_public/candidate.py": pin}
                registry.write_text(json.dumps(obj))
                with self.assertRaises(ValueError):
                    runner.optional_registration(registry, root)

    def test_optional_registry_rejects_symlinked_source_tree(self):
        with TemporaryDirectory() as folder, TemporaryDirectory() as outside:
            root = Path(folder)
            external = Path(outside) / "r34_linked"
            external.mkdir()
            source = external / "candidate.py"
            source.write_text("class Detector:\n    pass\n")
            (root / "experiments").mkdir()
            (root / "experiments" / "r34_linked").symlink_to(external, target_is_directory=True)
            pin = hashlib.sha256(source.read_bytes()).hexdigest()
            registry = root / "registry.json"
            registry.write_text(json.dumps({
                "schema": "r34-placement-opt-in-registry.v1",
                "public_sources_only": True, "kit_included": False,
                "candidate": {"entry": "experiments.r34_linked.candidate:Detector", "sha256": pin},
                "files": {"experiments/r34_linked/candidate.py": pin},
            }))
            with self.assertRaisesRegex(ValueError, "source differs"):
                runner.optional_registration(registry, root)

    def test_tool_copies_only_authenticated_optional_blobs(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            registry, relative = self.make_registry(root)
            obj, blobs = tool.load_optional(registry, root)
            self.assertEqual(obj["candidate"]["sha256"], hashlib.sha256(blobs[relative]).hexdigest())
            self.assertIn(tool.OPTIONAL_NAME, blobs)


if __name__ == "__main__":
    unittest.main()
