import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools import r4_report_links as L


ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / "results/v2_protocol/R4_REPORT_LINK_REPAIRS.json"


class PublicLinksTest(unittest.TestCase):
    def test_nested_images_reference_percent_fragment_and_external(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "media").mkdir()
            (root / "media/a b.gif").write_bytes(b"gif")
            (root / "media/a.mp4").write_bytes(b"mp4")
            (root / "docs/target.md").write_text("# Target\n")
            (root / "README.md").write_text(
                "[![preview](media/a%20b.gif)](media/a.mp4#time)\n"
                "[reference][ref]\n[ref]: docs/target.md#section\n"
                "[outside](https://example.invalid/not-fetched)\n")
            value = L.check_public_links(root)
            self.assertTrue(value["passed"])
            self.assertEqual(value["relative_links"], 3)
            self.assertEqual(value["external_links_skipped"], 1)

    def test_missing_root_escape_and_code_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "README.md").write_text(
                "```md\n[example](missing-in-code.md)\n```\n"
                "[missing](docs/missing.md)\n[escape](../outside.md)\n")
            value = L.check_public_links(root)
            self.assertEqual([item["reason"] for item in value["missing"]],
                             ["missing", "root_escape"])
            self.assertEqual(value["relative_links"], 2)

    def test_html_links_and_explicit_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "asset.png").write_bytes(b"png")
            (root / "docs/page.html").write_text(
                '<img src="../asset.png"><a href="#heading">same</a>'
                '<pre><a href="missing-code.html">sample</a></pre>')
            value = L.check_public_links(root, ["docs/page.html"])
            self.assertTrue(value["passed"])
            self.assertEqual(value["relative_links"], 1)
            self.assertEqual(value["anchor_only_links_skipped"], 1)

    def test_default_scope_includes_results_and_tools_not_only_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("results", "tools", "publish", ".venv"):
                (root / folder).mkdir()
            (root / "results/evidence.md").write_text("[missing](absent.json)\n")
            (root / "tools/guide.md").write_text("[missing](absent.py)\n")
            for folder in ("publish", ".venv"):
                (root / folder / "private.md").write_text("[private](absent)\n")
            value = L.check_public_links(root)
            self.assertEqual(value["document_count"], 2)
            self.assertEqual(len(value["missing"]), 2)

    def test_all_hidden_directories_are_excluded_by_general_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("# Public document\n")
            for name in (".hidden-state", ".training-env"):
                (root / name).mkdir()
                (root / name / "private.md").write_text("[not public](absent.md)\n")
            self.assertEqual(L.check_public_links(root)["documents"], ["README.md"])
            self.assertEqual(L.check_public_links(root, [".hidden-state/private.md"])["documents"], [])


class ArchiveProofTest(unittest.TestCase):
    def setUp(self):
        self.proof = json.loads(PROOF.read_text())

    def _copy_archives(self, destination):
        for relative in L.ARCHIVE_PATHS:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())

    def _originals_from_proof(self):
        originals = {}
        for item in self.proof["documents"]:
            text = (ROOT / item["path"]).read_text()
            spans = L._destination_spans(text)
            for change in reversed(item["changes"]):
                start, end, value = spans[change["ordinal"] - 1]
                self.assertEqual(value, change["new"])
                text = text[:start] + change["old"] + text[end:]
            originals[item["path"]] = text.encode()
        return originals

    def test_actual_proof_and_rebuild(self):
        self.assertTrue(L.verify_archive_link_proof(ROOT, self.proof)["passed"])
        rebuilt = L.build_archive_link_proof(ROOT, self._originals_from_proof())
        self.assertEqual(rebuilt, self.proof)

    def test_body_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._copy_archives(root)
            path = root / L.ARCHIVE_PATHS[0]
            path.write_text(path.read_text().replace("## 연구 범위", "## 변조된 범위"))
            altered = copy.deepcopy(self.proof)
            altered["documents"][0]["current_sha256"] = L._sha(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "outside approved URLs"):
                L.verify_archive_link_proof(root, altered)

    def test_url_removal_and_addition_are_rejected(self):
        for mutation in (lambda text: text.replace("](V2_METHODS.md)", "]", 1),
                         lambda text: text + "\n[extra](V2_METHODS.md)\n"):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    self._copy_archives(root)
                    path = root / L.ARCHIVE_PATHS[0]
                    path.write_text(mutation(path.read_text()))
                    altered = copy.deepcopy(self.proof)
                    altered["documents"][0]["current_sha256"] = L._sha(path.read_bytes())
                    with self.assertRaises(ValueError):
                        L.verify_archive_link_proof(root, altered)

    def test_existing_unrelated_target_cannot_be_resealed_as_path_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._copy_archives(root)
            path = root / L.ARCHIVE_PATHS[0]
            path.write_text(path.read_text().replace("](V2_METHODS.md)", "](R4_METHODS.md)", 1))
            (root / "docs/R4_METHODS.md").write_text("# Unrelated existing file\n")
            altered = copy.deepcopy(self.proof)
            altered["documents"][0]["changes"][0]["new"] = "R4_METHODS.md"
            altered["documents"][0]["current_sha256"] = L._sha(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "same repository target"):
                L.verify_archive_link_proof(root, altered)

    def test_query_and_fragment_cannot_change_in_path_only_repair(self):
        for suffix in ("?download=1", "#other-heading"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._copy_archives(root)
                path = root / L.ARCHIVE_PATHS[0]
                target = "V2_METHODS.md" + suffix
                path.write_text(path.read_text().replace("](V2_METHODS.md)", "](" + target + ")", 1))
                altered = copy.deepcopy(self.proof)
                altered["documents"][0]["changes"][0]["new"] = target
                altered["documents"][0]["current_sha256"] = L._sha(path.read_bytes())
                with self.assertRaisesRegex(ValueError, "query/fragment"):
                    L.verify_archive_link_proof(root, altered)


if __name__ == "__main__":
    unittest.main()
