import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import r25_weak_replay as replay


class WeakReplayTests(unittest.TestCase):
    def test_original_tree_is_not_a_replay_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model").mkdir()
            with patch.object(replay, "ROOT", root), self.assertRaisesRegex(ValueError, "kit-free"):
                replay.inputs()

    def test_does_not_accept_substituted_witness(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            name = "experiments/r25_weak/world_horizontal.py"
            (root / name).parent.mkdir(parents=True)
            (root / name).write_bytes(b"not the frozen implementation")
            (root / replay.LOCK).write_text(json.dumps(
                {"weak_source_pins": {name: "0" * 64}, "expected_raw_sha256": {}}))
            with (patch.object(replay, "ROOT", root),
                  patch.object(replay.base, "closure", return_value={}),
                  patch.object(replay.base, "source_blobs", return_value=({}, {})),
                  self.assertRaisesRegex(ValueError, "source changed")):
                replay.inputs()
