"""Strict R7 submission-contract output tests."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools import r7_contract as C


class R7ContractTests(unittest.TestCase):
    def test_contract_writer_uses_strict_numpy_scalar_conversion(self):
        value = {"bool": np.bool_(True), "int": np.int64(8),
                 "float64": np.float64(1.25), "float32": np.float32(2.5)}
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "contract.json"
            C.write_result(path, value)
            text = path.read_text(encoding="utf-8")
        self.assertIn('"bool": true', text)
        self.assertIn('"int": 8', text)
        self.assertIn('"float64": 1.25', text)
        self.assertIn('"float32": 2.5', text)

    def test_contract_writer_rejects_nonfinite_and_unsupported_values(self):
        for value in (np.float64(np.inf), np.asarray([1]), object()):
            with self.subTest(value=type(value).__name__), tempfile.TemporaryDirectory() as name:
                path = Path(name) / "contract.json"
                with self.assertRaises((TypeError, ValueError)):
                    C.write_result(path, {"value": value})
                self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
