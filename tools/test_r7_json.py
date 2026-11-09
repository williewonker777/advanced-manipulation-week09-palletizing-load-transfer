"""Regression tests for strict JSON record writing."""
import ast
from pathlib import Path
import inspect
import json
import math
import os
import tempfile
import textwrap
import unittest
from unittest import mock

import numpy as np

from tools import r7_json


def _legacy_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _function_body(source, name):
    module = ast.parse(source)
    function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.dump(ast.Module(body=function.body, type_ignores=[]), include_attributes=False)


class StrictJsonTests(unittest.TestCase):
    def test_legacy_writer_reproduces_numpy_scalar_failure(self):
        value = {
            "flag": np.bool_(True),
            "nested": [np.int64(-7), (np.float64(2.5), np.float32(0.125))],
        }
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "record.json"
            with self.assertRaises(TypeError):
                _legacy_write(path, value)

    def test_legacy_fixture_matches_source_when_available(self):
        source_path = Path(__file__).with_name("r6_study.py")
        fixture_body = _function_body(
            textwrap.dedent(inspect.getsource(_legacy_write)), "_legacy_write"
        )
        if source_path.is_file():
            source_body = _function_body(
                source_path.read_text(encoding="utf-8"), "write"
            )
            self.assertEqual(source_body, fixture_body)
        else:
            self.assertFalse(source_path.exists())

    def test_normalize_preserves_values_and_container_semantics(self):
        negative_zero = r7_json.normalize(-0.0)
        single = np.float32(0.1)
        value = {
            "none": None,
            "flag": np.bool_(True),
            "signed": np.int64(np.iinfo(np.int64).min),
            "unsigned": np.uint64(np.iinfo(np.uint64).max),
            "single": single,
            "double": np.float64(2.5),
            "nested": [1, (False, "text")],
        }
        normalized = r7_json.normalize(value)
        self.assertEqual(value["signed"].item(), normalized["signed"])
        self.assertEqual(value["unsigned"].item(), normalized["unsigned"])
        self.assertEqual(float(single), normalized["single"])
        self.assertIs(type(normalized["single"]), float)
        self.assertIs(type(normalized["flag"]), bool)
        self.assertIs(type(normalized["nested"]), list)
        self.assertIs(type(normalized["nested"][1]), tuple)
        self.assertEqual(-1.0, math.copysign(1.0, negative_zero))

    def test_mixed_nested_scalars_round_trip(self):
        source = {
            "flag": np.bool_(False),
            "items": [np.int64(-3), (np.float32(0.125), np.float64(4.5))],
        }
        decoded = json.loads(r7_json.dumps(source))
        self.assertEqual(
            {"flag": False, "items": [-3, [0.125, 4.5]]}, decoded
        )

    def test_round_trip_builtin_json_types(self):
        source = {
            "none": None,
            "bool": True,
            "int": -9,
            "float": -0.0,
            "str": "한글",
            "list": [1, 2],
            "tuple": (3, 4),
            "dict": {"nested": "value"},
        }
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "record.json"
            r7_json.write_json(path, source)
            decoded = r7_json.read_json(path)
        self.assertEqual(source | {"tuple": [3, 4]}, decoded)
        self.assertEqual(-1.0, math.copysign(1.0, decoded["float"]))

    def test_rejects_unsupported_values(self):
        class Custom:
            pass

        values = (
            np.asarray(1),
            np.asarray([1]),
            {1, 2},
            b"bytes",
            1 + 2j,
            np.complex64(1 + 2j),
            np.str_("text"),
            Path("record.json"),
            Custom(),
        )
        for value in values:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(TypeError):
                    r7_json.normalize(value)

    def test_rejects_non_string_object_keys(self):
        for key in (1, np.int64(1), ("tuple",)):
            with self.subTest(key=key):
                with self.assertRaises(TypeError):
                    r7_json.normalize({key: "value"})

    def test_rejects_python_and_numpy_non_finite_values(self):
        values = (
            float("nan"),
            float("inf"),
            float("-inf"),
            np.float32("nan"),
            np.float64("inf"),
            np.longdouble("-inf"),
        )
        for value in values:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    r7_json.dumps({"value": value})

    def test_rejects_circular_containers(self):
        value = []
        value.append(value)
        with self.assertRaises(TypeError):
            r7_json.normalize(value)

    def test_invalid_serialization_touches_no_destination_or_temporary(self):
        with tempfile.TemporaryDirectory() as name:
            parent = Path(name) / "new"
            path = parent / "record.json"
            with self.assertRaises(TypeError):
                r7_json.write_json(path, {"array": np.asarray([1])})
            self.assertFalse(parent.exists())
            self.assertFalse(path.exists())

    def test_exclusive_create_never_overwrites_existing_record(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "record.json"
            path.write_text('{"original": true}\n', encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                r7_json.write_json(path, {"replacement": True})
            self.assertEqual(before, path.read_bytes())
            self.assertEqual([], list(path.parent.glob(f".{path.name}.*.tmp")))

    def test_link_failure_leaves_no_final_or_owned_temporary(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "record.json"
            with mock.patch.object(r7_json.os, "link", side_effect=OSError("link failed")):
                with self.assertRaisesRegex(OSError, "link failed"):
                    r7_json.write_json(path, {"value": np.int64(7)})
            self.assertFalse(path.exists())
            self.assertEqual([], list(path.parent.glob(f".{path.name}.*.tmp")))

    def test_write_failure_leaves_no_final_or_owned_temporary(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "record.json"
            with mock.patch.object(r7_json.os, "fsync", side_effect=OSError("sync failed")):
                with self.assertRaisesRegex(OSError, "sync failed"):
                    r7_json.write_json(path, {"value": np.float32(0.5)})
            self.assertFalse(path.exists())
            self.assertEqual([], list(path.parent.glob(f".{path.name}.*.tmp")))

    def test_replace_is_explicit_and_uses_strict_conversion(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "record.json"
            path.write_text('{"value": 1}\n', encoding="utf-8")
            r7_json.replace_json(path, {"value": np.int64(2)})
            self.assertEqual({"value": 2}, r7_json.read_json(path))
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                r7_json.replace_json(path, {"value": np.float64("nan")})
            self.assertEqual(before, path.read_bytes())

    def test_written_record_is_utf8_and_newline_terminated(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "record.json"
            r7_json.write_json(path, {"label": "한글"})
            payload = path.read_bytes()
        self.assertTrue(payload.endswith(b"\n"))
        self.assertIn("한글", payload.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
