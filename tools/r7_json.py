"""Strict JSON normalization and crash-safe record publication."""
from __future__ import annotations

from pathlib import Path
import json
import math
import os
import tempfile
from typing import Any

import numpy as np


def normalize(value: Any) -> Any:
    """Return a JSON-compatible copy after strict scalar validation.

    Only native JSON primitives and native ``dict``/``list``/``tuple``
    containers are accepted. NumPy boolean, integer, and floating scalars are
    converted to their corresponding built-in types.
    """
    return _normalize(value, "$", set())


def _normalize(value: Any, location: str, active: set[int]) -> Any:
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return value
    if value_type is float:
        return _finite_float(value, location)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _finite_float(float(value), location)

    if value_type not in (dict, list, tuple):
        raise TypeError(f"unsupported JSON value at {location}: {value_type.__name__}")

    identity = id(value)
    if identity in active:
        raise TypeError(f"circular JSON container at {location}")
    active.add(identity)
    try:
        if value_type is dict:
            result = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(
                        f"JSON object key at {location} must be str, "
                        f"not {type(key).__name__}"
                    )
                result[key] = _normalize(item, f"{location}.{key}", active)
            return result
        converted = (
            _normalize(item, f"{location}[{index}]", active)
            for index, item in enumerate(value)
        )
        return list(converted) if value_type is list else tuple(converted)
    finally:
        active.remove(identity)


def _finite_float(value: float, location: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number at {location}")
    return value


def dumps(
    value: Any,
    *,
    ensure_ascii: bool = False,
    indent: int | None = 2,
    sort_keys: bool = False,
) -> str:
    """Serialize through :func:`normalize`, always rejecting non-finite data."""
    return json.dumps(
        normalize(value),
        ensure_ascii=ensure_ascii,
        indent=indent,
        sort_keys=sort_keys,
        allow_nan=False,
    )


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Atomically create *path* without replacing an existing record."""
    _publish(path, value, replace=False)


def replace_json(path: str | os.PathLike[str], value: Any) -> None:
    """Atomically replace *path* through the same strict conversion path."""
    _publish(path, value, replace=True)


def _publish(path: str | os.PathLike[str], value: Any, *, replace: bool) -> None:
    payload = (dumps(value) + "\n").encode("utf-8")
    destination = Path(path)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, destination)
            published = True
        else:
            os.link(temporary, destination)
            published = True
            temporary.unlink()
        _fsync_directory(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
            if published:
                _fsync_directory(parent)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path: str | os.PathLike[str]) -> Any:
    """Read a UTF-8 JSON record."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)
