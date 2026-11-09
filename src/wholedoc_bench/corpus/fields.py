import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import AfterValidator, Field


def _unique(values: list[str]) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError("values must be unique")
    return values


Text = Annotated[str, Field(min_length=1, pattern=r"^\S(?:.*\S)?$")]
DocumentId = Annotated[str, Field(pattern=r"^np-[0-9]{5}$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
MetricId = Annotated[str, Field(pattern=r"^T[1-9][0-9]*\.[a-z][a-z0-9_]*$")]
Metrics = Annotated[
    list[MetricId], Field(json_schema_extra={"uniqueItems": True}), AfterValidator(_unique)
]


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value}")


def strict_json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)


def safe_path(root: Path, relative: str) -> Path:
    parts = relative.split("/")
    if any(p in ("", ".", "..") or ":" in p or "\\" in p for p in parts):
        raise ValueError(f"unsafe corpus path {relative!r}")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink() or path.is_junction():
            raise ValueError(f"linked corpus path {relative!r} is not allowed")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"corpus path escapes its root: {relative!r}")
    return path


def read_safe(root: Path, relative: str) -> bytes:
    return safe_path(root, relative).read_bytes()
