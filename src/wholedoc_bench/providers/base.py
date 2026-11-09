import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, ClassVar

from wholedoc_bench.schema import Box, Prediction

CLIP_TOLERANCE = 0.01


@dataclass(frozen=True)
class RawResult:
    provider: str
    document_id: str
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)


def raw_path(predictions_dir: Path, document_id: str) -> Path:
    return predictions_dir / "raw" / f"{document_id}.json"


def save_raw(raw: RawResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(raw), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def load_raw(path: Path) -> RawResult:
    return RawResult(**json.loads(path.read_text(encoding="utf-8")))


class Provider(ABC):
    name: ClassVar[str]
    sends_documents: ClassVar[bool] = True
    formats: ClassVar[frozenset[str]] = frozenset({".pdf"})
    meters: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = self.validate_config({} if config is None else config)

    def prepare(self) -> None:
        return None

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise ValueError("provider config must be a JSON object")
        if config:
            raise ValueError(f"{cls.name}: unsupported config keys: {', '.join(sorted(config))}")
        return {}

    @abstractmethod
    def fetch(self, document_path: Path) -> RawResult: ...

    @abstractmethod
    def normalize(self, raw: RawResult) -> Prediction: ...

    def usage(self, raw: RawResult) -> dict[str, float]:
        return {}


def grid_place(row: Any, column: Any, rows: Any = 1, columns: Any = 1) -> dict[str, int]:
    place = {"row": row, "column": column, "rows": rows, "columns": columns}
    valid = all(type(value) is int for value in place.values())
    if not valid or row < 0 or column < 0 or rows < 1 or columns < 1:
        return {}
    return place


def clip_box(
    x: float, y: float, width: float, height: float, page_width: float, page_height: float
) -> Box | None:
    left, right = x / page_width, (x + width) / page_width
    top, bottom = y / page_height, (y + height) / page_height
    if not all(isfinite(edge) for edge in (left, top, right, bottom)):
        return None
    tol = CLIP_TOLERANCE
    if left < -tol or top < -tol or right > 1 + tol or bottom > 1 + tol:
        return None
    left, top = max(0.0, left), max(0.0, top)
    right, bottom = min(1.0, right), min(1.0, bottom)
    if right <= left or bottom <= top or (right - left) * (bottom - top) == 0:
        return None
    return (left, top, right - left, bottom - top)
