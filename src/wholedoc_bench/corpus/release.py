import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from wholedoc_bench.corpus.checks import check_groundtruth
from wholedoc_bench.corpus.fields import (
    DocumentId,
    Metrics,
    Sha256,
    Text,
    read_safe,
    safe_path,
    strict_json,
)
from wholedoc_bench.schema import SCHEMA_VERSION, Groundtruth
from wholedoc_bench.scorer.base import SubMetric
from wholedoc_bench.scorer.tracks import ALL_TRACKS

DOCUMENT_INDEX = "documents.jsonl"
LEDGER = "published.jsonl"
RELEASE = "release.json"
SUMS = "SHA256SUMS"
CARD = "README.md"
KEYS = "groundtruth"
COLUMNS = (
    "id",
    "document_path",
    "document_sha256",
    "format",
    "pages",
    "language",
    "scan",
    "category",
    "tracks",
    "supported_metrics",
    "limits",
    "groundtruth_source",
    "licence",
    "attribution",
    "groundtruth_path",
    "groundtruth_sha256",
)
TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ID = re.compile(r"^np-[0-9]{5}$")


class ReleaseError(ValueError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def read_ledger(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ReleaseError([f"previous ledger {path} does not exist"])
    ledger: dict[str, str] = {}
    for number, line in enumerate(path.read_text("utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not ID.match(str(row.get("id"))) or not TAG.match(str(row.get("first_release"))):
                raise ValueError("needs an id and a first_release tag")
        except (ValueError, AttributeError) as error:
            raise ReleaseError([f"{path}:{number}: not a ledger line ({error})"]) from None
        if row["id"] in ledger:
            raise ReleaseError([f"{path}:{number}: {row['id']} is listed twice"])
        ledger[row["id"]] = row["first_release"]
    return ledger


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ReleaseInfo(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    format_version: Annotated[str, Field(pattern=r"^1\.[0-9]+$")]
    corpus: Text
    tag: Annotated[str, Field(pattern=TAG.pattern)]
    source_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}([0-9a-f]{24})?$")]
    header_sha256: Sha256
    manifest_sha256: Sha256
    documents: Annotated[int, Field(ge=1)]


class ReleaseRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: DocumentId
    document_path: str
    document_sha256: Sha256
    format: Annotated[str, Field(pattern=r"^[a-z][a-z0-9]*$")]
    pages: Annotated[int, Field(ge=1)]
    language: list[str]
    scan: bool
    category: str | None
    tracks: list[str]
    supported_metrics: Metrics
    limits: str | None
    groundtruth_source: str
    licence: str | None
    attribution: str | None
    groundtruth_path: str
    groundtruth_sha256: Sha256


def release_schema() -> dict[str, dict[str, Any]]:
    return {"release-row.schema.json": ReleaseRow.model_json_schema()}


@dataclass(frozen=True)
class ReleaseDocument:
    row: ReleaseRow
    truth: Groundtruth
    source: Path


@dataclass(frozen=True)
class Release:
    info: ReleaseInfo
    documents: tuple[ReleaseDocument, ...]
    release_sha256: str
    sums_sha256: str


def is_release(folder: Path) -> bool:
    return (folder / RELEASE).is_file() and not (folder / "corpus.json").exists()


def _verified_sums(root: Path, problems: list[str]) -> dict[str, str]:
    sums: dict[str, str] = {}
    try:
        text = read_safe(root, SUMS).decode("utf-8")
    except (OSError, ValueError) as error:
        problems.append(f"{SUMS}: {error}")
        return sums
    for number, line in enumerate(text.split("\n"), 1):
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None or match[2] in sums:
            problems.append(f"{SUMS}:{number}: not a '<sha256>  <path>' line, or a repeated path")
            continue
        try:
            if _digest(read_safe(root, match[2])) != match[1]:
                problems.append(f"{match[2]}: SHA-256 differs from {SUMS}")
        except (OSError, ValueError) as error:
            problems.append(f"{match[2]}: {error}")
        sums[match[2]] = match[1]
    for name in (CARD, DOCUMENT_INDEX, RELEASE, LEDGER):
        if name not in sums:
            problems.append(f"{name}: not listed in {SUMS}")
    for folder in ("documents", KEYS):
        if (root / folder).is_dir():
            for path in (root / folder).rglob("*"):
                relative = path.relative_to(root).as_posix()
                if path.is_file() and relative not in sums:
                    problems.append(f"{relative}: not listed in {SUMS}")
    return sums


def _release_document(
    root: Path, row: ReleaseRow, sums: dict[str, str], metrics: dict[str, SubMetric]
) -> ReleaseDocument:
    if row.document_path != f"documents/{row.id}.{row.format}":
        raise ValueError(f"document_path must be documents/{row.id}.{row.format}")
    if sums.get(row.document_path) != row.document_sha256:
        raise ValueError("the document's SHA-256 differs from the row, or it is not listed")
    if row.groundtruth_path != f"{KEYS}/{row.id}.json":
        raise ValueError(f"groundtruth_path must be {KEYS}/{row.id}.json")
    if sums.get(row.groundtruth_path) != row.groundtruth_sha256:
        raise ValueError("the key's SHA-256 differs from the row, or it is not listed")
    data = strict_json(read_safe(root, row.groundtruth_path).decode("utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"groundtruth schema_version must be {SCHEMA_VERSION!r}")
    truth = Groundtruth.model_validate(data)
    check_groundtruth(truth)
    if truth.document.id != row.id or truth.document.sha256 != row.document_sha256:
        raise ValueError("groundtruth document.id and document.sha256 must match the row")
    if len(truth.document.pages) != row.pages:
        raise ValueError("groundtruth page count differs from pages")
    if truth.document.groundtruth_source != row.groundtruth_source:
        raise ValueError("groundtruth_source differs from the groundtruth")
    if not row.supported_metrics:
        raise ValueError("supported_metrics is empty")
    for name in row.supported_metrics:
        if name not in metrics:
            raise ValueError(f"unknown supported metric {name!r}")
        if not metrics[name].applies(truth):
            raise ValueError(f"supported metric {name} has nothing to score (applies is false)")
    if sorted(row.tracks) != sorted({name.split(".", 1)[0] for name in row.supported_metrics}):
        raise ValueError("tracks must be the tracks of supported_metrics")
    return ReleaseDocument(row, truth, safe_path(root, row.document_path))


def _refuse(problems: list[str]) -> ValueError:
    return ValueError("invalid release:\n  " + "\n  ".join(problems))


def read_release(folder: Path) -> Release:
    root = folder.resolve()
    problems: list[str] = []
    sums = _verified_sums(root, problems)
    if problems:
        raise _refuse(problems)
    try:
        raw_info = read_safe(root, RELEASE)
        info = ReleaseInfo.model_validate(strict_json(raw_info.decode("utf-8")))
    except (OSError, ValueError) as error:
        raise _refuse([f"{RELEASE}: {error}"]) from None
    metrics = {
        f"{track.id}.{metric.name}": metric for track in ALL_TRACKS for metric, _ in track.parts
    }
    documents: list[ReleaseDocument] = []
    seen: set[str] = set()
    for number, line in enumerate(read_safe(root, DOCUMENT_INDEX).decode("utf-8").split("\n"), 1):
        if not line.strip():
            continue
        where = f"{DOCUMENT_INDEX}:{number}"
        try:
            row = ReleaseRow.model_validate(strict_json(line))
        except ValueError as error:
            problems.append(f"{where}: {error}")
            continue
        if row.id in seen:
            problems.append(f"{where}: {row.id} is listed twice")
        seen.add(row.id)
        try:
            documents.append(_release_document(root, row, sums, metrics))
        except ValueError as error:
            problems.append(f"{where} ({row.id}): {error}")
    if not problems and len(documents) != info.documents:
        problems.append(
            f"{RELEASE} says {info.documents} documents; {DOCUMENT_INDEX} has {len(documents)}"
        )
    named = {d.row.document_path for d in documents} | {d.row.groundtruth_path for d in documents}
    rowless = {p for p in sums if p.startswith(("documents/", f"{KEYS}/"))} - named
    problems.extend(
        f"{path}: no row in {DOCUMENT_INDEX} names this file" for path in sorted(rowless)
    )
    try:
        ledger = read_ledger(root / LEDGER)
        unlisted = sorted(d.row.id for d in documents if d.row.id not in ledger)
        if unlisted:
            problems.append(f"{LEDGER} doesn't list released ids: {', '.join(unlisted)}")
    except ReleaseError as error:
        problems.extend(error.problems)
    if problems:
        raise _refuse(problems)
    return Release(info, tuple(documents), _digest(raw_info), _digest(read_safe(root, SUMS)))
