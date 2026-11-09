import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from wholedoc_bench.corpus.fields import DocumentId, Sha256


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ScorerIdentity(Evidence):
    commit: str | None
    dirty: bool | None
    source_sha256: Sha256
    python: str


class Attempt(Evidence):
    status: Literal["pending", "failed", "complete"] = "pending"
    raw_sha256: Sha256 | None = None
    prediction_sha256: Sha256 | None = None
    error: str | None = None
    version: str | None = None
    stage: Literal["input", "fetch", "normalize", "not_submitted"] | None = None
    response: Literal["new", "replayed", "retained"] | None = None

    @model_validator(mode="after")
    def consistent(self) -> "Attempt":
        if self.status == "complete" and (self.prediction_sha256 is None or self.error is not None):
            raise ValueError("complete attempt requires a prediction hash and no error")
        if self.status != "complete" and self.prediction_sha256 is not None:
            raise ValueError("only a complete attempt can have a prediction hash")
        if self.status == "failed" and not self.error:
            raise ValueError("failed attempt requires a reason")
        if self.stage is not None and self.status != "failed":
            raise ValueError("only a failed attempt has a failure stage")
        if self.response is not None and self.raw_sha256 is None:
            raise ValueError("a response origin requires a saved raw response")
        return self


_CREDENTIAL_WORDS = frozenset(
    {"key", "apikey", "token", "secret", "password", "passwd", "credential", "credentials"}
)


def _credential_keys(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{path}.{key}" if path else str(key)
            spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
            if _CREDENTIAL_WORDS & set(re.split(r"[^a-z0-9]+", spaced.lower())):
                found.append(here)
            found.extend(_credential_keys(item, here))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found.extend(_credential_keys(item, f"{path}[{i}]"))
    return found


class Submission(Evidence):
    parser: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=200)
    settings: dict[str, Any]
    ran_at: str = Field(min_length=1, max_length=200)
    ran_where: str = Field(min_length=1, max_length=2000)
    training_statement: str = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def no_credentials(self) -> "Submission":
        if found := _credential_keys(self.settings):
            raise ValueError(
                "settings must not hold credentials; remove: " + ", ".join(sorted(found))
            )
        return self


class ManifestRun(Evidence):
    schema_version: Literal[1] = 1
    origin: Literal["run", "import"] = "run"
    provider: str
    config: dict[str, Any]
    implementation: ScorerIdentity
    corpus: "CorpusIdentity"
    sources: dict[DocumentId, Sha256]
    attempts: dict[DocumentId, Attempt]
    submission: Submission | None = None

    @model_validator(mode="after")
    def matching_ids(self) -> "ManifestRun":
        if not self.sources or set(self.sources) != set(self.attempts):
            raise ValueError("sources and attempts must cover the same nonempty document set")
        if self.sources != {doc.id: doc.document_sha256 for doc in self.corpus.documents}:
            raise ValueError("run corpus scope must match its source documents")
        if (self.origin == "import") != (self.submission is not None):
            raise ValueError("imported predictions, and only those, carry a submission")
        if self.origin == "import" and any(
            a.raw_sha256 is not None or a.status == "pending" for a in self.attempts.values()
        ):
            raise ValueError("imported attempts are complete or failed and have no raw response")
        if self.origin != "import" and any(
            a.stage == "not_submitted" for a in self.attempts.values()
        ):
            raise ValueError("only an import can have a document that was not submitted")
        return self

    @model_serializer(mode="wrap")
    def _omit_run_defaults(self, handler: SerializerFunctionWrapHandler) -> Any:
        data = handler(self)
        if data.get("origin") == "run":
            data.pop("origin", None)
        if data.get("submission") is None:
            data.pop("submission", None)
        return data


def save_run(run: ManifestRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class DocumentIdentity(Evidence):
    id: DocumentId
    status: Literal["accepted", "draft", "in_review"]
    document_sha256: Sha256
    key_sha256: Sha256
    supported_metrics: list[str]


class ReleaseIdentity(Evidence):
    tag: str
    source_commit: str
    release_sha256: Sha256
    sums_sha256: Sha256


class CorpusIdentity(Evidence):
    format_version: str
    name: str
    header_sha256: Sha256
    manifest_sha256: Sha256
    include_drafts: bool
    excluded_ids: list[DocumentId]
    documents: list[DocumentIdentity] = Field(min_length=1)
    release: ReleaseIdentity | None = None

    @model_serializer(mode="wrap")
    def _omit_working_corpus_release(self, handler: SerializerFunctionWrapHandler) -> Any:
        data = handler(self)
        if data.get("release") is None:
            data.pop("release", None)
        return data


class ComparisonIdentity(Evidence):
    schema_version: Literal[1] = 1
    corpus: CorpusIdentity
    scorer: ScorerIdentity
    spec_version: str
    implemented_metrics: list[str]


def scorer_identity() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    root = Path(__file__).resolve().parents[2]
    checkout = (root / "src/wholedoc_bench").resolve() == package
    hashes = {
        p.relative_to(package).as_posix(): file_hash(p) for p in sorted(package.rglob("*.py"))
    }
    if checkout:
        hashes.update(
            {
                p.name: file_hash(p)
                for p in (root / "pyproject.toml", root / "uv.lock")
                if p.is_file()
            }
        )
    source_hash = digest(hashes)

    def git(*args: str) -> str | None:
        if not checkout:
            return None
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git(
        "status", "--porcelain", "--untracked-files=all", "--", "src", "pyproject.toml", "uv.lock"
    )
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "source_sha256": source_hash,
        "python": platform.python_version(),
    }


def comparison_identity(corpus: dict, spec_version: str, implemented_metrics: list[str]) -> dict:
    return ComparisonIdentity(
        corpus=CorpusIdentity.model_validate(corpus),
        scorer=ScorerIdentity.model_validate(scorer_identity()),
        spec_version=spec_version,
        implemented_metrics=sorted(implemented_metrics),
    ).model_dump()


_UNCOMPARED_CORPUS_FIELDS = {"format_version", "name", "header_sha256", "manifest_sha256"}


def comparison_key(value: dict) -> str:
    identity = ComparisonIdentity.model_validate(value)
    return digest(identity.model_dump(exclude={"corpus": _UNCOMPARED_CORPUS_FIELDS}))


def validate_comparison(value: dict, documents: int, spec_version: str) -> None:
    identity = ComparisonIdentity.model_validate(value)
    ids = [doc.id for doc in identity.corpus.documents]
    if len(ids) != len(set(ids)) or len(ids) != documents:
        raise ValueError("comparison identity does not match the document count")
    if identity.spec_version != spec_version:
        raise ValueError("comparison identity has a different spec version")
