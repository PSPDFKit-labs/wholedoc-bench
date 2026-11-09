import re
from pathlib import Path

from pydantic import ValidationError

from wholedoc_bench.corpus.loader import LoadedCorpus, prediction_path
from wholedoc_bench.execution import RUN_MANIFEST
from wholedoc_bench.provenance import (
    Attempt,
    CorpusIdentity,
    ManifestRun,
    ScorerIdentity,
    Submission,
    file_hash,
    save_run,
    scorer_identity,
)
from wholedoc_bench.schema import Prediction

NOT_SUBMITTED = "no prediction submitted"
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


def read_submission(path: Path) -> Submission:
    try:
        return Submission.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise ValueError(f"{path}: invalid claims: {_first_lines(error)}") from error


def import_predictions(
    corpus: LoadedCorpus,
    predictions: Path,
    out: Path,
    *,
    name: str,
    submission: Submission,
    missing_is_failure: bool = False,
) -> int:
    if not corpus.is_manifest:
        raise ValueError("import needs a public release")
    if not _NAME.fullmatch(name):
        raise ValueError(
            "--name must be 1-64 lower-case letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    if not predictions.is_dir():
        raise ValueError(f"{predictions}: no such predictions folder")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("--out must be a new or empty folder")

    ids = [d.truth.document.id for d in corpus.documents]
    expected = {f"{id_}.json": id_ for id_ in ids}
    problems: list[str] = []
    found: dict[str, Path] = {}
    for path in sorted(predictions.iterdir()):
        if path.is_symlink() or not path.is_file():
            problems.append(f"{path.name}: not a plain file; the folder holds <id>.json only")
        elif path.name in expected:
            found[expected[path.name]] = path
        elif path.name.lower() in {n.lower() for n in expected}:
            problems.append(f"{path.name}: wrong case; document ids are lower case")
        else:
            problems.append(f"{path.name}: not a prediction for a selected document")
    missing = [id_ for id_ in ids if id_ not in found]
    if missing and not missing_is_failure:
        problems.append(
            "no prediction for: "
            + ", ".join(missing)
            + " (pass --missing-is-failure to score them as failed parses)"
        )
    contents: dict[str, bytes] = {}
    for id_, path in found.items():
        data = path.read_bytes()
        try:
            Prediction.model_validate_json(data)
        except ValidationError as error:
            problems.append(f"{path.name}: not a valid prediction: {_first_lines(error)}")
        contents[id_] = data
    if problems:
        raise ValueError("import refused; nothing was written:\n  " + "\n  ".join(problems))

    out.mkdir(parents=True, exist_ok=True)
    attempts: dict[str, Attempt] = {}
    for id_ in ids:
        if id_ not in contents:
            attempts[id_] = Attempt(
                status="failed",
                error=NOT_SUBMITTED,
                version=submission.version,
                stage="not_submitted",
            )
            print(f"{id_}: {NOT_SUBMITTED}; scored as a failed parse")
            continue
        target = prediction_path(out, id_)
        target.write_bytes(contents[id_])
        attempts[id_] = Attempt(
            status="complete", prediction_sha256=file_hash(target), version=submission.version
        )
        print(f"{id_}: imported")
    assert corpus.identity is not None
    run = ManifestRun(
        origin="import",
        provider=name,
        config=submission.settings,
        implementation=ScorerIdentity.model_validate(scorer_identity()),
        corpus=CorpusIdentity.model_validate(corpus.identity),
        sources=corpus.sources,
        attempts=attempts,
        submission=submission,
    )
    (out / "VERSION.txt").write_text(submission.version + "\n", encoding="utf-8")
    save_run(run, out / RUN_MANIFEST)
    return 0


def _first_lines(error: ValidationError, limit: int = 3) -> str:
    details = [
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" if e["loc"] else e["msg"]
        for e in error.errors()[:limit]
    ]
    more = error.error_count() - limit
    return "; ".join(details) + (f"; and {more} more" if more > 0 else "")
