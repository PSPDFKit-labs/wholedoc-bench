import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from wholedoc_bench.corpus.loader import (
    LoadedCorpus,
    load_prediction,
    prediction_path,
    save_prediction,
)
from wholedoc_bench.prices import measure_usage
from wholedoc_bench.provenance import (
    Attempt,
    CorpusIdentity,
    ManifestRun,
    ScorerIdentity,
    comparison_identity,
    file_hash,
    save_run,
    scorer_identity,
)
from wholedoc_bench.providers import PROVIDERS
from wholedoc_bench.providers.base import load_raw, raw_path, save_raw
from wholedoc_bench.report import count_failures, summarize, write_summary
from wholedoc_bench.scorer.run import (
    SPEC_VERSION,
    CorpusResult,
    ParseFailure,
    score_corpus,
    write_scores_jsonl,
)
from wholedoc_bench.scorer.tracks import ALL_TRACKS

RUN_MANIFEST = Path(".parse-eval/run.json")


@contextmanager
def _provider_path(source: Path, document_id: str, sha256: str) -> Iterator[Path]:
    if source.parent.name == document_id:
        yield source
        return
    with tempfile.TemporaryDirectory() as folder:
        staged = Path(folder) / document_id / f"document{source.suffix}"
        staged.parent.mkdir()
        shutil.copyfile(source, staged)
        if file_hash(staged) != sha256:
            raise ValueError("source changed after corpus validation")
        yield staged


def run_manifest(
    provider_name: str,
    corpus: LoadedCorpus,
    out: Path,
    *,
    from_raw: bool,
    config: dict | None,
    previous: dict[str, Any] | None,
) -> int:
    if previous is None and out.exists() and any(out.iterdir()):
        raise ValueError("populated --out folder has no trusted run manifest; use a fresh folder")
    prior = ManifestRun.model_validate(previous) if previous is not None else None
    if prior is not None and prior.origin == "import":
        raise ValueError(
            "this folder holds imported predictions; they have no raw responses to replay, "
            "and a run must not mix with them; use a fresh --out folder"
        )
    if from_raw and prior is None:
        raise ValueError("manifest replay requires a saved run with source and raw hashes")
    if from_raw and config is None and prior is not None:
        config = prior.config
    provider = PROVIDERS[provider_name](config=config)
    if prior is not None:
        if prior.provider != provider.name or prior.config != provider.config:
            raise ValueError(
                "provider/config differs from the run manifest; use a separate --out folder"
            )
        if prior.sources != corpus.sources:
            raise ValueError(
                "source documents differ from the run manifest; use a fresh --out folder"
            )
    if provider.sends_documents and not from_raw:
        forbidden = [d.truth.document.id for d in corpus.documents if not d.cloud_allowed]
        if forbidden:
            raise ValueError("cloud parsing forbidden for: " + ", ".join(forbidden))

    if prior is not None:
        for doc in corpus.documents:
            id_ = doc.truth.document.id
            attempt = prior.attempts[id_]
            if from_raw and attempt.status == "pending" and attempt.raw_sha256 is None:
                raise ValueError(
                    f"{id_}: pending attempt has no saved raw response; replay refused"
                )
            if attempt.raw_sha256 is not None:
                saved = raw_path(out, id_)
                if file_hash(saved) != attempt.raw_sha256:
                    raise ValueError(f"{id_}: raw response hash mismatch; use a fresh --out folder")
                raw = load_raw(saved)
                if (
                    raw.document_id != id_
                    or raw.provider != provider.name
                    or raw.metadata.get("config") != provider.config
                    or raw.metadata.get("document_sha256") != corpus.sources[id_]
                ):
                    raise ValueError(f"{id_}: raw identity mismatch; use a fresh --out folder")
    if not from_raw:
        provider.prepare()
    run = ManifestRun(
        provider=provider.name,
        config=provider.config,
        implementation=ScorerIdentity.model_validate(scorer_identity()),
        corpus=CorpusIdentity.model_validate(corpus.identity),
        sources=corpus.sources,
        attempts={
            id_: Attempt(
                raw_sha256=prior.attempts[id_].raw_sha256 if prior else None,
                version=prior.attempts[id_].version if prior else None,
            )
            for id_ in corpus.sources
        },
    )
    path = out / RUN_MANIFEST
    save_run(run, path)
    (out / "VERSION.txt").unlink(missing_ok=True)
    for doc in corpus.documents:
        id_ = doc.truth.document.id
        prediction_file = prediction_path(out, id_)
        saved = raw_path(out, id_)
        prediction_file.unlink(missing_ok=True)
        attempt = run.attempts[id_]
        stage: Literal["input", "fetch", "normalize"] = "input"
        try:
            assert doc.source is not None
            if file_hash(doc.source) != corpus.sources[id_]:
                raise ValueError("source changed after corpus validation")
            if from_raw:
                assert prior is not None
                if prior.attempts[id_].raw_sha256 is None:
                    raise ValueError(prior.attempts[id_].error or "no saved raw response")
                raw = load_raw(saved)
                response: Literal["new", "replayed"] = "replayed"
            else:
                if doc.source.suffix not in provider.formats:
                    raise ValueError(f"{provider.name} does not read {doc.source.suffix}")
                with _provider_path(doc.source, id_, corpus.sources[id_]) as given:
                    stage = "fetch"
                    raw = provider.fetch(given)
                stage = "normalize"
                response = "new"
                if raw.provider != provider.name or raw.document_id != id_:
                    raise ValueError("raw provider/document id does not match the request")
                raw = replace(
                    raw,
                    metadata={
                        **raw.metadata,
                        "config": provider.config,
                        "document_sha256": corpus.sources[id_],
                    },
                )
                temporary = saved.with_suffix(".tmp")
                save_raw(raw, temporary)
                temporary.replace(saved)
            stage = "normalize"
            attempt.raw_sha256 = file_hash(saved)
            attempt.response = response
            version = raw.metadata.get("version")
            attempt.version = version if isinstance(version, str) else None
            save_run(run, path)
            prediction = provider.normalize(raw)
            save_prediction(prediction, prediction_file)
            attempt.prediction_sha256 = file_hash(prediction_file)
            attempt.status = "complete"
            print(f"{id_}: ok")
        except Exception as error:
            attempt.status = "failed"
            attempt.stage = stage
            attempt.error = f"{type(error).__name__}: {error}"
            if attempt.raw_sha256 is not None and attempt.response is None:
                attempt.response = "retained"
            print(f"{id_}: failed at {stage}: {attempt.error}")
        run.attempts[id_] = attempt
        save_run(run, path)
    versions = sorted({a.version for a in run.attempts.values() if a.version})
    if versions:
        (out / "VERSION.txt").write_text(", ".join(versions) + "\n", encoding="utf-8")
    return 0


def score_manifest(
    corpus: LoadedCorpus,
    predictions_dir: Path,
    provenance: dict | None,
    results_dir: Path | None,
) -> CorpusResult:
    if provenance is None or "schema_version" not in provenance:
        raise ValueError("this folder has no manifest run evidence; run it with this corpus first")
    run = ManifestRun.model_validate(provenance)
    items = []
    evidence = {}
    saved: list[tuple[Path, int]] = []
    for doc in corpus.documents:
        id_ = doc.truth.document.id
        if run.sources.get(id_) != doc.identity["document_sha256"]:
            raise ValueError(f"{id_}: prediction source missing or changed; run this corpus first")
        attempt = run.attempts[id_]
        if attempt.status == "pending":
            raise ValueError(f"{id_}: attempt is pending; evaluation is incomplete")
        path = prediction_path(predictions_dir, id_)
        if attempt.status == "complete":
            if not path.is_file() or file_hash(path) != attempt.prediction_sha256:
                raise ValueError(f"{id_}: prediction missing or hash mismatch")
            prediction = load_prediction(path)
        else:
            if path.exists():
                raise ValueError(f"{id_}: failed attempt has an unexpected prediction")
            prediction = ParseFailure(attempt.error or "parse failed")
        if attempt.raw_sha256 is not None:
            raw_file = raw_path(predictions_dir, id_)
            if not raw_file.is_file() or file_hash(raw_file) != attempt.raw_sha256:
                raise ValueError(f"{id_}: raw response missing or hash mismatch")
            saved.append((raw_file, len(doc.truth.document.pages)))
        items.append((doc.truth, prediction))
        evidence[id_] = attempt.model_dump()
    result = score_corpus(items, supported_metrics=corpus.metric_support)
    if results_dir is not None:
        name = predictions_dir.resolve().name
        scores_path, summary_path = results_dir / f"{name}.jsonl", results_dir / f"{name}.json"
        if scores_path.exists() or summary_path.exists():
            raise ValueError("evaluation output already exists; use a new --results folder")
        usage = None
        if run.origin == "run":
            raws = ((load_raw(path), pages) for path, pages in saved)
            usage = measure_usage(run.provider, run.config, raws)
        assert corpus.identity is not None
        comparison = comparison_identity(
            corpus.identity,
            SPEC_VERSION,
            [f"{t.id}.{m.name}" for t in ALL_TRACKS for m, _ in t.parts if m.implemented],
        )
        versions = sorted({a["version"] for a in evidence.values() if a["version"]})
        evaluation = {
            "id": str(uuid4()),
            "attempts": evidence,
            "normalization": run.implementation.model_dump(),
            "collection": run.corpus.model_dump(),
        }
        if run.submission is not None:
            evaluation["origin"] = run.origin
            evaluation["submission"] = run.submission.model_dump()
        write_scores_jsonl(result, scores_path, evaluation_id=evaluation["id"])
        evaluation["scores_sha256"] = file_hash(scores_path)
        summary = summarize(
            name,
            result,
            ", ".join(versions) or "unknown",
            provider=run.provider,
            provider_config=run.config,
            comparison=comparison,
            evaluation=evaluation,
            failure_stages=count_failures(evidence.values()),
            origin=run.origin,
            usage=usage,
        )
        write_summary(summary, summary_path)
    return result
