import argparse
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

from wholedoc_bench.corpus.loader import (
    load_corpus_context,
    load_prediction,
    prediction_path,
    save_prediction,
)
from wholedoc_bench.execution import run_manifest, score_manifest
from wholedoc_bench.prices import BOOK, load_book, measure_usage
from wholedoc_bench.provenance import ManifestRun
from wholedoc_bench.providers import PROVIDERS
from wholedoc_bench.providers.base import load_raw, raw_path, save_raw
from wholedoc_bench.report import (
    Prices,
    check_comparable,
    check_origins,
    check_rankable,
    read_summaries,
    render_markdown,
    render_text,
    summarize,
    write_summary,
)
from wholedoc_bench.schema import Groundtruth, Prediction
from wholedoc_bench.scorer.run import CorpusResult, ParseFailure, score_corpus, write_scores_jsonl
from wholedoc_bench.scorer.tracks import ALL_TRACKS
from wholedoc_bench.submission import import_predictions, read_submission

RUN_MANIFEST = Path(".parse-eval/run.json")


def read_run_manifest(predictions_dir: Path) -> dict[str, Any] | None:
    path = predictions_dir / RUN_MANIFEST
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected an object with provider and config")
        if "schema_version" in data:
            if ManifestRun.model_validate(data).origin == "import":
                return data
        elif set(data) != {"provider", "config"}:
            raise ValueError("expected an object with provider and config")
        name, config = data["provider"], data["config"]
        if not isinstance(name, str) or name not in PROVIDERS:
            raise ValueError("provider must name a registered provider")
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        if name in {"extend", "reducto"} and config == {}:
            data["config"] = config = PROVIDERS[name].validate_config(config)
        if PROVIDERS[name].validate_config(config) != config:
            raise ValueError("config must include effective defaults")
        return data
    except (OSError, ValueError, TypeError) as e:
        raise ValueError(f"{path}: invalid run manifest: {e}") from e


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wholedoc-bench", description="Document parsing benchmark"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a parser on every document of a corpus")
    run.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    run.add_argument("--config", type=Path, help="provider configuration as a JSON object file")
    run.add_argument("--corpus", type=Path, required=True, help="folder of <doc>/document.* files")
    run.add_argument("--out", type=Path, required=True, help="folder for <doc>.json predictions")
    run.add_argument(
        "--from-raw",
        action="store_true",
        help="convert the responses saved in <out>/raw again, without calling the parser",
    )
    score = commands.add_parser("score", help="score saved predictions against the groundtruth")
    score.add_argument(
        "--corpus", type=Path, required=True, help="folder of <doc>/groundtruth.json"
    )
    score.add_argument(
        "--predictions", type=Path, required=True, help="folder of <doc>.json predictions"
    )
    score.add_argument(
        "--results",
        type=Path,
        help="write <parser>.jsonl (per-document scores) and <parser>.json (the summary) "
        "here; the parser is named after the predictions folder",
    )
    imp = commands.add_parser(
        "import",
        help="import predictions made outside wholedoc-bench, so they can be scored",
        description="Check a folder of <id>.json predictions against a public release and "
        "the prediction schema, and copy them into a fresh folder with "
        "run evidence that says they were supplied. Score that folder with wholedoc-bench score.",
    )
    imp.add_argument("--corpus", type=Path, required=True, help="release folder")
    imp.add_argument(
        "--predictions", type=Path, required=True, help="folder of <id>.json predictions"
    )
    imp.add_argument("--out", type=Path, required=True, help="new folder for the import")
    imp.add_argument("--name", required=True, help="short name for the parser, such as my-parser")
    imp.add_argument(
        "--claims",
        type=Path,
        required=True,
        help="JSON file: parser, version, settings, ran_at, ran_where, training_statement",
    )
    imp.add_argument(
        "--missing-is-failure",
        action="store_true",
        help="score a selected document without a prediction file as a failed parse",
    )
    report = commands.add_parser("report", help="build the leaderboard from scored results")
    report.add_argument(
        "--results", type=Path, required=True, help="folder of <parser>.json summaries"
    )
    report.add_argument(
        "--out", type=Path, help="Markdown leaderboard (default: LEADERBOARD.md next to --results)"
    )
    report.add_argument(
        "--allow-mixed-origins",
        action="store_true",
        help="show wholedoc-bench runs and imported predictions in one labelled table",
    )
    report.add_argument(
        "--prices",
        type=Path,
        help="price book: a folder of <provider>.toml rates (default: prices/ in this checkout)",
    )
    report.add_argument(
        "--prices-as-of",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="price the runs' usage at the rates in force on this day (default: today)",
    )
    for command in (run, score, imp):
        command.add_argument(
            "--exclude-document",
            action="append",
            default=[],
            metavar="ID",
            help="exclude a release document explicitly (repeatable)",
        )
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            config = None if args.config is None else json.loads(args.config.read_text("utf-8"))
            if args.config is not None and not isinstance(config, dict):
                raise ValueError("provider config must be a JSON object")
            return _run(
                args.provider,
                args.corpus,
                args.out,
                from_raw=args.from_raw,
                config=config,
                exclude=frozenset(args.exclude_document),
            )
        except (OSError, ValueError, TypeError) as e:
            print(f"run: {e}", file=sys.stderr)
            return 2
    if args.command == "score":
        try:
            return _score(
                args.corpus,
                args.predictions,
                args.results,
                exclude=frozenset(args.exclude_document),
            )
        except (OSError, ValueError, TypeError) as e:
            print(f"score: {e}", file=sys.stderr)
            return 2
    if args.command == "import":
        try:
            loaded = load_corpus_context(args.corpus, exclude=frozenset(args.exclude_document))
            return import_predictions(
                loaded,
                args.predictions,
                args.out,
                name=args.name,
                submission=read_submission(args.claims),
                missing_is_failure=args.missing_is_failure,
            )
        except (OSError, ValueError, TypeError) as e:
            print(f"import: {e}", file=sys.stderr)
            return 2
    return _report(
        args.results,
        args.out,
        allow_mixed_origins=args.allow_mixed_origins,
        prices=args.prices,
        as_of=args.prices_as_of,
    )


def _run(
    provider_name: str,
    corpus_dir: Path,
    out: Path,
    *,
    from_raw: bool = False,
    config: dict | None = None,
    exclude: frozenset[str] = frozenset(),
) -> int:
    loaded = load_corpus_context(corpus_dir, exclude=exclude)
    manifest_path = out / RUN_MANIFEST
    previous = read_run_manifest(out)
    if previous is not None and previous.get("origin") == "import":
        raise ValueError(
            "this folder holds imported predictions; they have no raw responses to replay, "
            "and a run must not mix with them; use a fresh --out folder"
        )
    if loaded.is_manifest:
        return run_manifest(
            provider_name, loaded, out, from_raw=from_raw, config=config, previous=previous
        )
    corpus = [document.truth for document in loaded.documents]
    if previous is None and out.exists() and any(out.iterdir()):
        raise ValueError("populated --out folder has no trusted run manifest; use a fresh folder")
    if previous is not None and from_raw and config is None:
        config = previous["config"]
    provider = PROVIDERS[provider_name](config=config)
    manifest = {"provider": provider.name, "config": provider.config}
    if previous is not None and previous != manifest:
        raise ValueError(
            "provider/config differs from the run manifest; use a separate --out folder"
        )
    if not from_raw:
        provider.prepare()
    if from_raw:
        for saved in sorted((out / "raw").glob("*.json")):
            raw = load_raw(saved)
            raw_config = raw.metadata.get("config")
            if raw.provider in {"extend", "reducto"} and raw_config == {}:
                raw_config = PROVIDERS[raw.provider].validate_config({})
            if raw.provider != provider.name or raw_config != provider.config:
                raise ValueError(
                    f"{saved}: raw provider/config missing or mismatched; replay refused"
                )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    versions: set[str] = set()
    (out / "VERSION.txt").unlink(missing_ok=True)
    for truth in corpus:
        doc_id = truth.document.id
        prediction_path(out, doc_id).unlink(missing_ok=True)
        saved = raw_path(out, doc_id)
        found = sorted((corpus_dir / doc_id).glob("document.*"))
        if not found:
            print(f"{doc_id}: no document file")
            continue
        if found[0].suffix not in provider.formats:
            print(f"{doc_id}: {provider.name} does not read {found[0].suffix}")
            continue
        if from_raw and not saved.exists():
            print(f"{doc_id}: no raw response in {saved}")
            continue
        try:
            if from_raw:
                raw = load_raw(saved)
            else:
                saved.unlink(missing_ok=True)
                raw = provider.fetch(found[0])
                if raw.provider != provider.name:
                    raise ValueError("raw provider does not match the selected provider")
                raw = replace(raw, metadata={**raw.metadata, "config": provider.config})
                save_raw(raw, saved)
        except Exception as e:
            print(f"{doc_id}: failed: {type(e).__name__}: {e}")
            continue
        try:
            prediction = provider.normalize(raw)
        except Exception as e:
            print(f"{doc_id}: failed: {type(e).__name__}: {e}")
            print(f"  the response is kept in {saved}; after a fix, run again with --from-raw")
            continue
        save_prediction(prediction, prediction_path(out, doc_id))
        versions.update(v for v in [raw.metadata.get("version")] if v)
        print(f"{doc_id}: ok")
    if versions:
        (out / "VERSION.txt").write_text(", ".join(sorted(versions)) + "\n", encoding="utf-8")
    return 0


def _score(
    corpus_dir: Path,
    predictions_dir: Path,
    results_dir: Path | None,
    *,
    exclude: frozenset[str] = frozenset(),
) -> int:
    if not predictions_dir.is_dir():
        print(f"{predictions_dir}: no such predictions folder", file=sys.stderr)
        return 1
    try:
        provenance = read_run_manifest(predictions_dir) or {}
    except ValueError as e:
        print(f"score: {e}", file=sys.stderr)
        return 2
    loaded = load_corpus_context(corpus_dir, exclude=exclude)
    if provenance.get("origin") == "import" and not loaded.is_manifest:
        print(
            "score: imported predictions are scored against the release they were imported for",
            file=sys.stderr,
        )
        return 2
    if loaded.is_manifest:
        result = score_manifest(loaded, predictions_dir, provenance, results_dir)
        print(_summary(result))
        return 0
    items: list[tuple[Groundtruth, Prediction | ParseFailure]] = []
    for document in loaded.documents:
        truth = document.truth
        path = prediction_path(predictions_dir, truth.document.id)
        items.append((truth, read_prediction(path)))
    result = score_corpus(items)
    if results_dir is not None:
        name = predictions_dir.resolve().name
        version_file = predictions_dir / "VERSION.txt"
        version = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else ""
        provider, config = provenance.get("provider"), provenance.get("config")
        raws = []
        for document in loaded.documents:
            id_ = document.truth.document.id
            try:
                raw = load_raw(raw_path(predictions_dir, id_))
            except (OSError, ValueError, TypeError):
                continue
            if raw.provider == provider and raw.document_id == id_:
                raws.append((raw, len(document.truth.document.pages)))
        summary = summarize(
            name,
            result,
            version or "unknown",
            provider=provider,
            provider_config=config,
            usage=measure_usage(provider, config, raws),
        )
        write_scores_jsonl(result, results_dir / f"{name}.jsonl")
        write_summary(summary, results_dir / f"{name}.json")
    print(_summary(result))
    return 0


def _report(
    results_dir: Path,
    out: Path | None,
    *,
    allow_mixed_origins: bool = False,
    prices: Path | None = None,
    as_of: date | None = None,
) -> int:
    try:
        summaries = read_summaries(results_dir)
        check_comparable(summaries)
        check_rankable(summaries)
        check_origins(summaries, allow_mixed=allow_mixed_origins)
        if not summaries:
            print(
                f"{results_dir}: no <parser>.json summaries; "
                "run wholedoc-bench score --results first"
            )
            return 1
        if prices is None and BOOK.is_dir():
            prices = BOOK
        if prices is None and as_of is not None:
            raise ValueError(
                "--prices-as-of needs a price book: this scorer has none; use --prices"
            )
        book = None if prices is None else Prices(load_book(prices), as_of or date.today())
        text, markdown = render_text(summaries, book), render_markdown(summaries, book)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    out = out or results_dir.resolve().parent / "LEADERBOARD.md"
    print(text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def read_prediction(path: Path) -> Prediction | ParseFailure:
    if not path.exists():
        return ParseFailure(f"{path}: no prediction file")
    try:
        return load_prediction(path)
    except (OSError, ValueError) as e:
        first_line = str(e).splitlines()[0]
        return ParseFailure(f"{path}: {type(e).__name__}: {first_line}")


def _summary(result: CorpusResult) -> str:
    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    lines = [f"{'Track':<6} {'Name':<38} Score"]
    for track in ALL_TRACKS:
        lines.append(f"{track.id:<6} {track.name:<38} {fmt(result.track_scores[track.id])}")
    lines.append(f"{'':<6} {'Overall':<38} {fmt(result.overall)}")
    cov = result.coverage
    parsed = sum(not d.failed for d in result.documents)
    lines.append(
        f"\nG3 coverage: {parsed}/{len(result.documents)} documents "
        f"({cov.value:.0%}), {'pass' if cov.passed else 'FAIL'}"
    )
    for doc in result.documents:
        if doc.error is not None:
            lines.append(f"  failed: {doc.document_id}: {doc.error}")
    if result.not_implemented:
        lines.append(
            f"Not implemented yet: {len(result.not_implemented)} sub-metrics, "
            "left out of every mean"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
