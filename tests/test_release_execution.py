import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

from tests.builders import NAME, memo
from tests.release_helpers import copy_release, edit_document, edit_key, rows
from tests.scorer.test_base import Fixed, OtherFixed, Stub
from wholedoc_bench.cli import RUN_MANIFEST, main
from wholedoc_bench.corpus.loader import load_corpus, load_corpus_context
from wholedoc_bench.execution import run_manifest
from wholedoc_bench.provenance import ManifestRun, file_hash
from wholedoc_bench.providers import MarkitdownProvider, NutrientProvider, RawResult
from wholedoc_bench.report import check_comparable, read_summary
from wholedoc_bench.schema import Prediction
from wholedoc_bench.scorer.base import Source, SubMetric, Track
from wholedoc_bench.scorer.run import score_corpus, score_document


@pytest.fixture
def release(tmp_path: Path) -> Path:
    return copy_release(tmp_path / "release")


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    seen: list[Path] = []

    def fetch(self: MarkitdownProvider, path: Path) -> RawResult:
        seen.append(path)
        return RawResult(
            self.name,
            path.parent.name,
            "# Library notice\n\nThe library opens at nine.\n\n"
            "| Item | Count |\n| --- | --- |\n| Blue pens | 3 |\n",
            {"version": "fixture-parser 1.0"},
        )

    monkeypatch.setattr(MarkitdownProvider, "fetch", fetch)
    return seen


def run(corpus: Path, out: Path, *extra: str, provider: str = "markitdown") -> int:
    return main(["run", "--provider", provider, "--corpus", str(corpus), "--out", str(out), *extra])


def score(corpus: Path, out: Path, results: Path, *extra: str) -> int:
    return main(
        [
            "score",
            "--corpus",
            str(corpus),
            "--predictions",
            str(out),
            "--results",
            str(results),
            *extra,
        ]
    )


def files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_run_score_and_report_a_release(release: Path, tmp_path: Path, calls: list[Path]) -> None:
    out, results = tmp_path / "parser", tmp_path / "results"
    assert run(release, out) == 0
    assert [p.parent.name + "/" + p.name for p in calls] == [
        "np-00001/document.pdf",
        "np-00002/document.pdf",
    ]
    record = ManifestRun.model_validate_json((out / RUN_MANIFEST).read_bytes())
    assert all(a.status == "complete" for a in record.attempts.values())
    assert record.corpus.include_drafts is False
    assert [d.id for d in record.corpus.documents] == ["np-00001", "np-00002"]
    assert score(release, out, results) == 0
    summary = read_summary(results / "parser.json")
    assert summary.documents == summary.parsed == 2
    assert summary.eligible_documents["T1.word_bag_f1"] == 2
    assert summary.track_scores["T3"] is None
    assert summary.comparison
    corpus = summary.comparison["corpus"]
    assert corpus["documents"][0]["key_sha256"] == rows(release)[0]["groundtruth_sha256"]
    assert corpus["release"]["tag"] == "v1.0"
    scores = [json.loads(line) for line in (results / "parser.jsonl").read_text().splitlines()]
    assert summary.evaluation
    assert summary.evaluation["scores_sha256"] == file_hash(results / "parser.jsonl")
    assert {row["evaluation_id"] for row in scores} == {summary.evaluation["id"]}
    assert all(
        r["score"] is None and r["status"] == "unsupported" for r in scores if r["track"] == "T3"
    )
    assert main(["report", "--results", str(results)]) == 0


def test_a_working_corpus_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = tmp_path / "working"
    corpus.mkdir()
    (corpus / "corpus.json").write_text("{}")
    with pytest.raises(ValueError, match="working corpus"):
        load_corpus_context(corpus)
    assert run(corpus, tmp_path / "out") == 2
    assert "not a release" in capsys.readouterr().err


def test_explicit_exclusion_records_a_different_scope(
    release: Path, tmp_path: Path, calls: list[Path]
) -> None:
    out, results = tmp_path / "parser", tmp_path / "results"
    options = ("--exclude-document", "np-00002")
    assert run(release, out, *options) == 0
    assert len(calls) == 1
    assert score(release, out, results, *options) == 0
    summary = read_summary(results / "parser.json")
    assert summary.comparison and summary.comparison["corpus"]["excluded_ids"] == ["np-00002"]
    assert score(release, out, tmp_path / "full") == 2


def test_a_corrected_key_reuses_predictions_but_changes_the_comparison(
    release: Path, tmp_path: Path, calls: list[Path]
) -> None:
    out = tmp_path / "parser"
    assert run(release, out) == 0
    assert score(release, out, tmp_path / "old") == 0
    before = files(out)
    edit_key(release, lambda truth: truth["elements"][1].update(text="The library opens at ten."))
    assert score(release, out, tmp_path / "new") == 0
    assert len(calls) == 2 and files(out) == before
    summary = read_summary(tmp_path / "new/parser.json")
    assert summary.evaluation and summary.comparison
    assert (
        summary.evaluation["collection"]["documents"][0]["key_sha256"]
        != summary.comparison["corpus"]["documents"][0]["key_sha256"]
    )
    with pytest.raises(ValueError, match="comparison identities"):
        check_comparable(
            [read_summary(tmp_path / folder / "parser.json") for folder in ("old", "new")]
        )


def test_source_change_cannot_reuse_predictions_or_raw(
    release: Path, tmp_path: Path, calls: list[Path]
) -> None:
    out = tmp_path / "parser"
    assert run(release, out) == 0
    before = files(out)
    document = release / rows(release)[0]["document_path"]
    edit_document(release, document.read_bytes() + b"\n")
    assert score(release, out, tmp_path / "results") == 2
    assert run(release, out, "--from-raw") == 2
    assert run(release, out) == 2
    assert len(calls) == 2 and files(out) == before


@pytest.mark.parametrize("artifact", ["np-00001.json", "raw/np-00001.json"])
def test_modified_artifact_is_not_trusted(
    release: Path, tmp_path: Path, calls: list[Path], artifact: str
) -> None:
    out = tmp_path / "parser"
    assert run(release, out) == 0
    path = out / artifact
    path.write_bytes(path.read_bytes() + b" ")
    before = files(out)
    assert score(release, out, tmp_path / "results") == 2
    if artifact.startswith("raw/"):
        assert run(release, out, "--from-raw") == 2
        assert files(out) == before


def test_raw_replay_is_offline(release: Path, tmp_path: Path, calls: list[Path]) -> None:
    out = tmp_path / "parser"
    assert run(release, out) == 0
    calls.clear()
    assert run(release, out, "--from-raw") == 0
    assert calls == []
    assert score(release, out, tmp_path / "results") == 0


@pytest.mark.parametrize("mode", ["initial", "replay", "live"])
def test_interrupted_normalization_preserves_raw_for_offline_recovery(
    release: Path,
    tmp_path: Path,
    calls: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    out = tmp_path / "parser"
    if mode != "initial":
        assert run(release, out) == 0
    original = MarkitdownProvider.normalize
    interrupted_id = "np-00002" if mode == "initial" else "np-00001"

    def interrupted(self: MarkitdownProvider, raw: RawResult) -> Prediction:
        if raw.document_id == interrupted_id:
            raise KeyboardInterrupt
        return original(self, raw)

    monkeypatch.setattr(MarkitdownProvider, "normalize", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(release, out, *(["--from-raw"] if mode == "replay" else []))
    record = ManifestRun.model_validate_json((out / RUN_MANIFEST).read_bytes())
    assert record.attempts[interrupted_id].status == "pending"
    assert all(
        attempt.raw_sha256 == file_hash(out / "raw" / f"{id_}.json")
        for id_, attempt in record.attempts.items()
    )
    assert score(release, out, tmp_path / "incomplete") == 2
    monkeypatch.setattr(MarkitdownProvider, "normalize", original)
    calls.clear()
    assert run(release, out, "--from-raw") == 0
    assert calls == []
    assert score(release, out, tmp_path / "recovered") == 0


def test_interrupted_live_fetch_preserves_previous_responses(
    release: Path,
    tmp_path: Path,
    calls: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "parser"
    assert run(release, out) == 0

    def interrupted(self: MarkitdownProvider, path: Path) -> RawResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(MarkitdownProvider, "fetch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(release, out)
    assert score(release, out, tmp_path / "incomplete") == 2
    assert run(release, out, "--from-raw") == 0
    assert score(release, out, tmp_path / "recovered") == 0


@pytest.mark.parametrize("replay", [False, True])
def test_pending_raw_is_verified_before_any_output_changes(
    release: Path, tmp_path: Path, calls: list[Path], replay: bool
) -> None:
    out = tmp_path / "parser"
    assert run(release, out) == 0
    path = out / RUN_MANIFEST
    manifest = json.loads(path.read_bytes())
    attempt = manifest["attempts"]["np-00002"]
    attempt.update(status="pending", prediction_sha256=None)
    path.write_text(json.dumps(manifest))
    raw = out / "raw/np-00002.json"
    raw.write_bytes(raw.read_bytes() + b" ")
    before = files(out)
    calls.clear()
    assert run(release, out, *(["--from-raw"] if replay else [])) == 2
    assert calls == [] and files(out) == before


def test_keys_only_loader_refuses_release_metric_loss(release: Path) -> None:
    with pytest.raises(ValueError, match="load_corpus_context.*metric_support"):
        load_corpus(release)


@pytest.mark.parametrize("legacy", [False, True])
def test_missing_run_evidence_has_a_short_diagnostic(
    release: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], legacy: bool
) -> None:
    out = tmp_path / "parser"
    out.mkdir()
    if legacy:
        path = out / RUN_MANIFEST
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"provider": "markitdown", "config": {}}))
    assert score(release, out, tmp_path / "results") == 2
    assert capsys.readouterr().err.strip() == (
        "score: this folder has no manifest run evidence; run it with this corpus first"
    )
    assert not (tmp_path / "results").exists()


def test_pending_attempt_is_not_scored_as_failure(
    release: Path, tmp_path: Path, calls: list[Path]
) -> None:
    out = tmp_path / "parser"
    assert run(release, out) == 0
    path = out / RUN_MANIFEST
    manifest = json.loads(path.read_bytes())
    manifest["attempts"]["np-00001"] = {"status": "pending"}
    path.write_text(json.dumps(manifest))
    assert score(release, out, tmp_path / "results") == 2
    before = files(out)
    assert run(release, out, "--from-raw") == 2
    assert files(out) == before


def test_actual_failure_counts_zero_only_on_supported_metrics(
    release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(self: MarkitdownProvider, path: Path) -> RawResult:
        raise RuntimeError("test parser failed")

    monkeypatch.setattr(MarkitdownProvider, "fetch", crash)
    out, results = tmp_path / "parser", tmp_path / "results"
    assert run(release, out) == 0
    assert score(release, out, results) == 0
    summary = read_summary(results / "parser.json")
    assert summary.parsed == 0 and summary.overall == 0 and not summary.coverage.passed
    assert summary.track_scores["T3"] is None and summary.eligible_documents["T1"] == 2


def test_evaluations_are_not_overwritten(release: Path, tmp_path: Path, calls: list[Path]) -> None:
    out, results = tmp_path / "parser", tmp_path / "results"
    assert run(release, out) == score(release, out, results) == 0
    before = files(results)
    assert score(release, out, results) == 2
    assert files(results) == before


def _document(comparison: dict[str, Any], **change: Any) -> None:
    comparison["corpus"]["documents"][0].update(change)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda c: c["scorer"].update(source_sha256="0" * 64), id="scorer"),
        pytest.param(lambda c: c.update(spec_version="0.1"), id="spec"),
        pytest.param(lambda c: c.update(implemented_metrics=[]), id="metrics"),
        pytest.param(lambda c: c["corpus"].update(include_drafts=True), id="drafts"),
        pytest.param(lambda c: c["corpus"].update(excluded_ids=["np-00003"]), id="exclusions"),
        pytest.param(lambda c: _document(c, id="np-00009"), id="document-id"),
        pytest.param(lambda c: _document(c, status="draft"), id="document-status"),
        pytest.param(lambda c: _document(c, document_sha256="0" * 64), id="document-hash"),
        pytest.param(lambda c: _document(c, key_sha256="0" * 64), id="key-hash"),
        pytest.param(lambda c: _document(c, supported_metrics=[]), id="supported-metrics"),
        pytest.param(lambda c: c["corpus"]["release"].update(tag="v1.1"), id="release"),
        pytest.param(lambda c: c["corpus"].pop("release"), id="not-a-release"),
        pytest.param(lambda c: c.clear(), id="missing"),
    ],
)
def test_comparisons_require_identical_scope_and_scorer(
    release: Path, tmp_path: Path, calls: list[Path], change: Callable[[dict[str, Any]], object]
) -> None:
    out, results = tmp_path / "parser", tmp_path / "results"
    assert run(release, out) == score(release, out, results) == 0
    original = read_summary(results / "parser.json")
    comparison = json.loads(json.dumps(original.comparison))
    change(comparison)
    check_comparable([original, replace(original, parser="same-scope-other-parser")])
    with pytest.raises(ValueError, match="comparison identit"):
        check_comparable([original, replace(original, comparison=comparison or None)])


def test_a_manifest_edit_outside_the_scored_scope_keeps_results_comparable(
    release: Path, tmp_path: Path, calls: list[Path]
) -> None:
    out, results = tmp_path / "parser", tmp_path / "results"
    assert run(release, out) == score(release, out, results) == 0
    original = read_summary(results / "parser.json")
    assert original.comparison
    recorded = original.comparison["corpus"]
    info = json.loads((release / "release.json").read_text(encoding="utf-8"))
    assert recorded["manifest_sha256"] == info["manifest_sha256"]
    assert recorded["header_sha256"] == info["header_sha256"]
    before = json.loads(json.dumps(original.comparison))
    before["corpus"].pop("release")
    after = json.loads(json.dumps(before))
    after["corpus"].update(
        name="renamed", format_version="9.9", header_sha256="1" * 64, manifest_sha256="2" * 64
    )
    check_comparable(
        [
            replace(original, comparison=before),
            replace(original, parser="after-manifest-edit", comparison=after),
        ]
    )


def test_unsupported_metric_never_runs_or_builds_details(release: Path) -> None:
    class MustSkip(SubMetric):
        name: ClassVar[str] = "skip"
        source: ClassVar[Source] = Source.MARKDOWN

        def applies(self, truth):
            raise AssertionError("unsupported applies called")

        def score(self, ctx):
            raise AssertionError("unsupported score called")

        def details(self, truth, ctx, status):
            raise AssertionError("unsupported details called")

    truth = load_corpus_context(release).documents[0].truth
    tracks = [Track("TX", "partial", ((Fixed(1), 1), (MustSkip(), 1)))]
    result = score_document(
        truth, Prediction(markdown="text"), tracks, supported_metrics=frozenset({"TX.fixed"})
    )
    assert result.track_scores["TX"] == 1
    assert result.statuses["TX"] == {"fixed": "scored", "skip": "unsupported"}


def test_supported_metric_must_apply_and_exist(release: Path) -> None:
    truth = load_corpus_context(release).documents[0].truth
    tracks = [Track("TX", "partial", ((Fixed(None), 1),))]
    with pytest.raises(ValueError, match="nothing to score"):
        score_document(truth, None, tracks, supported_metrics=frozenset({"TX.fixed"}))
    with pytest.raises(ValueError, match="unknown supported"):
        score_document(truth, None, tracks, supported_metrics=frozenset({"TX.unknown"}))
    with pytest.raises(ValueError, match="exactly match"):
        score_corpus([(truth, None)], tracks, supported_metrics={})


def test_statuses_distinguish_unimplemented_and_not_applicable(release: Path) -> None:
    truth = load_corpus_context(release).documents[0].truth
    tracks = [Track("TX", "partial", ((Fixed(1), 1), (OtherFixed(None), 1), (Stub(1), 1)))]
    result = score_document(truth, None, tracks)
    assert result.statuses["TX"] == {
        "fixed": "parse_failed",
        "other_fixed": "not_applicable",
        "stub": "not_implemented",
    }


def test_empty_support_keeps_gates_but_has_no_numeric_score() -> None:
    truth = memo(NAME)
    result = score_corpus(
        [(truth, Prediction(markdown=NAME["printed"]))],
        supported_metrics={truth.document.id: frozenset()},
    )
    assert result.overall is None and result.coverage.passed
    assert all(value == 0 for value in result.eligible_documents.values())


def _nutrient(monkeypatch: pytest.MonkeyPatch, calls: list[Path]) -> None:
    def fetch(self: NutrientProvider, path: Path) -> RawResult:
        calls.append(path)
        return RawResult(self.name, path.parent.name, "saved text")

    monkeypatch.setattr(NutrientProvider, "prepare", lambda self: None)
    monkeypatch.setattr(NutrientProvider, "fetch", fetch)
    monkeypatch.setattr(
        NutrientProvider, "normalize", lambda self, raw: Prediction(markdown=raw.payload)
    )


def _forbid_cloud(release: Path):
    loaded = load_corpus_context(release)
    documents = tuple(replace(d, cloud_allowed=False) for d in loaded.documents[:1])
    return replace(loaded, documents=documents + loaded.documents[1:])


def test_a_cloud_forbidden_document_aborts_every_upload_before_prepare(
    release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []
    monkeypatch.setattr(NutrientProvider, "prepare", lambda self: seen.append("prepare"))
    monkeypatch.setattr(NutrientProvider, "fetch", lambda self, path: seen.append("fetch"))
    out = tmp_path / "cloud"
    with pytest.raises(ValueError, match="cloud parsing forbidden for: np-00001"):
        run_manifest(
            "nutrient", _forbid_cloud(release), out, from_raw=False, config=None, previous=None
        )
    assert seen == [] and not out.exists()


def test_cloud_raw_replay_does_not_upload_even_when_documents_now_forbid_it(
    release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    _nutrient(monkeypatch, calls)
    out = tmp_path / "cloud"
    assert run(release, out, provider="nutrient") == 0
    assert len(calls) == 2

    def reject_prepare(self: NutrientProvider) -> None:
        raise AssertionError("offline replay must not prepare live clients")

    monkeypatch.setattr(NutrientProvider, "prepare", reject_prepare)
    calls.clear()
    previous = json.loads((out / RUN_MANIFEST).read_bytes())
    forbidden = _forbid_cloud(release)
    replayed = run_manifest(
        "nutrient", forbidden, out, from_raw=True, config=None, previous=previous
    )
    assert replayed == 0 and calls == []


def test_conversion_failure_keeps_raw_for_offline_repair(
    release: Path,
    tmp_path: Path,
    calls: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = MarkitdownProvider.normalize

    def broken(self: MarkitdownProvider, raw: RawResult) -> Prediction:
        raise ValueError("test conversion failed")

    monkeypatch.setattr(MarkitdownProvider, "normalize", broken)
    out = tmp_path / "parser"
    assert run(release, out) == 0
    record = ManifestRun.model_validate_json((out / RUN_MANIFEST).read_bytes())
    assert all(a.status == "failed" and a.raw_sha256 for a in record.attempts.values())
    assert score(release, out, tmp_path / "failed") == 0
    monkeypatch.setattr(MarkitdownProvider, "normalize", original)
    calls.clear()
    assert run(release, out, "--from-raw") == 0
    assert calls == []
    assert score(release, out, tmp_path / "fixed") == 0
    assert read_summary(tmp_path / "fixed/parser.json").parsed == 2
