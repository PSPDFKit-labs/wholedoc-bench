import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest
from jsonschema import Draft202012Validator

from tests.release_helpers import copy_release
from tests.test_release_execution import calls, run, score  # noqa: F401  (the calls fixture)
from wholedoc_bench.cli import RUN_MANIFEST, main
from wholedoc_bench.provenance import ManifestRun, Submission, digest
from wholedoc_bench.report import Gate, Summary, check_origins, read_summary, write_summary
from wholedoc_bench.schema import Prediction, prediction_schema
from wholedoc_bench.scorer.run import SPEC_VERSION

ROOT = Path(__file__).parents[1]
IDS = ["np-00001", "np-00002"]
MARKDOWN = "# Library notice\n\nThe library opens at nine.\n"
CLAIMS: dict[str, Any] = {
    "parser": "My Parser",
    "version": "2.0",
    "settings": {"mode": "accurate", "max_tokens": 4096},
    "ran_at": "2026-10-18",
    "ran_where": "a laptop, offline",
    "training_statement": "Not trained or tuned on this release.",
}


@pytest.fixture
def release(tmp_path: Path) -> Path:
    return copy_release(tmp_path / "release")


def supplied(folder: Path, ids: list[str] = IDS, text: str = MARKDOWN) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for id_ in ids:
        (folder / f"{id_}.json").write_text(json.dumps({"markdown": text}), encoding="utf-8")
    return folder


def claims(tmp_path: Path, **change: Any) -> Path:
    path = tmp_path / "claims.json"
    path.write_text(json.dumps({**CLAIMS, **change}), encoding="utf-8")
    return path


def import_(release: Path, predictions: Path, out: Path, claims_file: Path, *extra: str) -> int:
    args = ["--corpus", str(release), "--predictions", str(predictions), "--out", str(out)]
    return main(["import", *args, "--name", "mine", "--claims", str(claims_file), *extra])


def report(results: Path, *extra: str) -> int:
    return main(["report", "--results", str(results), *extra])


def test_a_release_is_imported_scored_and_reported_as_imported(
    release: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, results = tmp_path / "mine", tmp_path / "results"
    assert import_(release, supplied(tmp_path / "outputs"), out, claims(tmp_path)) == 0
    record = ManifestRun.model_validate_json((out / RUN_MANIFEST).read_bytes())
    assert record.origin == "import" and record.provider == "mine"
    assert record.config == CLAIMS["settings"]
    assert record.submission == Submission.model_validate(CLAIMS)
    assert record.corpus.release is not None
    assert all(a.status == "complete" and a.raw_sha256 is None for a in record.attempts.values())

    assert score(release, out, results) == 0
    summary = read_summary(results / "mine.json")
    assert summary.origin == "import"
    assert summary.parser_version == "2.0"
    assert summary.provider_config == CLAIMS["settings"]
    assert summary.evaluation and summary.evaluation["origin"] == "import"
    assert summary.evaluation["submission"] == CLAIMS
    assert json.loads((results / "mine.json").read_text("utf-8"))["origin"] == "import"

    capsys.readouterr()
    assert report(results) == 0
    printed = capsys.readouterr().out
    assert "mine (imported)" in printed and "claimed, not verified" in printed
    board = (tmp_path / "LEADERBOARD.md").read_text("utf-8")
    assert "| mine (imported) |" in board and "claimed, not verified" in board


def test_imports_share_the_comparison_identity_but_never_rank_with_runs_by_default(
    release: Path,
    tmp_path: Path,
    calls: list[Path],  # noqa: F811
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = tmp_path / "results"
    ran = tmp_path / "markitdown"
    assert run(release, ran) == 0
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    for id_ in IDS:
        shutil.copy(ran / f"{id_}.json", outputs / f"{id_}.json")
    imported = tmp_path / "mine"
    assert import_(release, outputs, imported, claims(tmp_path)) == 0
    assert score(release, ran, results) == 0
    assert score(release, imported, results) == 0
    by_run, by_import = (
        read_summary(results / "markitdown.json"),
        read_summary(results / "mine.json"),
    )
    assert by_run.comparison and by_import.comparison
    assert digest(by_run.comparison) == digest(by_import.comparison)
    assert by_run.track_scores == by_import.track_scores
    assert (by_run.origin, by_import.origin) == ("run", "import")

    capsys.readouterr()
    assert report(results) == 2
    assert "mix wholedoc-bench runs with imported predictions" in capsys.readouterr().err
    assert report(results, "--allow-mixed-origins") == 0
    printed = capsys.readouterr().out
    assert "mine (imported)" in printed and "markitdown (imported)" not in printed


def test_run_evidence_and_run_summaries_are_written_exactly_as_before(
    release: Path,
    tmp_path: Path,
    calls: list[Path],  # noqa: F811
) -> None:
    out, results = tmp_path / "parser", tmp_path / "results"
    assert run(release, out) == 0
    assert score(release, out, results) == 0
    evidence = json.loads((out / RUN_MANIFEST).read_text("utf-8"))
    assert "origin" not in evidence and "submission" not in evidence
    summary = json.loads((results / "parser.json").read_text("utf-8"))
    assert "origin" not in summary
    assert not {"origin", "submission"} & set(summary["evaluation"])
    assert read_summary(results / "parser.json").origin == "run"


def test_the_whole_import_is_refused_and_nothing_written(
    release: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outputs = supplied(tmp_path / "outputs", ["np-00001"])
    (outputs / "NP-00002.json").write_text(json.dumps({"markdown": "x"}), "utf-8")
    (outputs / "np-09999.json").write_text(json.dumps({"markdown": "x"}), "utf-8")
    (outputs / "nested").mkdir()
    out = tmp_path / "mine"
    capsys.readouterr()
    assert import_(release, outputs, out, claims(tmp_path)) == 2
    err = capsys.readouterr().err
    assert "nothing was written" in err
    assert "NP-00002.json: wrong case" in err
    assert "np-09999.json: not a prediction for a selected document" in err
    assert "nested: not a plain file" in err
    assert "no prediction for: np-00002" in err
    assert not out.exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "np-00001.json: not a valid prediction"),
        (json.dumps({"markdown": "x", "extra": 1}), "extra: Extra inputs are not permitted"),
        (json.dumps({"layout": []}), "markdown: Field required"),
    ],
)
def test_one_invalid_prediction_refuses_the_import(
    release: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: str,
    message: str,
) -> None:
    outputs = supplied(tmp_path / "outputs")
    (outputs / "np-00001.json").write_text(content, "utf-8")
    assert import_(release, outputs, tmp_path / "mine", claims(tmp_path)) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "mine").exists()


def test_a_selection_and_the_output_folder_are_checked(
    release: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outputs = supplied(tmp_path / "outputs")
    args = ("--exclude-document", "np-00002")
    assert import_(release, outputs, tmp_path / "a", claims(tmp_path), *args) == 2
    assert "np-00002.json: not a prediction for a selected document" in capsys.readouterr().err
    populated = tmp_path / "populated"
    populated.mkdir()
    (populated / "notes.txt").write_text("x", "utf-8")
    assert import_(release, outputs, populated, claims(tmp_path)) == 2
    assert "new or empty folder" in capsys.readouterr().err
    legacy = tmp_path / "legacy"
    shutil.copytree(ROOT / "tests" / "fixtures" / "plain_continuation_demo", legacy / "plain")
    assert import_(legacy, outputs, tmp_path / "c", claims(tmp_path)) == 2
    assert "import needs a public release" in capsys.readouterr().err
    bad_name = ["--name", "My Parser"]
    args = ["--corpus", str(release), "--predictions", str(outputs), "--out", str(tmp_path / "d")]
    assert main(["import", *args, *bad_name, "--claims", str(claims(tmp_path))]) == 2
    assert "--name must be" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"version": ""}, "version"),
        ({"settings": "fast"}, "settings"),
        ({"comment": "extra"}, "comment: Extra inputs are not permitted"),
        ({"settings": {"api_key": "x"}}, "settings must not hold credentials; remove: api_key"),
        ({"settings": {"auth": {"accessToken": "x"}}}, "remove: auth.accessToken"),
        ({"settings": {"steps": [{"password": "x"}]}}, "remove: steps[0].password"),
    ],
)
def test_claims_have_a_fixed_shape_and_no_credentials(
    release: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    change: dict[str, Any],
    message: str,
) -> None:
    outputs = supplied(tmp_path / "outputs")
    assert import_(release, outputs, tmp_path / "mine", claims(tmp_path, **change)) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "mine").exists()


def test_claims_missing_a_field_are_refused(
    release: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "claims.json"
    path.write_text(json.dumps({k: v for k, v in CLAIMS.items() if k != "training_statement"}))
    assert import_(release, supplied(tmp_path / "outputs"), tmp_path / "mine", path) == 2
    assert "training_statement: Field required" in capsys.readouterr().err


def test_settings_words_are_not_mistaken_for_credentials() -> None:
    settings = {"max_tokens": 1, "keyboard_layout": "uk", "monkey": True, "tokenizer": "bpe"}
    assert Submission.model_validate({**CLAIMS, "settings": settings}).settings == settings


def test_a_missing_document_is_a_failed_parse_only_when_asked(
    release: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outputs = supplied(tmp_path / "outputs", ["np-00001"])
    out, results = tmp_path / "mine", tmp_path / "results"
    assert import_(release, outputs, out, claims(tmp_path), "--missing-is-failure") == 0
    record = ManifestRun.model_validate_json((out / RUN_MANIFEST).read_bytes())
    attempt = record.attempts["np-00002"]
    assert (attempt.status, attempt.error) == ("failed", "no prediction submitted")
    assert attempt.stage == "not_submitted"
    assert not (out / "np-00002.json").exists()
    assert score(release, out, results) == 0
    summary = read_summary(results / "mine.json")
    assert (summary.parsed, summary.documents) == (1, 2)
    assert summary.failure_stages == {"not_submitted": 1}
    assert "np-00002: no prediction submitted" in capsys.readouterr().out
    assert report(results) == 0


@pytest.mark.parametrize("out_exists", [False, True])
def test_an_import_with_every_document_missing_records_a_total_failure(
    release: Path, tmp_path: Path, out_exists: bool
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    out, results = tmp_path / "new" / "mine", tmp_path / "results"
    if out_exists:
        out.mkdir(parents=True)
    assert import_(release, outputs, out, claims(tmp_path), "--missing-is-failure") == 0
    record = ManifestRun.model_validate_json((out / RUN_MANIFEST).read_bytes())
    assert all(a.status == "failed" for a in record.attempts.values())
    assert {a.stage for a in record.attempts.values()} == {"not_submitted"}
    assert score(release, out, results) == 0
    summary = read_summary(results / "mine.json")
    assert (summary.parsed, summary.documents) == (0, 2)
    assert summary.coverage.passed is False
    assert summary.failure_stages == {"not_submitted": 2}
    assert report(results) == 0


def test_only_an_import_can_have_a_document_that_was_not_submitted(
    release: Path, tmp_path: Path
) -> None:
    out = tmp_path / "mine"
    outputs = supplied(tmp_path / "outputs", ["np-00001"])
    assert import_(release, outputs, out, claims(tmp_path), "--missing-is-failure") == 0
    data = json.loads((out / RUN_MANIFEST).read_bytes())
    data = {k: v for k, v in data.items() if k not in ("origin", "submission")}
    with pytest.raises(ValueError, match="only an import can have a document that was not"):
        ManifestRun.model_validate(data)


def test_the_imported_copy_is_what_gets_scored(
    release: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outputs = supplied(tmp_path / "outputs")
    out = tmp_path / "mine"
    assert import_(release, outputs, out, claims(tmp_path)) == 0
    assert (out / "np-00001.json").read_bytes() == (outputs / "np-00001.json").read_bytes()
    (outputs / "np-00001.json").write_text(json.dumps({"markdown": "changed"}), "utf-8")
    assert score(release, out, tmp_path / "results") == 0
    (out / "np-00001.json").write_text(json.dumps({"markdown": "changed"}), "utf-8")
    capsys.readouterr()
    assert score(release, out, tmp_path / "results2") == 2
    assert "np-00001: prediction missing or hash mismatch" in capsys.readouterr().err


def test_run_and_replay_refuse_an_import_folder_even_under_a_built_in_name(
    release: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "pdftotext"
    args = ["--corpus", str(release), "--predictions", str(supplied(tmp_path / "outputs"))]
    claims_file = claims(tmp_path, settings={})
    assert (
        main(
            [
                "import",
                *args,
                "--out",
                str(out),
                "--name",
                "pdftotext",
                "--claims",
                str(claims_file),
            ]
        )
        == 0
    )
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    capsys.readouterr()
    assert run(release, out, provider="pdftotext") == 2
    assert "holds imported predictions" in capsys.readouterr().err
    assert run(release, out, "--from-raw", provider="pdftotext") == 2
    assert "holds imported predictions" in capsys.readouterr().err
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before


def test_an_unregistered_provider_is_accepted_only_as_an_import(
    release: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "mine"
    assert import_(release, supplied(tmp_path / "outputs"), out, claims(tmp_path)) == 0
    evidence = json.loads((out / RUN_MANIFEST).read_text("utf-8"))
    del evidence["origin"], evidence["submission"]
    (out / RUN_MANIFEST).write_text(json.dumps(evidence), "utf-8")
    capsys.readouterr()
    assert score(release, out, tmp_path / "results") == 2
    assert "provider must name a registered provider" in capsys.readouterr().err


def test_an_import_must_carry_its_claims_and_no_raw_responses(
    release: Path, tmp_path: Path
) -> None:
    out = tmp_path / "mine"
    assert import_(release, supplied(tmp_path / "outputs"), out, claims(tmp_path)) == 0
    evidence = json.loads((out / RUN_MANIFEST).read_text("utf-8"))
    with pytest.raises(ValueError, match="carry a submission"):
        ManifestRun.model_validate({k: v for k, v in evidence.items() if k != "submission"})
    evidence["attempts"]["np-00001"]["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="no raw response"):
        ManifestRun.model_validate(evidence)


def test_check_origins_and_unknown_origins(tmp_path: Path) -> None:
    def one(parser: str, origin: Literal["run", "import"] = "run") -> Summary:
        summary = Summary(parser, "1", SPEC_VERSION, 1, 1, {}, None, Gate(1.0, True))
        return replace(summary, origin=origin)

    ran, imported = one("a"), one("b", "import")
    check_origins([ran, ran])
    check_origins([imported])
    with pytest.raises(ValueError, match="mix wholedoc-bench runs"):
        check_origins([ran, imported])
    check_origins([ran, imported], allow_mixed=True)
    path = tmp_path / "x.json"
    write_summary(imported, path)
    data = json.loads(path.read_text("utf-8"))
    path.write_text(json.dumps({**data, "origin": "emailed"}), "utf-8")
    with pytest.raises(ValueError, match="unknown origin"):
        read_summary(path)


def prediction_validator() -> Draft202012Validator:
    return Draft202012Validator(
        json.loads((ROOT / "schemas" / "prediction.schema.json").read_bytes())
    )


def assert_valid_predictions(files: list[Path]) -> None:
    validator = prediction_validator()
    for path in files:
        data = json.loads(path.read_bytes())
        validator.validate(data)
        Prediction.model_validate_json(path.read_bytes())


def test_the_published_prediction_schema_is_current_and_accepts_its_example() -> None:
    for name, expected in prediction_schema().items():
        schema = json.loads((ROOT / "schemas" / name).read_bytes())
        assert schema == {"$schema": "https://json-schema.org/draft/2020-12/schema", **expected}
        Draft202012Validator.check_schema(schema)
    assert_valid_predictions([ROOT / "schemas" / "examples" / "prediction-minimal.json"])
    validator = prediction_validator()
    assert not validator.is_valid({"markdown": "x", "extra": 1})
    assert not validator.is_valid({"layout": []})


def test_an_import_under_a_cloud_providers_name_records_no_usage(
    release: Path, tmp_path: Path
) -> None:
    out, results = tmp_path / "textract", tmp_path / "results"
    args = ["--corpus", str(release), "--predictions", str(supplied(tmp_path / "outputs"))]
    imported = [*args, "--out", str(out), "--name", "textract", "--claims", str(claims(tmp_path))]
    assert main(["import", *imported]) == 0
    assert score(release, out, results) == 0
    assert read_summary(results / "textract.json").usage is None
