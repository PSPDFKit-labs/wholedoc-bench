import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from tests.builders import ACCOUNT, NAME, VISIBLE, memo, truth
from tests.conftest import DEMO
from wholedoc_bench.cli import main
from wholedoc_bench.corpus.loader import prediction_path, save_prediction
from wholedoc_bench.providers import NutrientProvider, RawResult
from wholedoc_bench.report import ranked, read_summary, summarize
from wholedoc_bench.schema import Prediction
from wholedoc_bench.scorer.run import SPEC_VERSION, score_corpus


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    folder = tmp_path / "corpus" / DEMO.name
    folder.mkdir(parents=True)
    shutil.copy(DEMO / "groundtruth.json", folder / "groundtruth.json")
    return folder.parent


def score(corpus: Path, predictions: Path, results: Path) -> int:
    args = ["score", "--corpus", str(corpus), "--predictions", str(predictions)]
    return main([*args, "--results", str(results)])


def parser_folder(tmp_path: Path, name: str, prediction: Prediction, version: str = "") -> Path:
    folder = tmp_path / "predictions" / name
    save_prediction(prediction, prediction_path(folder, DEMO.name))
    if version:
        (folder / "VERSION.txt").write_text(version + "\n", encoding="utf-8")
    return folder


def test_score_writes_the_scores_and_a_summary_named_after_the_predictions(
    tmp_path: Path, corpus: Path, demo_nutrient_raw: RawResult
) -> None:
    pred = NutrientProvider().normalize(demo_nutrient_raw)
    folder = parser_folder(tmp_path, "nutrient", pred, version="nutrient 1.0")
    results = tmp_path / "results"
    assert score(corpus, folder, results) == 0

    assert (results / "nutrient.jsonl").exists()
    summary = read_summary(results / "nutrient.json")
    assert (summary.parser, summary.parser_version) == ("nutrient", "nutrient 1.0")
    assert summary.spec_version == SPEC_VERSION
    assert (summary.documents, summary.parsed, summary.coverage.passed) == (1, 1, True)
    assert summary.overall is not None and summary.track_scores["T1"] is not None


def test_report_ranks_the_parsers_and_writes_one_markdown_table(
    tmp_path: Path,
    corpus: Path,
    demo_nutrient_raw: RawResult,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = tmp_path / "results"
    good = parser_folder(tmp_path, "good", NutrientProvider().normalize(demo_nutrient_raw))
    weak = parser_folder(tmp_path, "weak", Prediction(markdown="Plain page continuation\n"))
    score(corpus, weak, results)
    score(corpus, good, results)
    capsys.readouterr()

    assert main(["report", "--results", str(results)]) == 0

    printed = capsys.readouterr().out
    table = printed.splitlines()
    assert table[0].split()[:2] == ["Parser", "T1"]
    assert [line.split()[0] for line in table[2:4]] == ["good", "weak"]
    markdown = (tmp_path / "LEADERBOARD.md").read_text(encoding="utf-8")
    rows = [line for line in markdown.splitlines() if line.startswith("| ")]
    assert [row.split(" | ")[0] for row in rows] == ["| Parser", "| good", "| weak"]
    assert "### " not in markdown


def test_report_writes_the_markdown_where_asked(
    tmp_path: Path, corpus: Path, demo_nutrient_raw: RawResult
) -> None:
    results = tmp_path / "results"
    score(corpus, parser_folder(tmp_path, "p", Prediction(markdown="x\n")), results)
    out = tmp_path / "site" / "board.md"
    assert main(["report", "--results", str(results), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# Leaderboard")


def test_a_failed_gate_ranks_below_a_higher_overall(
    tmp_path: Path, corpus: Path, demo_nutrient_raw: RawResult, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    score(corpus, parser_folder(tmp_path, "weak", Prediction(markdown="Plain\n")), results)
    good = parser_folder(tmp_path, "good", NutrientProvider().normalize(demo_nutrient_raw))
    score(corpus, good, results)
    summary = json.loads((results / "good.json").read_text(encoding="utf-8"))
    summary["coverage"]["passed"] = False
    (results / "good.json").write_text(json.dumps(summary), encoding="utf-8")
    capsys.readouterr()

    main(["report", "--results", str(results)])
    rows = capsys.readouterr().out.splitlines()[2:4]
    assert [row.split()[0] for row in rows] == ["weak", "good"]


def test_report_refuses_results_from_another_spec_version(
    tmp_path: Path, corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    for name in ("a", "b", "c"):
        score(corpus, parser_folder(tmp_path, name, Prediction(markdown="x\n")), results)
    stale = json.loads((results / "b.json").read_text(encoding="utf-8"))
    stale["spec_version"] = "0.1"
    (results / "b.json").write_text(json.dumps(stale), encoding="utf-8")
    capsys.readouterr()

    assert main(["report", "--results", str(results)]) == 2
    error = capsys.readouterr().err
    assert "  b: spec 0.1" in error
    assert "a:" not in error and "c:" not in error
    assert not (tmp_path / "LEADERBOARD.md").exists()


def test_report_without_summaries_says_what_to_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "results").mkdir()
    assert main(["report", "--results", str(tmp_path / "results")]) == 1
    assert "wholedoc-bench score --results" in capsys.readouterr().out


def test_hidden_text_in_the_output_does_not_change_the_ranking() -> None:
    revealing = score_corpus([(memo(ACCOUNT, NAME), Prediction(markdown=f"{VISIBLE} Jane Roe"))])
    clean = score_corpus([(memo(ACCOUNT, NAME), Prediction(markdown=VISIBLE))])
    summaries = [
        summarize("revealing", replace(revealing, overall=0.9), "v"),
        summarize("clean", replace(clean, overall=0.1), "v"),
    ]
    assert [s.parser for s in ranked(summaries)] == ["revealing", "clean"]


def _set(results: Path, name: str, **changes: object) -> None:
    path = results / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, **changes}), encoding="utf-8")


@pytest.mark.parametrize(
    ("old", "named"),
    [
        (("a", "b", "c"), "  a: spec 0.4\n  b: spec 0.4\n  c: spec 0.4"),
        (("a", "b", "c", "d"), "d: spec"),
    ],
    ids=["three old, one current", "all old"],
)
def test_report_names_every_summary_from_another_spec_than_the_scorer(
    tmp_path: Path,
    corpus: Path,
    capsys: pytest.CaptureFixture[str],
    old: tuple[str, ...],
    named: str,
) -> None:
    results = tmp_path / "results"
    for name in ("a", "b", "c", "d"):
        score(corpus, parser_folder(tmp_path, name, Prediction(markdown="x\n")), results)
    for name in old:
        _set(results, name, spec_version="0.4")
    capsys.readouterr()

    assert main(["report", "--results", str(results)]) == 2
    error = capsys.readouterr().err
    assert named in error
    assert f"this scorer: spec {SPEC_VERSION}" in error
    if "d" not in old:
        assert "d:" not in error


@pytest.mark.parametrize(
    ("name", "content", "named"),
    [
        ("b", {"spec_version": "999.0", "added_later": 1}, "b: spec 999.0"),
        ("b", {"spec_version": "0.4", "parsed": None}, "b: spec 0.4"),
        ("notes", {"todo": "rerun docling"}, "notes.json: not a parser summary"),
        ("b", {"parsed": None}, "b.json: not a parser summary"),
    ],
    ids=["newer with a new field", "older without a field", "another file", "a field missing"],
)
def test_report_refuses_a_file_it_cannot_read_as_a_summary_without_a_traceback(
    tmp_path: Path,
    corpus: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    content: dict[str, object],
    named: str,
) -> None:
    results = tmp_path / "results"
    for parser in ("a", "b"):
        score(corpus, parser_folder(tmp_path, parser, Prediction(markdown="x\n")), results)
    path = results / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data = {**data, **content}
    data = {key: value for key, value in data.items() if key not in content or value is not None}
    path.write_text(json.dumps(data), encoding="utf-8")
    capsys.readouterr()

    assert main(["report", "--results", str(results)]) == 2
    assert named in capsys.readouterr().err


def test_report_refuses_summaries_scored_on_corpora_of_different_sizes(
    tmp_path: Path, corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    for name in ("a", "b", "c"):
        score(corpus, parser_folder(tmp_path, name, Prediction(markdown="x\n")), results)
    _set(results, "a", documents=4, parsed=4)
    _set(results, "c", documents=4, parsed=4)
    capsys.readouterr()

    assert main(["report", "--results", str(results)]) == 2
    error = capsys.readouterr().err
    assert "  a: 4 documents\n  b: 1 document\n  c: 4 documents" in error
    assert not (tmp_path / "LEADERBOARD.md").exists()


def test_legacy_summary_without_provenance_remains_readable(tmp_path: Path) -> None:
    data = asdict(summarize("old", score_corpus([(truth(), Prediction(markdown=""))]), "v1"))
    del data["provider"]
    del data["provider_config"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(data))
    summary = read_summary(path)
    assert summary.provider is None
    assert summary.provider_config is None
