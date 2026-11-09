from pathlib import Path

import pytest

from tests.test_release_execution import calls, release, run, score  # noqa: F401  (fixtures)
from wholedoc_bench.cli import RUN_MANIFEST, main
from wholedoc_bench.provenance import Attempt, ManifestRun
from wholedoc_bench.providers import MarkitdownProvider, RawResult
from wholedoc_bench.report import Gate, Summary, check_rankable, read_summary
from wholedoc_bench.scorer.run import SPEC_VERSION


def attempts(out: Path) -> dict[str, Attempt]:
    return ManifestRun.model_validate_json((out / RUN_MANIFEST).read_bytes()).attempts


def report(results: Path) -> int:
    return main(["report", "--results", str(results)])


def broken_fetch(self: MarkitdownProvider, path: Path) -> RawResult:
    raise ConnectionError("service unavailable")


def broken_normalize(self: MarkitdownProvider, raw: RawResult):
    raise KeyError("blocks")


def test_a_fetch_failure_is_recorded_and_still_counts_as_a_parse_failure(
    release: Path,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MarkitdownProvider, "fetch", broken_fetch)
    out, results = tmp_path / "parser", tmp_path / "results"
    assert run(release, out) == 0
    found = attempts(out)
    assert {a.stage for a in found.values()} == {"fetch"}
    assert {a.response for a in found.values()} == {None}
    assert score(release, out, results) == 0
    summary = read_summary(results / "parser.json")
    assert summary.failure_stages == {"fetch": 2} and summary.parsed == 0
    assert report(results) == 0


def test_an_adapter_failure_blocks_ranking_until_replayed(
    release: Path,  # noqa: F811
    tmp_path: Path,
    calls: list[Path],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "parser"
    original = MarkitdownProvider.normalize
    monkeypatch.setattr(MarkitdownProvider, "normalize", broken_normalize)
    assert run(release, out) == 0
    found = attempts(out)
    assert {a.stage for a in found.values()} == {"normalize"}
    assert {a.response for a in found.values()} == {"new"}
    assert all(a.raw_sha256 for a in found.values())
    assert score(release, out, tmp_path / "blocked") == 0
    assert read_summary(tmp_path / "blocked/parser.json").failure_stages == {"normalize": 2}
    assert report(tmp_path / "blocked") == 2
    assert "adapter or unclassified failures" in capsys.readouterr().err

    monkeypatch.setattr(MarkitdownProvider, "normalize", original)
    assert run(release, out, "--from-raw") == 0
    found = attempts(out)
    assert {a.status for a in found.values()} == {"complete"}
    assert {a.response for a in found.values()} == {"replayed"}
    assert {a.stage for a in found.values()} == {None}
    assert score(release, out, tmp_path / "fixed") == 0
    assert read_summary(tmp_path / "fixed/parser.json").failure_stages == {}
    assert report(tmp_path / "fixed") == 0


def test_a_failed_refetch_keeps_the_earlier_response_and_says_so(
    release: Path,  # noqa: F811
    tmp_path: Path,
    calls: list[Path],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "parser"
    assert run(release, out) == 0
    earlier = {i: a.raw_sha256 for i, a in attempts(out).items()}
    assert {a.response for a in attempts(out).values()} == {"new"}
    monkeypatch.setattr(MarkitdownProvider, "fetch", broken_fetch)
    assert run(release, out) == 0
    found = attempts(out)
    assert {a.stage for a in found.values()} == {"fetch"}
    assert {a.response for a in found.values()} == {"retained"}
    assert {i: a.raw_sha256 for i, a in found.items()} == earlier


def test_an_unreadable_format_fails_before_the_parser_is_asked(
    release: Path,  # noqa: F811
    tmp_path: Path,
    calls: list[Path],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MarkitdownProvider, "formats", frozenset({".docx"}))
    out = tmp_path / "parser"
    assert run(release, out) == 0
    assert calls == []
    assert {a.stage for a in attempts(out).values()} == {"input"}


def summary_with(*failed: dict) -> Summary:
    found = {f"np-0000{n}": attempt for n, attempt in enumerate(failed, 1)}
    return Summary(
        parser="p",
        parser_version="1",
        spec_version=SPEC_VERSION,
        documents=len(found),
        parsed=0,
        track_scores={},
        overall=None,
        coverage=Gate(0.0, False),
        evaluation={"attempts": found},
    )


def test_unclassified_failures_from_older_runs_are_not_ranked() -> None:
    check_rankable([summary_with({"status": "failed", "error": "x", "stage": "fetch"})])
    with pytest.raises(ValueError, match="p: 1 unclassified"):
        check_rankable([summary_with({"status": "failed", "error": "x"})])
    legacy = summary_with()
    check_rankable([Summary(**{**legacy.__dict__, "evaluation": None})])


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"status": "complete", "prediction_sha256": "0" * 64, "stage": "fetch"}, "failure stage"),
        ({"status": "failed", "error": "x", "response": "new"}, "response origin"),
    ],
)
def test_attempt_fields_must_agree(fields: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Attempt.model_validate(fields)
