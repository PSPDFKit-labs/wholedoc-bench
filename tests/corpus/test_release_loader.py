import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from tests.release_helpers import copy_release, edit_key, edit_rows, resum, rows
from tests.test_release_execution import calls, run, score  # noqa: F401  (the calls fixture)
from wholedoc_bench.cli import main
from wholedoc_bench.corpus.loader import load_corpus, load_corpus_context
from wholedoc_bench.corpus.release import COLUMNS, DOCUMENT_INDEX, release_schema
from wholedoc_bench.provenance import CorpusIdentity


@pytest.fixture
def release(tmp_path: Path) -> Path:
    return copy_release(tmp_path / "release")


def test_a_release_loads_with_its_rows_and_identity(release: Path) -> None:
    loaded = load_corpus_context(release)
    assert [d.truth.document.id for d in loaded.documents] == ["np-00001", "np-00002"]
    assert loaded.metric_support == {
        row["id"]: frozenset(row["supported_metrics"]) for row in rows(release)
    }
    assert loaded.documents[0].source == (release / "documents/np-00001.pdf").resolve()
    assert all(d.cloud_allowed for d in loaded.documents)
    assert loaded.identity
    assert [d["key_sha256"] for d in loaded.identity["documents"]] == [
        row["groundtruth_sha256"] for row in rows(release)
    ]
    info = (release / "release.json").read_bytes()
    assert loaded.identity["manifest_sha256"] == json.loads(info)["manifest_sha256"]
    assert loaded.identity["release"] == {
        "tag": "v1.0",
        "source_commit": json.loads(info)["source_commit"],
        "release_sha256": hashlib.sha256(info).hexdigest(),
        "sums_sha256": hashlib.sha256((release / "SHA256SUMS").read_bytes()).hexdigest(),
    }
    assert CorpusIdentity.model_validate(loaded.identity).model_dump() == loaded.identity
    working = {k: v for k, v in loaded.identity.items() if k != "release"}
    assert "release" not in CorpusIdentity.model_validate(working).model_dump()


def test_release_options(release: Path) -> None:
    assert [
        d.truth.document.id
        for d in load_corpus_context(release, exclude=frozenset({"np-00002"})).documents
    ] == ["np-00001"]
    with pytest.raises(ValueError, match="accepted documents only"):
        load_corpus_context(release, include_drafts=True)
    with pytest.raises(ValueError, match="unknown excluded"):
        load_corpus_context(release, exclude=frozenset({"np-09999"}))
    with pytest.raises(ValueError, match="no documents selected"):
        load_corpus_context(release, exclude=frozenset({"np-00001", "np-00002"}))
    with pytest.raises(ValueError, match="releases require load_corpus_context"):
        load_corpus(release)


def _extra_key(release: Path) -> None:
    (release / "groundtruth/np-09999.json").write_text("{}")
    resum(release)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda r: (r / "documents/np-00001.pdf").write_bytes(b"%PDF-1.4 changed"), "differs from"),
        (lambda r: (r / "documents/extra.pdf").write_bytes(b"x"), "not listed in SHA256SUMS"),
        (lambda r: (r / "SHA256SUMS").unlink(), "SHA256SUMS"),
        (
            lambda r: edit_rows(r, lambda rows: rows[0].update(supported_metrics=["T9.nothing"])),
            "unknown supported metric",
        ),
        (
            lambda r: edit_rows(r, lambda rows: rows[0].update(tracks=["T1", "T4"])),
            "tracks must be",
        ),
        (lambda r: edit_rows(r, lambda rows: rows[0].pop("licence")), "licence"),
        (lambda r: edit_rows(r, lambda rows: rows[0].update(extra=1)), "extra"),
        (lambda r: edit_rows(r, lambda rows: rows.append(rows[0])), "listed twice"),
        (lambda r: edit_rows(r, lambda rows: rows.pop()), "says 2 documents"),
        (
            lambda r: edit_key(r, lambda t: t["document"].update(id="np-00002")),
            "must match the row",
        ),
        (lambda r: edit_key(r, lambda t: t.update(schema_version="0.0")), "schema_version"),
        (
            lambda r: edit_rows(
                r, lambda rows: rows[0].update(document_sha256=rows[1]["document_sha256"])
            ),
            "document's SHA-256 differs from the row",
        ),
        (lambda r: (r / "groundtruth/np-00001.json").write_text("{}"), "differs from SHA256SUMS"),
        (lambda r: (r / "groundtruth/np-09999.json").write_text("{}"), "not listed in SHA256SUMS"),
        (_extra_key, f"groundtruth/np-09999.json: no row in {DOCUMENT_INDEX} names this file"),
        (
            lambda r: edit_rows(r, lambda rows: rows[0].update(groundtruth_path="keys/x.json")),
            "groundtruth_path must be groundtruth/np-00001.json",
        ),
        (
            lambda r: edit_rows(
                r, lambda rows: rows[0].update(groundtruth_sha256=rows[1]["groundtruth_sha256"])
            ),
            "key's SHA-256 differs from the row",
        ),
        (
            lambda r: edit_rows(r, lambda rows: rows[0].pop("groundtruth_sha256")),
            "groundtruth_sha256",
        ),
    ],
)
def test_a_damaged_release_is_refused_whole(release: Path, damage: Any, message: str) -> None:
    damage(release)
    with pytest.raises(ValueError, match=message):
        load_corpus_context(release)


def unlist(release: Path, name: str) -> None:
    sums = release / "SHA256SUMS"
    lines = [line for line in sums.read_text().split("\n") if line and not line.endswith(name)]
    sums.write_text("".join(f"{line}\n" for line in lines), newline="\n")


def test_every_release_file_must_be_listed(release: Path) -> None:
    unlist(release, "README.md")
    (release / "README.md").write_text("edited card")
    with pytest.raises(ValueError, match="README.md: not listed in SHA256SUMS"):
        load_corpus_context(release)


def test_the_ledger_must_be_present_and_list_every_row(release: Path) -> None:
    unlist(release, "published.jsonl")
    (release / "published.jsonl").unlink()
    with pytest.raises(ValueError, match="published.jsonl: not listed"):
        load_corpus_context(release)
    (release / "published.jsonl").write_text('{"id":"np-00001","first_release":"v1.0"}\n')
    resum(release)
    with pytest.raises(ValueError, match="doesn't list released ids: np-00002"):
        load_corpus_context(release)


def test_the_published_row_schema_is_current_and_matches_released_rows(release: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    (name, expected), *others = release_schema().items()
    assert not others
    schema = json.loads((root / "schemas" / name).read_bytes())
    assert schema == {"$schema": "https://json-schema.org/draft/2020-12/schema", **expected}
    assert list(schema["properties"]) == list(COLUMNS)
    assert set(schema["required"]) == set(COLUMNS)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    found = rows(release)
    assert len(found) == 2
    for row in found:
        validator.validate(row)
    row = found[0]
    without_licence = {k: v for k, v in row.items() if k != "licence"}
    for bad in ({**row, "extra": 1}, {**row, "pages": 0}, {**row, "scan": "no"}, without_licence):
        assert not validator.is_valid(bad)


def test_line_and_paragraph_separators_in_rows_and_keys_survive_loading(release: Path) -> None:
    edit_rows(release, lambda found: found[0].update(limits="first part second part"))
    edit_key(
        release,
        lambda truth: truth["elements"][1].update(text="The library opens. Second paragraph"),
    )
    loaded = load_corpus_context(release)
    assert loaded.documents[0].truth.elements[1].text.endswith(" Second paragraph")
    assert rows(release)[0]["limits"] == "first part second part"


def _as_plain_folder(release: Path, destination: Path) -> Path:
    for row in rows(release):
        folder = destination / row["id"]
        folder.mkdir(parents=True)
        shutil.copyfile(release / row["document_path"], folder / f"document.{row['format']}")
        shutil.copyfile(release / row["groundtruth_path"], folder / "groundtruth.json")
    return destination


def test_scoring_a_release_records_it(
    release: Path,
    tmp_path: Path,
    calls: list[Path],  # noqa: F811
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run(release, tmp_path / "out") == 0
    assert score(release, tmp_path / "out", tmp_path / "results") == 0
    summary = json.loads((tmp_path / "results/out.json").read_text("utf-8"))
    assert summary["comparison"]["corpus"]["release"]["tag"] == "v1.0"
    plain = _as_plain_folder(release, tmp_path / "plain")
    assert run(plain, tmp_path / "plain-out") == 0
    assert score(plain, tmp_path / "plain-out", tmp_path / "results") == 0
    capsys.readouterr()
    assert main(["report", "--results", str(tmp_path / "results")]) == 2
    assert "comparison identities" in capsys.readouterr().err
