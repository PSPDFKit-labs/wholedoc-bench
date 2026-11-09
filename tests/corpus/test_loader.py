import json
import shutil
from pathlib import Path

import pytest

from tests.conftest import DEMO
from wholedoc_bench.corpus.checks import CorpusError
from wholedoc_bench.corpus.loader import SchemaVersionError, load_corpus, load_groundtruth


def test_other_schema_version_is_rejected(tmp_path: Path) -> None:
    data = json.loads((DEMO / "groundtruth.json").read_text(encoding="utf-8"))
    data["schema_version"] = "0.0"
    path = tmp_path / "groundtruth.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SchemaVersionError):
        load_groundtruth(path)


def test_duplicate_document_ids_are_rejected(tmp_path: Path) -> None:
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        shutil.copy(DEMO / "groundtruth.json", tmp_path / folder / "groundtruth.json")
    with pytest.raises(ValueError, match="used twice: .*a and .*b"):
        load_corpus(tmp_path)


def test_loader_runs_the_corpus_checks(tmp_path: Path) -> None:
    data = json.loads((DEMO / "groundtruth.json").read_text(encoding="utf-8"))
    data["planted"] = [{"kind": "exact", "printed": "Intake"}]
    path = tmp_path / "groundtruth.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(
        CorpusError, match=r"groundtruth\.json: .*'Intake' is not printed in body text"
    ):
        load_groundtruth(path)
