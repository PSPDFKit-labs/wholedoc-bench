import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.builders import CHART_BOX, chart_element, groundtruth
from tests.conftest import DEMO
from wholedoc_bench.providers import (
    DoclingProvider,
    MarkitdownProvider,
    PdftotextProvider,
    RawResult,
)
from wholedoc_bench.schema import ChartKind
from wholedoc_bench.scorer.run import score_document


def test_a_markdown_provider_keeps_the_markdown_as_it_came() -> None:
    raw = RawResult(provider="markitdown", document_id="d", payload="# Title\n\nText.\n")
    assert MarkitdownProvider().normalize(raw).markdown == "# Title\n\nText.\n"
    assert MarkitdownProvider().normalize(raw).layout == []


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="needs poppler's pdftotext")
def test_pdftotext_fetch_gives_the_page_text() -> None:
    raw = PdftotextProvider().fetch(DEMO / "document.pdf")
    assert raw.document_id == "plain_continuation_demo"
    assert "Page 3 begins with this fragment." in raw.payload


def _install_fake_pdftotext(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, version_stream: str | None
) -> None:
    print_version = (
        f"print('pdftotext version 4.04', file=sys.{version_stream})" if version_stream else "pass"
    )
    script = tmp_path / "pdftotext.py"
    script.write_text(
        "import sys\n"
        "if sys.argv[1] == '-v':\n"
        f"    {print_version}\n"
        "else:\n"
        "    sys.stdout.buffer.write('José Müller\\f'.encode('utf-8'))\n",
        encoding="utf-8",
    )
    real_run = subprocess.run

    def run_with_a_cp1252_default(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        if kwargs.pop("text", False):
            kwargs.setdefault("encoding", "cp1252")
        return real_run([sys.executable, str(script), *argv[1:]], **kwargs)

    monkeypatch.setattr(shutil, "which", lambda name: name)
    monkeypatch.setattr(subprocess, "run", run_with_a_cp1252_default)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_pdftotext_version_is_read_from_the_stream_that_has_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stream: str
) -> None:
    _install_fake_pdftotext(monkeypatch, tmp_path, stream)
    raw = PdftotextProvider().fetch(tmp_path / "doc" / "document.pdf")
    assert raw.metadata["version"] == "pdftotext version 4.04"


def test_pdftotext_without_a_version_still_parses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_pdftotext(monkeypatch, tmp_path, None)
    raw = PdftotextProvider().fetch(tmp_path / "doc" / "document.pdf")
    assert raw.payload == "José Müller\n"
    assert raw.metadata["version"] == "pdftotext (version unknown)"


@pytest.mark.parametrize(
    ("config", "effective"),
    [
        ({}, {"chart_extraction": True}),
        ({"chart_extraction": True}, {"chart_extraction": True}),
        ({"chart_extraction": False}, {"chart_extraction": False}),
    ],
)
def test_docling_extracts_charts_by_default_and_always_names_the_setting(
    config: dict[str, Any], effective: dict[str, Any]
) -> None:
    assert DoclingProvider.validate_config(config) == effective


@pytest.mark.parametrize(
    ("config", "error"),
    [
        ({"chart_extraction": "yes"}, "true or false"),
        ({"chart_extraction": 1}, "true or false"),
        ({"charts": True}, "unsupported config keys: charts"),
    ],
)
def test_docling_rejects_a_bad_config(config: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        DoclingProvider(config=config)


@pytest.mark.parametrize(("config", "charts"), [({}, True), ({"chart_extraction": False}, False)])
def test_docling_asks_for_charts_unless_switched_off_and_writes_no_picture_classes(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any], charts: bool
) -> None:
    converter_module = pytest.importorskip("docling.document_converter")
    from docling.datamodel.base_models import InputFormat

    seen: dict[str, Any] = {}

    class Document:
        def export_to_markdown(self, **options: Any) -> str:
            seen["export"] = options
            return "Text.\n"

    class Converter:
        def __init__(self, format_options: dict[Any, Any] | None = None) -> None:
            seen["format_options"] = format_options
            seen["built"] = seen.get("built", 0) + 1

        def convert(self, source: str) -> SimpleNamespace:
            seen["converted"] = [*seen.get("converted", []), source]
            return SimpleNamespace(document=Document())

    monkeypatch.setattr(converter_module, "DocumentConverter", Converter)
    provider = DoclingProvider(config=config)
    raw = provider.fetch(Path("doc") / "document.pdf")
    provider.fetch(Path("other") / "document.pdf")
    assert raw.payload == "Text.\n"
    assert seen["built"] == 1 and len(seen["converted"]) == 2
    assert seen["export"] == {"include_picture_classification": False}
    if charts:
        options = seen["format_options"][InputFormat.PDF].pipeline_options
        assert options.do_chart_extraction is True
    else:
        assert seen["format_options"] is None


def test_a_docling_chart_with_its_data_scores_t6_and_its_class_stays_out() -> None:
    pytest.importorskip("docling_core")
    from docling_core.types.doc import DoclingDocument, TableCell, TableData
    from docling_core.types.doc.items.picture.meta import (
        PictureClassificationMetaField,
        PictureClassificationPrediction,
        PictureMeta,
        TabularChartMetaField,
    )

    rows = [["Region", "Sales"], ["North", "10"], ["South", "20"]]
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=r == 0,
        )
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
    ]
    document = DoclingDocument(name="charts")
    picture = document.add_picture()
    picture.meta = PictureMeta(
        classification=PictureClassificationMetaField(
            predictions=[PictureClassificationPrediction(class_name="bar_chart", confidence=0.9)]
        ),
        tabular_chart=TabularChartMetaField(
            chart_data=TableData(num_rows=3, num_cols=2, table_cells=cells)
        ),
    )
    assert "Bar chart" in document.export_to_markdown()
    markdown = document.export_to_markdown(include_picture_classification=False)
    assert "Bar chart" not in markdown

    truth = groundtruth(
        [
            chart_element(
                "c1",
                kind=ChartKind.BAR,
                box=CHART_BOX,
                series={"Sales": [("North", 10), ("South", 20)]},
            )
        ]
    )
    result = score_document(truth, DoclingProvider().normalize(raw_markdown(markdown)))
    plain = DoclingProvider().normalize(raw_markdown("<!-- image -->\n"))
    assert score_document(truth, plain).sub_scores["T6"]["data_points"] == 0.0
    assert result.sub_scores["T6"]["data_points"] == 1.0
    assert result.sub_scores["T6"]["series"] == 1.0


def raw_markdown(markdown: str) -> RawResult:
    return RawResult(provider="docling", document_id="charts", payload=markdown)
