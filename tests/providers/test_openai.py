import json
from pathlib import Path

import pytest

from tests.conftest import CHARTS_OPENAI, TWO_COLUMN_OPENAI
from wholedoc_bench.providers import OpenAIProvider, RawResult
from wholedoc_bench.schema import Channel, ChartKind, ElementClass


def real(path: Path, document_id: str) -> RawResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RawResult(provider="openai", document_id=document_id, payload=payload)


def test_a_real_answer_reads_into_classes_links_and_charts() -> None:
    prediction = OpenAIProvider().normalize(real(CHARTS_OPENAI, "charts_report_pdf"))
    found = {item.id: item for item in prediction.layout}
    assert [(i.page, i.cls) for i in prediction.layout if i.channel is Channel.BODY] == [
        (0, ElementClass.HEADING),
        (0, ElementClass.PARAGRAPH),
        (0, ElementClass.HEADING),
        (0, ElementClass.CHART),
        (0, ElementClass.CAPTION),
        (0, ElementClass.CAPTION),
        (0, ElementClass.TABLE),
        (1, ElementClass.HEADING),
        (1, ElementClass.PARAGRAPH),
        (1, ElementClass.CHART),
        (1, ElementClass.CAPTION),
        (1, ElementClass.PARAGRAPH),
    ]
    charts = [i.chart for i in prediction.layout if i.chart is not None]
    assert [(c.kind, c.title) for c in charts] == [
        (ChartKind.BAR, "Revenue by region (EUR m)"),
        (ChartKind.LINE, "Energy cost per tonne (EUR)"),
    ]
    assert [(found[r.source].text, found[r.target].cls) for r in prediction.relations] == [
        ("Figure 1: Revenue by region, 2025 and 2026.", ElementClass.CHART),
        ("Table 1: Revenue by region (EUR m)", ElementClass.TABLE),
        ("Figure 2: Monthly energy cost per tonne moved.", ElementClass.CHART),
    ]
    assert "Harbour Energy" not in prediction.markdown
    assert prediction.page_offsets is not None
    assert prediction.markdown[prediction.page_offsets[1] :].startswith("## Energy costs")
    assert all(i.box is not None for i in prediction.layout)
    assert prediction.unscored_text == []


def test_usage_counts_cached_and_written_input_apart() -> None:
    usage = OpenAIProvider().usage(real(TWO_COLUMN_OPENAI, "report_two_column_pdf"))
    assert usage == {
        "uncached_input_tokens": 9,
        "cached_tokens": 0,
        "cache_write_tokens": 7800,
        "output_tokens": 725 + 641 + 396,
    }


def test_a_page_without_usage_has_no_usage() -> None:
    raw = RawResult("openai", "d", {"pages": [{"usage": None}]})
    with pytest.raises(ValueError):
        OpenAIProvider().usage(raw)
