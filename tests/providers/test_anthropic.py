import json
from pathlib import Path

import pytest

from tests.conftest import CHARTS_ANTHROPIC, TWO_COLUMN_ANTHROPIC
from wholedoc_bench.providers import AnthropicProvider, RawResult
from wholedoc_bench.schema import Channel, ChartKind, ElementClass


def real(path: Path, document_id: str) -> RawResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RawResult(provider="anthropic", document_id=document_id, payload=payload)


def test_a_real_answer_reads_into_classes_links_and_charts() -> None:
    prediction = AnthropicProvider().normalize(real(CHARTS_ANTHROPIC, "charts_report_pdf"))
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


def test_usage_sums_every_pages_tokens() -> None:
    usage = AnthropicProvider().usage(real(TWO_COLUMN_ANTHROPIC, "report_two_column_pdf"))
    assert usage == {
        "input_tokens": 9177,
        "output_tokens": 2138,
        "cache_read_input_tokens": 0,
        "ephemeral_5m_input_tokens": 0,
        "ephemeral_1h_input_tokens": 0,
    }


def test_usage_counts_a_cache_write_without_its_split_as_five_minutes() -> None:
    page = {"usage": {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 7}}
    usage = AnthropicProvider().usage(RawResult("anthropic", "d", {"pages": [page]}))
    assert (usage["ephemeral_5m_input_tokens"], usage["ephemeral_1h_input_tokens"]) == (7, 0)


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"input_tokens": 10, "output_tokens": 5, "speed": "fast"},
        {"input_tokens": 10, "output_tokens": 5, "inference_geo": "us"},
        {"input_tokens": 10, "output_tokens": 5, "service_tier": "priority"},
    ],
    ids=["no usage", "fast mode", "US-only inference", "priority tier"],
)
def test_a_page_the_list_rates_do_not_price_has_no_usage(usage: dict | None) -> None:
    raw = RawResult("anthropic", "d", {"pages": [{"usage": usage}]})
    with pytest.raises(ValueError):
        AnthropicProvider().usage(raw)
