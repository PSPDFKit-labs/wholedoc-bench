import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest

from wholedoc_bench.cli import main
from wholedoc_bench.execution import RUN_MANIFEST
from wholedoc_bench.providers import PROVIDERS, CredentialError, LlamaParseProvider, RawResult
from wholedoc_bench.providers.base import load_raw, save_raw
from wholedoc_bench.schema import Channel, ElementClass

RESULT = {
    "job": {"id": "job-1", "status": "COMPLETED"},
    "markdown": {"pages": [{"page_number": 1, "success": True, "markdown": "# Report\n\nHello"}]},
    "items": {
        "pages": [
            {
                "page_number": 1,
                "success": True,
                "page_width": 100,
                "page_height": 200,
                "items": [
                    {
                        "type": "heading",
                        "value": "Report",
                        "md": "# Report",
                        "bbox": [
                            {"x": 10, "y": 20, "w": 50, "h": 10},
                        ],
                    },
                    {"type": "text", "value": "Hello", "md": "Hello"},
                ],
            }
        ]
    },
}


def raw(payload: dict[str, Any]) -> RawResult:
    return RawResult("llamaparse", "test", payload)


def answering(*responses: httpx.Response) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    requests = []
    left = iter(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return next(left)

    return httpx.MockTransport(handle), requests


def setup_responses() -> list[httpx.Response]:
    return [
        httpx.Response(200, json={"id": "file-1"}),
        httpx.Response(200, json={"id": "job-1", "status": "PENDING"}),
    ]


@pytest.fixture
def document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", " secret-do-not-store ")
    path = tmp_path / "test" / "document.pdf"
    path.parent.mkdir()
    path.write_bytes(b"%PDF-1.7 test")
    return path


def test_fetch_uploads_creates_polls_and_records_config(document: Path, tmp_path: Path) -> None:
    transport, requests = answering(
        *setup_responses(),
        httpx.Response(200, json={"job": {"status": "RUNNING"}}),
        httpx.Response(200, json=RESULT),
    )
    sleeps = []
    provider = LlamaParseProvider(transport=transport, sleep=sleeps.append)
    result = provider.fetch(document)
    assert [r.method for r in requests] == ["POST", "POST", "GET", "GET"]
    assert all(r.url.host == "api.cloud.llamaindex.ai" for r in requests)
    assert all(r.headers["authorization"] == "Bearer secret-do-not-store" for r in requests)
    assert requests[0].url.path == "/api/v1/beta/files"
    assert b"%PDF-1.7 test" in requests[0].read()
    assert b'name="purpose"\r\n\r\nparse' in requests[0].read()
    body = json.loads(requests[1].read())
    assert body["file_id"] == "file-1"
    assert body["tier"] == "agentic"
    assert body["version"] == "2026-09-28"
    assert body["disable_cache"] is True
    assert "processing_options" not in body
    assert "specialized_chart_parsing" not in body
    assert body["output_options"]["markdown"]["tables"] == {
        "output_tables_as_markdown": False,
        "merge_continued_tables": True,
    }
    assert requests[2].url.params.get_list("expand") == ["markdown", "items"]
    assert sleeps == [2]
    assert result.payload == RESULT
    assert result.metadata["config"] == provider.config
    assert result.metadata["version"] == "llamaparse v2 agentic 2026-09-28 merged-tables"
    saved = tmp_path / "raw.json"
    save_raw(result, saved)
    assert "secret-do-not-store" not in saved.read_text()
    assert load_raw(saved) == result


@pytest.mark.parametrize(
    ("preset", "tier", "mode"),
    [
        ("cost-effective", "cost_effective", "efficient"),
        ("agentic", "agentic", "agentic"),
        ("agentic-plus", "agentic_plus", "agentic_plus"),
    ],
)
def test_chart_presets_send_nested_option_and_record_provenance(
    document: Path, tmp_path: Path, preset: str, tier: str, mode: str
):
    config = json.loads(Path(f"configs/llamaparse/{preset}.json").read_text())
    expected = {"tier": tier, "version": "2026-09-28", "specialized_chart_parsing": mode}
    assert config == expected
    transport, requests = answering(*setup_responses(), httpx.Response(200, json=RESULT))
    result = LlamaParseProvider(config, transport=transport).fetch(document)
    body = json.loads(requests[1].read())
    assert body["tier"] == tier
    assert body["processing_options"] == {"specialized_chart_parsing": mode}
    assert "specialized_chart_parsing" not in body
    assert result.metadata["config"] == expected
    assert (
        result.metadata["version"] == f"llamaparse v2 {tier} 2026-09-28 merged-tables charts={mode}"
    )
    saved = tmp_path / "raw.json"
    save_raw(result, saved)
    assert load_raw(saved).metadata == result.metadata


@pytest.mark.parametrize("value", [None, "", " \n"])
def test_missing_credentials(document: Path, monkeypatch: pytest.MonkeyPatch, value: str | None):
    if value is None:
        monkeypatch.delenv("LLAMA_CLOUD_API_KEY")
    else:
        monkeypatch.setenv("LLAMA_CLOUD_API_KEY", value)
    provider = LlamaParseProvider()
    for action in (provider.prepare, lambda: provider.fetch(document)):
        with pytest.raises(CredentialError, match="LLAMA_CLOUD_API_KEY"):
            action()


@pytest.mark.parametrize(
    "config",
    [
        {"api_key": "secret"},
        {"base_url": "secret"},
        {"tier": "invalid"},
        {"version": ""},
        {"version": "2026-02-30"},
        {"version": 1},
        {"tier": "fast", "version": "2026-06-14"},
        {"tier": "fast", "version": "2026-09-28"},
        *[{"specialized_chart_parsing": value} for value in (None, True, 1, [], {}, "secret")],
        *[
            {"tier": "fast", "specialized_chart_parsing": mode}
            for mode in ("efficient", "agentic", "agentic_plus")
        ],
    ],
)
def test_invalid_config(config: dict[str, Any]):
    with pytest.raises(ValueError) as error:
        LlamaParseProvider(config)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    ("tier", "version"),
    [
        ("fast", "2026-06-15"),
        ("cost_effective", "2026-09-28"),
        ("agentic", "2026-09-28"),
        ("agentic_plus", "2026-09-28"),
    ],
)
def test_tiers(tier: str, version: str):
    assert LlamaParseProvider({"tier": tier}).config == {"tier": tier, "version": version}


def test_fast_accepts_latest():
    config = {"tier": "fast", "version": "latest"}
    assert LlamaParseProvider(config).config == config


@pytest.mark.parametrize("status", [401, 403, 429, 500, 307])
def test_upload_errors_are_sanitized_and_not_retried(document: Path, status: int):
    transport, requests = answering(
        httpx.Response(
            status, text="secret-do-not-store", headers={"Location": "https://attacker.example"}
        )
    )
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as error:
        LlamaParseProvider(transport=transport).fetch(document)
    assert "secret-do-not-store" not in str(error.value)
    assert len(requests) == 1


def test_transport_errors_are_sanitized(document: Path):
    def handle(request: httpx.Request):
        raise httpx.ConnectError("secret-do-not-store", request=request)

    with pytest.raises(RuntimeError, match="network connectivity") as error:
        LlamaParseProvider(transport=httpx.MockTransport(handle)).fetch(document)
    assert "secret-do-not-store" not in str(error.value)


@pytest.mark.parametrize("status", ["FAILED", "CANCELLED", "secret-do-not-store"])
def test_job_errors_are_sanitized(document: Path, status: str):
    transport, _ = answering(
        *setup_responses(),
        httpx.Response(
            200,
            json={
                "job": {"status": status, "error_message": "secret-do-not-store"},
            },
        ),
    )
    with pytest.raises(RuntimeError) as error:
        LlamaParseProvider(transport=transport).fetch(document)
    assert "secret-do-not-store" not in str(error.value)


def test_poll_retry_honors_retry_after(document: Path):
    transport, requests = answering(
        *setup_responses(),
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(503),
        httpx.Response(200, json=RESULT),
    )
    waits = []
    assert (
        LlamaParseProvider(transport=transport, sleep=waits.append).fetch(document).payload
        == RESULT
    )
    assert waits == [7, 4]
    assert len(requests) == 5


def test_deadline_stops_running_job(document: Path):
    now = [0.0]
    transport, requests = answering(
        *setup_responses(),
        httpx.Response(200, json={"job": {"status": "RUNNING"}}),
        httpx.Response(200, json={"job": {"status": "RUNNING"}}),
    )

    def sleep(_: float):
        now[0] = 599.0

    with pytest.raises(RuntimeError, match="timed out"):
        LlamaParseProvider(transport=transport, sleep=sleep, clock=lambda: now[0]).fetch(document)
    assert len(requests) == 4


def test_native_layout_and_markdown_spans():
    prediction = LlamaParseProvider().normalize(raw(RESULT))
    assert prediction.markdown == "# Report\n\nHello"
    assert prediction.page_offsets == [0]
    heading, paragraph = prediction.layout
    assert heading.cls == ElementClass.HEADING
    assert heading.text == "Report"
    assert heading.page == 0
    assert heading.box == pytest.approx((0.1, 0.1, 0.5, 0.05))
    assert (heading.md_start, heading.md_end) == (0, 8)
    assert (paragraph.md_start, paragraph.md_end) == (10, 15)
    assert paragraph.box is None


def test_chart_parsing_keeps_native_table_claim():
    payload = deepcopy(RESULT)
    table = "| Year | Revenue |\n| --- | --- |\n| 2025 | 10 |"
    payload["markdown"]["pages"][0]["markdown"] = table
    payload["items"]["pages"][0]["items"] = [
        {"type": "table", "md": table, "rows": [["Year", "Revenue"], ["2025", "10"]]}
    ]
    prediction = LlamaParseProvider({"specialized_chart_parsing": "agentic"}).normalize(
        raw(payload)
    )
    assert prediction.markdown == table
    assert [item.cls for item in prediction.layout] == [ElementClass.TABLE]


def test_html_table_spans_come_from_the_item_html():
    payload = deepcopy(RESULT)
    html = "<table><tr><th>Item</th></tr><tr><td>Total</td></tr></table>"
    page = f"Intro\n\n{html}\n\nTotal\n\n<table><tr><td>Kept</td></tr></table>"
    payload["markdown"]["pages"][0]["markdown"] = page
    payload["items"]["pages"][0]["items"] = [
        {"type": "text", "value": "Intro", "md": "Intro"},
        {"type": "table", "md": "| Item |\n| --- |\n| Total |", "html": html},
        {"type": "text", "value": "Total", "md": "Total"},
        {
            "type": "table",
            "md": "| Lost |\n| --- |",
            "html": "<table><tr><td>Lost</td></tr></table>",
        },
    ]
    _, table, total, lost = LlamaParseProvider().normalize(raw(payload)).layout
    assert (table.md_start, table.md_end) == (7, 7 + len(html))
    assert (total.md_start, total.md_end) == (
        page.index("\n\nTotal") + 2,
        page.index("\n\nTotal") + 7,
    )
    assert (lost.md_start, lost.md_end) == (None, None)


def test_a_table_whose_html_keeps_colour_tags_is_found_without_them():
    payload = deepcopy(RESULT)
    written = "<table><tr><th>Bonds</th></tr><tr><td>Germany</td></tr></table>"
    page = f"Intro\n\n{written}\n\n<table><tr><td>Other</td></tr></table>"
    payload["markdown"]["pages"][0]["markdown"] = page
    coloured = (
        '<table><tr><th><span style="color:#f9a91b">Bonds</span></th></tr>'
        '<tr><td><FONT color="blue">Germany</FONT></td></tr></table>'
    )
    payload["items"]["pages"][0]["items"] = [
        {"type": "text", "value": "Intro", "md": "Intro"},
        {"type": "table", "md": "| Bonds |\n| --- |\n| Germany |", "html": coloured},
        {
            "type": "table",
            "md": "| Lost |",
            "html": '<table><tr><td><span style="color:red">Lost</span></td></tr></table>',
        },
    ]
    _, table, lost = LlamaParseProvider().normalize(raw(payload)).layout
    assert (table.md_start, table.md_end) == (7, 7 + len(written))
    assert (lost.md_start, lost.md_end) == (None, None)


def table(cell: str) -> str:
    return f"<table><tr><td>{cell}</td></tr></table>"


def spans(page: str, *htmls: str) -> list[tuple[int | None, int | None]]:
    payload = deepcopy(RESULT)
    payload["markdown"]["pages"][0]["markdown"] = page
    payload["items"]["pages"][0]["items"] = [
        {"type": "table", "md": f"| {n} |", "html": html} for n, html in enumerate(htmls)
    ]
    prediction = LlamaParseProvider().normalize(raw(payload))
    return [(item.md_start, item.md_end) for item in prediction.layout]


def at(page: str, written: str, after: int = 0) -> tuple[int, int]:
    start = page.index(written, after)
    return start, start + len(written)


def test_only_colour_markup_is_left_out_and_later_tables_keep_their_spans():
    kept = table('<span class="keep">X</span>')
    page = "\n\n".join((kept, table("Y"), table("X")))
    coloured = table('<span class="keep"><font color="red">X</font></span>')
    assert spans(page, coloured, table("Y"), table("X")) == [
        at(page, kept),
        at(page, table("Y")),
        at(page, table("X"), len(kept)),
    ]


def test_a_quoted_angle_bracket_does_not_cut_the_colour_markup_short():
    leftover = table('b" style="color:red">X')
    page = "\n\n".join((table("X"), leftover))
    coloured = table('<span title="a>b" style="color:red">X</span>')
    assert spans(page, coloured, leftover) == [at(page, table("X")), at(page, leftover)]


@pytest.mark.parametrize(
    "page",
    [
        f"<table><tr><td>{table('X')}</td></tr></table>",
        f"{table('X')}\n\n{table('X')}",
    ],
)
def test_a_coloured_table_with_no_single_whole_table_to_be_keeps_no_span(page):
    assert spans(page, table('<font color="red">X</font>')) == [(None, None)]


def test_a_coloured_table_does_not_take_a_table_another_item_has():
    page = table("X")
    assert spans(page, table("X"), table('<font color="red">X</font>')) == [
        at(page, table("X")),
        (None, None),
    ]


def merged_pages(*later: tuple[int, dict[str, Any]], holders: int = 1) -> dict[str, Any]:
    payload = deepcopy(RESULT)
    html = "<table><tr><td>A</td></tr><tr><td>B</td></tr><tr><td>C</td></tr></table>"
    head = {"type": "table", "md": "| A |", "html": html, "merged_from_pages": [1, 2, 3]}
    payload["markdown"]["pages"] = [
        {"page_number": 1, "markdown": html},
        {"page_number": 2, "markdown": "Two"},
        {"page_number": 3, "markdown": "Three"},
    ]
    payload["items"]["pages"] = [
        {"page_number": 1, "items": [head] * holders},
        *({"page_number": n, "items": [piece]} for n, piece in later),
    ]
    return payload


def later_piece() -> dict[str, Any]:
    return {"type": "table", "md": "", "html": "", "rows": 0, "merged_into_page": 1}


def test_pieces_of_a_table_merged_across_pages_are_linked_in_order():
    payload = merged_pages((2, later_piece()), (3, later_piece()))
    prediction = LlamaParseProvider().normalize(raw(payload))
    first, second, third = prediction.layout
    assert (first.page, second.page, third.page) == (0, 1, 2)
    assert [(r.kind.value, r.source, r.target) for r in prediction.relations] == [
        ("continues_in", first.id, second.id),
        ("continues_in", second.id, third.id),
    ]


@pytest.mark.parametrize(
    ("pieces", "holders"),
    [
        (((2, later_piece()),), 2),
        (((3, later_piece()),), 1),
    ],
)
def test_a_piece_whose_table_is_not_certain_is_not_linked(pieces, holders):
    prediction = LlamaParseProvider().normalize(raw(merged_pages(*pieces, holders=holders)))
    assert prediction.relations == []


def test_page_offsets_nested_lists_furniture_and_multiple_boxes():
    payload = deepcopy(RESULT)
    payload["markdown"]["pages"].append({"page_number": 2, "markdown": "- One\n- Two"})
    payload["items"]["pages"].append(
        {
            "page_number": 2,
            "page_width": 100,
            "page_height": 100,
            "items": [
                {"type": "header", "items": [{"type": "text", "value": "Page 2", "md": "Page 2"}]},
                {
                    "type": "list",
                    "items": [
                        {
                            "type": "text",
                            "value": "One",
                            "md": "- One",
                            "bbox": [
                                {"x": 10, "y": 10, "w": 10, "h": 5},
                                {"x": 20, "y": 15, "w": 10, "h": 5},
                            ],
                        },
                        {
                            "type": "list",
                            "items": [{"type": "text", "value": "Two", "md": "- Two"}],
                        },
                    ],
                },
            ],
        }
    )
    result = LlamaParseProvider().normalize(raw(payload))
    assert result.page_offsets == [0, 17]
    header, first, second = result.layout[2:]
    assert header.cls == ElementClass.HEADER
    assert header.channel == Channel.FURNITURE
    assert header.md_start is None
    assert first.cls == second.cls == ElementClass.LIST_ITEM
    assert first.box == pytest.approx((0.1, 0.1, 0.2, 0.1))
    assert (first.md_start, second.md_start) == (17, 23)
    assert first.page == second.page == 1


@pytest.mark.parametrize(
    "box",
    [
        None,
        [],
        [{"x": 1}],
        [{"x": -20, "y": 0, "w": 30, "h": 2}],
        [{"x": 0, "y": 0, "w": float("nan"), "h": 2}],
    ],
)
def test_bad_boxes_keep_text(box: Any):
    payload = deepcopy(RESULT)
    payload["items"]["pages"][0]["items"][0]["bbox"] = box
    heading = LlamaParseProvider().normalize(raw(payload)).layout[0]
    assert heading.box is None
    assert heading.text == "Report"


def test_rotated_box():
    payload = deepcopy(RESULT)
    payload["items"]["pages"][0]["items"][0]["bbox"] = [
        {"x": 20, "y": 20, "w": 20, "h": 10, "r": 90},
    ]
    assert LlamaParseProvider().normalize(raw(payload)).layout[0].box == pytest.approx(
        (0.25, 0.075, 0.1, 0.1)
    )


@pytest.mark.parametrize("field", ["markdown", "items"])
def test_failed_pages_are_not_silently_dropped(field: str):
    payload = deepcopy(RESULT)
    payload[field]["pages"][0] = {"page_number": 1, "success": False, "error": "secret"}
    with pytest.raises(ValueError, match="failed or invalid page") as error:
        LlamaParseProvider().normalize(raw(payload))
    assert "secret" not in str(error.value)


def test_markdown_only_replay_needs_no_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLAMA_CLOUD_API_KEY", raising=False)
    payload = deepcopy(RESULT)
    payload["items"] = None
    prediction = LlamaParseProvider().normalize(raw(payload))
    assert prediction.markdown == "# Report\n\nHello"
    assert prediction.layout == []


@pytest.mark.parametrize("mode", [None, "efficient", "agentic", "agentic_plus"])
def test_cli_replay_and_artifacts(
    document: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str | None
):
    from tests.test_cli import _corpus

    transport, requests = answering(*setup_responses(), httpx.Response(200, json=RESULT))

    class MockProvider(LlamaParseProvider):
        def __init__(self, config=None):
            super().__init__(config, transport=transport)

    monkeypatch.setitem(PROVIDERS, "llamaparse", MockProvider)
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    args = ["run", "--provider", "llamaparse", "--corpus", str(corpus), "--out", str(out)]
    config = {"tier": "agentic", "version": "2026-09-28"}
    if mode is not None:
        config["specialized_chart_parsing"] = mode
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    assert main([*args, "--config", str(config_path)]) == 0
    assert json.loads((out / RUN_MANIFEST).read_text())["config"] == config
    assert load_raw(out / "raw" / "plain_continuation_demo.json").metadata["config"] == config
    assert (out / "plain_continuation_demo.json").exists()
    assert all(b"secret-do-not-store" not in p.read_bytes() for p in out.rglob("*") if p.is_file())
    monkeypatch.delenv("LLAMA_CLOUD_API_KEY")
    assert main([*args, "--from-raw"]) == 0
    assert len(requests) == 3
    config["specialized_chart_parsing"] = "efficient" if mode != "efficient" else "agentic"
    config_path.write_text(json.dumps(config))
    assert main([*args, "--from-raw", "--config", str(config_path)]) == 2
    assert len(requests) == 3


@pytest.mark.parametrize("native", [False, True])
def test_markdown_furniture_is_preserved_without_duplicating_native_items(native: bool):
    payload = deepcopy(RESULT)
    payload["markdown"]["pages"][0].update({"header": "**Report header**", "footer": "Page 1"})
    if native:
        payload["items"]["pages"][0]["items"].append(
            {
                "type": "header",
                "items": [{"type": "text", "value": "Report header"}],
            }
        )
    result = LlamaParseProvider().normalize(raw(payload))
    assert [i.text for i in result.layout if i.cls == ElementClass.HEADER] == ["Report header"]
    assert [i.text for i in result.layout if i.cls == ElementClass.FOOTER] == ["Page 1"]
    assert "Report header" not in result.markdown


def test_blank_page_offsets_follow_the_next_page_or_document_end():
    payload = {
        "markdown": {
            "pages": [
                {"page_number": number, "markdown": text}
                for number, text in enumerate(["", "A", "", "B", ""], 1)
            ]
        }
    }
    result = LlamaParseProvider().normalize(raw(payload))
    assert result.markdown == "A\n\nB"
    assert result.page_offsets == [0, 0, 3, 3, 4]


def test_usage_counts_the_pages_parsed() -> None:
    payload = deepcopy(RESULT)
    payload["markdown"]["pages"].append({"page_number": 2, "success": True, "markdown": "Two"})
    assert LlamaParseProvider().usage(raw(payload)) == {"pages": 2}
