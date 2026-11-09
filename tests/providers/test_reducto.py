import json

import httpx
import pytest

from wholedoc_bench.providers.base import RawResult, load_raw, save_raw
from wholedoc_bench.providers.reducto import CONFIG, ReductoProvider, _piece_cells


def parsed():
    return {
        "job_id": "job-1",
        "result": {
            "type": "full",
            "chunks": [
                {
                    "content": "Hello",
                    "blocks": [
                        {
                            "type": "Text",
                            "content": "Hello",
                            "bbox": {
                                "page": 1,
                                "original_page": 1,
                                "left": 0.1,
                                "top": 0.2,
                                "width": 0.3,
                                "height": 0.4,
                            },
                        }
                    ],
                }
            ],
        },
    }


@pytest.mark.parametrize("suffix,mime", [(".pdf", "application/pdf"), (".png", "image/png")])
def test_upload_poll_save_normalize_offline(tmp_path, monkeypatch, suffix, mime):
    monkeypatch.setenv("REDUCTO_API_KEY", "secret")
    document = tmp_path / ("sample" + suffix)
    document.write_bytes(b"sample-document")
    calls = []

    def respond(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer secret"
        if request.url.path == "/upload":
            assert mime.encode() in request.content and b"sample-document" in request.content
            return httpx.Response(200, json={"file_id": "reducto://file-1"})
        if request.url.path == "/parse_async":
            assert json.loads(request.content) == {"input": "reducto://file-1", **CONFIG}
            assert CONFIG["settings"]["model"] == "r-1"
            return httpx.Response(200, json={"job_id": "job-1"})
        if len(calls) == 3:
            return httpx.Response(200, json={"status": "InProgress"})
        return httpx.Response(200, json={"status": "Completed", "result": parsed()})

    provider = ReductoProvider(transport=httpx.MockTransport(respond), sleep=lambda _: None)
    raw = provider.fetch(document)
    path = tmp_path / "raw.json"
    save_raw(raw, path)
    assert "secret" not in path.read_text()
    monkeypatch.delenv("REDUCTO_API_KEY")
    prediction = ReductoProvider().normalize(load_raw(path))
    assert prediction.markdown == "Hello"
    assert prediction.layout[0].box == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert prediction.layout[0].page == 0
    assert prediction.page_offsets is None
    assert len(calls) == 4


def test_normalization_keeps_tables_unknowns_and_original_pages():
    payload = parsed()
    chunk = payload["result"]["chunks"][0]
    table = '<table><tr><td colspan="2">A</td></tr></table>'
    chunk["content"] = "Header\n" + table + "\nOdd"
    chunk["blocks"] = [
        {"type": "Header", "content": "Header", "bbox": {"page": 1, "original_page": 3}},
        {"type": "Table", "content": table, "bbox": {"page": 3}},
        {"type": "Future", "content": "Odd", "bbox": {"page": 3}},
    ]
    prediction = ReductoProvider().normalize(RawResult("reducto", "doc", payload))
    assert prediction.layout[0].page == 2
    assert prediction.layout[0].channel is not None
    assert prediction.layout[0].channel.value == "furniture"
    assert prediction.layout[0].box is None
    assert prediction.layout[1].cls.value == "table"
    assert prediction.layout[1].cells is None
    assert table in prediction.markdown
    assert prediction.unscored_text == ["Odd"]


def merged_table_payload(second_page=3):
    header = "<tr><th>Region</th><th>Revenue</th></tr>"
    north, metro = "<tr><td>North</td><td>42.1</td></tr>", "<tr><td>Metro</td><td>51.2</td></tr>"
    first = f"<table>{header}{north}</table>"
    second = f"<table>{header}{metro}</table>"
    merged = f"<table>{header}{north}{metro}</table>"

    def box(page, top):
        return {
            "left": 0.16,
            "top": top,
            "width": 0.67,
            "height": 0.1,
            "page": page,
            "original_page": page,
        }

    payload = parsed()
    chunk = payload["result"]["chunks"][0]
    chunk["content"] = "Intro\n" + merged
    chunk["blocks"] = [
        {"type": "Text", "content": "Intro", "bbox": box(2, 0.1)},
        {
            "type": "Table",
            "content": merged,
            "bbox": box(2, 0.2),
            "merged_tables": [
                {"content": first, "bbox": box(2, 0.2), "image_url": None},
                {"content": second, "bbox": box(second_page, 0.07), "image_url": None},
            ],
        },
    ]
    return payload, merged


def test_a_table_merged_across_pages_is_one_table_with_a_linked_piece_per_page():
    payload, merged = merged_table_payload()
    prediction = ReductoProvider().normalize(RawResult("reducto", "doc", payload))
    first, second = prediction.layout[1:]
    assert prediction.markdown.count("<table>") == 1 and merged in prediction.markdown
    assert (first.cls.value, first.page, second.cls.value, second.page) == ("table", 1, "table", 2)
    assert "North" in first.text and "Metro" not in first.text
    assert "Metro" in second.text and "North" not in second.text
    assert first.box == pytest.approx((0.16, 0.2, 0.67, 0.1))
    assert second.box == pytest.approx((0.16, 0.07, 0.67, 0.1))
    assert prediction.markdown[first.md_start : first.md_end] == merged
    assert second.md_start is None
    [link] = prediction.relations
    assert (link.kind.value, link.source, link.target) == ("continues_in", first.id, second.id)


def test_each_piece_of_a_merged_table_keeps_its_own_cells():
    payload, _ = merged_table_payload()
    first, second = ReductoProvider().normalize(RawResult("reducto", "doc", payload)).layout[1:]
    assert first.cells is not None and second.cells is not None
    places = [(c.text, c.row, c.column, c.rows, c.columns) for c in second.cells]
    assert places == [
        ("Region", 0, 0, 1, 1),
        ("Revenue", 0, 1, 1, 1),
        ("Metro", 1, 0, 1, 1),
        ("51.2", 1, 1, 1, 1),
    ]
    assert [c.text for c in first.cells] == ["Region", "Revenue", "North", "42.1"]
    assert all(c.box is None for c in first.cells + second.cells)


def test_piece_cells_keep_their_spans_and_text_as_written():
    cells = _piece_cells(
        "<table><tr><th>Mode</th><th>Range</th></tr>"
        '<tr><td rowspan="2"><b>QCIF</b></td><td>385&ndash;1024 mm²</td></tr>'
        "<tr><td>129</td></tr>"
        '<tr><td colspan="2">Total <table><tr><td>n</td></tr></table></td></tr>'
        '<tr><td rowspan="9">Last</td><td>x</td></tr></table>'
    )
    assert cells is not None
    places = [(c.text, c.row, c.column, c.rows, c.columns) for c in cells]
    assert places[:4] == [
        ("Mode", 0, 0, 1, 1),
        ("Range", 0, 1, 1, 1),
        ("QCIF", 1, 0, 2, 1),
        ("385–1024 mm²", 1, 1, 1, 1),
    ]
    assert places[4] == ("129", 2, 1, 1, 1)
    assert places[5][1:] == (3, 0, 1, 2) and places[5][0].startswith("Total")
    assert places[6:] == [("Last", 4, 0, 1, 1), ("x", 4, 1, 1, 1)]


@pytest.mark.parametrize("second_page", [2, 4])
def test_merged_pieces_not_on_consecutive_pages_stay_one_item(second_page):
    payload, merged = merged_table_payload(second_page)
    prediction = ReductoProvider().normalize(RawResult("reducto", "doc", payload))
    [table] = [item for item in prediction.layout if item.cls.value == "table"]
    assert table.page == 1 and prediction.relations == []
    assert prediction.markdown[table.md_start : table.md_end] == merged


def test_key_value_blocks_keep_their_boxes():
    payload = parsed()
    chunk = payload["result"]["chunks"][0]
    chunk["content"] = "Rated power: 30 kw Start wind speed: 3 m/s"
    chunk["blocks"] = [
        {
            "type": "Key Value",
            "content": chunk["content"],
            "bbox": {"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.05, "page": 1},
        }
    ]
    prediction = ReductoProvider().normalize(RawResult("reducto", "doc", payload))
    item = prediction.layout[0]
    assert item.cls.value == "paragraph"
    assert item.box == pytest.approx((0.1, 0.2, 0.3, 0.05))
    assert (item.md_start, item.md_end) == (0, len(chunk["content"]))
    assert prediction.unscored_text == []


def test_an_empty_key_value_block_gives_no_item():
    payload = parsed()
    chunk = payload["result"]["chunks"][0]
    chunk["content"] = "<empty/>"
    chunk["blocks"] = [
        {
            "type": "Key Value",
            "content": "<empty/>",
            "bbox": {"left": 0.5, "top": 0.2, "width": 0.3, "height": 0.1, "page": 1},
        }
    ]
    prediction = ReductoProvider().normalize(RawResult("reducto", "doc", payload))
    assert prediction.layout == [] and prediction.unscored_text == []


@pytest.mark.parametrize(
    "bounds", [{}, {"left": -3}, {"left": float("nan")}, {"width": -1}, {"height": 0}]
)
def test_bad_geometry_preserves_text(bounds):
    payload = parsed()
    block = payload["result"]["chunks"][0]["blocks"][0]
    if bounds:
        block["bbox"].update(bounds)
    else:
        block["bbox"].pop("width")
    item = ReductoProvider().normalize(RawResult("reducto", "doc", payload)).layout[0]
    assert item.box is None and item.text == "Hello"


def test_key_not_part_of_cache_but_configuration_is(monkeypatch):
    provider = ReductoProvider()
    monkeypatch.delenv("REDUCTO_API_KEY", raising=False)
    assert provider.configuration_error()
    before = provider.raw_cache_key()
    monkeypatch.setenv("REDUCTO_API_KEY", "secret")
    assert provider.configuration_error() is None
    assert before == provider.raw_cache_key()
    monkeypatch.setitem(CONFIG, "extra", True)
    assert before != provider.raw_cache_key()


@pytest.mark.parametrize("status", [401, 403, 429, 500, 302])
def test_http_error_does_not_expose_body_or_retry_upload(status, tmp_path, monkeypatch):
    monkeypatch.setenv("REDUCTO_API_KEY", "secret")
    document = tmp_path / "a.pdf"
    document.touch()
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(status, text="secret document contents")

    with pytest.raises(RuntimeError, match=f"HTTP {status}") as exc:
        ReductoProvider(transport=httpx.MockTransport(respond)).fetch(document)
    assert "secret" not in str(exc.value) and len(calls) == 1


@pytest.mark.parametrize("cancel", [True, False])
def test_cancel_or_deadline_requests_remote_cancellation(cancel, tmp_path, monkeypatch):
    monkeypatch.setenv("REDUCTO_API_KEY", "secret")
    document = tmp_path / "a.pdf"
    document.touch()
    state = [False, 0.0]
    calls = []

    def respond(request):
        calls.append(request.url.path)
        if request.url.path == "/upload":
            return httpx.Response(200, json={"file_id": "reducto://file-1"})
        if request.url.path == "/parse_async":
            return httpx.Response(200, json={"job_id": "job-1"})
        return httpx.Response(200, json={"status": "Pending"})

    def sleep(_):
        state[0] = cancel
        state[1] = 1000

    provider = ReductoProvider(
        transport=httpx.MockTransport(respond), sleep=sleep, clock=lambda: state[1]
    )
    provider.set_cancelled(lambda: state[0])
    with pytest.raises(RuntimeError, match="cancelled" if cancel else "timed out"):
        provider.fetch(document)
    assert calls[-1] == "/cancel/job-1"


@pytest.mark.parametrize("terminal", ["Failed", "unknown"])
def test_terminal_failure(tmp_path, monkeypatch, terminal):
    monkeypatch.setenv("REDUCTO_API_KEY", "secret")
    document = tmp_path / "a.pdf"
    document.touch()
    responses = iter(
        [
            {"file_id": "reducto://file-1"},
            {"job_id": "job-1"},
            {"status": terminal, "reason": "secret"},
        ]
    )
    with pytest.raises(RuntimeError, match="parsing failed") as exc:
        ReductoProvider(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=next(responses)))
        ).fetch(document)
    assert "secret" not in str(exc.value)


def test_signed_output_download_has_no_bearer(tmp_path, monkeypatch):
    monkeypatch.setenv("REDUCTO_API_KEY", "secret")
    document = tmp_path / "a.pdf"
    document.touch()

    def respond(request):
        if request.url.host == "output.example":
            assert "Authorization" not in request.headers
            return httpx.Response(200, json=parsed()["result"])
        if request.url.path == "/upload":
            return httpx.Response(200, json={"file_id": "reducto://file-1"})
        if request.url.path == "/parse_async":
            return httpx.Response(200, json={"job_id": "job-1"})
        return httpx.Response(
            200,
            json={
                "status": "Completed",
                "result": {"result": {"type": "url", "url": "https://output.example/file?secret"}},
            },
        )

    raw = ReductoProvider(transport=httpx.MockTransport(respond)).fetch(document)
    assert raw.payload["result"] == parsed()["result"]
    assert "secret" not in json.dumps(raw.payload)


@pytest.mark.parametrize("value", [[], {"file_id": "wrong"}, None])
def test_invalid_upload_response(tmp_path, monkeypatch, value):
    monkeypatch.setenv("REDUCTO_API_KEY", "secret")
    document = tmp_path / "a.pdf"
    document.touch()
    with pytest.raises(RuntimeError, match="invalid"):
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=value))
        ReductoProvider(transport=transport).fetch(document)


@pytest.mark.parametrize(
    "url",
    [
        "http://output.example/file",
        "file:///tmp/file",
        None,
        "https://user:secret@output.example/file",
    ],
)
def test_invalid_output_url(tmp_path, monkeypatch, url):
    monkeypatch.setenv("REDUCTO_API_KEY", "secret")
    document = tmp_path / "a.pdf"
    document.touch()
    responses = iter(
        [
            {"file_id": "reducto://file-1"},
            {"job_id": "job-1"},
            {"status": "Completed", "result": {"result": {"type": "url", "url": url}}},
        ]
    )
    with pytest.raises(RuntimeError, match="invalid output URL"):
        ReductoProvider(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=next(responses)))
        ).fetch(document)


def test_transport_errors_do_not_expose_request_details(tmp_path, monkeypatch):
    monkeypatch.setenv("REDUCTO_API_KEY", "secret")
    document = tmp_path / "a.pdf"
    document.touch()

    def fail(_):
        raise httpx.ReadTimeout("Bearer secret must not surface")

    with pytest.raises(RuntimeError, match="request failed or timed out") as error:
        ReductoProvider(transport=httpx.MockTransport(fail)).fetch(document)
    assert "secret" not in str(error.value)


def test_list_item_uses_benchmark_list_class():
    payload = parsed()
    payload["result"]["chunks"][0]["blocks"][0]["type"] = "List Item"
    prediction = ReductoProvider().normalize(RawResult("reducto", "doc", payload))
    assert prediction.layout[0].cls.value == "list_item"


@pytest.mark.parametrize("mode", ["r-1", "legacy", "legacy-agentic"])
def test_mode_request(mode, tmp_path, monkeypatch):
    monkeypatch.setenv("REDUCTO_API_KEY", "secret")
    path = tmp_path / "a.pdf"
    path.write_bytes(b"sample")

    def respond(request):
        if request.url.path == "/upload":
            return httpx.Response(200, json={"file_id": "reducto://file-1"})
        if request.url.path == "/parse_async":
            config = json.loads(request.content)
            assert config["settings"] == {
                "persist_results": False,
                "model": "r-1" if mode == "r-1" else "legacy",
            }
            assert config["formatting"] == {"table_output_format": "html", "merge_tables": True}
            if mode == "r-1":
                assert config["enhance"] == {
                    "agentic": [{"scope": "figure", "advanced_chart_agent": True}]
                }
            elif mode == "legacy-agentic":
                assert config["enhance"] == {
                    "agentic": [{"scope": "text"}, {"scope": "table"}, {"scope": "figure"}],
                    "summarize_figures": True,
                }
            else:
                assert "enhance" not in config
            return httpx.Response(200, json={"job_id": "job-1"})
        return httpx.Response(200, json={"status": "Completed", "result": parsed()})

    provider = ReductoProvider(
        {"mode": mode}, transport=httpx.MockTransport(respond), sleep=lambda _: None
    )
    assert provider.fetch(path).metadata["raw_cache_key"] == provider.raw_cache_key()
    assert CONFIG["settings"]["model"] == "r-1"
    assert CONFIG["enhance"] == {"agentic": [{"scope": "figure", "advanced_chart_agent": True}]}


@pytest.mark.parametrize(
    "config", [[], {"mode": "agentic-plus"}, {"mode": []}, {"mode": None}, {"model": "r-1"}]
)
def test_invalid_mode_config(config):
    with pytest.raises(ValueError):
        ReductoProvider(config)


def test_mode_cache_identities_preserve_default():
    import hashlib

    digest = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]
    expected = f"reducto REST r-1 {digest}"
    assert ReductoProvider().raw_cache_key() == expected
    assert ReductoProvider({"mode": "r-1"}).raw_cache_key() == expected
    providers = [ReductoProvider({"mode": m}) for m in ("r-1", "legacy", "legacy-agentic")]
    assert len({p.raw_cache_key() for p in providers}) == 3
    assert len({p.version() for p in providers}) == 3


CHART_TABLE = (
    "<table><tr><th>Region</th><th>Sales</th></tr><tr><td>North</td><td>10</td></tr>"
    "<tr><td>South</td><td>20</td></tr></table>"
)
BOUNDS = {"page": 1, "left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4}


def chart_truth(title=None):
    from wholedoc_bench.schema import Groundtruth

    chart = {
        "kind": "bar",
        "series": [
            {
                "name": "Sales",
                "points": [{"label": "North", "value": 10}, {"label": "South", "value": 20}],
            }
        ],
    }
    if title:
        chart["title"] = title
    return Groundtruth.model_validate(
        {
            "schema_version": "0.1",
            "document": {
                "id": "doc",
                "source": "born-digital",
                "groundtruth_source": "nutrient_seeded_reviewed",
                "language": ["en"],
                "pages": [{"index": 0, "width": 100, "height": 200}],
            },
            "elements": [
                {
                    "id": "c1",
                    "class": "chart",
                    "page": 0,
                    "box": [0.1, 0.2, 0.3, 0.4],
                    "text": "",
                    "chart": chart,
                }
            ],
        }
    )


def test_charts_are_requested_as_their_data():
    assert CONFIG["enhance"] == {"agentic": [{"scope": "figure", "advanced_chart_agent": True}]}


@pytest.mark.parametrize(
    "content,extra,expected",
    [
        ("A bar chart of sales.", {"extra": {"is_chart": True}}, "chart"),
        ("A bar chart of sales.", {"chart_data": [{"series": []}]}, "chart"),
        (CHART_TABLE, {"extra": {"is_chart": True}}, "chart"),
        (CHART_TABLE, {"extra": {"is_chart": False}}, "figure"),
        ("| Region | Sales |\n| --- | --- |\n| North | 10 |", {}, "figure"),
        ("A photo of the office.", {"chart_data": None, "extra": {}}, "figure"),
    ],
)
def test_a_figure_is_a_chart_when_reducto_flags_it_or_gives_its_data(content, extra, expected):
    payload = parsed()
    chunk = payload["result"]["chunks"][0]
    chunk["content"] = content
    chunk["blocks"] = [{"type": "Figure", "content": content, **extra, "bbox": BOUNDS}]
    prediction = ReductoProvider().normalize(RawResult("reducto", "doc", payload))
    assert [x.cls.value for x in prediction.layout] == [expected]


def chart_job(links):
    payload = parsed()
    payload["result"]["chunks"][0]["blocks"][0].update(
        {
            "type": "Figure",
            "content": "A bar chart of sales.",
            "chart_data": links,
            "extra": {
                "is_chart": True,
                "chart_reconstruction": "https://charts.example/image.png?secret",
            },
        }
    )
    return payload


def fetch_with_charts(tmp_path, monkeypatch, links, respond_chart):
    monkeypatch.setenv("REDUCTO_API_KEY", "secret")
    document = tmp_path / "a.pdf"
    document.touch()

    def respond(request):
        if request.url.host == "charts.example":
            assert "Authorization" not in request.headers
            return respond_chart(request)
        if request.url.path == "/upload":
            return httpx.Response(200, json={"file_id": "reducto://file-1"})
        if request.url.path == "/parse_async":
            return httpx.Response(200, json={"job_id": "job-1"})
        return httpx.Response(200, json={"status": "Completed", "result": chart_job(links)})

    return ReductoProvider(transport=httpx.MockTransport(respond)).fetch(document)


def test_chart_data_links_are_saved_as_the_data_they_point_to(tmp_path, monkeypatch):
    links = ["https://charts.example/1.json?secret", "https://charts.example/2.json?secret"]
    raw = fetch_with_charts(
        tmp_path,
        monkeypatch,
        links,
        lambda request: httpx.Response(200, json={"chart": request.url.path}),
    )
    block = raw.payload["result"]["chunks"][0]["blocks"][0]
    assert block["chart_data"] == [{"chart": "/1.json"}, {"chart": "/2.json"}]
    assert block["extra"] == {"is_chart": True}
    assert "secret" not in json.dumps(raw.payload)
    path = tmp_path / "raw.json"
    save_raw(raw, path)
    monkeypatch.delenv("REDUCTO_API_KEY")
    prediction = ReductoProvider().normalize(load_raw(path))
    assert [x.cls.value for x in prediction.layout] == ["chart"]


@pytest.mark.parametrize(
    "bad,status,error",
    [
        ("http://charts.example/2.json", 200, "invalid chart data URL"),
        ("https://user:secret@charts.example/2.json", 200, "invalid chart data URL"),
        ("https://charts.example/2.json?secret", 403, "HTTP 403"),
    ],
)
def test_a_bad_chart_link_loses_that_chart_not_the_parse(tmp_path, monkeypatch, bad, status, error):
    def respond_chart(request):
        if request.url.path == "/1.json":
            return httpx.Response(200, json=CHART_JSON)
        return httpx.Response(status, json={"chart": 2})

    raw = fetch_with_charts(
        tmp_path, monkeypatch, ["https://charts.example/1.json", bad], respond_chart
    )
    good, failed = raw.payload["result"]["chunks"][0]["blocks"][0]["chart_data"]
    assert good == CHART_JSON and error in failed["download_failed"]
    assert "secret" not in json.dumps(raw.payload)
    item = ReductoProvider().normalize(raw).layout[0]
    assert item.cls.value == "chart" and item.chart is not None and item.chart.title == "Sales"


def test_chart_links_not_in_a_list_are_still_read(tmp_path, monkeypatch):
    raw = fetch_with_charts(
        tmp_path,
        monkeypatch,
        "https://charts.example/1.json",
        lambda _: httpx.Response(200, json=CHART_JSON),
    )
    assert raw.payload["result"]["chunks"][0]["blocks"][0]["chart_data"] == [CHART_JSON]


@pytest.mark.parametrize("cancel", [True, False])
def test_a_cancel_or_timeout_during_a_chart_download_still_stops(tmp_path, monkeypatch, cancel):
    monkeypatch.setenv("REDUCTO_API_KEY", "secret")
    document = tmp_path / "a.pdf"
    document.touch()
    state = [False, 0.0]
    calls = []

    def respond(request):
        calls.append(request.url.path)
        if request.url.host == "charts.example":
            state[0], state[1] = cancel, (0.0 if cancel else 1000.0)
            return httpx.Response(500)
        if request.url.path == "/upload":
            return httpx.Response(200, json={"file_id": "reducto://file-1"})
        if request.url.path == "/parse_async":
            return httpx.Response(200, json={"job_id": "job-1"})
        if request.url.path.startswith("/cancel/"):
            return httpx.Response(200, json={})
        links = ["https://charts.example/1.json"]
        return httpx.Response(200, json={"status": "Completed", "result": chart_job(links)})

    provider = ReductoProvider(transport=httpx.MockTransport(respond), clock=lambda: state[1])
    provider.set_cancelled(lambda: state[0])
    with pytest.raises(RuntimeError, match="cancelled" if cancel else "timed out"):
        provider.fetch(document)
    assert calls[-2:] == ["/1.json", "/cancel/job-1"]


CHART_JSON = {
    "chart_title": "Sales",
    "source": "",
    "axes": {
        "y_primary": {"label": "Units", "min": 0, "max": 30, "ticks": [0, 10, 20, 30]},
        "x": {"type": "category", "labels": ["North", "South"]},
    },
    "series": [
        {
            "label": "Sales",
            "type": "bar",
            "corresponding_y": "primary",
            "color": "#000",
            "data": [
                {"category": "North", "value": 10},
                {"category": "South", "value": 20},
                {"category": "East", "value": [5, 9]},
            ],
        }
    ],
    "printed_labels": [],
}


def chart_figure(content, chart_data):
    payload = parsed()
    chunk = payload["result"]["chunks"][0]
    chunk["content"] = content
    chunk["blocks"] = [
        {
            "type": "Figure",
            "content": content,
            "chart_data": chart_data,
            "extra": {"is_chart": True},
            "bbox": BOUNDS,
        }
    ]
    return ReductoProvider().normalize(RawResult("reducto", "doc", payload))


def test_chart_json_becomes_the_benchmark_chart():
    chart = chart_figure("A bar chart of sales.", [CHART_JSON]).layout[0].chart
    assert chart is not None and chart.value_axis is not None
    assert chart.kind == "bar" and chart.title == "Sales"
    assert (chart.value_axis.min, chart.value_axis.max) == (0, 30)
    assert [(s.name, [(p.label, p.value) for p in s.points]) for s in chart.series] == [
        ("Sales", [("North", 10), ("South", 20)])
    ]


@pytest.mark.parametrize(
    "chart_data",
    [
        None,
        ["https://charts.example/1.json"],
        [{}],
        [{"download_failed": "Reducto request failed (HTTP 403)"}],
        [{"series": [{"type": "text", "data": [{"value": 1}]}]}],
    ],
)
def test_chart_without_usable_json_has_no_chart_data(chart_data):
    item = chart_figure("A bar chart of sales.", chart_data).layout[0]
    assert item.cls.value == "chart" and item.chart is None


@pytest.mark.parametrize("other,kind", [("line", None), ("text", "bar")])
def test_a_combination_chart_names_no_kind_but_printed_values_do_not_count(other, kind):
    extra = {**CHART_JSON["series"][0], "type": other, "label": "Change"}
    prediction = chart_figure(
        "A chart.", [{**CHART_JSON, "series": [*CHART_JSON["series"], extra]}]
    )
    chart = prediction.layout[0].chart
    assert chart is not None and chart.kind == kind and len(chart.series) == 2


def test_a_chart_of_ranges_keeps_its_kind_and_title():
    from wholedoc_bench.scorer.run import score_document

    ranges = [{"category": "North", "value": [0, 10]}, {"category": "South", "value": [0, 20]}]
    chart_data = [{**CHART_JSON, "series": [{**CHART_JSON["series"][0], "data": ranges}]}]
    prediction = chart_figure("A bar chart of sales.", chart_data)
    chart = prediction.layout[0].chart
    assert chart is not None and (chart.kind, chart.title, chart.series) == ("bar", "Sales", [])
    t6 = score_document(chart_truth("Sales"), prediction).sub_scores["T6"]
    assert t6["chart_type"] == 1.0 and t6["chart_title"] == 1.0


def test_chart_json_described_in_words_scores_its_points():
    from wholedoc_bench.scorer.run import score_document

    prediction = chart_figure("A bar chart of sales by region.", [CHART_JSON])
    t6 = score_document(chart_truth(), prediction).sub_scores["T6"]
    assert t6["data_points"] == 1.0 and t6["series"] == 1.0 and t6["chart_type"] == 1.0


def test_a_chart_extracted_as_its_data_table_scores_its_points():
    from wholedoc_bench.scorer.run import score_document

    payload = parsed()
    chunk = payload["result"]["chunks"][0]
    chunk["content"] = CHART_TABLE
    chunk["blocks"] = [
        {"type": "Figure", "content": CHART_TABLE, "extra": {"is_chart": True}, "bbox": BOUNDS}
    ]
    prediction = ReductoProvider().normalize(RawResult("reducto", "doc", payload))
    t6 = score_document(chart_truth(), prediction).sub_scores["T6"]
    assert t6["data_points"] == 1.0 and t6["series"] == 1.0


def test_usage_counts_the_billed_pages_and_the_charts_the_agent_read():
    payload = parsed()
    payload["usage"] = {"num_pages": 3, "credits": None}
    read = {"type": "Figure", "bbox": {"page": 1}, "chart_data": [{"series": []}]}
    flagged = {"type": "Figure", "bbox": {"page": 1}, "extra": {"is_chart": True}}
    picture = {"type": "Figure", "bbox": {"page": 1}}
    payload["result"]["chunks"][0]["blocks"] += [read, flagged, picture]
    usage = ReductoProvider().usage(RawResult("reducto", "doc", payload))
    assert usage == {"pages": 3, "charts": 1}


def test_a_response_without_a_page_count_has_no_usage():
    with pytest.raises((KeyError, ValueError)):
        ReductoProvider().usage(RawResult("reducto", "doc", parsed()))
