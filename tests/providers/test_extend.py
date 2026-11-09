import json

import httpx
import pytest

from wholedoc_bench.providers.base import RawResult, load_raw, save_raw
from wholedoc_bench.providers.extend import API_VERSION, CONFIG, ExtendProvider


def block(kind="text", content="Hello", **extra):
    return {
        "type": kind,
        "content": content,
        "metadata": {"page": {"number": 1, "width": 100, "height": 200}},
        "boundingBox": {"left": 10, "top": 20, "right": 70, "bottom": 60},
        **extra,
    }


def result():
    return {
        "id": "pr_123",
        "status": "PROCESSED",
        "output": {
            "chunks": [
                {
                    "content": "Hello",
                    "metadata": {"pageRange": {"start": 1, "end": 1}},
                    "blocks": [block()],
                }
            ]
        },
    }


@pytest.mark.parametrize("suffix,mime", [(".pdf", "application/pdf"), (".png", "image/png")])
def test_upload_poll_raw_offline(tmp_path, monkeypatch, suffix, mime):
    monkeypatch.setenv("EXTEND_API_KEY", "secret-test-key")
    monkeypatch.setenv("EXTEND_WORKSPACE_ID", "workspace-test")
    document = tmp_path / ("sample" + suffix)
    document.write_bytes(b"test-document")
    requests = []

    def respond(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer secret-test-key"
        assert request.headers["x-extend-api-version"] == API_VERSION
        assert request.headers["x-extend-workspace-id"] == "workspace-test"
        if request.url.path == "/files/upload":
            assert mime.encode() in request.content and b"test-document" in request.content
            return httpx.Response(200, json={"id": "file_123"})
        if request.method == "POST":
            assert json.loads(request.content) == {"file": {"id": "file_123"}, "config": CONFIG}
            return httpx.Response(200, json={"id": "pr_123", "status": "PENDING"})
        if len(requests) == 3:
            return httpx.Response(200, json={"id": "pr_123", "status": "PROCESSING"})
        return httpx.Response(200, json=result())

    provider = ExtendProvider(transport=httpx.MockTransport(respond), sleep=lambda _: None)
    raw = provider.fetch(document)
    assert raw.document_id == tmp_path.name and raw.payload == result()
    path = tmp_path / "raw.json"
    save_raw(raw, path)
    assert "secret-test-key" not in path.read_text()
    monkeypatch.delenv("EXTEND_API_KEY")
    prediction = ExtendProvider().normalize(load_raw(path))
    assert prediction.markdown == "Hello"
    assert prediction.layout[0].box == pytest.approx((0.1, 0.1, 0.6, 0.2))
    assert prediction.page_offsets == [0]
    assert len(requests) == 4


def test_cached_image_with_null_page_metadata(tmp_path, monkeypatch):
    payload = result()
    payload["output"]["metadata"] = {"originalMimeType": "image/png", "pages": None}
    raw = RawResult("extend", "image", payload)
    path = tmp_path / "raw.json"
    save_raw(raw, path)
    monkeypatch.delenv("EXTEND_API_KEY", raising=False)

    prediction = ExtendProvider().normalize(load_raw(path))

    assert prediction.markdown == "Hello"
    assert prediction.page_offsets == [0]
    assert len(prediction.layout) == 1
    assert prediction.layout[0].page == 0
    assert prediction.layout[0].text == "Hello"
    assert prediction.layout[0].box == pytest.approx((0.1, 0.1, 0.6, 0.2))


def test_configuration(monkeypatch, tmp_path):
    monkeypatch.delenv("EXTEND_API_KEY", raising=False)
    provider = ExtendProvider()
    assert provider.configuration_error()
    with pytest.raises(RuntimeError, match="EXTEND_API_KEY"):
        provider.fetch(tmp_path / "a.pdf")
    initial = provider.raw_cache_key()
    monkeypatch.setenv("EXTEND_API_KEY", "changed")
    assert initial == provider.raw_cache_key()
    monkeypatch.setenv("EXTEND_WORKSPACE_ID", "different")
    assert initial != provider.raw_cache_key()
    assert "different" not in provider.version()
    assert provider.configuration_error() is None


@pytest.mark.parametrize("status", [401, 403, 429, 500, 302])
def test_http_errors_safe_no_repeated_upload(status, monkeypatch, tmp_path):
    monkeypatch.setenv("EXTEND_API_KEY", "secret-test-key")
    path = tmp_path / "a.pdf"
    path.write_bytes(b"sample")
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(status, text="secret-test-key document contents")

    with pytest.raises(RuntimeError, match=f"HTTP {status}") as error:
        ExtendProvider(transport=httpx.MockTransport(respond)).fetch(path)
    assert "secret-test-key" not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("terminal", ["FAILED", "CANCELLED", "unknown"])
def test_terminal_failure(terminal, monkeypatch, tmp_path):
    monkeypatch.setenv("EXTEND_API_KEY", "secret")
    path = tmp_path / "a.pdf"
    path.touch()

    def respond(request):
        if request.url.path == "/files/upload":
            return httpx.Response(200, json={"id": "file_123"})
        return httpx.Response(
            200, json={"id": "pr_123", "status": terminal, "failureMessage": "secret"}
        )

    with pytest.raises(RuntimeError, match="Extend parsing failed") as error:
        ExtendProvider(transport=httpx.MockTransport(respond)).fetch(path)
    assert "secret" not in str(error.value)


def test_poll_deadline(monkeypatch, tmp_path):
    monkeypatch.setenv("EXTEND_API_KEY", "secret")
    path = tmp_path / "a.pdf"
    path.touch()
    now = [0.0]

    def respond(request):
        return httpx.Response(
            200,
            json={"id": "file_123"}
            if request.url.path == "/files/upload"
            else {"id": "pr_123", "status": "PROCESSING"},
        )

    def sleep(_):
        now[0] += 1000

    provider = ExtendProvider(
        transport=httpx.MockTransport(respond), sleep=sleep, clock=lambda: now[0]
    )
    with pytest.raises(RuntimeError, match="timed out"):
        provider.fetch(path)


@pytest.mark.parametrize("content", ["", "   "])
def test_an_empty_key_value_block_gives_no_item(content):
    chunks = [
        {
            "content": content,
            "blocks": [block("key_value", content)],
            "metadata": {"pageRange": {"start": 1, "end": 1}},
        }
    ]
    prediction = ExtendProvider().normalize(
        RawResult("extend", "doc", {"output": {"chunks": chunks}})
    )
    assert prediction.layout == []


def test_layout_tables_unknown_and_sparse_pages():
    table = "<table><tr><th>Name</th></tr><tr><td>A</td></tr></table>"
    blocks = [
        block("heading", "# Title"),
        block("header", "Running header"),
        block("table", table, children=[block("table_cell", "A")]),
        block("figure", "Chart", details={"figureType": "chart"}),
        block("key_value", "Field: Value"),
        block("future_type", "Unknown"),
    ]
    chunks = [
        {
            "content": "\n\n".join(b["content"] for b in blocks),
            "blocks": blocks,
            "metadata": {"pageRange": {"start": 2, "end": 2}},
        }
    ]
    raw = RawResult("extend", "doc", {"output": {"chunks": chunks}})
    prediction = ExtendProvider().normalize(raw)
    assert [x.cls.value for x in prediction.layout] == [
        "heading",
        "header",
        "table",
        "chart",
        "paragraph",
    ]
    assert prediction.layout[0].text == "Title"
    assert prediction.layout[1].channel is not None
    assert prediction.layout[1].channel.value == "furniture"
    assert prediction.layout[2].cells is not None
    assert prediction.layout[2].cells[0].text == "A"
    assert table in prediction.markdown
    key_value = prediction.layout[4]
    assert (key_value.text, key_value.box is not None) == ("Field: Value", True)
    assert prediction.unscored_text == ["Unknown"]
    assert prediction.page_offsets is None


@pytest.mark.parametrize(
    "angle,expected",
    [
        (90, (0.1, 0.3, 0.2, 0.6)),
        (180, (0.3, 0.7, 0.6, 0.2)),
        (-90, (0.7, 0.1, 0.2, 0.6)),
        (0, (0.1, 0.1, 0.6, 0.2)),
    ],
)
def test_original_page_rotation(angle, expected):
    payload = result()
    payload["output"]["metadata"] = {
        "pages": [
            {
                "number": 1,
                "rotationApplied": angle,
                "originalPageWidth": 200 if angle % 180 else 100,
                "originalPageHeight": 100 if angle % 180 else 200,
                "dpi": 72,
            }
        ]
    }
    item = ExtendProvider().normalize(RawResult("extend", "doc", payload)).layout[0]
    assert item.box == pytest.approx(expected)


def test_invalid_box_keeps_text():
    payload = result()
    item = payload["output"]["chunks"][0]["blocks"][0]
    item["boundingBox"]["left"] = -100
    provider = ExtendProvider()
    assert provider.normalize(RawResult("extend", "doc", payload)).layout[0].box is None
    item["metadata"]["page"]["width"] = 0
    assert provider.normalize(RawResult("extend", "doc", payload)).layout[0].text == "Hello"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=[]),
        httpx.Response(200, json={"id": "../../secret"}),
    ],
)
def test_malformed_response(response, monkeypatch, tmp_path):
    monkeypatch.setenv("EXTEND_API_KEY", "secret")
    path = tmp_path / "a.pdf"
    path.touch()
    with pytest.raises(RuntimeError, match="invalid"):
        ExtendProvider(transport=httpx.MockTransport(lambda _: response)).fetch(path)


def test_transport_error_is_safe(monkeypatch, tmp_path):
    monkeypatch.setenv("EXTEND_API_KEY", "secret")
    path = tmp_path / "a.pdf"
    path.touch()

    def fail(_):
        raise httpx.ReadTimeout("secret headers should not surface")

    with pytest.raises(RuntimeError, match="request failed or timed out") as error:
        ExtendProvider(transport=httpx.MockTransport(fail)).fetch(path)
    assert "secret" not in str(error.value)


def test_signed_output_url_never_receives_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("EXTEND_API_KEY", "secret")
    path = tmp_path / "a.pdf"
    path.touch()
    calls = []

    def respond(request):
        calls.append(request)
        if request.url.path == "/files/upload":
            return httpx.Response(200, json={"id": "file_123"})
        if request.url.host == "other.example":
            assert "Authorization" not in request.headers
            assert "x-extend-workspace-id" not in request.headers
            return httpx.Response(200, json=result()["output"])
        return httpx.Response(
            200,
            json={
                "id": "pr_123",
                "status": "PROCESSED",
                "output": None,
                "outputUrl": "https://other.example/private?token=secret",
            },
        )

    raw = ExtendProvider(transport=httpx.MockTransport(respond)).fetch(path)
    assert len(calls) == 3
    assert raw.payload["output"] == result()["output"]
    assert raw.payload["outputUrl"] is None
    assert ExtendProvider().normalize(raw).markdown == "Hello"


@pytest.mark.parametrize(
    "url",
    [
        "http://other.example/file",
        "file:///tmp/file",
        None,
        "https://user:password@other.example/file",
    ],
)
def test_invalid_output_url(url, monkeypatch, tmp_path):
    monkeypatch.setenv("EXTEND_API_KEY", "secret")
    path = tmp_path / "a.pdf"
    path.touch()

    def respond(request):
        if request.url.path == "/files/upload":
            return httpx.Response(200, json={"id": "file_123"})
        return httpx.Response(200, json={"id": "pr_123", "status": "PROCESSED", "outputUrl": url})

    with pytest.raises(RuntimeError, match="no usable parse output"):
        ExtendProvider(transport=httpx.MockTransport(respond)).fetch(path)


@pytest.mark.parametrize(
    "engine,version",
    [("parse_performance", "2.0.0"), ("parse_light", "1.0.0"), ("parse_auto", "1.0.0")],
)
@pytest.mark.parametrize("agentic", [False, True])
def test_engine_and_agentic_request(engine, version, agentic, tmp_path, monkeypatch):
    monkeypatch.setenv("EXTEND_API_KEY", "secret")
    path = tmp_path / "a.pdf"
    path.write_bytes(b"sample")

    def respond(request):
        if request.url.path == "/files/upload":
            return httpx.Response(200, json={"id": "file_123"})
        config = json.loads(request.content)["config"]
        assert config["engine"] == engine
        assert config["engineVersion"] == version
        assert config["blockOptions"]["tables"] == {"targetFormat": "html"}
        assert config["advancedOptions"] == {"pageRotationEnabled": False}
        if engine == "parse_performance":
            assert config["blockOptions"]["figures"] == CONFIG["blockOptions"]["figures"]
        else:
            assert "figures" not in config["blockOptions"]
        if agentic:
            assert config["blockOptions"]["text"] == {"agentic": {"enabled": True}}
        else:
            assert "text" not in config["blockOptions"]
        return httpx.Response(200, json=result())

    provider = ExtendProvider(
        {"engine": engine, "agentic_text": agentic}, transport=httpx.MockTransport(respond)
    )
    assert provider.fetch(path).metadata["raw_cache_key"] == provider.raw_cache_key()
    assert CONFIG["engine"] == "parse_performance"
    assert "text" not in CONFIG["blockOptions"]


@pytest.mark.parametrize(
    "config",
    [
        [],
        {"engine": "cheap"},
        {"engine": []},
        {"agentic_text": "true"},
        {"agentic_text": 1},
        {"agentic_text": None},
        {"mode": "auto"},
    ],
)
def test_invalid_mode_config(config):
    with pytest.raises(ValueError):
        ExtendProvider(config)


def test_mode_cache_identities_preserve_default(monkeypatch):
    import hashlib

    monkeypatch.delenv("EXTEND_WORKSPACE_ID", raising=False)
    old = {"api": API_VERSION, "config": CONFIG, "workspace": ""}
    digest = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()[:16]
    expected = f"extend {API_VERSION} parse_performance 2.0.0 {digest}"
    assert ExtendProvider().raw_cache_key() == expected
    assert (
        ExtendProvider({"engine": "parse_performance", "agentic_text": False}).raw_cache_key()
        == expected
    )
    providers = [
        ExtendProvider({"engine": e, "agentic_text": a})
        for e in ("parse_performance", "parse_light", "parse_auto")
        for a in (False, True)
    ]
    assert len({p.raw_cache_key() for p in providers}) == 6
    assert len({p.version() for p in providers}) == 6


CHART_TABLE = (
    "<table><tr><th>Region</th><th>Sales</th></tr><tr><td>North</td><td>10</td></tr>"
    "<tr><td>South</td><td>20</td></tr></table>"
)


def chart_truth():
    from wholedoc_bench.schema import Groundtruth

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
                    "box": [0.1, 0.1, 0.6, 0.2],
                    "text": "",
                    "chart": {
                        "kind": "bar",
                        "series": [
                            {
                                "name": "Sales",
                                "points": [
                                    {"label": "North", "value": 10},
                                    {"label": "South", "value": 20},
                                ],
                            }
                        ],
                    },
                }
            ],
        }
    )


def test_charts_are_requested_as_their_data():
    assert CONFIG["blockOptions"]["figures"] == {
        "enabled": True,
        "advancedChartExtractionEnabled": True,
        "figureImageClippingEnabled": False,
    }
    for engine in ("parse_light", "parse_auto"):
        options = ExtendProvider({"engine": engine})._request_config()["blockOptions"]
        assert "figures" not in options


def test_a_chart_extracted_as_its_data_table_scores_its_points():
    from wholedoc_bench.scorer.run import score_document

    chart = block("figure", CHART_TABLE, details={"type": "figure_details", "figureType": "chart"})
    chunks = [
        {
            "content": chart["content"],
            "blocks": [chart],
            "metadata": {"pageRange": {"start": 1, "end": 1}},
        }
    ]
    raw = RawResult("extend", "doc", {"output": {"chunks": chunks}})
    prediction = ExtendProvider().normalize(raw)
    assert [x.cls.value for x in prediction.layout] == ["chart"]
    t6 = score_document(chart_truth(), prediction).sub_scores["T6"]
    assert t6["data_points"] == 1.0 and t6["series"] == 1.0


@pytest.mark.parametrize(
    "extra,usage",
    [
        ({"usage": {"credits": 4.5}, "metrics": {"pageCount": 2}}, {"credits": 4.5}),
        ({"usage": None, "metrics": {"pageCount": 2}}, {"pages": 2}),
    ],
    ids=["billed credits", "legacy billing"],
)
def test_usage_is_the_billed_credits_or_else_the_pages(extra, usage):
    assert ExtendProvider().usage(RawResult("extend", "doc", {**result(), **extra})) == usage


def test_a_run_without_credits_or_pages_has_no_usage():
    with pytest.raises(ValueError):
        ExtendProvider().usage(RawResult("extend", "doc", result()))
