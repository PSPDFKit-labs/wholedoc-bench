import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from wholedoc_bench.providers import CredentialError, NutrientProvider
from wholedoc_bench.providers.nutrient import API_URL, RETRIES

RESPONSE = {"status": 200, "output": {"elements": [], "markdown": "# Report\n"}}


@pytest.fixture
def document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("NUTRIENT_API_KEY", "pdf_live_test")
    path = tmp_path / "invoice_scan" / "document.pdf"
    path.parent.mkdir()
    path.write_bytes(b"%PDF-1.7 test")
    return path


def answering(*responses: httpx.Response) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    requests: list[httpx.Request] = []
    left = list(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return left.pop(0)

    return httpx.MockTransport(handle), requests


def instructions(request: httpx.Request) -> dict[str, Any]:
    body = request.read().decode("latin-1")
    start = body.index('name="instructions"') + len('name="instructions"')
    return json.loads(body[start:].split("\r\n\r\n", 1)[1].split("\r\n--", 1)[0])


def test_fetch_posts_the_document_with_the_pinned_versions(document: Path) -> None:
    transport, requests = answering(httpx.Response(200, json=RESPONSE))
    raw = NutrientProvider(transport=transport).fetch(document)

    request = requests[0]
    assert (request.method, str(request.url)) == ("POST", API_URL)
    assert request.headers["Authorization"] == "Bearer pdf_live_test"
    assert request.headers["x-nutrient-api-version"] == "2026-05-25"
    assert request.headers["x-nutrient-engine-version"] == "latest"
    assert b"%PDF-1.7 test" in request.read()
    assert instructions(request) == {
        "mode": "understand",
        "output": {
            "formats": ["spatial", "markdown"],
            "includeHeadersAndFooters": False,
            "useHtmlTables": True,
            "enableSemanticBlockFormatting": True,
            "enableInlineFormatting": True,
        },
    }
    assert (raw.provider, raw.document_id, raw.payload) == (
        "nutrient",
        "invoice_scan",
        RESPONSE,
    )
    assert raw.metadata["version"] == "nutrient understand mode, API 2026-05-25, engine latest"


@pytest.mark.parametrize("mode", ["structure", "understand", "agentic"])
def test_config_sends_its_mode(document: Path, mode: str) -> None:
    transport, requests = answering(httpx.Response(200, json=RESPONSE))
    raw = NutrientProvider({"mode": mode}, transport=transport).fetch(document)
    assert instructions(requests[0])["mode"] == mode
    assert raw.provider == "nutrient"
    assert raw.metadata["config"] == {"mode": mode}


def test_a_missing_api_key_fails_before_any_request(
    document: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NUTRIENT_API_KEY")
    transport, requests = answering()
    with pytest.raises(CredentialError, match="NUTRIENT_API_KEY is not set"):
        NutrientProvider(transport=transport).fetch(document)
    assert requests == []


def test_a_rate_limit_or_backend_error_is_retried(document: Path) -> None:
    waits: list[float] = []
    transport, requests = answering(
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(503),
        httpx.Response(200, json=RESPONSE),
    )
    raw = NutrientProvider(transport=transport, sleep=waits.append).fetch(document)
    assert raw.payload == RESPONSE
    assert len(requests) == 3
    assert waits == [7.0, 4.0]


def test_retries_stop_after_the_last_try(document: Path) -> None:
    transport, requests = answering(*[httpx.Response(500)] * (RETRIES + 1))
    with pytest.raises(RuntimeError, match="Nutrient API 500"):
        NutrientProvider(transport=transport, sleep=lambda _: None).fetch(document)
    assert len(requests) == RETRIES + 1


@pytest.mark.parametrize("status", [400, 401, 402, 408, 413])
def test_other_errors_fail_at_once_without_echoing_remote_messages(
    document: Path, status: int
) -> None:
    error = {"status": status, "errorMessage": "Authorization: Bearer pdf_live_test"}
    transport, requests = answering(httpx.Response(status, json=error))
    with pytest.raises(RuntimeError, match=f"Nutrient API {status}: request failed") as error_info:
        NutrientProvider(transport=transport).fetch(document)
    assert len(requests) == 1
    assert "pdf_live_test" not in str(error_info.value)
