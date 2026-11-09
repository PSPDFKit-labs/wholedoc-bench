import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.conftest import TWO_COLUMN_AZURE
from wholedoc_bench.providers import AzureDIProvider, CredentialError
from wholedoc_bench.providers.azure_di import CHECK_SECONDS, FEATURES, MAX_WAIT_SECONDS, RETRIES

ENDPOINT = "https://example-resource.cognitiveservices.azure.com"
KEY = "test-key-do-not-persist-42"
ANALYZE = f"{ENDPOINT}/documentintelligence/documentModels/prebuilt-layout:analyze"
STATUS = (
    f"{ENDPOINT}/documentintelligence/documentModels/prebuilt-layout/analyzeResults/r1"
    "?api-version=2024-11-30"
)
RESULT = json.loads(TWO_COLUMN_AZURE.read_text(encoding="utf-8"))
ALL_FEATURES = "barcodes,formulas,ocrHighResolution,styleFont"


@pytest.fixture
def document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", f"{ENDPOINT}/")
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", KEY)
    path = tmp_path / "invoice_scan" / "document.pdf"
    path.parent.mkdir()
    path.write_bytes(b"%PDF-1.7 test")
    return path


class Azure:
    def __init__(self, *responses: httpx.Response | Exception, request_seconds: float = 0.0):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []
        self.waits: list[float] = []
        self.now = 0.0
        self.request_seconds = request_seconds

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.now += self.request_seconds
        answer = self.responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds

    def provider(self, config: dict[str, Any] | None = None) -> AzureDIProvider:
        transport = httpx.MockTransport(self.handle)
        return AzureDIProvider(
            config, transport=transport, sleep=self.sleep, clock=lambda: self.now
        )


def started(location: str = STATUS, retry_after: str | None = None) -> httpx.Response:
    headers = {"Operation-Location": location}
    if retry_after:
        headers["Retry-After"] = retry_after
    return httpx.Response(202, headers=headers)


def running(retry_after: str | None = "1") -> httpx.Response:
    headers = {"Retry-After": retry_after} if retry_after else {}
    return httpx.Response(200, headers=headers, json={"status": "running"})


def succeeded() -> httpx.Response:
    return httpx.Response(200, json=RESULT)


def error(status: int, code: str, message: str, inner: str | None = None) -> httpx.Response:
    body: dict[str, Any] = {"code": code, "message": message}
    if inner:
        body["innererror"] = {"code": inner, "message": message}
    return httpx.Response(status, json={"error": body})


def test_fetch_posts_the_document_and_polls_until_it_succeeds(document: Path) -> None:
    waiting = httpx.Response(200, json={"status": "notStarted"})
    azure = Azure(started(), running("1"), waiting, succeeded())
    raw = azure.provider().fetch(document)

    post, *polls = azure.requests
    assert (post.method, str(post.url).split("?")[0]) == ("POST", ANALYZE)
    assert dict(post.url.params) == {
        "api-version": "2024-11-30",
        "outputContentFormat": "markdown",
        "stringIndexType": "unicodeCodePoint",
        "features": ALL_FEATURES,
    }
    assert post.headers["Ocp-Apim-Subscription-Key"] == KEY
    assert post.headers["Content-Type"] == "application/octet-stream"
    assert post.read() == b"%PDF-1.7 test"
    assert [(r.method, str(r.url)) for r in polls] == [("GET", STATUS)] * 3
    assert all(r.headers["Ocp-Apim-Subscription-Key"] == KEY for r in polls)
    assert azure.waits == [2.0, 1.0, 2.0]
    assert (raw.provider, raw.document_id, raw.payload) == ("azure-di", "invoice_scan", RESULT)
    assert raw.metadata == {
        "version": "azure-di prebuilt-layout; API 2024-11-30; "
        "features barcodes+formulas+ocrHighResolution+styleFont",
        "config": {"features": list(FEATURES)},
    }


@pytest.mark.parametrize(
    ("features", "sent"), [(["styleFont", "barcodes"], "barcodes,styleFont"), ([], None)]
)
def test_the_configured_features_are_sent(
    document: Path, features: list[str], sent: str | None
) -> None:
    azure = Azure(started(), succeeded())
    raw = azure.provider({"features": features}).fetch(document)
    assert azure.requests[0].url.params.get("features") == sent
    assert raw.metadata["version"].endswith(f"features {(sent or 'none').replace(',', '+')}")


def test_an_image_is_sent_with_the_features(document: Path) -> None:
    image = document.with_name("document.PNG")
    image.write_bytes(b"\x89PNG image")
    azure = Azure(started(), succeeded())
    azure.provider().fetch(image)
    assert azure.requests[0].url.params["features"] == ALL_FEATURES


def test_an_office_file_is_sent_without_add_on_features(document: Path) -> None:
    docx = document.with_name("document.docx")
    docx.write_bytes(b"PK docx")
    azure = Azure(started(), succeeded())
    raw = azure.provider().fetch(docx)
    assert "features" not in azure.requests[0].url.params
    assert raw.metadata["version"].endswith("features none")


def test_rate_limits_are_tried_again_after_retry_after(document: Path) -> None:
    limited = httpx.Response(429, headers={"Retry-After": "7"})
    azure = Azure(limited, started(), httpx.Response(503), succeeded())
    azure.provider().fetch(document)
    assert [r.method for r in azure.requests] == ["POST", "POST", "GET", "GET"]
    assert azure.waits == [7.0, 2.0, 2.0]


def test_a_billed_start_is_not_tried_again_after_a_gateway_error(document: Path) -> None:
    azure = Azure(httpx.Response(504))
    with pytest.raises(RuntimeError, match=r"failed \(HTTP 504\)"):
        azure.provider().fetch(document)
    assert len(azure.requests) == 1


def test_a_poll_is_tried_again_after_a_dropped_connection(document: Path) -> None:
    azure = Azure(started(), httpx.ReadError("connection reset"), succeeded())
    raw = azure.provider().fetch(document)
    assert [r.method for r in azure.requests] == ["POST", "GET", "GET"]
    assert azure.waits == [2.0, 2.0]
    assert raw.payload == RESULT


def test_a_lasting_rate_limit_fails_after_the_retries(document: Path) -> None:
    azure = Azure(*[httpx.Response(429)] * (RETRIES + 1))
    with pytest.raises(RuntimeError, match=r"failed \(HTTP 429\)"):
        azure.provider().fetch(document)
    assert len(azure.requests) == RETRIES + 1


def test_a_rejected_document_reports_its_codes_without_the_message(document: Path) -> None:
    rejected = error(400, "InvalidRequest", f"bad file from {KEY}", inner="InvalidContent")
    azure = Azure(rejected)
    with pytest.raises(RuntimeError, match=r"failed \(HTTP 400, InvalidContent\)") as e:
        azure.provider().fetch(document)
    assert KEY not in str(e.value)


@pytest.mark.parametrize("status", ["failed", "canceled"])
def test_a_failed_analysis_reports_its_code_without_the_message(
    document: Path, status: str
) -> None:
    failed = httpx.Response(
        200, json={"status": status, "error": {"code": "InternalServerError", "message": KEY}}
    )
    azure = Azure(started(), failed)
    with pytest.raises(RuntimeError, match=rf"ended as '{status}' \(InternalServerError\)") as e:
        azure.provider().fetch(document)
    assert KEY not in str(e.value)


@pytest.mark.parametrize(
    "location",
    [
        "https://elsewhere.example.com/analyzeResults/r1",
        STATUS.replace("https://", "http://"),
        STATUS.replace(".com/", ".com:8443/"),
        None,
    ],
)
def test_a_status_url_off_the_endpoint_gets_no_key(document: Path, location: str | None) -> None:
    azure = Azure(started(location) if location else httpx.Response(202))
    with pytest.raises(RuntimeError, match="no status URL on the endpoint's host"):
        azure.provider().fetch(document)
    assert len(azure.requests) == 1


@pytest.mark.parametrize(("retry_after", "polls"), [("60", 10), ("0", 600), ("86400", 10)])
def test_an_analysis_that_does_not_finish_fails(
    document: Path, retry_after: str, polls: int
) -> None:
    azure = Azure(started(retry_after=retry_after), *[running(retry_after)] * polls)
    with pytest.raises(RuntimeError, match="did not finish within 600 seconds"):
        azure.provider().fetch(document)
    assert len(azure.waits) == polls and sum(azure.waits) == MAX_WAIT_SECONDS


def test_retry_waits_count_towards_the_time_limit(document: Path) -> None:
    limited = httpx.Response(429, headers={"Retry-After": "60"})
    azure = Azure(started(), *[limited, limited, limited, running("1")] * 20)
    with pytest.raises(RuntimeError, match=r"HTTP 429|did not finish within 600 seconds"):
        azure.provider().fetch(document)
    assert azure.now <= MAX_WAIT_SECONDS


def test_a_poll_does_not_wait_past_the_time_limit(document: Path) -> None:
    azure = Azure(started(retry_after="60"), *[running("60")] * 20, request_seconds=10.0)
    with pytest.raises(RuntimeError, match="did not finish within 600 seconds"):
        azure.provider().fetch(document)
    assert azure.now <= MAX_WAIT_SECONDS


def test_request_time_counts_towards_the_time_limit(document: Path) -> None:
    azure = Azure(started(), *[running("1")] * 100, request_seconds=100.0)
    with pytest.raises(RuntimeError, match="did not finish within 600 seconds"):
        azure.provider().fetch(document)
    assert len(azure.requests) == 6
    assert azure.requests[-1].extensions["timeout"]["read"] == pytest.approx(94.0)


def test_a_transport_error_hides_its_message(document: Path) -> None:
    azure = Azure(httpx.ConnectError(f"proxy echoed {KEY}"))
    with pytest.raises(RuntimeError, match="check network connectivity") as e:
        azure.provider().fetch(document)
    assert KEY not in str(e.value) and len(azure.requests) == 1
    assert e.value.__cause__ is None and e.value.__suppress_context__


@pytest.mark.parametrize(
    "variable", ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "AZURE_DOCUMENT_INTELLIGENCE_KEY"]
)
def test_missing_settings_fail_before_any_request(
    document: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv(variable, " ")
    azure = Azure()
    for step in (azure.provider().prepare, lambda: azure.provider().fetch(document)):
        with pytest.raises(CredentialError, match=f"{variable} is not set or is blank"):
            step()
    assert azure.requests == []


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example-resource.cognitiveservices.azure.com/",
        "https://example-resource.cognitiveservices.azure.com/documentintelligence",
        "https://user:secret@example-resource.cognitiveservices.azure.com/",
        "example-resource",
    ],
)
def test_an_endpoint_that_is_not_an_https_host_is_refused(
    document: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", endpoint)
    azure = Azure()
    with pytest.raises(CredentialError, match="must be the resource's https endpoint") as e:
        azure.provider().prepare()
    assert "secret" not in str(e.value)
    assert azure.requests == []


def test_a_key_is_sent_without_surrounding_whitespace(
    document: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", f" {KEY}\n")
    azure = Azure(httpx.Response(200, json={}))
    azure.provider().prepare()
    assert azure.requests[0].headers["Ocp-Apim-Subscription-Key"] == KEY


def test_a_key_with_characters_no_key_has_fails_without_echoing_it(
    document: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", f"\u201c{KEY}\u201d")
    azure = Azure()
    with pytest.raises(CredentialError, match="holds characters that no key has") as e:
        azure.provider().prepare()
    assert KEY not in str(e.value) and azure.requests == []


def test_prepare_checks_the_key_with_the_free_info_call(document: Path) -> None:
    azure = Azure(httpx.Response(200, json={"customDocumentModels": {"count": 0}}))
    azure.provider().prepare()
    [request] = azure.requests
    assert (request.method, str(request.url)) == (
        "GET",
        f"{ENDPOINT}/documentintelligence/info?api-version=2024-11-30",
    )
    assert request.headers["Ocp-Apim-Subscription-Key"] == KEY


def test_a_rejected_key_fails_prepare_without_echoing_it(document: Path) -> None:
    azure = Azure(error(401, "401", f"Access denied for {KEY}"))
    with pytest.raises(CredentialError, match=r"did not accept the key \(HTTP 401\)") as e:
        azure.provider().prepare()
    assert KEY not in str(e.value)
    assert "Do not put credentials in --config" in str(e.value)


@pytest.mark.parametrize(
    ("status", "message"),
    [(429, "is limiting requests"), (503, "is unavailable"), (500, "is unavailable")],
)
def test_throttling_or_an_outage_at_prepare_does_not_blame_the_key(
    document: Path, status: int, message: str
) -> None:
    azure = Azure(*[httpx.Response(status, headers={"Retry-After": "1"})] * (RETRIES + 1))
    with pytest.raises(ValueError, match=rf"{message} \(HTTP {status}\); try again later") as e:
        azure.provider().prepare()
    assert not isinstance(e.value, CredentialError)
    assert "did not accept the key" not in str(e.value)


def test_a_wrong_endpoint_path_at_prepare_names_the_endpoint(document: Path) -> None:
    azure = Azure(error(404, "NotFound", f"no route for {KEY}"))
    with pytest.raises(CredentialError, match=r"answered HTTP 404, NotFound at the endpoint") as e:
        azure.provider().prepare()
    assert "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT" in str(e.value) and KEY not in str(e.value)


def test_prepare_gives_up_after_its_time_limit(document: Path) -> None:
    azure = Azure(*[httpx.ReadTimeout("slow")] * (RETRIES + 1), request_seconds=CHECK_SECONDS)
    with pytest.raises(ValueError, match="did not answer within 120 seconds"):
        azure.provider().prepare()
    assert len(azure.requests) == 1


def test_an_unreachable_endpoint_fails_prepare(document: Path) -> None:
    azure = Azure(*[httpx.ConnectError(f"DNS failure, key {KEY}")] * (RETRIES + 1))
    with pytest.raises(CredentialError, match="could not be reached") as e:
        azure.provider().prepare()
    assert KEY not in str(e.value) and len(azure.requests) == RETRIES + 1


def test_the_provider_reads_documents_and_images() -> None:
    assert {".pdf", ".png", ".jpg", ".tiff", ".docx", ".xlsx", ".pptx"} <= AzureDIProvider.formats


def test_every_measured_feature_is_on_by_default_and_can_be_left_out() -> None:
    assert AzureDIProvider().config == {
        "features": ["barcodes", "formulas", "ocrHighResolution", "styleFont"]
    }
    chosen = AzureDIProvider({"features": ["styleFont", "barcodes", "styleFont"]})
    assert chosen.config == {"features": ["barcodes", "styleFont"]}
    assert AzureDIProvider.validate_config(chosen.config) == chosen.config
    assert AzureDIProvider({"features": []}).config == {"features": []}


@pytest.mark.parametrize(
    ("config", "error"),
    [
        ({"key": KEY}, "unsupported config keys: key"),
        ({"features": ["queryFields"]}, "unsupported features: queryFields; choose from"),
        ({"features": "barcodes"}, "features must be a list of feature names"),
    ],
)
def test_other_options_and_features_are_refused(config: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error) as e:
        AzureDIProvider(config)
    assert KEY not in str(e.value)
