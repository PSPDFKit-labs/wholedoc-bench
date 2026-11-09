import asyncio
import base64
import io
import json
import shutil
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pypdf import PdfReader

from tests.conftest import DEMO
from wholedoc_bench.providers import AnthropicProvider, CredentialError
from wholedoc_bench.providers import anthropic as provider_module
from wholedoc_bench.providers.anthropic import (
    MAX_RETRIES,
    MAX_TOKENS,
    PAGE_SECONDS,
    PROMPT_DIGEST,
    SYSTEM_PROMPT,
    USER_PROMPT,
    previous_page_prompt,
)

KEY = "test-key-do-not-persist-42"
MODEL = "claude-opus-5-5"
ANSWER = '<div data-bbox="[100, 80, 900, 120]" data-label="Title"># Plain page</div>'
ENDING = '<div data-bbox="[100, 800, 900, 900]" data-label="Text">The table begins</div>'
VERSION = f"anthropic {MODEL}; effort medium; prompt {PROMPT_DIGEST}"


def answer(
    text: str = ANSWER,
    *,
    stop_reason: str = "end_turn",
    stop_details: dict[str, Any] | None = None,
    model: str = MODEL,
    thinking: bool = False,
) -> httpx2.Response:
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 1500, "output_tokens": 1},
    }
    events: list[dict[str, Any]] = [{"type": "message_start", "message": message}]
    blocks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if thinking:
        empty = {"type": "thinking", "thinking": "", "signature": ""}
        blocks.append((empty, {"type": "signature_delta", "signature": "opaque"}))
    blocks.append(({"type": "text", "text": ""}, {"type": "text_delta", "text": text}))
    for index, (block, delta) in enumerate(blocks):
        events += [
            {"type": "content_block_start", "index": index, "content_block": block},
            {"type": "content_block_delta", "index": index, "delta": delta},
            {"type": "content_block_stop", "index": index},
        ]
    events += [
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "stop_details": stop_details,
            },
            "usage": {"output_tokens": 420},
        },
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    headers = {"content-type": "text/event-stream", "request-id": "req_1"}
    return httpx2.Response(200, headers=headers, content=body.encode())


def stream_error(kind: str, message: str) -> httpx2.Response:
    started = answer().content.decode().split("event: content_block_start")[0]
    error = {"type": "error", "error": {"type": kind, "message": message}}
    body = f"{started}event: error\ndata: {json.dumps(error)}\n\n"
    headers = {"content-type": "text/event-stream", "request-id": "req_3"}
    return httpx2.Response(200, headers=headers, content=body.encode())


class _Dropped(httpx2.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield answer().content.split(b"\n\n")[0] + b"\n\n"
        raise httpx2.ReadError(f"connection reset while sending {KEY}")


def dropped() -> httpx2.Response:
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=_Dropped())


def failure(status: int, kind: str, message: str, retry: bool = False) -> httpx2.Response:
    headers = {"request-id": "req_2", "x-should-retry": "true" if retry else "false"}
    if retry:
        headers["retry-after-ms"] = "1"
    body = {"type": "error", "error": {"type": kind, "message": message}}
    return httpx2.Response(status, headers=headers, json=body)


class Anthropic:
    def __init__(
        self, *responses: httpx2.Response | Exception, request_seconds: float = 0.0
    ) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []
        self.waits: list[float] = []
        self.now = 0.0
        self.request_seconds = request_seconds

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        self.now += self.request_seconds
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds

    def provider(self, config: dict[str, Any] | None = None) -> AnthropicProvider:
        transport = httpx2.MockTransport(self.handle)
        return AnthropicProvider(
            config, transport=transport, sleep=self.sleep, clock=lambda: self.now
        )

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-that-must-not-be-sent")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    return monkeypatch


@pytest.fixture
def pdf(tmp_path: Path, key: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "plain_continuation_demo" / "document.pdf"
    path.parent.mkdir()
    shutil.copy(DEMO / "document.pdf", path)
    return path


def test_fetch_sends_each_pdf_page_with_the_layout_prompt(pdf: Path) -> None:
    api = Anthropic(answer(ENDING), answer(ANSWER, thinking=True), answer("page 3"))
    raw = api.provider().fetch(pdf)

    assert [str(r.url) for r in api.requests] == ["https://api.anthropic.com/v1/messages"] * 3
    for request, body in zip(api.requests, api.bodies(), strict=True):
        assert request.headers["x-api-key"] == KEY
        assert "authorization" not in request.headers
        assert body.keys() == {
            "model",
            "max_tokens",
            "system",
            "messages",
            "output_config",
            "stream",
        }
        assert (body["model"], body["max_tokens"], body["stream"]) == (MODEL, MAX_TOKENS, True)
        assert body["system"] == SYSTEM_PROMPT
        assert body["output_config"] == {"effort": "medium"}
        [message] = body["messages"]
        document, prompt, *_ = message["content"]
        assert prompt == {"type": "text", "text": USER_PROMPT}
        assert document["source"]["media_type"] == "application/pdf"
        page = base64.standard_b64decode(document["source"]["data"])
        assert len(PdfReader(io.BytesIO(page)).pages) == 1
    quotes = [body["messages"][0]["content"][2:] for body in api.bodies()]
    assert quotes == [[], [{"type": "text", "text": previous_page_prompt(ENDING)}], []]
    assert (raw.provider, raw.document_id) == ("anthropic", "plain_continuation_demo")
    assert [page["text"] for page in raw.payload["pages"]] == [ENDING, ANSWER, "page 3"]
    first = raw.payload["pages"][0]
    assert (first["model"], first["stop_reason"]) == (MODEL, "end_turn")
    assert first["usage"]["input_tokens"] == 1500
    assert first["usage"]["output_tokens"] == 420
    assert raw.metadata == {"version": VERSION, "config": {"model": MODEL, "effort": "medium"}}


@pytest.mark.parametrize(("suffix", "media_type"), [(".png", "image/png"), (".jpg", "image/jpeg")])
def test_an_image_is_sent_as_it_is(
    tmp_path: Path, key: pytest.MonkeyPatch, suffix: str, media_type: str
) -> None:
    path = tmp_path / "receipt_photo" / f"document{suffix}"
    path.parent.mkdir()
    path.write_bytes(b"image bytes")
    api = Anthropic(answer())
    raw = api.provider().fetch(path)

    [body] = api.bodies()
    image = body["messages"][0]["content"][0]
    assert image == {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": "aW1hZ2UgYnl0ZXM="},
    }
    assert len(raw.payload["pages"]) == 1


def test_the_configured_model_and_effort_are_sent(pdf: Path) -> None:
    sonnet = "claude-sonnet-5-5"
    api = Anthropic(*[answer(model=sonnet)] * 3)
    raw = api.provider({"model": sonnet, "effort": "high"}).fetch(pdf)

    assert {b["model"] for b in api.bodies()} == {sonnet}
    assert all(b["output_config"] == {"effort": "high"} for b in api.bodies())
    assert raw.metadata["version"] == f"anthropic {sonnet}; effort high; prompt {PROMPT_DIGEST}"


def test_no_effort_leaves_it_to_the_model(pdf: Path) -> None:
    api = Anthropic(*[answer(model="claude-haiku-4-5")] * 3)
    raw = api.provider({"model": "claude-haiku-4-5", "effort": None}).fetch(pdf)

    assert all("output_config" not in b for b in api.bodies())
    assert raw.metadata["version"].startswith("anthropic claude-haiku-4-5; effort default;")


def test_a_refusal_fails_the_document(pdf: Path) -> None:
    refused = answer(
        "", stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber"}
    )
    api = Anthropic(answer(), refused)
    with pytest.raises(RuntimeError, match=r"Claude declined page 2 \(cyber\)"):
        api.provider().fetch(pdf)
    assert len(api.requests) == 2


def test_an_answer_cut_at_the_token_cap_fails_the_document(pdf: Path) -> None:
    api = Anthropic(answer("<div data-label=", stop_reason="max_tokens"))
    with pytest.raises(RuntimeError, match=f"page 1 reached the {MAX_TOKENS} token cap"):
        api.provider().fetch(pdf)


def test_an_api_error_reports_its_status_type_and_request_only(pdf: Path) -> None:
    api = Anthropic(failure(400, "invalid_request_error", f"bad request echoing {KEY}"))
    with pytest.raises(RuntimeError) as caught:
        api.provider().fetch(pdf)
    message = str(caught.value)
    assert "page 1 (HTTP 400, invalid_request_error, request req_2)" in message
    assert KEY not in message
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


def test_an_overloaded_api_is_tried_again(pdf: Path) -> None:
    api = Anthropic(failure(529, "overloaded_error", "busy", retry=True), *[answer()] * 3)
    raw = api.provider().fetch(pdf)
    assert len(api.requests) == 4
    assert len(raw.payload["pages"]) == 3


def test_each_page_quotes_how_the_answer_for_the_page_before_it_ended(pdf: Path) -> None:
    second = ENDING.replace("The table begins", "on page two")
    busy = failure(529, "overloaded_error", "busy", retry=True)
    api = Anthropic(answer(ENDING), busy, answer(second), answer())
    api.provider().fetch(pdf)

    quotes = [body["messages"][0]["content"][2:] for body in api.bodies()]
    assert quotes == [
        [],
        [{"type": "text", "text": previous_page_prompt(ENDING)}],
        [{"type": "text", "text": previous_page_prompt(ENDING)}],
        [{"type": "text", "text": previous_page_prompt(second)}],
    ]
    assert "on page two" in quotes[3][0]["text"]


@pytest.mark.parametrize(
    ("headers", "waits"),
    [
        ({}, [2.0, 4.0]),
        ({"retry-after": "30"}, [30.0, 30.0]),
        ({"retry-after": "900"}, [60.0, 60.0]),
        ({"retry-after-ms": "1500", "retry-after": "30"}, [1.5, 1.5]),
    ],
)
def test_a_rate_limit_waits_for_the_apis_retry_after(
    pdf: Path, headers: dict[str, str], waits: list[float]
) -> None:
    limited = [
        httpx2.Response(429, headers=headers, json={"error": {"type": "rate_limit_error"}})
        for _ in range(2)
    ]
    api = Anthropic(*limited, *[answer()] * 3)
    api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (5, waits)


@pytest.mark.parametrize("status", [500, 503, 408])
def test_a_server_error_or_timeout_status_is_tried_again(pdf: Path, status: int) -> None:
    api = Anthropic(httpx2.Response(status, json={}), *[answer()] * 3)
    api.provider().fetch(pdf)
    assert len(api.requests) == 4


def test_the_apis_should_retry_hint_wins(pdf: Path) -> None:
    api = Anthropic(failure(503, "api_error", "do not retry", retry=False))
    with pytest.raises(RuntimeError, match=r"HTTP 503, api_error.*try again later"):
        api.provider().fetch(pdf)
    assert len(api.requests) == 1


def limited(retry_after: str) -> httpx2.Response:
    body = {"error": {"type": "rate_limit_error"}}
    return httpx2.Response(429, headers={"retry-after": retry_after}, json=body)


def test_each_request_may_take_only_the_time_left(pdf: Path) -> None:
    api = Anthropic(limited("60"), limited("60"), *[answer()] * 3, request_seconds=250.0)
    api.provider().fetch(pdf)
    timeouts = [r.extensions["timeout"] for r in api.requests]
    assert [t["read"] for t in timeouts] == [600.0, 600.0, PAGE_SECONDS - 620, 600.0, 600.0]
    assert timeouts[0]["connect"] == 5.0


def test_a_stream_past_the_page_time_is_stopped(pdf: Path) -> None:
    api = Anthropic(answer(), request_seconds=PAGE_SECONDS + 1)
    with pytest.raises(RuntimeError, match="page 1 took more than 20 minutes"):
        api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (1, [])


def test_no_retry_starts_that_would_pass_the_page_time(pdf: Path) -> None:
    api = Anthropic(limited("60"), request_seconds=1150.0)
    with pytest.raises(RuntimeError, match=r"page 1 \(HTTP 429, rate_limit_error\)"):
        api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (1, [])


class _Pings(httpx2.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield answer().content.split(b"\n\n")[0] + b"\n\n"
        while True:
            await asyncio.sleep(0.02)
            yield b'event: ping\ndata: {"type": "ping"}\n\n'


def test_a_stream_kept_alive_by_pings_is_stopped_at_the_page_time(
    pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider_module, "PAGE_SECONDS", 0.3)
    pings = httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=_Pings())
    provider = AnthropicProvider(transport=httpx2.MockTransport(lambda _: pings))
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="page 1 took more than"):
        provider.fetch(pdf)
    assert time.monotonic() - started < 2.0


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_a_redirect_is_not_followed_with_the_key(pdf: Path, status: int) -> None:
    moved = httpx2.Response(status, headers={"location": "https://elsewhere.example.com/v1"})
    api = Anthropic(moved, moved)
    with pytest.raises(ValueError, match=f"refused the check \\(HTTP {status}"):
        api.provider().prepare()
    with pytest.raises(RuntimeError, match=f"page 1 \\(HTTP {status}"):
        api.provider().fetch(pdf)
    assert {r.url.host for r in api.requests} == {"api.anthropic.com"}


def test_a_dropped_connection_fails_without_echoing_the_request(
    pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider_module, "MAX_RETRIES", 0)
    api = Anthropic(httpx2.ConnectError(f"connection reset while sending {KEY}"))
    with pytest.raises(RuntimeError, match="check network connectivity") as caught:
        api.provider().fetch(pdf)
    assert KEY not in str(caught.value)


def test_an_error_event_in_the_stream_is_tried_again(pdf: Path) -> None:
    api = Anthropic(stream_error("overloaded_error", "busy"), *[answer()] * 3)
    raw = api.provider().fetch(pdf)
    assert len(api.requests) == 4
    assert api.waits == [2.0]
    assert len(raw.payload["pages"]) == 3


def test_an_error_event_that_persists_fails_without_its_message(pdf: Path) -> None:
    api = Anthropic(*[stream_error("overloaded_error", f"busy {KEY}")] * (MAX_RETRIES + 1))
    with pytest.raises(RuntimeError) as caught:
        api.provider().fetch(pdf)
    assert str(caught.value) == (
        "Anthropic API request failed on page 1 (overloaded_error, request req_3); try again later."
    )
    assert api.waits == [2.0, 4.0, 8.0]


def test_an_error_event_of_another_type_is_not_tried_again(pdf: Path) -> None:
    api = Anthropic(stream_error("invalid_request_error", "bad"))
    with pytest.raises(RuntimeError, match=r"page 1 \(invalid_request_error, request req_3\)"):
        api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (1, [])


def cut_short() -> httpx2.Response:
    body = answer().content.split(b"\n\n")[0] + b"\n\n"
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def unreadable() -> httpx2.Response:
    body = b"event: message_start\ndata: {not json\n\n"
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)


@pytest.mark.parametrize("broken", [cut_short, unreadable])
def test_an_incomplete_or_unreadable_stream_is_tried_again(pdf: Path, broken: Any) -> None:
    api = Anthropic(broken(), *[answer()] * 3)
    raw = api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (4, [2.0])
    assert len(raw.payload["pages"]) == 3
    api = Anthropic(*[broken() for _ in range(MAX_RETRIES + 1)])
    with pytest.raises(RuntimeError, match="incomplete or unreadable answer for page 1"):
        api.provider().fetch(pdf)


def test_an_error_without_a_type_says_so(pdf: Path) -> None:
    body = b'event: error\ndata: {"type": "error", "error": {}}\n\n'
    response = httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    with pytest.raises(RuntimeError, match=r"page 1 \(no error type\)"):
        Anthropic(response).provider().fetch(pdf)


def test_a_connection_dropped_in_the_stream_is_tried_again(pdf: Path) -> None:
    api = Anthropic(dropped(), *[answer()] * 3)
    raw = api.provider().fetch(pdf)
    assert len(api.requests) == 4
    assert len(raw.payload["pages"]) == 3


def test_a_connection_that_keeps_dropping_fails_without_echoing_it(pdf: Path) -> None:
    api = Anthropic(*[dropped() for _ in range(MAX_RETRIES + 1)])
    with pytest.raises(RuntimeError, match="check network connectivity") as caught:
        api.provider().fetch(pdf)
    assert KEY not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


def test_custom_headers_from_the_environment_are_refused(
    pdf: Path, key: pytest.MonkeyPatch
) -> None:
    key.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Api-Key: other-key")
    api = Anthropic()
    for call in (api.provider().prepare, lambda: api.provider().fetch(pdf)):
        with pytest.raises(CredentialError, match="ANTHROPIC_CUSTOM_HEADERS is set") as caught:
            call()
        assert "other-key" not in str(caught.value)
    assert api.requests == []


@pytest.mark.parametrize("value", [None, "", " \t\n"])
def test_a_missing_api_key_fails_before_any_request(
    pdf: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY")
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    api = Anthropic()
    for call in (api.provider().prepare, lambda: api.provider().fetch(pdf)):
        with pytest.raises(CredentialError, match="ANTHROPIC_API_KEY is not set or is blank"):
            call()
    assert api.requests == []


def test_prepare_checks_the_key_and_model_with_a_free_call(key: pytest.MonkeyPatch) -> None:
    api = Anthropic(httpx2.Response(200, json={"id": MODEL, "type": "model"}))
    api.provider().prepare()
    [request] = api.requests
    assert (request.method, str(request.url)) == (
        "GET",
        f"https://api.anthropic.com/v1/models/{MODEL}",
    )
    assert request.headers["x-api-key"] == KEY


def model_info(effort: dict[str, Any]) -> httpx2.Response:
    levels = {level: {"supported": True} for level in ("low", "medium", "high", "max")}
    capability = {"supported": True, **levels, **effort}
    return httpx2.Response(200, json={"id": MODEL, "capabilities": {"effort": capability}})


@pytest.mark.parametrize(
    ("config", "effort", "accepted"),
    [
        ({}, {}, True),
        ({"effort": "xhigh"}, {}, False),
        ({"effort": "xhigh"}, {"xhigh": {"supported": True}}, True),
        ({"effort": "low"}, {"low": {"supported": False}}, False),
        ({}, {"supported": False}, False),
        ({"effort": None}, {"supported": False}, True),
    ],
)
def test_prepare_checks_that_the_model_takes_the_effort(
    key: pytest.MonkeyPatch, config: dict[str, Any], effort: dict[str, Any], accepted: bool
) -> None:
    provider = Anthropic(model_info(effort)).provider(config)
    if accepted:
        provider.prepare()
    else:
        with pytest.raises(ValueError, match="does not take effort"):
            provider.prepare()


def test_prepare_waits_at_most_a_minute_for_a_rate_limit(key: pytest.MonkeyPatch) -> None:
    api = Anthropic(limited("3600"), model_info({}))
    api.provider().prepare()
    assert api.waits == [60.0]


def test_prepare_stops_trying_after_two_minutes(key: pytest.MonkeyPatch) -> None:
    api = Anthropic(limited("60"), limited("60"))
    with pytest.raises(ValueError, match=r"busy or unavailable \(HTTP 429"):
        api.provider().prepare()
    assert (len(api.requests), api.waits) == (2, [60.0])


def test_a_key_is_sent_without_surrounding_whitespace(key: pytest.MonkeyPatch) -> None:
    key.setenv("ANTHROPIC_API_KEY", f" {KEY}\n")
    api = Anthropic(model_info({}))
    api.provider().prepare()
    assert api.requests[0].headers["x-api-key"] == KEY


def test_a_key_with_characters_no_key_has_fails_without_echoing_it(
    key: pytest.MonkeyPatch,
) -> None:
    key.setenv("ANTHROPIC_API_KEY", "sk-ant-clé")
    api = Anthropic()
    with pytest.raises(CredentialError, match="holds characters that no key has") as caught:
        api.provider().prepare()
    assert "clé" not in str(caught.value)
    assert api.requests == []


def test_a_rejected_key_fails_prepare_without_echoing_it(key: pytest.MonkeyPatch) -> None:
    api = Anthropic(failure(401, "authentication_error", f"invalid x-api-key {KEY}"))
    with pytest.raises(CredentialError, match="did not accept the key") as caught:
        api.provider().prepare()
    assert "HTTP 401, authentication_error, request req_2" in str(caught.value)
    assert KEY not in str(caught.value)


def test_a_key_without_access_to_the_model_is_not_called_rejected(
    pdf: Path, key: pytest.MonkeyPatch
) -> None:
    api = Anthropic(failure(403, "permission_error", "no access"))
    with pytest.raises(CredentialError, match="has no access to claude-opus-5-5") as caught:
        api.provider().prepare()
    assert "did not accept" not in str(caught.value)
    api = Anthropic(failure(403, "permission_error", "no access"))
    with pytest.raises(RuntimeError, match="check the key's access to the model"):
        api.provider().fetch(pdf)


def test_an_unknown_model_fails_prepare(key: pytest.MonkeyPatch) -> None:
    api = Anthropic(failure(404, "not_found_error", "model: claude-nope"))
    with pytest.raises(ValueError, match="no model 'claude-nope'.*check the model in --config"):
        api.provider({"model": "claude-nope"}).prepare()


@pytest.mark.parametrize(("status", "kind"), [(429, "rate_limit_error"), (529, "overloaded_error")])
def test_a_rate_limit_or_an_outage_at_prepare_does_not_blame_the_key(
    key: pytest.MonkeyPatch, status: int, kind: str
) -> None:
    api = Anthropic(failure(status, kind, "busy"))
    with pytest.raises(ValueError, match="busy or unavailable.*try again later") as caught:
        api.provider().prepare()
    assert not isinstance(caught.value, CredentialError)


def test_an_unreachable_api_or_another_refusal_fails_prepare(
    key: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider_module, "MAX_RETRIES", 0)
    with pytest.raises(ValueError, match="could not be reached; check the network"):
        Anthropic(httpx2.ConnectError(f"reset {KEY}")).provider().prepare()
    api = Anthropic(failure(400, "invalid_request_error", f"bad {KEY}"))
    with pytest.raises(ValueError, match=r"refused the check \(HTTP 400") as caught:
        api.provider().prepare()
    assert KEY not in str(caught.value)


def test_config_defaults_and_choices() -> None:
    assert AnthropicProvider().config == {"model": MODEL, "effort": "medium"}
    assert AnthropicProvider({"effort": None}).config == {"model": MODEL, "effort": None}
    with pytest.raises(ValueError, match="unsupported config keys: api_key"):
        AnthropicProvider({"api_key": KEY})
    with pytest.raises(ValueError, match="model must be a Claude model id"):
        AnthropicProvider({"model": "claude opus"})
    with pytest.raises(ValueError, match="effort must be one of low, medium, high, xhigh, max"):
        AnthropicProvider({"effort": "extreme"})
