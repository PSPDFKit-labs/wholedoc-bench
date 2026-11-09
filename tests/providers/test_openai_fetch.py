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
from wholedoc_bench.providers import CredentialError, OpenAIProvider
from wholedoc_bench.providers import openai as provider_module
from wholedoc_bench.providers.openai import (
    MAX_RETRIES,
    MAX_TOKENS,
    PAGE_SECONDS,
    PROMPT_DIGEST,
    SYSTEM_PROMPT,
    USER_PROMPT,
    previous_page_prompt,
)

KEY = "test-key-do-not-persist-42"
MODEL = "gpt-6.1-sol"
ANSWER = '<div data-bbox="[100, 80, 900, 120]" data-label="Title"># Plain page</div>'
ENDING = '<div data-bbox="[100, 800, 900, 900]" data-label="Text">The table begins</div>'
VERSION = f"openai {MODEL}; effort medium; prompt {PROMPT_DIGEST}"
USAGE = {
    "input_tokens": 1500,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 420,
    "output_tokens_details": {"reasoning_tokens": 300},
    "total_tokens": 1920,
}


def response(
    status: str = "completed",
    output: list[dict[str, Any]] | None = None,
    *,
    model: str = MODEL,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1760000000,
        "status": status,
        "model": model,
        "output": [] if output is None else output,
        "usage": USAGE if status != "in_progress" else None,
        "incomplete_details": None,
        "error": None,
        **fields,
    }


def message(*parts: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": list(parts),
    }


def text(value: str) -> dict[str, Any]:
    return {"type": "output_text", "text": value, "annotations": []}


REASONING = {"type": "reasoning", "id": "rs_1", "summary": []}


def sse(*events: dict[str, Any], request_id: str = "req_1") -> httpx2.Response:
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    headers = {"content-type": "text/event-stream", "x-request-id": request_id}
    return httpx2.Response(200, headers=headers, content=body.encode())


def started() -> list[dict[str, Any]]:
    return [
        {"type": "response.created", "sequence_number": 0, "response": response("in_progress")},
        {"type": "response.in_progress", "sequence_number": 1, "response": response("in_progress")},
    ]


def answer(value: str = ANSWER, *, model: str = MODEL, reasoning: bool = False) -> httpx2.Response:
    output = [*([REASONING] if reasoning else []), message(text(value))]
    half = len(value) // 2
    deltas = [
        {
            "type": "response.output_text.delta",
            "sequence_number": 2 + index,
            "item_id": "msg_1",
            "output_index": len(output) - 1,
            "content_index": 0,
            "delta": delta,
            "logprobs": [],
        }
        for index, delta in enumerate((value[:half], value[half:]))
    ]
    done = {
        "type": "response.completed",
        "sequence_number": 4,
        "response": response(output=output, model=model),
    }
    return sse(*started(), *deltas, done)


def ended(kind: str, ending: dict[str, Any], request_id: str = "req_3") -> httpx2.Response:
    return sse(*started(), {"type": kind, "sequence_number": 2, **ending}, request_id=request_id)


def stream_error(code: str | None, message_text: str) -> httpx2.Response:
    return ended("error", {"code": code, "message": message_text, "param": None})


def failed(code: str, message_text: str) -> httpx2.Response:
    error = {"code": code, "message": message_text}
    return ended("response.failed", {"response": response("failed", error=error)})


def incomplete(reason: str, value: str = "<div data-label=") -> httpx2.Response:
    details = {"reason": reason}
    out = response("incomplete", [message(text(value))], incomplete_details=details)
    return ended("response.incomplete", {"response": out})


class _Dropped(httpx2.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield answer().content.split(b"\n\n")[0] + b"\n\n"
        raise httpx2.ReadError(f"connection reset while sending {KEY}")


def dropped() -> httpx2.Response:
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=_Dropped())


def failure(
    status: int, kind: str, message_text: str, *, code: str | None = None, retry: bool = False
) -> httpx2.Response:
    headers = {"x-request-id": "req_2", "x-should-retry": "true" if retry else "false"}
    if retry:
        headers["retry-after-ms"] = "1"
    error = {"type": kind, "code": code, "message": message_text, "param": None}
    return httpx2.Response(status, headers=headers, json={"error": error})


class OpenAI:
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

    def provider(self, config: dict[str, Any] | None = None) -> OpenAIProvider:
        transport = httpx2.MockTransport(self.handle)
        return OpenAIProvider(config, transport=transport, sleep=self.sleep, clock=lambda: self.now)

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.com/v1")
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
    monkeypatch.delenv("OPENAI_PROJECT_ID", raising=False)
    return monkeypatch


@pytest.fixture
def pdf(tmp_path: Path, key: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "plain_continuation_demo" / "document.pdf"
    path.parent.mkdir()
    shutil.copy(DEMO / "document.pdf", path)
    return path


def test_fetch_sends_each_pdf_page_with_the_layout_prompt(pdf: Path) -> None:
    api = OpenAI(answer(ENDING), answer(ANSWER, reasoning=True), answer("page 3"))
    raw = api.provider().fetch(pdf)

    assert [str(r.url) for r in api.requests] == ["https://api.openai.com/v1/responses"] * 3
    for request, body in zip(api.requests, api.bodies(), strict=True):
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert body.keys() == {
            "model",
            "max_output_tokens",
            "input",
            "reasoning",
            "store",
            "stream",
        }
        assert (body["model"], body["max_output_tokens"]) == (MODEL, MAX_TOKENS)
        assert (body["stream"], body["store"]) == (True, False)
        assert body["reasoning"] == {"effort": "medium"}
        system, user = body["input"]
        assert system == {"role": "system", "content": SYSTEM_PROMPT}
        assert user["role"] == "user"
        document, prompt, *_ = user["content"]
        assert prompt == {"type": "input_text", "text": USER_PROMPT}
        assert (document["type"], document["filename"], document["detail"]) == (
            "input_file",
            "page.pdf",
            "auto",
        )
        prefix = "data:application/pdf;base64,"
        assert document["file_data"].startswith(prefix)
        page = base64.standard_b64decode(document["file_data"].removeprefix(prefix))
        assert len(PdfReader(io.BytesIO(page)).pages) == 1
    quotes = [body["input"][1]["content"][2:] for body in api.bodies()]
    assert quotes == [[], [{"type": "input_text", "text": previous_page_prompt(ENDING)}], []]
    assert (raw.provider, raw.document_id) == ("openai", "plain_continuation_demo")
    assert [page["text"] for page in raw.payload["pages"]] == [ENDING, ANSWER, "page 3"]
    first = raw.payload["pages"][0]
    assert (first["model"], first["status"]) == (MODEL, "completed")
    assert first["usage"]["input_tokens"] == 1500
    assert first["usage"]["output_tokens_details"]["reasoning_tokens"] == 300
    assert raw.metadata == {"version": VERSION, "config": {"model": MODEL, "effort": "medium"}}


@pytest.mark.parametrize(("suffix", "media_type"), [(".png", "image/png"), (".jpg", "image/jpeg")])
def test_an_image_is_sent_as_it_is(
    tmp_path: Path, key: pytest.MonkeyPatch, suffix: str, media_type: str
) -> None:
    path = tmp_path / "receipt_photo" / f"document{suffix}"
    path.parent.mkdir()
    path.write_bytes(b"image bytes")
    api = OpenAI(answer())
    raw = api.provider().fetch(path)

    [body] = api.bodies()
    image = body["input"][1]["content"][0]
    assert image == {
        "type": "input_image",
        "image_url": f"data:{media_type};base64,aW1hZ2UgYnl0ZXM=",
        "detail": "auto",
    }
    assert len(raw.payload["pages"]) == 1


def test_the_configured_model_and_effort_are_sent(pdf: Path) -> None:
    luna = "gpt-6-luna"
    api = OpenAI(*[answer(model=luna)] * 3)
    raw = api.provider({"model": luna, "effort": "high"}).fetch(pdf)

    assert {b["model"] for b in api.bodies()} == {luna}
    assert all(b["reasoning"] == {"effort": "high"} for b in api.bodies())
    assert raw.metadata["version"] == f"openai {luna}; effort high; prompt {PROMPT_DIGEST}"


def test_no_effort_leaves_it_to_the_model(pdf: Path) -> None:
    api = OpenAI(*[answer(model="gpt-4.1")] * 3)
    raw = api.provider({"model": "gpt-4.1", "effort": None}).fetch(pdf)

    assert all("reasoning" not in b for b in api.bodies())
    assert raw.metadata["version"].startswith("openai gpt-4.1; effort default;")


def test_the_answer_is_the_text_of_every_message_in_order(pdf: Path) -> None:
    commentary = {**message(text("Let me parse this page. ")), "phase": "commentary"}
    final = {**message(text("first "), text("second")), "phase": "final_answer"}
    output = [REASONING, commentary, final, message(text(" third"))]
    done = {"type": "response.completed", "sequence_number": 2, "response": response(output=output)}
    api = OpenAI(sse(*started(), done), *[answer()] * 2)
    raw = api.provider().fetch(pdf)
    assert raw.payload["pages"][0]["text"] == "first second third"


@pytest.mark.parametrize(
    "broken",
    [
        {"type": "response.completed", "sequence_number": 2},
        {"type": "response.completed", "sequence_number": 2, "response": None},
        {
            "type": "response.completed",
            "sequence_number": 2,
            "response": response(output=[message({"type": "output_text"})]),
        },
    ],
    ids=["no response", "null response", "text without text"],
)
def test_a_final_response_that_cannot_be_read_is_tried_again(
    pdf: Path, broken: dict[str, Any]
) -> None:
    api = OpenAI(sse(*started(), broken), *[answer()] * 3)
    api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (4, [2.0])


def test_an_event_that_is_not_an_object_is_tried_again(pdf: Path) -> None:
    body = b"event: response.created\ndata: 42\n\n"
    odd = httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    api = OpenAI(odd, *[answer()] * 3)
    api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (4, [2.0])


def test_each_page_quotes_how_the_answer_for_the_page_before_it_ended(pdf: Path) -> None:
    second = ENDING.replace("The table begins", "on page two")
    busy = failure(500, "server_error", "busy", retry=True)
    api = OpenAI(answer(ENDING), busy, answer(second), answer())
    api.provider().fetch(pdf)

    quotes = [body["input"][1]["content"][2:] for body in api.bodies()]
    assert quotes == [
        [],
        [{"type": "input_text", "text": previous_page_prompt(ENDING)}],
        [{"type": "input_text", "text": previous_page_prompt(ENDING)}],
        [{"type": "input_text", "text": previous_page_prompt(second)}],
    ]
    assert "on page two" in quotes[3][0]["text"]


def test_a_model_name_that_is_no_model_id_stays_out_of_the_version(pdf: Path) -> None:
    api = OpenAI(*[answer(model="gpt,6 sol")] * 3)
    raw = api.provider().fetch(pdf)
    assert raw.metadata["version"] == VERSION
    assert raw.payload["pages"][0]["model"] == "gpt,6 sol"


def test_a_refusal_fails_the_document_without_its_text(pdf: Path) -> None:
    refusal = {"type": "refusal", "refusal": f"I cannot help with {KEY}"}
    done = {
        "type": "response.completed",
        "sequence_number": 2,
        "response": response(output=[message(refusal)]),
    }
    api = OpenAI(answer(), sse(*started(), done))
    with pytest.raises(RuntimeError, match=r"^The model declined page 2\.$"):
        api.provider().fetch(pdf)
    assert len(api.requests) == 2


def test_an_answer_cut_at_the_token_cap_fails_the_document(pdf: Path) -> None:
    api = OpenAI(incomplete("max_output_tokens"))
    with pytest.raises(RuntimeError, match=f"page 1 reached the {MAX_TOKENS} token cap"):
        api.provider().fetch(pdf)
    assert api.waits == []


def test_an_answer_stopped_by_the_content_filter_fails_the_document(pdf: Path) -> None:
    api = OpenAI(incomplete("content_filter"))
    with pytest.raises(RuntimeError, match=r"OpenAI stopped page 1 \(content_filter\)"):
        api.provider().fetch(pdf)


def test_an_api_error_reports_its_status_type_code_and_request_only(pdf: Path) -> None:
    api = OpenAI(
        failure(400, "invalid_request_error", f"bad request echoing {KEY}", code="invalid_value")
    )
    with pytest.raises(RuntimeError) as caught:
        api.provider().fetch(pdf)
    message_text = str(caught.value)
    assert "page 1 (HTTP 400, invalid_request_error, invalid_value, request req_2)" in message_text
    assert KEY not in message_text
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


def test_an_error_type_and_code_that_agree_are_named_once(pdf: Path) -> None:
    api = OpenAI(failure(429, "insufficient_quota", "no credit", code="insufficient_quota"))
    with pytest.raises(RuntimeError) as caught:
        api.provider().fetch(pdf)
    assert str(caught.value) == (
        "OpenAI API request failed on page 1 (HTTP 429, insufficient_quota, request req_2); "
        "check the account's billing."
    )


def test_a_used_up_quota_is_not_tried_again(pdf: Path) -> None:
    quota = failure(429, "insufficient_quota", "no credit", code="insufficient_quota", retry=True)
    api = OpenAI(quota)
    with pytest.raises(RuntimeError, match="check the account's billing"):
        api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (1, [])


def test_an_overloaded_api_is_tried_again(pdf: Path) -> None:
    busy = failure(503, "server_error", "busy", code="server_is_overloaded", retry=True)
    api = OpenAI(busy, *[answer()] * 3)
    raw = api.provider().fetch(pdf)
    assert len(api.requests) == 4
    assert len(raw.payload["pages"]) == 3


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
        httpx2.Response(429, headers=headers, json={"error": {"code": "rate_limit_exceeded"}})
        for _ in range(2)
    ]
    api = OpenAI(*limited, *[answer()] * 3)
    api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (5, waits)


@pytest.mark.parametrize("status", [500, 503, 408])
def test_a_server_error_or_timeout_status_is_tried_again(pdf: Path, status: int) -> None:
    api = OpenAI(httpx2.Response(status, json={}), *[answer()] * 3)
    api.provider().fetch(pdf)
    assert len(api.requests) == 4


def test_the_apis_should_retry_hint_wins(pdf: Path) -> None:
    api = OpenAI(failure(503, "server_error", "do not retry", retry=False))
    with pytest.raises(RuntimeError, match=r"HTTP 503, server_error.*try again later"):
        api.provider().fetch(pdf)
    assert len(api.requests) == 1
    api = OpenAI(failure(400, "invalid_request_error", "retry", retry=True), *[answer()] * 3)
    api.provider().fetch(pdf)
    assert len(api.requests) == 4


def limited(retry_after: str) -> httpx2.Response:
    body = {"error": {"type": "requests", "code": "rate_limit_exceeded"}}
    return httpx2.Response(429, headers={"retry-after": retry_after}, json=body)


def test_each_request_may_take_only_the_time_left(pdf: Path) -> None:
    api = OpenAI(limited("60"), limited("60"), *[answer()] * 3, request_seconds=250.0)
    api.provider().fetch(pdf)
    timeouts = [r.extensions["timeout"] for r in api.requests]
    assert [t["read"] for t in timeouts] == [600.0, 600.0, PAGE_SECONDS - 620, 600.0, 600.0]
    assert timeouts[0]["connect"] == 5.0


def test_a_stream_past_the_page_time_is_stopped(pdf: Path) -> None:
    api = OpenAI(answer(), request_seconds=PAGE_SECONDS + 1)
    with pytest.raises(RuntimeError, match="page 1 took more than 20 minutes"):
        api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (1, [])


def test_no_retry_starts_that_would_pass_the_page_time(pdf: Path) -> None:
    api = OpenAI(limited("60"), request_seconds=1150.0)
    with pytest.raises(RuntimeError, match=r"page 1 \(HTTP 429, requests, rate_limit_exceeded\)"):
        api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (1, [])


class _Endless(httpx2.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield answer().content.split(b"\n\n")[0] + b"\n\n"
        while True:
            await asyncio.sleep(0.02)
            event = {"type": "response.in_progress", "sequence_number": 1}
            yield f"event: response.in_progress\ndata: {json.dumps(event)}\n\n".encode()


def test_a_stream_that_never_ends_is_stopped_at_the_page_time(
    pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider_module, "PAGE_SECONDS", 0.3)
    endless = httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=_Endless())
    provider = OpenAIProvider(transport=httpx2.MockTransport(lambda _: endless))
    started_at = time.monotonic()
    with pytest.raises(RuntimeError, match="page 1 took more than"):
        provider.fetch(pdf)
    assert time.monotonic() - started_at < 2.0


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_a_redirect_is_not_followed_with_the_key(pdf: Path, status: int) -> None:
    moved = httpx2.Response(status, headers={"location": "https://elsewhere.example.com/v1"})
    api = OpenAI(moved, moved)
    with pytest.raises(ValueError, match=f"refused the check \\(HTTP {status}"):
        api.provider().prepare()
    with pytest.raises(RuntimeError, match=f"page 1 \\(HTTP {status}"):
        api.provider().fetch(pdf)
    assert {r.url.host for r in api.requests} == {"api.openai.com"}


def test_a_dropped_connection_fails_without_echoing_the_request(
    pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider_module, "MAX_RETRIES", 0)
    api = OpenAI(httpx2.ConnectError(f"connection reset while sending {KEY}"))
    with pytest.raises(RuntimeError, match="check network connectivity") as caught:
        api.provider().fetch(pdf)
    assert KEY not in str(caught.value)


@pytest.mark.parametrize("broken", [stream_error, failed], ids=["error event", "failed response"])
def test_a_server_error_in_the_stream_is_tried_again(pdf: Path, broken: Any) -> None:
    api = OpenAI(broken("server_error", "busy"), *[answer()] * 3)
    raw = api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (4, [2.0])
    assert len(raw.payload["pages"]) == 3


def test_an_error_in_the_stream_that_persists_fails_without_its_message(pdf: Path) -> None:
    api = OpenAI(*[failed("rate_limit_exceeded", f"slow down {KEY}")] * (MAX_RETRIES + 1))
    with pytest.raises(RuntimeError) as caught:
        api.provider().fetch(pdf)
    assert str(caught.value) == (
        "OpenAI API request failed on page 1 (rate_limit_exceeded, request req_3); try again later."
    )
    assert api.waits == [2.0, 4.0, 8.0]


def test_an_error_in_the_stream_of_another_kind_is_not_tried_again(pdf: Path) -> None:
    api = OpenAI(failed("invalid_image", "cannot read"))
    with pytest.raises(RuntimeError, match=r"page 1 \(invalid_image, request req_3\); check"):
        api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (1, [])


def nested(error: dict[str, Any]) -> httpx2.Response:
    body = f"event: error\ndata: {json.dumps({'type': 'error', 'error': error})}\n\n".encode()
    headers = {"content-type": "text/event-stream", "x-request-id": "req_4"}
    return httpx2.Response(200, headers=headers, content=body)


def test_an_error_object_in_an_event_is_read_as_a_stream_error(pdf: Path) -> None:
    api = OpenAI(nested({"type": "server_error", "code": "server_error"}), *[answer()] * 3)
    api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (4, [2.0])
    api = OpenAI(nested({"type": "invalid_request_error", "code": "invalid_value"}))
    with pytest.raises(RuntimeError, match=r"page 1 \(invalid_value, request req_4\); check"):
        api.provider().fetch(pdf)
    api = OpenAI(nested({"type": "invalid_request_error", "message": f"bad {KEY}"}))
    with pytest.raises(RuntimeError, match=r"page 1 \(invalid_request_error, request req_4\)"):
        api.provider().fetch(pdf)
    assert len(api.requests) == 1


def test_a_used_up_quota_in_the_stream_says_to_check_billing(pdf: Path) -> None:
    api = OpenAI(failed("insufficient_quota", "no credit"))
    with pytest.raises(RuntimeError, match=r"\(insufficient_quota, request req_3\); check the acc"):
        api.provider().fetch(pdf)
    assert len(api.requests) == 1


def test_an_error_without_a_code_says_so(pdf: Path) -> None:
    api = OpenAI(stream_error(None, "something"))
    with pytest.raises(RuntimeError, match=r"page 1 \(request req_3\)"):
        api.provider().fetch(pdf)
    api = OpenAI(stream_error("Not a code!", "something"))
    with pytest.raises(RuntimeError, match=r"page 1 \(request req_3\)"):
        api.provider().fetch(pdf)


def cut_short() -> httpx2.Response:
    return sse(*started())


def unreadable() -> httpx2.Response:
    body = b"event: response.created\ndata: {not json\n\n"
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)


@pytest.mark.parametrize("broken", [cut_short, unreadable])
def test_an_incomplete_or_unreadable_stream_is_tried_again(pdf: Path, broken: Any) -> None:
    api = OpenAI(broken(), *[answer()] * 3)
    raw = api.provider().fetch(pdf)
    assert (len(api.requests), api.waits) == (4, [2.0])
    assert len(raw.payload["pages"]) == 3
    api = OpenAI(*[broken() for _ in range(MAX_RETRIES + 1)])
    with pytest.raises(RuntimeError, match="incomplete or unreadable answer for page 1"):
        api.provider().fetch(pdf)


def test_a_connection_dropped_in_the_stream_is_tried_again(pdf: Path) -> None:
    api = OpenAI(dropped(), *[answer()] * 3)
    raw = api.provider().fetch(pdf)
    assert len(api.requests) == 4
    assert len(raw.payload["pages"]) == 3


def test_a_connection_that_keeps_dropping_fails_without_echoing_it(pdf: Path) -> None:
    api = OpenAI(*[dropped() for _ in range(MAX_RETRIES + 1)])
    with pytest.raises(RuntimeError, match="check network connectivity") as caught:
        api.provider().fetch(pdf)
    assert KEY not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


def test_custom_headers_from_the_environment_are_refused(
    pdf: Path, key: pytest.MonkeyPatch
) -> None:
    key.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer other-key")
    api = OpenAI()
    for call in (api.provider().prepare, lambda: api.provider().fetch(pdf)):
        with pytest.raises(CredentialError, match="OPENAI_CUSTOM_HEADERS is set") as caught:
            call()
        assert "other-key" not in str(caught.value)
    assert api.requests == []


def test_an_admin_key_in_the_environment_is_not_sent(pdf: Path, key: pytest.MonkeyPatch) -> None:
    key.setenv("OPENAI_ADMIN_KEY", "admin-key-that-must-not-be-sent")
    api = OpenAI(model_info(), *[answer()] * 3)
    api.provider().prepare()
    api.provider().fetch(pdf)
    assert {r.headers["authorization"] for r in api.requests} == {f"Bearer {KEY}"}


@pytest.mark.parametrize("name", ["OPENAI_ORG_ID", "OPENAI_PROJECT_ID"])
def test_an_account_id_with_characters_no_id_has_is_refused(
    pdf: Path, key: pytest.MonkeyPatch, name: str
) -> None:
    key.setenv(name, "org-clé")
    api = OpenAI()
    for call in (api.provider().prepare, lambda: api.provider().fetch(pdf)):
        with pytest.raises(CredentialError, match=f"{name} holds characters") as caught:
            call()
        assert "clé" not in str(caught.value)
    assert api.requests == []


def test_the_organization_and_project_from_the_environment_are_sent(
    pdf: Path, key: pytest.MonkeyPatch
) -> None:
    key.setenv("OPENAI_ORG_ID", "org-benchmark")
    key.setenv("OPENAI_PROJECT_ID", "proj_benchmark")
    api = OpenAI(*[answer()] * 3)
    api.provider().fetch(pdf)
    headers = api.requests[0].headers
    assert (headers["openai-organization"], headers["openai-project"]) == (
        "org-benchmark",
        "proj_benchmark",
    )


@pytest.mark.parametrize("value", [None, "", " \t\n"])
def test_a_missing_api_key_fails_before_any_request(
    pdf: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("OPENAI_API_KEY")
    else:
        monkeypatch.setenv("OPENAI_API_KEY", value)
    api = OpenAI()
    for call in (api.provider().prepare, lambda: api.provider().fetch(pdf)):
        with pytest.raises(CredentialError, match="OPENAI_API_KEY is not set or is blank"):
            call()
    assert api.requests == []


def model_info() -> httpx2.Response:
    body = {"id": MODEL, "object": "model", "created": 1759100000, "owned_by": "system"}
    return httpx2.Response(200, json=body)


def test_prepare_checks_the_key_and_model_with_a_free_call(key: pytest.MonkeyPatch) -> None:
    api = OpenAI(model_info())
    api.provider().prepare()
    [request] = api.requests
    assert (request.method, str(request.url)) == (
        "GET",
        f"https://api.openai.com/v1/models/{MODEL}",
    )
    assert request.headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize(
    "answered",
    [
        lambda: httpx2.Response(200, headers={"content-type": "text/html"}, text="<p>Sign in</p>"),
        lambda: httpx2.Response(200, json={"object": "list"}),
    ],
    ids=["html", "not a model"],
)
def test_prepare_refuses_an_answer_that_is_not_the_models_details(
    key: pytest.MonkeyPatch, answered: Any
) -> None:
    api = OpenAI(*[answered() for _ in range(MAX_RETRIES + 1)])
    with pytest.raises(ValueError, match="did not answer with the model's details"):
        api.provider().prepare()
    assert api.waits == [2.0, 4.0, 8.0]


def test_prepare_waits_at_most_a_minute_for_a_rate_limit(key: pytest.MonkeyPatch) -> None:
    api = OpenAI(limited("3600"), model_info())
    api.provider().prepare()
    assert api.waits == [60.0]


def test_prepare_stops_trying_after_two_minutes(key: pytest.MonkeyPatch) -> None:
    api = OpenAI(limited("60"), limited("60"))
    with pytest.raises(ValueError, match=r"busy or unavailable \(HTTP 429"):
        api.provider().prepare()
    assert (len(api.requests), api.waits) == (2, [60.0])


def test_a_key_is_sent_without_surrounding_whitespace(key: pytest.MonkeyPatch) -> None:
    key.setenv("OPENAI_API_KEY", f" {KEY}\n")
    api = OpenAI(model_info())
    api.provider().prepare()
    assert api.requests[0].headers["authorization"] == f"Bearer {KEY}"


def test_a_key_with_characters_no_key_has_fails_without_echoing_it(
    key: pytest.MonkeyPatch,
) -> None:
    key.setenv("OPENAI_API_KEY", "sk-proj-clé")
    api = OpenAI()
    with pytest.raises(CredentialError, match="holds characters that no key has") as caught:
        api.provider().prepare()
    assert "clé" not in str(caught.value)
    assert api.requests == []


def test_a_rejected_key_fails_prepare_without_echoing_it(key: pytest.MonkeyPatch) -> None:
    api = OpenAI(
        failure(401, "invalid_request_error", f"Incorrect API key {KEY}", code="invalid_api_key")
    )
    with pytest.raises(CredentialError, match="did not accept the key") as caught:
        api.provider().prepare()
    assert "HTTP 401, invalid_request_error, invalid_api_key, request req_2" in str(caught.value)
    assert KEY not in str(caught.value)


def test_a_key_without_access_to_the_model_is_not_called_rejected(
    pdf: Path, key: pytest.MonkeyPatch
) -> None:
    api = OpenAI(failure(403, "invalid_request_error", "no access"))
    with pytest.raises(CredentialError, match="has no access to gpt-6.1-sol") as caught:
        api.provider().prepare()
    assert "did not accept" not in str(caught.value)
    api = OpenAI(failure(403, "invalid_request_error", "no access"))
    with pytest.raises(RuntimeError, match="check the key's access to the model"):
        api.provider().fetch(pdf)


def test_an_unknown_model_fails_prepare(key: pytest.MonkeyPatch) -> None:
    api = OpenAI(failure(404, "invalid_request_error", "gpt-nope", code="model_not_found"))
    with pytest.raises(ValueError, match="no model 'gpt-nope'.*check the model in --config"):
        api.provider({"model": "gpt-nope"}).prepare()


@pytest.mark.parametrize(
    ("status", "code"), [(429, "rate_limit_exceeded"), (503, "server_is_overloaded")]
)
def test_a_rate_limit_or_an_outage_at_prepare_does_not_blame_the_key(
    key: pytest.MonkeyPatch, status: int, code: str
) -> None:
    api = OpenAI(failure(status, "requests", "busy", code=code))
    with pytest.raises(ValueError, match="busy or unavailable.*try again later") as caught:
        api.provider().prepare()
    assert not isinstance(caught.value, CredentialError)


def test_an_unreachable_api_or_another_refusal_fails_prepare(
    key: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider_module, "MAX_RETRIES", 0)
    with pytest.raises(ValueError, match="could not be reached; check the network"):
        OpenAI(httpx2.ConnectError(f"reset {KEY}")).provider().prepare()
    api = OpenAI(failure(400, "invalid_request_error", f"bad {KEY}"))
    with pytest.raises(ValueError, match=r"refused the check \(HTTP 400") as caught:
        api.provider().prepare()
    assert KEY not in str(caught.value)


def test_config_defaults_and_choices() -> None:
    assert OpenAIProvider().config == {"model": MODEL, "effort": "medium"}
    assert OpenAIProvider({"effort": None}).config == {"model": MODEL, "effort": None}
    assert OpenAIProvider({"effort": "none"}).config == {"model": MODEL, "effort": "none"}
    with pytest.raises(ValueError, match="unsupported config keys: api_key"):
        OpenAIProvider({"api_key": KEY})
    with pytest.raises(ValueError, match="model must be an OpenAI model id"):
        OpenAIProvider({"model": "gpt 6"})
    with pytest.raises(
        ValueError, match="effort must be one of none, minimal, low, medium, high, xhigh, max"
    ):
        OpenAIProvider({"effort": "extreme"})
