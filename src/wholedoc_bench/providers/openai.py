import asyncio
import base64
import os
import re
import time
from collections.abc import Awaitable, Callable, Iterator
from math import isfinite
from pathlib import Path
from typing import Any, ClassVar

from wholedoc_bench.providers.base import Provider, RawResult
from wholedoc_bench.providers.credentials import CredentialError, require_env
from wholedoc_bench.providers.layout import (
    PROMPT_DIGEST,
    SYSTEM_PROMPT,
    USER_PROMPT,
    previous_page_prompt,
    read_pages,
)
from wholedoc_bench.providers.pdf import pdf_pages, require_pdf_reader
from wholedoc_bench.schema import Prediction

API_URL = "https://api.openai.com/v1"
API_KEY_VARIABLE = "OPENAI_API_KEY"
DEFAULT_MODEL = "gpt-6.1-sol"
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "medium"
MAX_TOKENS = 32768
MAX_RETRIES = 3
RETRY_STATUSES = frozenset({408, 409, 429})
RETRY_CODES = frozenset(
    {"server_error", "rate_limit_exceeded", "server_is_overloaded", "slow_down"}
)
QUOTA_CODE = "insufficient_quota"
MAX_RETRY_AFTER_SECONDS = 60.0
PAGE_SECONDS = 1200.0
CHECK_SECONDS = 120.0
READ_SECONDS = 600.0
CONNECT_SECONDS = 5.0
CUSTOM_HEADERS_VARIABLE = "OPENAI_CUSTOM_HEADERS"
ACCOUNT_VARIABLES = ("OPENAI_ORG_ID", "OPENAI_PROJECT_ID")
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/-]*")
_INCOMPLETE = (
    "OpenAI API sent an incomplete or unreadable answer for page {number}; try again later."
)
_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_ENDS = frozenset({"response.completed", "response.incomplete", "response.failed"})
USAGE_METERS = ("uncached_input_tokens", "cached_tokens", "cache_write_tokens", "output_tokens")


class OpenAIProvider(Provider):
    name: ClassVar[str] = "openai"
    formats: ClassVar[frozenset[str]] = frozenset({".pdf", *IMAGE_TYPES})
    meters: ClassVar[frozenset[str]] = frozenset(USAGE_METERS)

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        transport: Any = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(config)
        self._transport = transport
        self._sleep = sleep
        self._clock = clock

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise ValueError("provider config must be a JSON object")
        if unknown := config.keys() - {"model", "effort"}:
            raise ValueError(f"{cls.name}: unsupported config keys: {', '.join(sorted(unknown))}")
        model = config.get("model", DEFAULT_MODEL)
        if not isinstance(model, str) or not _MODEL.fullmatch(model):
            raise ValueError(
                f"{cls.name}: model must be an OpenAI model id, such as {DEFAULT_MODEL}"
            )
        effort = config.get("effort", DEFAULT_EFFORT)
        if effort is not None and effort not in EFFORTS:
            raise ValueError(
                f"{cls.name}: effort must be one of {', '.join(EFFORTS)}, or null for a model "
                "without reasoning"
            )
        return {"model": model, "effort": effort}

    def prepare(self) -> None:
        key = _key()
        self._sdk()
        require_pdf_reader(self.name)
        asyncio.run(self._check(key))

    async def _check(self, key: str) -> None:
        import openai

        model = self.config["model"]
        async with self._client(key) as client:

            async def call(timeout: Any, _: Callable[[], None]) -> None:
                try:
                    info = await client.models.retrieve(model, timeout=timeout)
                except ValueError:
                    raise _Incomplete from None
                if not isinstance(getattr(info, "id", None), str):
                    raise _Incomplete

            try:
                await self._retrying(call, CHECK_SECONDS)
            except _TimeUp:
                raise ValueError(
                    f"The OpenAI API did not answer within {CHECK_SECONDS:.0f} seconds; try "
                    "again later."
                ) from None
            except openai.AuthenticationError as e:
                raise CredentialError(
                    f"OpenAI did not accept the key ({_reason(e)}); check {API_KEY_VARIABLE}. "
                    "Do not put credentials in --config."
                ) from None
            except openai.PermissionDeniedError as e:
                raise CredentialError(
                    f"The key has no access to {model} ({_reason(e)}); check the key's project "
                    "and the model in --config."
                ) from None
            except openai.NotFoundError as e:
                raise ValueError(
                    f"OpenAI has no model {model!r} for this key ({_reason(e)}); check the "
                    "model in --config."
                ) from None
            except openai.APIStatusError as e:
                if _busy(e):
                    raise ValueError(
                        f"The OpenAI API is busy or unavailable ({_reason(e)}); try again later."
                    ) from None
                raise ValueError(f"The OpenAI API refused the check ({_reason(e)}).") from None
            except (openai.APIConnectionError, _transport_error()):
                raise ValueError(
                    "The OpenAI API could not be reached; check the network."
                ) from None
            except _Incomplete:
                raise ValueError(
                    "The OpenAI API did not answer with the model's details; check the network."
                ) from None

    def fetch(self, document_path: Path) -> RawResult:
        key = _key()
        self._sdk()
        pages = asyncio.run(self._pages(key, document_path))
        models = sorted({_model_id(p["model"]) or self.config["model"] for p in pages})
        models = models or [self.config["model"]]
        effort = self.config["effort"] or "default"
        return RawResult(
            provider=self.name,
            document_id=document_path.parent.name,
            payload={"pages": pages},
            metadata={
                "version": f"openai {'+'.join(models)}; effort {effort}; prompt {PROMPT_DIGEST}",
                "config": dict(self.config),
            },
        )

    def _sdk(self) -> None:
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ValueError(
                f"{self.name} needs the openai SDK: run `uv sync --extra {self.name}`"
            ) from None

    def _client(self, key: str) -> Any:
        import openai

        if CUSTOM_HEADERS_VARIABLE in os.environ:
            raise CredentialError(
                f"{CUSTOM_HEADERS_VARIABLE} is set: unset it, as the SDK would send its headers "
                "with every request and they could replace the key."
            )
        for name in ACCOUNT_VARIABLES:
            value = os.environ.get(name)
            if value is not None and not (value.isascii() and value.isprintable()):
                raise CredentialError(f"{name} holds characters that no OpenAI id has.")
        transport = {} if self._transport is None else {"transport": self._transport}
        http_client = openai.DefaultAsyncHttpxClient(follow_redirects=False, **transport)
        return openai.AsyncOpenAI(
            api_key=key, base_url=API_URL, max_retries=0, http_client=http_client
        )

    async def _pages(self, key: str, document_path: Path) -> list[dict[str, Any]]:
        async with self._client(key) as client:
            pages: list[dict[str, Any]] = []
            for number, source in enumerate(_sources(document_path), start=1):
                previous = pages[-1]["text"] if pages else ""
                pages.append(await self._page(client, number, source, previous))
            return pages

    async def _retrying(
        self, call: Callable[[Any, Callable[[], None]], Awaitable[Any]], seconds: float
    ) -> Any:
        import httpx2
        import openai

        deadline = self._clock() + seconds

        def check() -> None:
            if self._clock() > deadline:
                raise _TimeUp

        attempt = 0
        while True:
            wait = 2.0 * 2**attempt
            left = deadline - self._clock()
            timeout = httpx2.Timeout(min(READ_SECONDS, left), connect=min(CONNECT_SECONDS, left))
            limit = asyncio.timeout(left)
            try:
                async with limit:
                    return await call(timeout, check)
            except openai.APIStatusError as e:
                if not _worth_trying_again(e):
                    raise
                error: Exception = e
                wait = _retry_after(e.response.headers, wait)
            except _Failed as e:
                if e.code not in RETRY_CODES:
                    raise
                error = e
            except (openai.APIConnectionError, httpx2.TransportError, _Incomplete) as e:
                error = e
            except TimeoutError:
                if not limit.expired():
                    raise
                raise _TimeUp from None
            if self._clock() >= deadline:
                raise _TimeUp
            if attempt == MAX_RETRIES or self._clock() + wait >= deadline:
                raise error
            await self._sleep(wait)
            attempt += 1

    async def _page(
        self, client: Any, number: int, source: dict[str, Any], previous: str
    ) -> dict[str, Any]:
        import openai

        effort = self.config["effort"]
        content = [source, {"type": "input_text", "text": USER_PROMPT}]
        if (quote := previous_page_prompt(previous)) is not None:
            content.append({"type": "input_text", "text": quote})

        async def call(timeout: Any, check: Callable[[], None]) -> Any:
            stream = await client.responses.create(
                model=self.config["model"],
                max_output_tokens=MAX_TOKENS,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                store=False,
                stream=True,
                timeout=timeout,
                **({"reasoning": {"effort": effort}} if effort else {}),
            )
            async with stream:
                request_id = stream.response.headers.get("x-request-id")
                try:
                    async for event in stream:
                        check()
                        if event.type == "error":
                            raise _Failed(event.code, request_id)
                        if event.type in _ENDS:
                            response = event.response
                            break
                    else:
                        raise _Incomplete
                except openai.APIError as e:
                    if isinstance(e, openai.APIConnectionError | openai.APIStatusError):
                        raise
                    raise _Failed(e.code or e.type, request_id) from None
                except (ValueError, AttributeError):
                    raise _Incomplete from None
            answer = _answer(response)
            if answer["status"] == "failed":
                raise _Failed(answer["error"], request_id)
            return answer

        try:
            answer = await self._retrying(call, PAGE_SECONDS)
        except _TimeUp:
            raise RuntimeError(
                f"The answer for page {number} took more than {PAGE_SECONDS / 60:.0f} "
                "minutes, its retries included."
            ) from None
        except openai.APIStatusError as e:
            raise RuntimeError(_failure(e, number)) from None
        except _Failed as e:
            if e.code in RETRY_CODES:
                advice = "try again later"
            elif e.code == QUOTA_CODE:
                advice = "check the account's billing"
            else:
                advice = "check the document"
            raise RuntimeError(
                f"OpenAI API request failed on page {number} ({_stream_reason(e)}); {advice}."
            ) from None
        except (openai.APIConnectionError, _transport_error()):
            raise RuntimeError("OpenAI API request failed; check network connectivity.") from None
        except _Incomplete:
            raise RuntimeError(_INCOMPLETE.format(number=number)) from None
        except openai.OpenAIError as e:
            raise RuntimeError(
                f"OpenAI API request failed on page {number} ({type(e).__name__})."
            ) from None
        reason, refused = _safe(answer.pop("reason")), answer.pop("refused")
        del answer["error"]
        if answer["status"] == "incomplete":
            if reason == "max_output_tokens":
                raise RuntimeError(
                    f"The answer for page {number} reached the {MAX_TOKENS} token cap."
                )
            raise RuntimeError(f"OpenAI stopped page {number} ({reason or 'no reason'}).")
        if answer["status"] != "completed":
            raise RuntimeError(
                f"OpenAI ended page {number} with status {_safe(answer['status']) or 'none'}."
            )
        if refused:
            raise RuntimeError(f"The model declined page {number}.")
        return answer

    def normalize(self, raw: RawResult) -> Prediction:
        return read_pages(answer["text"] for answer in raw.payload["pages"])

    def usage(self, raw: RawResult) -> dict[str, float]:
        totals: dict[str, float] = dict.fromkeys(USAGE_METERS, 0)
        for page in raw.payload["pages"]:
            usage = page.get("usage")
            if not isinstance(usage, dict):
                raise ValueError("a page has no token usage")
            details = usage.get("input_tokens_details") or {}
            cached = details.get("cached_tokens") or 0
            written = details.get("cache_write_tokens") or 0
            totals["uncached_input_tokens"] += usage["input_tokens"] - cached - written
            totals["cached_tokens"] += cached
            totals["cache_write_tokens"] += written
            totals["output_tokens"] += usage["output_tokens"]
        return totals


class _TimeUp(Exception): ...


class _Incomplete(Exception): ...


class _Failed(Exception):
    def __init__(self, code: Any, request_id: Any) -> None:
        super().__init__()
        self.code = _safe(code)
        self.request_id = request_id


def _transport_error() -> type[Exception]:
    import httpx2

    return httpx2.TransportError


def _answer(response: Any) -> dict[str, Any]:
    try:
        parts = [
            part
            for item in response.output
            if item.type == "message" and getattr(item, "phase", None) != "commentary"
            for part in item.content
        ]
        details, error = response.incomplete_details, response.error
        return {
            "model": response.model,
            "status": response.status,
            "usage": response.usage.to_dict() if response.usage is not None else None,
            "text": "".join(part.text for part in parts if part.type == "output_text"),
            "reason": details.reason if details is not None else None,
            "error": error.code if error is not None else None,
            "refused": any(part.type == "refusal" for part in parts),
        }
    except (AttributeError, TypeError):
        raise _Incomplete from None


def _model_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _MODEL.fullmatch(value) else None


def _key() -> str:
    key = require_env(API_KEY_VARIABLE).strip()
    if not (key.isascii() and key.isprintable()):
        raise CredentialError(
            f"{API_KEY_VARIABLE} holds characters that no key has; copy it again from the "
            "OpenAI platform."
        )
    return key


def _sources(document_path: Path) -> Iterator[dict[str, Any]]:
    suffix = document_path.suffix.lower()
    if suffix != ".pdf":
        data = _base64(document_path.read_bytes())
        yield {
            "type": "input_image",
            "image_url": f"data:{IMAGE_TYPES[suffix]};base64,{data}",
            "detail": "auto",
        }
        return
    for page in pdf_pages(document_path):
        yield {
            "type": "input_file",
            "filename": "page.pdf",
            "file_data": f"data:application/pdf;base64,{_base64(page)}",
            "detail": "auto",
        }


def _base64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


def _safe(value: Any) -> str | None:
    return value if isinstance(value, str) and _CODE.fullmatch(value) else None


def _reason(error: Any) -> str:
    parts = [f"HTTP {error.status_code}"]
    for name in (_safe(error.type), _safe(error.code)):
        if name is not None and name not in parts:
            parts.append(name)
    if isinstance(error.request_id, str) and _REQUEST_ID.fullmatch(error.request_id):
        parts.append(f"request {error.request_id}")
    return ", ".join(parts)


def _stream_reason(error: _Failed) -> str:
    parts = [error.code] if error.code else []
    if isinstance(error.request_id, str) and _REQUEST_ID.fullmatch(error.request_id):
        parts.append(f"request {error.request_id}")
    return ", ".join(parts) or "no error code"


def _busy(error: Any) -> bool:
    status = error.status_code
    return (status in RETRY_STATUSES or status >= 500) and error.code != QUOTA_CODE


def _worth_trying_again(error: Any) -> bool:
    if error.code == QUOTA_CODE:
        return False
    hint = error.response.headers.get("x-should-retry")
    if hint in ("true", "false"):
        return hint == "true"
    return _busy(error)


def _retry_after(headers: Any, default: float) -> float:
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        try:
            wait = float(headers.get(name, "")) / scale
        except ValueError:
            continue
        if isfinite(wait) and wait >= 0:
            return min(wait, MAX_RETRY_AFTER_SECONDS)
    return default


def _failure(error: Any, number: int) -> str:
    if error.status_code == 401:
        advice = f"check {API_KEY_VARIABLE}"
    elif error.status_code == 403:
        advice = "check the key's access to the model"
    elif error.code == QUOTA_CODE:
        advice = "check the account's billing"
    elif _busy(error):
        advice = "try again later"
    else:
        advice = "check the account's limits and the document requirements"
    return f"OpenAI API request failed on page {number} ({_reason(error)}); {advice}."
