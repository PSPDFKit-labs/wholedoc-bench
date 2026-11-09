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

API_URL = "https://api.anthropic.com"
API_KEY_VARIABLE = "ANTHROPIC_API_KEY"
DEFAULT_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "medium"
MAX_TOKENS = 32768
MAX_RETRIES = 3
RETRY_STATUSES = frozenset({408, 409, 429})
RETRY_TYPES = frozenset({"rate_limit_error", "api_error", "overloaded_error", "timeout_error"})
MAX_RETRY_AFTER_SECONDS = 60.0
PAGE_SECONDS = 1200.0
CHECK_SECONDS = 120.0
READ_SECONDS = 600.0
CONNECT_SECONDS = 5.0
CUSTOM_HEADERS_VARIABLE = "ANTHROPIC_CUSTOM_HEADERS"
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/-]*")
_INCOMPLETE = (
    "Anthropic API sent an incomplete or unreadable answer for page {number}; try again later."
)
_ERROR_TYPE = re.compile(r"[a-z][a-z_]{0,63}")
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
USAGE_METERS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "ephemeral_5m_input_tokens",
    "ephemeral_1h_input_tokens",
)


class AnthropicProvider(Provider):
    name: ClassVar[str] = "anthropic"
    sends_documents: ClassVar[bool] = True
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
                f"{cls.name}: model must be a Claude model id, such as {DEFAULT_MODEL}"
            )
        effort = config.get("effort", DEFAULT_EFFORT)
        if effort is not None and effort not in EFFORTS:
            raise ValueError(
                f"{cls.name}: effort must be one of {', '.join(EFFORTS)}, or null for a model "
                "without effort"
            )
        return {"model": model, "effort": effort}

    def prepare(self) -> None:
        key = _key()
        self._sdk()
        require_pdf_reader(self.name)
        asyncio.run(self._check(key))

    async def _check(self, key: str) -> None:
        import anthropic

        model = self.config["model"]
        async with self._client(key) as client:
            try:
                info = await self._retrying(
                    lambda timeout, _: client.models.retrieve(model, timeout=timeout),
                    CHECK_SECONDS,
                )
            except _TimeUp:
                raise ValueError(
                    f"The Anthropic API did not answer within {CHECK_SECONDS:.0f} seconds; try "
                    "again later."
                ) from None
            except anthropic.AuthenticationError as e:
                raise CredentialError(
                    f"Anthropic did not accept the key ({_reason(e)}); check "
                    f"{API_KEY_VARIABLE}. Do not put credentials in --config."
                ) from None
            except anthropic.PermissionDeniedError as e:
                raise CredentialError(
                    f"The key has no access to {model} ({_reason(e)}); check the key's "
                    "workspace and the model in --config."
                ) from None
            except anthropic.NotFoundError as e:
                raise ValueError(
                    f"Anthropic has no model {model!r} for this key ({_reason(e)}); check the "
                    "model in --config."
                ) from None
            except anthropic.APIStatusError as e:
                if e.status_code in RETRY_STATUSES or e.status_code >= 500:
                    raise ValueError(
                        f"The Anthropic API is busy or unavailable ({_reason(e)}); try again later."
                    ) from None
                raise ValueError(f"The Anthropic API refused the check ({_reason(e)}).") from None
            except (anthropic.APIConnectionError, _transport_error()):
                raise ValueError(
                    "The Anthropic API could not be reached; check the network."
                ) from None
        effort = self.config["effort"]
        capabilities = info.capabilities.effort if info.capabilities is not None else None
        if effort is not None and capabilities is not None:
            level = getattr(capabilities, effort, None)
            if not capabilities.supported or level is None or not level.supported:
                raise ValueError(
                    f"{model} does not take effort {effort!r}; choose another effort, or null "
                    "for none, in --config."
                )

    def fetch(self, document_path: Path) -> RawResult:
        key = _key()
        self._sdk()
        pages = asyncio.run(self._pages(key, document_path))
        models = sorted({page["model"] for page in pages}) or [self.config["model"]]
        effort = self.config["effort"] or "default"
        return RawResult(
            provider=self.name,
            document_id=document_path.parent.name,
            payload={"pages": pages},
            metadata={
                "version": f"anthropic {'+'.join(models)}; effort {effort}; prompt {PROMPT_DIGEST}",
                "config": dict(self.config),
            },
        )

    def _sdk(self) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ValueError(
                f"{self.name} needs the anthropic SDK: run `uv sync --extra {self.name}`"
            ) from None

    def _client(self, key: str) -> Any:
        import anthropic

        if CUSTOM_HEADERS_VARIABLE in os.environ:
            raise CredentialError(
                f"{CUSTOM_HEADERS_VARIABLE} is set: unset it, as the SDK would send its headers "
                "with every request and they could replace the key."
            )
        transport = {} if self._transport is None else {"transport": self._transport}
        http_client = anthropic.DefaultAsyncHttpxClient(follow_redirects=False, **transport)
        return anthropic.AsyncAnthropic(
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
        import anthropic
        import httpx2

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
            except anthropic.APIStatusError as e:
                if not _worth_trying_again(e):
                    raise
                error: Exception = e
                wait = _retry_after(e.response.headers, wait)
            except (
                anthropic.APIConnectionError,
                httpx2.TransportError,
                _Incomplete,
                ValueError,
            ) as e:
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
        import anthropic

        effort = self.config["effort"]
        content = [source, {"type": "text", "text": USER_PROMPT}]
        if (quote := previous_page_prompt(previous)) is not None:
            content.append({"type": "text", "text": quote})

        async def call(timeout: Any, check: Callable[[], None]) -> Any:
            async with client.messages.stream(
                model=self.config["model"],
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
                timeout=timeout,
                **({"output_config": {"effort": effort}} if effort else {}),
            ) as stream:
                async for _ in stream:
                    check()
                message = await stream.get_final_message()
            if message.stop_reason is None:
                raise _Incomplete
            return message

        try:
            message = await self._retrying(call, PAGE_SECONDS)
        except _TimeUp:
            raise RuntimeError(
                f"Claude's answer for page {number} took more than {PAGE_SECONDS / 60:.0f} "
                "minutes, its retries included."
            ) from None
        except anthropic.APIStatusError as e:
            raise RuntimeError(_failure(e, number)) from None
        except (anthropic.APIConnectionError, _transport_error()):
            raise RuntimeError(
                "Anthropic API request failed; check network connectivity."
            ) from None
        except (_Incomplete, ValueError):
            raise RuntimeError(_INCOMPLETE.format(number=number)) from None
        except anthropic.AnthropicError as e:
            raise RuntimeError(
                f"Anthropic API request failed on page {number} ({type(e).__name__})."
            ) from None
        if message.stop_reason == "refusal":
            details = message.stop_details
            category = details.category if details is not None else None
            raise RuntimeError(f"Claude declined page {number} ({category or 'no category'}).")
        if message.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Claude's answer for page {number} reached the {MAX_TOKENS} token cap."
            )
        if message.stop_reason not in ("end_turn", "stop_sequence"):
            raise RuntimeError(f"Claude stopped page {number} with {message.stop_reason!r}.")
        return {
            "model": message.model,
            "stop_reason": message.stop_reason,
            "usage": message.usage.to_dict(),
            "text": "".join(block.text for block in message.content if block.type == "text"),
        }

    def normalize(self, raw: RawResult) -> Prediction:
        return read_pages(answer["text"] for answer in raw.payload["pages"])

    def usage(self, raw: RawResult) -> dict[str, float]:
        totals: dict[str, float] = dict.fromkeys(USAGE_METERS, 0)
        for page in raw.payload["pages"]:
            usage = page.get("usage")
            if not isinstance(usage, dict):
                raise ValueError("a page has no token usage")
            if usage.get("speed", "standard") != "standard":
                raise ValueError(f"a page ran at {usage['speed']!r} speed")
            if usage.get("service_tier", "standard") != "standard":
                raise ValueError(f"a page ran on the {usage['service_tier']!r} tier")
            if usage.get("inference_geo", "global") != "global":
                raise ValueError(f"a page ran with inference_geo {usage['inference_geo']!r}")
            writes = usage.get("cache_creation") or {}
            counts = {
                **usage,
                "ephemeral_5m_input_tokens": writes.get(
                    "ephemeral_5m_input_tokens", usage.get("cache_creation_input_tokens") or 0
                ),
                "ephemeral_1h_input_tokens": writes.get("ephemeral_1h_input_tokens", 0),
            }
            for meter in USAGE_METERS:
                totals[meter] += counts.get(meter) or 0
        return totals


class _TimeUp(Exception): ...


class _Incomplete(Exception): ...


def _transport_error() -> type[Exception]:
    import httpx2

    return httpx2.TransportError


def _key() -> str:
    key = require_env(API_KEY_VARIABLE).strip()
    if not (key.isascii() and key.isprintable()):
        raise CredentialError(
            f"{API_KEY_VARIABLE} holds characters that no key has; copy it again from the "
            "Claude Console."
        )
    return key


def _sources(document_path: Path) -> Iterator[dict[str, Any]]:
    suffix = document_path.suffix.lower()
    if suffix != ".pdf":
        yield {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": IMAGE_TYPES[suffix],
                "data": _base64(document_path.read_bytes()),
            },
        }
        return
    for page in pdf_pages(document_path):
        yield {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": _base64(page)},
        }


def _base64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


def _reason(error: Any) -> str:
    parts = [f"HTTP {error.status_code}"] if error.status_code != 200 else []
    if isinstance(error.type, str) and _ERROR_TYPE.fullmatch(error.type):
        parts.append(error.type)
    if isinstance(error.request_id, str) and _REQUEST_ID.fullmatch(error.request_id):
        parts.append(f"request {error.request_id}")
    return ", ".join(parts) or "no error type"


def _worth_trying_again(error: Any) -> bool:
    hint = error.response.headers.get("x-should-retry")
    if hint in ("true", "false"):
        return hint == "true"
    status = error.status_code
    return (
        status in RETRY_STATUSES or status >= 500 or (status == 200 and error.type in RETRY_TYPES)
    )


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
    elif (
        error.status_code in RETRY_STATUSES or error.status_code >= 500 or error.type in RETRY_TYPES
    ):
        advice = "try again later"
    else:
        advice = "check the account's limits and the document requirements"
    return f"Anthropic API request failed on page {number} ({_reason(error)}); {advice}."
