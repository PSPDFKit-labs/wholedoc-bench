import json
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from wholedoc_bench.provenance import digest, file_hash
from wholedoc_bench.providers import PROVIDERS, RawResult

BOOK = Path(__file__).resolve().parents[2] / "prices"
CREDITS = "credits"
STALE_AFTER = timedelta(days=90)


class Rate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    meter: str
    when: dict[str, Any] = Field(default_factory=dict)
    usd: float | None = Field(default=None, ge=0)
    credits: float | None = Field(default=None, ge=0)
    per: float = Field(default=1, gt=0)
    estimate: bool = False
    effective: date
    checked: date
    source: str = Field(pattern=r"^https://\S+$")

    @model_validator(mode="after")
    def one_price(self) -> "Rate":
        settings = [v for value in self.when.values() for v in _items(value)]
        if not all(isinstance(v, str | int | float | bool) for v in settings):
            raise ValueError("a when setting is a string, number, boolean or list of them")
        if (self.usd is None) == (self.credits is None):
            raise ValueError("a rate has either usd or credits")
        if self.meter == CREDITS and self.usd is None:
            raise ValueError("the credits rate is in usd")
        return self

    @property
    def settings(self) -> str:
        return json.dumps(self.when, sort_keys=True)

    def applies(self, config: dict[str, Any], as_of: date) -> bool:
        return self.effective <= as_of and all(
            k in config and config[k] == v for k, v in self.when.items()
        )


@dataclass(frozen=True)
class PriceBook:
    rates: dict[str, tuple[Rate, ...]]
    sha256: str

    def rate(self, provider: str, config: dict[str, Any], meter: str, as_of: date) -> Rate | None:
        current: dict[str, Rate] = {}
        for r in self.rates.get(provider, ()):
            if r.meter == meter and r.applies(config, as_of):
                if r.settings not in current or current[r.settings].effective < r.effective:
                    current[r.settings] = r
        if not current:
            return None
        most = max(len(r.when) for r in current.values())
        best = [r for r in current.values() if len(r.when) == most]
        if len(best) > 1:
            raise ValueError(f"prices/{provider}.toml: {len(best)} rates for {meter} tie")
        return best[0]


@dataclass(frozen=True)
class Cost:
    usd: float | None
    estimated: bool
    missing: tuple[str, ...]
    rates: tuple[Rate, ...]


def load_book(folder: Path) -> PriceBook:
    if not folder.is_dir():
        raise ValueError(f"{folder}: no price book folder")
    rates: dict[str, tuple[Rate, ...]] = {}
    files = sorted(folder.glob("*.toml"))
    for path in files:
        provider = PROVIDERS.get(path.stem)
        if provider is None or not provider.meters:
            raise ValueError(f"{path.name}: no provider with usage meters is named {path.stem}")
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            if set(data) != {"rate"} or not isinstance(data["rate"], list):
                raise ValueError("expected [[rate]] tables only")
            entries = tuple(Rate.model_validate(entry) for entry in data["rate"])
        except (tomllib.TOMLDecodeError, ValidationError, ValueError) as e:
            raise ValueError(f"{path.name}: {e}") from e
        seen = set()
        for entry in entries:
            if entry.meter not in provider.meters | {CREDITS}:
                raise ValueError(f"{path.name}: {provider.name} has no meter {entry.meter!r}")
            key = (entry.meter, entry.settings, entry.effective)
            if key in seen:
                raise ValueError(
                    f"{path.name}: two {entry.meter} rates with the same settings take effect "
                    f"on {entry.effective}"
                )
            seen.add(key)
        rates[path.stem] = entries
    return PriceBook(rates, digest({p.name: file_hash(p) for p in files}))


def measure_usage(
    provider: str | None, config: dict[str, Any] | None, documents: Iterable[tuple[RawResult, int]]
) -> dict[str, Any] | None:
    if provider not in PROVIDERS or config is None or not PROVIDERS[provider].meters:
        return None
    adapter = PROVIDERS[provider](config=config)
    measured, pages = 0, 0
    meters: dict[str, float] = dict.fromkeys(sorted(adapter.meters), 0)
    for raw, page_count in documents:
        try:
            usage = adapter.usage(raw)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if unknown := usage.keys() - adapter.meters:
            raise ValueError(f"{provider} usage has undeclared meters: {', '.join(unknown)}")
        for meter, quantity in usage.items():
            meters[meter] += quantity
        measured, pages = measured + 1, pages + page_count
    return {"documents": measured, "pages": pages, "meters": meters}


def price(
    book: PriceBook, provider: str, config: dict[str, Any], meters: Mapping[str, float], as_of: date
) -> Cost:
    usd, credits = 0.0, float(meters.get(CREDITS, 0))
    missing: list[str] = []
    used: list[Rate] = []
    for meter, quantity in sorted(meters.items()):
        if meter == CREDITS or not quantity:
            continue
        rate = book.rate(provider, config, meter, as_of)
        if rate is None:
            missing.append(meter)
        elif rate.usd is not None:
            usd += quantity * rate.usd / rate.per
            used.append(rate)
        else:
            assert rate.credits is not None
            credits += quantity * rate.credits / rate.per
            used.append(rate)
    if credits:
        rate = book.rate(provider, config, CREDITS, as_of)
        if rate is None:
            missing.append(CREDITS)
        else:
            assert rate.usd is not None
            usd += credits * rate.usd / rate.per
            used.append(rate)
    estimated = any(rate.estimate for rate in used)
    return Cost(None if missing else usd, estimated, tuple(missing), tuple(used))


def _items(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def is_stale(rate: Rate, as_of: date) -> bool:
    return rate.checked < as_of - STALE_AFTER
