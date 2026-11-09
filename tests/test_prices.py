import json
import shutil
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import DEMO
from tests.release_helpers import copy_release
from wholedoc_bench.cli import RUN_MANIFEST, main
from wholedoc_bench.corpus.loader import load_groundtruth, prediction_path, save_prediction
from wholedoc_bench.prices import BOOK, load_book, measure_usage, price
from wholedoc_bench.providers import PROVIDERS, MarkitdownProvider, NutrientProvider, RawResult
from wholedoc_bench.providers.base import raw_path, save_raw
from wholedoc_bench.report import read_summary, summarize, write_summary
from wholedoc_bench.schema import Prediction
from wholedoc_bench.scorer.run import score_corpus

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*/*.json"))
PRICED = [(p.parent.name, json.loads(p.read_text("utf-8"))) for p in CONFIGS] + [
    (name, provider.validate_config({})) for name, provider in PROVIDERS.items() if provider.meters
]
SOURCE = "https://example.com/pricing"


def rate(meter: str, *, effective: str = "2026-01-01", checked: str = "2026-10-01", **fields: Any):
    entry = {"meter": meter, "effective": effective, "checked": checked, "source": SOURCE}
    return {**entry, **fields}


def book_with(tmp_path: Path, **files: list[dict[str, Any]]) -> Path:
    folder = tmp_path / "prices"
    folder.mkdir(exist_ok=True)
    for provider, rates in files.items():
        tables = []
        for entry in rates:
            lines = ["[[rate]]"]
            for key, value in entry.items():
                if key in ("effective", "checked"):
                    lines.append(f"{key} = {value}")
                else:
                    lines.append(f"{key} = {_toml(value)}")
            tables.append("\n".join(lines))
        (folder / f"{provider}.toml").write_text("\n\n".join(tables) + "\n", "utf-8")
    return folder


def _toml(value: Any) -> str:
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {_toml(v)}" for k, v in value.items()) + " }"
    return json.dumps(value)


@pytest.mark.parametrize("provider,config", PRICED, ids=[f"{n} {c}" for n, c in PRICED])
def test_the_price_book_prices_every_meter_of_every_benchmark_configuration(
    provider: str, config: dict[str, Any]
) -> None:
    meters = dict.fromkeys(PROVIDERS[provider].meters, 1)
    cost = price(load_book(BOOK), provider, config, meters, date.today())
    assert cost.missing == ()
    assert cost.usd is not None


def test_every_when_setting_is_a_setting_of_its_provider() -> None:
    settings: dict[str, set[str]] = {}
    for name, config in PRICED:
        settings.setdefault(name, set()).update(config)
    for name, rates in load_book(BOOK).rates.items():
        for entry in rates:
            assert set(entry.when) <= settings[name], f"{name}: {entry}"


@pytest.mark.parametrize(
    "files,error",
    [
        ({"acme": [rate("pages", usd=1)]}, "no provider with usage meters is named acme"),
        ({"pdftotext": [rate("pages", usd=1)]}, "no provider with usage meters"),
        ({"textract": [rate("tokens", usd=1)]}, "textract has no meter 'tokens'"),
        ({"textract": [rate("pages", usd=1, credits=1)]}, "either usd or credits"),
        ({"textract": [rate("pages")]}, "either usd or credits"),
        ({"nutrient": [rate("credits", credits=2)]}, "the credits rate is in usd"),
        ({"textract": [rate("pages", usd=1, source="http://example.com")]}, "source"),
        ({"textract": [rate("pages", usd=1, per=0)]}, "per"),
        ({"textract": [rate("pages", usd=-1)]}, "usd"),
        ({"textract": [{**rate("pages", usd=1), "efective": "x"}]}, "efective"),
        ({"llamaparse": [rate("pages", credits=1, when={"version": {"x": 1}})]}, "a when setting"),
    ],
)
def test_a_wrong_rate_is_refused_by_file_name(
    tmp_path: Path, files: dict[str, list[dict[str, Any]]], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        load_book(book_with(tmp_path, **files))


def test_a_file_that_is_not_toml_is_refused_by_name(tmp_path: Path) -> None:
    folder = book_with(tmp_path)
    (folder / "textract.toml").write_text("[[rate]\n", "utf-8")
    with pytest.raises(ValueError, match="textract.toml"):
        load_book(folder)


def test_a_new_rate_replaces_the_old_one_from_its_effective_day(tmp_path: Path) -> None:
    folder = book_with(
        tmp_path,
        textract=[
            rate("pages", usd=15, per=1000, effective="2026-01-01"),
            rate("pages", usd=12, per=1000, effective="2026-11-01"),
        ],
    )
    book = load_book(folder)
    days = [date(2026, 10, 31), date(2026, 11, 1)]
    assert [price(book, "textract", {}, {"pages": 1000}, day).usd for day in days] == [15, 12]
    assert price(book, "textract", {}, {"pages": 1000}, date(2025, 12, 31)).missing == ("pages",)


def test_the_rate_with_the_most_matching_settings_wins(tmp_path: Path) -> None:
    folder = book_with(
        tmp_path,
        extend=[
            rate("credits", usd=0.01),
            rate("pages", credits=2, when={"engine": "parse_performance"}),
            rate("pages", credits=3, when={"engine": "parse_performance", "agentic_text": True}),
        ],
    )
    book, day = load_book(folder), date(2026, 10, 22)
    plain = {"engine": "parse_performance", "agentic_text": False}
    agentic = {"engine": "parse_performance", "agentic_text": True}
    light = {"engine": "parse_light", "agentic_text": False}
    assert price(book, "extend", plain, {"pages": 100}, day).usd == pytest.approx(2.0)
    assert price(book, "extend", agentic, {"pages": 100}, day).usd == pytest.approx(3.0)
    assert price(book, "extend", light, {"pages": 100}, day).missing == ("pages",)


def test_a_new_price_for_every_configuration_keeps_a_specific_rate(tmp_path: Path) -> None:
    folder = book_with(
        tmp_path,
        extend=[
            rate("credits", usd=0.01),
            rate("pages", credits=2, when={"engine": "parse_performance"}),
            rate(
                "pages",
                credits=3,
                estimate=True,
                when={"engine": "parse_performance", "agentic_text": True},
            ),
            rate(
                "pages", credits=2.5, when={"engine": "parse_performance"}, effective="2027-01-01"
            ),
        ],
    )
    book, day = load_book(folder), date(2027, 2, 1)
    agentic = {"engine": "parse_performance", "agentic_text": True}
    plain = {"engine": "parse_performance", "agentic_text": False}
    surcharge = price(book, "extend", agentic, {"pages": 100}, day)
    assert (surcharge.usd, surcharge.estimated) == (pytest.approx(3.0), True)
    assert price(book, "extend", plain, {"pages": 100}, day).usd == pytest.approx(2.5)


def test_two_rates_as_specific_for_one_configuration_are_a_book_error(tmp_path: Path) -> None:
    folder = book_with(
        tmp_path,
        extend=[
            rate("pages", credits=2, when={"engine": "parse_performance"}),
            rate("pages", credits=3, when={"agentic_text": True}),
        ],
    )
    config = {"engine": "parse_performance", "agentic_text": True}
    with pytest.raises(ValueError, match="2 rates for pages tie"):
        price(load_book(folder), "extend", config, {"pages": 1}, date(2026, 10, 22))


@pytest.mark.parametrize("later", [False, True], ids=["alone", "with a later rate"])
def test_two_rates_with_the_same_settings_and_day_are_a_book_error(
    tmp_path: Path, later: bool
) -> None:
    rates = [rate("pages", usd=15), rate("pages", usd=12)]
    if later:
        rates.insert(0, rate("pages", usd=10, effective="2026-06-01"))
    with pytest.raises(ValueError, match="two pages rates with the same settings take effect"):
        load_book(book_with(tmp_path, textract=rates))


def test_credits_are_billed_or_converted_from_pages(tmp_path: Path) -> None:
    folder = book_with(
        tmp_path,
        nutrient=[
            rate("credits", usd=0.001),
            rate("pages", credits=9, when={"mode": "understand"}),
        ],
    )
    book, day, config = load_book(folder), date(2026, 10, 22), {"mode": "understand"}
    billed = price(book, "nutrient", config, {"credits": 27, "pages": 0}, day)
    converted = price(book, "nutrient", config, {"credits": 0, "pages": 3}, day)
    assert billed.usd == converted.usd == pytest.approx(0.027)
    assert not billed.estimated and not converted.estimated


def test_an_estimated_rate_makes_the_cost_an_estimate(tmp_path: Path) -> None:
    folder = book_with(
        tmp_path,
        reducto=[
            rate("pages", usd=10, per=1000, when={"mode": "r-1"}),
            rate("charts", usd=0.06, estimate=True, when={"mode": "r-1"}),
        ],
    )
    book, day, config = load_book(folder), date(2026, 10, 22), {"mode": "r-1"}
    assert not price(book, "reducto", config, {"pages": 1000, "charts": 0}, day).estimated
    cost = price(book, "reducto", config, {"pages": 1000, "charts": 2}, day)
    assert cost.estimated and cost.usd == pytest.approx(10.12)


def test_a_meter_without_a_rate_leaves_the_cost_unknown(tmp_path: Path) -> None:
    folder = book_with(tmp_path, nutrient=[rate("pages", credits=9)])
    cost = price(load_book(folder), "nutrient", {}, {"pages": 3}, date(2026, 10, 22))
    assert (cost.usd, cost.missing) == (None, ("credits",))


def test_usage_is_measured_only_for_registered_providers_with_meters(
    demo_nutrient_raw: RawResult,
) -> None:
    documents = [(demo_nutrient_raw, 3)]
    assert measure_usage("acme", {}, documents) is None
    assert measure_usage("pdftotext", {}, documents) is None
    assert measure_usage("nutrient", None, documents) is None
    usage = measure_usage("nutrient", {"mode": "understand"}, documents)
    assert usage == {"documents": 1, "pages": 3, "meters": {"credits": 27, "pages": 0}}


def test_a_response_without_usage_is_left_out_with_its_pages(demo_nutrient_raw: RawResult) -> None:
    silent = RawResult("nutrient", "other", {"output": {}})
    documents = [(demo_nutrient_raw, 3), (silent, 5)]
    usage = measure_usage("nutrient", {"mode": "understand"}, documents)
    assert usage is not None
    assert (usage["documents"], usage["pages"]) == (1, 3)


def test_an_adapter_reporting_an_undeclared_meter_is_an_error(
    monkeypatch: pytest.MonkeyPatch, demo_nutrient_raw: RawResult
) -> None:
    monkeypatch.setattr(NutrientProvider, "usage", lambda self, raw: {"tokens": 1})
    with pytest.raises(ValueError, match="undeclared meters: tokens"):
        measure_usage("nutrient", {"mode": "understand"}, [(demo_nutrient_raw, 3)])


def test_a_response_of_an_unexpected_shape_is_left_out() -> None:
    odd = RawResult("reducto", "d", {"usage": {"num_pages": 1}, "result": {"chunks": ["x"]}})
    usage = measure_usage("reducto", {"mode": "r-1"}, [(odd, 1)])
    assert usage is not None and usage["documents"] == 0


def test_score_records_the_usage_of_the_saved_responses(
    tmp_path: Path, demo_nutrient_raw: RawResult
) -> None:
    corpus = tmp_path / "corpus" / DEMO.name
    corpus.mkdir(parents=True)
    shutil.copy(DEMO / "groundtruth.json", corpus / "groundtruth.json")
    out = tmp_path / "nutrient"
    save_prediction(
        NutrientProvider().normalize(demo_nutrient_raw), prediction_path(out, DEMO.name)
    )
    save_raw(demo_nutrient_raw, raw_path(out, DEMO.name))
    manifest = {"provider": "nutrient", "config": {"mode": "understand"}}
    (out / RUN_MANIFEST).parent.mkdir()
    (out / RUN_MANIFEST).write_text(json.dumps(manifest), "utf-8")
    args = ["--corpus", str(corpus.parent), "--predictions", str(out)]
    assert main(["score", *args, "--results", str(tmp_path / "results")]) == 0
    summary = read_summary(tmp_path / "results" / "nutrient.json")
    assert summary.usage == {"documents": 1, "pages": 3, "meters": {"credits": 27, "pages": 0}}


def test_score_records_the_usage_of_a_release_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fetch(self: MarkitdownProvider, path: Path) -> RawResult:
        return RawResult(self.name, path.parent.name, "# Notice\n", {"version": "fixture 1.0"})

    monkeypatch.setattr(MarkitdownProvider, "fetch", fetch)
    monkeypatch.setattr(MarkitdownProvider, "meters", frozenset({"pages"}))
    monkeypatch.setattr(MarkitdownProvider, "usage", lambda self, raw: {"pages": 2})
    release, out = copy_release(tmp_path / "release"), tmp_path / "parser"
    run = ["run", "--provider", "markitdown", "--corpus", str(release), "--out", str(out)]
    assert main(run) == 0
    args = ["--corpus", str(release), "--predictions", str(out)]
    assert main(["score", *args, "--results", str(tmp_path / "results")]) == 0
    usage = read_summary(tmp_path / "results" / "parser.json").usage
    assert usage is not None and usage["meters"] == {"pages": 4}
    assert usage["documents"] == 2


def test_score_leaves_out_a_saved_response_of_another_provider_or_document(
    tmp_path: Path, demo_nutrient_raw: RawResult
) -> None:
    corpus = tmp_path / "corpus" / DEMO.name
    corpus.mkdir(parents=True)
    shutil.copy(DEMO / "groundtruth.json", corpus / "groundtruth.json")
    out = tmp_path / "nutrient"
    save_prediction(
        NutrientProvider().normalize(demo_nutrient_raw), prediction_path(out, DEMO.name)
    )
    save_raw(replace(demo_nutrient_raw, provider="reducto"), raw_path(out, DEMO.name))
    manifest = {"provider": "nutrient", "config": {"mode": "understand"}}
    (out / RUN_MANIFEST).parent.mkdir()
    (out / RUN_MANIFEST).write_text(json.dumps(manifest), "utf-8")
    args = ["--corpus", str(corpus.parent), "--predictions", str(out)]
    assert main(["score", *args, "--results", str(tmp_path / "results")]) == 0
    usage = read_summary(tmp_path / "results" / "nutrient.json").usage
    assert usage is not None and usage["documents"] == 0


def test_a_usage_error_writes_no_result_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fetch(self: MarkitdownProvider, path: Path) -> RawResult:
        return RawResult(self.name, path.parent.name, "# Notice\n", {"version": "fixture 1.0"})

    monkeypatch.setattr(MarkitdownProvider, "fetch", fetch)
    monkeypatch.setattr(MarkitdownProvider, "meters", frozenset({"pages"}))
    monkeypatch.setattr(MarkitdownProvider, "usage", lambda self, raw: {"tokens": 2})
    release, out, results = copy_release(tmp_path / "release"), tmp_path / "parser", tmp_path / "r"
    assert (
        main(["run", "--provider", "markitdown", "--corpus", str(release), "--out", str(out)]) == 0
    )
    args = ["--corpus", str(release), "--predictions", str(out), "--results", str(results)]
    assert main(["score", *args]) == 2
    assert not results.exists() or not any(results.iterdir())


def test_a_summary_without_usage_is_written_as_before(tmp_path: Path) -> None:
    write_summary(_summary("p", None), tmp_path / "p.json")
    assert "usage" not in json.loads((tmp_path / "p.json").read_text("utf-8"))


def _summary(
    parser: str, usage: dict[str, Any] | None, prediction: Prediction | None = None, **fields: Any
):
    truth = load_groundtruth(DEMO / "groundtruth.json")
    result = score_corpus([(truth, prediction or Prediction(markdown="x\n"))])
    return summarize(parser, result, "v1", usage=usage, **fields)


def leaderboard(tmp_path: Path, prices: Path | None, *summaries, as_of: str = "2026-10-22") -> str:
    results = tmp_path / "results"
    for summary in summaries:
        write_summary(summary, results / f"{summary.parser}.json")
    args = ["report", "--results", str(results)]
    if prices is not None:
        args += ["--prices", str(prices), "--prices-as-of", as_of]
    assert main(args) == 0
    return (tmp_path / "LEADERBOARD.md").read_text("utf-8")


def row(markdown: str, parser: str) -> list[str]:
    line = next(line for line in markdown.splitlines() if line.startswith(f"| {parser} "))
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_the_leaderboard_shows_each_runs_cost_per_1000_pages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prices = book_with(
        tmp_path,
        nutrient=[rate("credits", usd=0.001), rate("pages", credits=9)],
        reducto=[rate("pages", usd=10, per=1000), rate("charts", usd=0.06, estimate=True)],
    )
    nutrient = _summary(
        "nutrient",
        {"documents": 1, "pages": 3, "meters": {"credits": 27, "pages": 0}},
        provider="nutrient",
        provider_config={"mode": "understand"},
    )
    reducto = _summary(
        "reducto",
        {"documents": 1, "pages": 3, "meters": {"pages": 3, "charts": 1}},
        provider="reducto",
        provider_config={"mode": "r-1"},
    )
    local = _summary("pdftotext", None, provider="pdftotext", provider_config={})
    markdown = leaderboard(tmp_path, prices, nutrient, reducto, local)
    assert "| $/1k pages |" in markdown
    assert row(markdown, "nutrient")[-1] == "9.00"
    assert row(markdown, "reducto")[-1] == "~30.00"
    assert row(markdown, "pdftotext")[-1] == "local"
    assert "in force on 2026-10-22" in markdown
    assert "~ estimated" in markdown
    table = capsys.readouterr().out.splitlines()
    assert table[0].endswith("$/1k pages")
    cells = {line.split()[0]: line[-len("$/1k pages") :] for line in table[2:5]}
    assert cells == {"nutrient": "      9.00", "pdftotext": "     local", "reducto": "    ~30.00"}


def test_the_cost_never_changes_the_ranking(tmp_path: Path, demo_nutrient_raw: RawResult) -> None:
    prices = book_with(tmp_path, textract=[rate("pages", usd=15, per=1000)])
    good = NutrientProvider().normalize(demo_nutrient_raw)
    orders = []
    for good_pages, weak_pages in ((3000, 3), (3, 3000)):
        rows = [
            _summary("good", usage(good_pages), good, provider="textract", provider_config={}),
            _summary("weak", usage(weak_pages), provider="textract", provider_config={}),
        ]
        markdown = leaderboard(tmp_path / str(good_pages), prices, *rows)
        orders.append([line.split(" | ")[0] for line in markdown.splitlines() if line[:2] == "| "])
    assert orders == [["| Parser", "| good", "| weak"]] * 2


def usage(pages: int) -> dict[str, Any]:
    return {"documents": 1, "pages": 3, "meters": {"pages": pages}}


def test_the_leaderboard_names_partial_usage_missing_and_stale_prices(tmp_path: Path) -> None:
    prices = book_with(
        tmp_path,
        textract=[rate("pages", usd=15, per=1000, checked="2026-01-15")],
        nutrient=[rate("pages", credits=9)],
    )
    textract = _summary(
        "textract",
        {"documents": 0, "pages": 0, "meters": {"pages": 0}},
        provider="textract",
        provider_config={},
    )
    stale = _summary(
        "stale",
        {"documents": 1, "pages": 3, "meters": {"pages": 3}},
        provider="textract",
        provider_config={},
    )
    nutrient = _summary(
        "nutrient",
        {"documents": 1, "pages": 3, "meters": {"credits": 0, "pages": 3}},
        provider="nutrient",
        provider_config={"mode": "understand"},
    )
    markdown = leaderboard(tmp_path, prices, textract, stale, nutrient)
    assert row(markdown, "textract")[-1] == "n/a"
    assert row(markdown, "nutrient")[-1] == "no price"
    assert "Usage from fewer documents than scored: textract (0/1)." in markdown
    assert "No price in the book for: nutrient (credits)." in markdown
    assert "Prices not checked in the 90 days before 2026-10-22: textract pages" in markdown


def test_no_cost_column_without_usage(tmp_path: Path) -> None:
    prices = book_with(tmp_path, textract=[rate("pages", usd=15, per=1000)])
    local = _summary("pdftotext", None, provider="pdftotext", provider_config={})
    assert "$/1k pages" not in leaderboard(tmp_path, prices, local)


def test_an_installed_scorer_without_a_book_reports_no_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("wholedoc_bench.cli.BOOK", tmp_path / "no-book")
    usage = {"documents": 1, "pages": 3, "meters": {"pages": 3}}
    cloud = _summary("textract", usage, provider="textract", provider_config={})
    assert "$/1k pages" not in leaderboard(tmp_path, None, cloud)


def test_the_checkouts_book_is_the_default(tmp_path: Path) -> None:
    usage = {"documents": 1, "pages": 1000, "meters": {"pages": 1000}}
    cloud = _summary("textract", usage, provider="textract", provider_config={})
    assert row(leaderboard(tmp_path, None, cloud), "textract")[-1] == "15.00"


def test_a_broken_price_book_stops_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prices = book_with(tmp_path, textract=[rate("tokens", usd=1)])
    results = tmp_path / "results"
    write_summary(_summary("p", None), results / "p.json")
    assert main(["report", "--results", str(results), "--prices", str(prices)]) == 2
    assert "textract has no meter 'tokens'" in capsys.readouterr().err


def test_prices_as_of_without_a_price_book_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("wholedoc_bench.cli.BOOK", tmp_path / "no-book")
    results = tmp_path / "results"
    write_summary(_summary("p", None), results / "p.json")
    assert main(["report", "--results", str(results), "--prices-as-of", "2026-10-22"]) == 2
    assert "--prices-as-of needs a price book" in capsys.readouterr().err


@pytest.mark.parametrize(
    "usage",
    [
        {"documents": 1},
        {"documents": 1, "pages": 3, "meters": {"pages": "3"}},
        {"documents": True, "pages": 3, "meters": {}},
        [1],
    ],
    ids=[
        "missing counts",
        "a meter that is not a number",
        "a count that is a boolean",
        "not an object",
    ],
)
def test_a_summary_with_malformed_usage_is_refused(
    tmp_path: Path, usage: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    write_summary(_summary("p", None), results / "p.json")
    data = json.loads((results / "p.json").read_text("utf-8"))
    (results / "p.json").write_text(json.dumps({**data, "usage": usage}), "utf-8")
    assert main(["report", "--results", str(results)]) == 2
    assert "p.json: not a parser summary" in capsys.readouterr().err
