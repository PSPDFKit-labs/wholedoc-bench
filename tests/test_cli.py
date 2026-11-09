import json
import shutil
from pathlib import Path
from typing import Any

import httpx
import httpx2
import pytest

from tests.builders import para
from tests.conftest import DEMO, FIXTURES, TWO_COLUMN_AZURE, TWO_COLUMN_TEXTRACT, StubbedSts
from wholedoc_bench.cli import RUN_MANIFEST, main, read_prediction
from wholedoc_bench.corpus.loader import load_prediction, prediction_path, save_prediction
from wholedoc_bench.providers import (
    PROVIDERS,
    AnthropicProvider,
    AzureDIProvider,
    MarkitdownProvider,
    NutrientProvider,
    OpenAIProvider,
    RawResult,
    TextractProvider,
)
from wholedoc_bench.schema import Prediction
from wholedoc_bench.scorer.run import ParseFailure
from wholedoc_bench.scorer.tracks import ALL_TRACKS

SECRET = "zebra quartz 9931"


def test_every_provider_is_registered_by_its_name() -> None:
    assert sorted(PROVIDERS) == [
        "anthropic",
        "azure-di",
        "databricks",
        "docling",
        "extend",
        "llamaparse",
        "markitdown",
        "nutrient",
        "openai",
        "pdftotext",
        "pymupdf",
        "reducto",
        "textract",
    ]
    assert all(PROVIDERS[name].name == name for name in PROVIDERS)


def _corpus(tmp_path: Path, suffix: str) -> Path:
    corpus = tmp_path / "corpus"
    folder = corpus / "plain_continuation_demo"
    folder.mkdir(parents=True)
    shutil.copy(DEMO / "groundtruth.json", folder / "groundtruth.json")
    shutil.copy(DEMO / "document.pdf", folder / f"document{suffix}")
    return corpus


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="needs poppler's pdftotext")
def test_run_writes_a_prediction_and_the_parser_version(tmp_path: Path) -> None:
    out = tmp_path / "predictions"
    corpus = _corpus(tmp_path, ".pdf")
    assert main(["run", "--provider", "pdftotext", "--corpus", str(corpus), "--out", str(out)]) == 0
    prediction = load_prediction(out / "plain_continuation_demo.json")
    assert "Plain page continuation" in prediction.markdown
    assert (out / "VERSION.txt").read_text().startswith("pdftotext version")


def test_run_skips_a_format_the_parser_does_not_read(tmp_path: Path) -> None:
    out = tmp_path / "predictions"
    corpus = _corpus(tmp_path, ".docx")
    assert main(["run", "--provider", "pdftotext", "--corpus", str(corpus), "--out", str(out)]) == 0
    assert not (out / "plain_continuation_demo.json").exists()


def test_a_failed_rerun_into_the_same_folder_counts_a_failed_parse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "predictions"
    corpus = _corpus(tmp_path, ".pdf")
    args = ["run", "--provider", "markitdown", "--corpus", str(corpus), "--out", str(out)]

    def parse(self: MarkitdownProvider, document_path: Path) -> RawResult:
        return RawResult(
            provider="markitdown",
            document_id=document_path.parent.name,
            payload="# Old run\n",
            metadata={"version": "markitdown 1.0"},
        )

    def crash(self: MarkitdownProvider, document_path: Path) -> RawResult:
        raise RuntimeError("parser crashed")

    monkeypatch.setattr(MarkitdownProvider, "fetch", parse)
    assert main(args) == 0
    monkeypatch.setattr(MarkitdownProvider, "fetch", crash)
    assert main(args) == 0
    assert isinstance(read_prediction(out / "plain_continuation_demo.json"), ParseFailure)
    assert not (out / "VERSION.txt").exists()


def test_a_conversion_failure_keeps_the_raw_response_for_a_rerun_without_the_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "predictions"
    corpus = _corpus(tmp_path, ".pdf")
    args = ["run", "--provider", "markitdown", "--corpus", str(corpus), "--out", str(out)]
    calls: list[Path] = []

    def fetch(self: MarkitdownProvider, document_path: Path) -> RawResult:
        calls.append(document_path)
        return RawResult("markitdown", document_path.parent.name, "# Paid\n", {"version": "m 1"})

    def broken(self: MarkitdownProvider, raw: RawResult) -> Any:
        raise ValueError("no class mapping for 'sticker'")

    monkeypatch.setattr(MarkitdownProvider, "fetch", fetch)
    monkeypatch.setattr(MarkitdownProvider, "normalize", broken)
    assert main(args) == 0
    assert not (out / "plain_continuation_demo.json").exists()
    assert "--from-raw" in capsys.readouterr().out

    monkeypatch.undo()
    monkeypatch.setattr(MarkitdownProvider, "fetch", fetch)
    calls.clear()
    assert main([*args, "--from-raw"]) == 0
    assert calls == []
    assert load_prediction(out / "plain_continuation_demo.json").markdown == "# Paid\n"
    assert (out / "VERSION.txt").read_text() == "m 1\n"


def test_from_raw_without_a_saved_response_writes_no_prediction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "predictions"
    corpus = _corpus(tmp_path, ".pdf")
    args = ["run", "--provider", "markitdown", "--corpus", str(corpus), "--out", str(out)]
    assert main([*args, "--from-raw"]) == 0
    assert "no raw response" in capsys.readouterr().out
    assert not (out / "plain_continuation_demo.json").exists()


def test_cli_score_runs_end_to_end(
    demo_nutrient_raw: RawResult, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    predictions = tmp_path / "predictions" / "nutrient"
    pred = NutrientProvider().normalize(demo_nutrient_raw)
    save_prediction(pred, prediction_path(predictions, "plain_continuation_demo"))
    results = tmp_path / "results"
    out = results / "nutrient.jsonl"

    code = main(
        ["score", "--corpus", str(FIXTURES), "--predictions", str(predictions)]
        + ["--results", str(results)]
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert "G3 coverage: 1/1 documents (100%), pass" in printed
    stubs = any(not metric.implemented for t in ALL_TRACKS for metric, _ in t.parts)
    assert ("Not implemented yet" in printed) is stubs
    assert len(out.read_text().splitlines()) == sum(len(t.parts) for t in ALL_TRACKS)
    assert (results / "nutrient.json").exists()


@pytest.mark.parametrize("content", ["{not json", b"\xff\xfe"])
def test_cli_score_counts_an_invalid_prediction_as_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | bytes
) -> None:
    path = prediction_path(tmp_path, "plain_continuation_demo")
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    code = main(["score", "--corpus", str(FIXTURES), "--predictions", str(tmp_path)])
    printed = capsys.readouterr().out
    assert code == 0
    assert "G3 coverage: 0/1 documents (0%), FAIL" in printed
    assert "failed: plain_continuation_demo:" in printed


def corpus_with(tmp_path: Path, kind: str) -> Path:
    data = json.loads((DEMO / "groundtruth.json").read_text(encoding="utf-8"))
    held = {**para("secret", SECRET), "channel": "hidden"}
    data["elements"].append(held)
    data["planted"] = [{"kind": kind, "printed": SECRET}]
    folder = tmp_path / "corpus" / DEMO.name
    folder.mkdir(parents=True)
    (folder / "groundtruth.json").write_text(json.dumps(data), encoding="utf-8")
    return folder.parent


def score(tmp_path: Path, corpus: Path, markdown: str | None) -> int:
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    if markdown is not None:
        save_prediction(Prediction(markdown=markdown), prediction_path(predictions, DEMO.name))
    return main(["score", "--corpus", str(corpus), "--predictions", str(predictions)])


def test_cli_reports_no_leaks_for_a_written_redacted_string(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = corpus_with(tmp_path, "redacted")
    assert score(tmp_path, corpus, "Intro text. Zebra Quartz 9931.") == 0
    printed = capsys.readouterr().out
    assert "G1" not in printed
    assert "leak" not in printed.lower()


def test_score_stops_on_a_predictions_folder_that_does_not_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    args = ["score", "--corpus", str(FIXTURES), "--predictions", str(tmp_path / "nope")]
    assert main([*args, "--results", str(results)]) == 1
    assert "nope: no such predictions folder" in capsys.readouterr().err
    assert not results.exists()


@pytest.mark.parametrize("config", [[], None, {"mode": "text"}, {"mode": 2}, {"typo": True}])
def test_invalid_config_preserves_outputs(tmp_path: Path, config: Any) -> None:
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "VERSION.txt"
    sentinel.write_text("previous")
    manifest = out / RUN_MANIFEST
    manifest.parent.mkdir()
    manifest.write_text('{"provider":"nutrient","config":{"mode":"understand"}}')
    before = _snapshot(out)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    assert (
        main(
            [
                "run",
                "--provider",
                "nutrient",
                "--corpus",
                str(corpus),
                "--out",
                str(out),
                "--config",
                str(config_path),
            ]
        )
        == 2
    )
    assert _snapshot(out) == before


def test_config_provenance_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUTRIENT_API_KEY", "test-key")
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "agentic"
    config = tmp_path / "config.json"
    config.write_text('{"mode": "agentic"}')
    calls = []

    def fetch(self: NutrientProvider, path: Path) -> RawResult:
        calls.append(self.mode)
        return RawResult(
            self.name, path.parent.name, {"output": {"elements": [], "markdown": "hi"}}
        )

    monkeypatch.setattr(NutrientProvider, "fetch", fetch)
    args = ["run", "--provider", "nutrient", "--corpus", str(corpus), "--out", str(out)]
    assert main([*args, "--config", str(config)]) == 0
    assert calls == ["agentic"]
    assert json.loads((out / RUN_MANIFEST).read_text()) == {
        "provider": "nutrient",
        "config": {"mode": "agentic"},
    }
    raw_path = out / "raw" / "plain_continuation_demo.json"
    raw = json.loads(raw_path.read_text())
    assert raw["metadata"]["config"] == {"mode": "agentic"}
    monkeypatch.delenv("NUTRIENT_API_KEY")
    assert main([*args, "--from-raw"]) == 0
    assert calls == ["agentic"]
    results = tmp_path / "results"
    assert (
        main(
            ["score", "--corpus", str(corpus), "--predictions", str(out), "--results", str(results)]
        )
        == 0
    )
    summary = json.loads((results / "agentic.json").read_text())
    assert summary["provider"] == "nutrient"
    assert summary["provider_config"] == {"mode": "agentic"}
    prediction = out / "plain_continuation_demo.json"
    original = prediction.read_bytes()
    config.write_text('{"mode": "structure"}')
    assert main([*args, "--from-raw", "--config", str(config)]) == 2
    assert prediction.read_bytes() == original
    raw["metadata"]["config"] = {"mode": "structure"}
    raw_path.write_text(json.dumps(raw))
    assert main([*args, "--from-raw"]) == 2
    assert prediction.read_bytes() == original


@pytest.mark.parametrize("key", [None, "", " \t\n"])
@pytest.mark.parametrize("existing", [False, True])
def test_missing_credentials_preserve_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    key: str | None,
    existing: bool,
) -> None:
    if key is None:
        monkeypatch.delenv("NUTRIENT_API_KEY", raising=False)
    else:
        monkeypatch.setenv("NUTRIENT_API_KEY", key)
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    if existing:
        manifest = out / RUN_MANIFEST
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{"provider":"nutrient","config":{"mode":"understand"}}')
        (out / "VERSION.txt").write_text("previous version")
        (out / "plain_continuation_demo.json").write_text('{"markdown":"previous"}')
        saved = out / "raw" / "plain_continuation_demo.json"
        saved.parent.mkdir()
        saved.write_text("previous raw response")
    before = _snapshot(out)

    def unexpected_fetch(self: NutrientProvider, document_path: Path) -> RawResult:
        pytest.fail("credential preflight must run before fetch")

    monkeypatch.setattr(NutrientProvider, "fetch", unexpected_fetch)
    args = ["run", "--provider", "nutrient", "--corpus", str(corpus), "--out", str(out)]
    assert main(args) == 2
    assert "NUTRIENT_API_KEY is not set or is blank" in capsys.readouterr().err
    assert _snapshot(out) == before
    assert out.exists() is existing


@pytest.mark.parametrize("outcome", ["success", "http_error", "transport_error"])
def test_nutrient_credentials_stay_out_of_artifacts_and_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
) -> None:
    key = "test-credential-do-not-persist-42"
    monkeypatch.setenv("NUTRIENT_API_KEY", key)
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == f"Bearer {key}"
        if outcome == "transport_error":
            raise httpx.ConnectError(f"transport echoed {key}", request=request)
        if outcome == "http_error":
            return httpx.Response(401, json={"errorMessage": f"invalid credential: {key}"})
        return httpx.Response(200, json={"output": {"elements": [], "markdown": "parsed"}})

    class MockNutrientProvider(NutrientProvider):
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            super().__init__(config, transport=httpx.MockTransport(handle))

    monkeypatch.setitem(PROVIDERS, "nutrient", MockNutrientProvider)
    args = ["run", "--provider", "nutrient", "--corpus", str(corpus), "--out", str(out)]
    assert main(args) == 0
    assert len(requests) == 1
    printed = capsys.readouterr()
    assert key not in printed.out + printed.err
    assert all(key.encode() not in data for data in _snapshot(out).values())
    prediction = out / "plain_continuation_demo.json"
    assert prediction.exists() is (outcome == "success")
    if outcome == "success":
        monkeypatch.delenv("NUTRIENT_API_KEY")
        assert main([*args, "--from-raw"]) == 0
        assert len(requests) == 1
    else:
        assert not (out / "raw" / "plain_continuation_demo.json").exists()
        assert "Nutrient API" in printed.out


@pytest.mark.parametrize("existing", [False, True])
def test_missing_aws_credentials_preserve_outputs(
    tmp_path: Path,
    isolated_aws: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    existing: bool,
) -> None:
    isolated_aws.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    if existing:
        manifest = out / RUN_MANIFEST
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{"provider":"textract","config":{}}')
        (out / "plain_continuation_demo.json").write_text('{"markdown":"previous"}')
    before = _snapshot(out)

    def unexpected_fetch(self: TextractProvider, document_path: Path) -> RawResult:
        pytest.fail("credential preflight must run before fetch")

    isolated_aws.setattr(TextractProvider, "fetch", unexpected_fetch)
    args = ["run", "--provider", "textract", "--corpus", str(corpus), "--out", str(out)]
    assert main(args) == 2
    assert "No AWS credentials found" in capsys.readouterr().err
    assert _snapshot(out) == before
    assert out.exists() is existing


def test_rejected_aws_keys_preserve_outputs(
    tmp_path: Path,
    isolated_aws: pytest.MonkeyPatch,
    sts: StubbedSts,
    capsys: pytest.CaptureFixture[str],
) -> None:
    isolated_aws.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    isolated_aws.setenv("AWS_SECRET_ACCESS_KEY", "expired-secret-do-not-persist")
    isolated_aws.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    sts.errors.append("ExpiredToken")
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    manifest = out / RUN_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"provider":"textract","config":{}}')
    (out / "plain_continuation_demo.json").write_text('{"markdown":"previous"}')
    saved = out / "raw" / "plain_continuation_demo.json"
    saved.parent.mkdir()
    saved.write_text("previous paid response")
    before = _snapshot(out)

    def unexpected_fetch(self: TextractProvider, document_path: Path) -> RawResult:
        pytest.fail("the credential check must run before fetch")

    isolated_aws.setattr(TextractProvider, "fetch", unexpected_fetch)
    args = ["run", "--provider", "textract", "--corpus", str(corpus), "--out", str(out)]
    assert main(args) == 2
    printed = capsys.readouterr()
    assert "ExpiredToken" in printed.err
    assert "expired-secret-do-not-persist" not in printed.out + printed.err
    assert _snapshot(out) == before


@pytest.mark.parametrize("outcome", ["success", "service_error"])
def test_textract_credentials_stay_out_of_artifacts_and_diagnostics(
    tmp_path: Path,
    isolated_aws: pytest.MonkeyPatch,
    sts: StubbedSts,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
) -> None:
    from botocore.stub import Stubber

    key_id, secret = "AKIAIOSFODNN7EXAMPLE", "test-credential-do-not-persist-42"
    isolated_aws.setenv("AWS_ACCESS_KEY_ID", key_id)
    isolated_aws.setenv("AWS_SECRET_ACCESS_KEY", secret)
    isolated_aws.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    pages = json.loads(TWO_COLUMN_TEXTRACT.read_text(encoding="utf-8"))["pages"]
    stubbers: list[Stubber] = []

    class StubbedTextractProvider(TextractProvider):
        def _textract(self) -> Any:
            client = super()._textract()
            if not stubbers:
                stubber = Stubber(client)
                for page in pages:
                    if outcome == "success":
                        stubber.add_response("analyze_document", page)
                    else:
                        stubber.add_client_error(
                            "analyze_document",
                            service_error_code="InvalidSignatureException",
                            service_message=f"signature for {key_id} with {secret}",
                        )
                stubber.activate()
                stubbers.append(stubber)
            return client

    isolated_aws.setitem(PROVIDERS, "textract", StubbedTextractProvider)
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    args = ["run", "--provider", "textract", "--corpus", str(corpus), "--out", str(out)]
    assert main(args) == 0
    printed = capsys.readouterr()
    for value in (key_id, secret):
        assert value not in printed.out + printed.err
        assert all(value.encode() not in data for data in _snapshot(out).values())
    prediction = out / "plain_continuation_demo.json"
    assert prediction.exists() is (outcome == "success")
    if outcome == "success":
        stubbers[0].assert_no_pending_responses()
        sts.assert_checked()
        assert (out / "VERSION.txt").read_text() == (
            "textract AnalyzeDocument LAYOUT+TABLES, model 1.0\n"
        )
        isolated_aws.delenv("AWS_ACCESS_KEY_ID")
        isolated_aws.delenv("AWS_SECRET_ACCESS_KEY")
        assert main([*args, "--from-raw"]) == 0
        assert "Annual Review 2026" in load_prediction(prediction).markdown
    else:
        assert "InvalidSignatureException" in printed.out
        assert not (out / "raw" / "plain_continuation_demo.json").exists()


AZURE_ENDPOINT = "https://example-resource.cognitiveservices.azure.com"


@pytest.mark.parametrize("missing", ["endpoint", "key", "both"])
def test_missing_azure_settings_preserve_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    for name, value in (("ENDPOINT", AZURE_ENDPOINT), ("KEY", "test-key")):
        variable = f"AZURE_DOCUMENT_INTELLIGENCE_{name}"
        if missing in (name.lower(), "both"):
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, value)
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    manifest = out / RUN_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"provider": "azure-di", "config": AzureDIProvider().config}))
    (out / "plain_continuation_demo.json").write_text('{"markdown":"previous"}')
    before = _snapshot(out)

    def unexpected_fetch(self: AzureDIProvider, document_path: Path) -> RawResult:
        pytest.fail("credential preflight must run before fetch")

    monkeypatch.setattr(AzureDIProvider, "fetch", unexpected_fetch)
    args = ["run", "--provider", "azure-di", "--corpus", str(corpus), "--out", str(out)]
    assert main(args) == 2
    assert "AZURE_DOCUMENT_INTELLIGENCE_" in capsys.readouterr().err
    assert _snapshot(out) == before


@pytest.mark.parametrize("outcome", ["success", "rejected_key", "service_error"])
def test_azure_key_stays_out_of_artifacts_and_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
) -> None:
    key = "test-credential-do-not-persist-42"
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", AZURE_ENDPOINT)
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", key)
    status = (
        f"{AZURE_ENDPOINT}/documentintelligence/documentModels/prebuilt-layout/analyzeResults/r1"
    )
    result = json.loads(TWO_COLUMN_AZURE.read_text(encoding="utf-8"))
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Ocp-Apim-Subscription-Key"] == key
        denied = {"error": {"code": "401", "message": f"Access denied for {key}"}}
        if request.url.path.endswith("/info"):
            return httpx.Response(401 if outcome == "rejected_key" else 200, json=denied)
        if outcome == "service_error":
            error = {"code": "InvalidRequest", "message": f"bad request with {key}"}
            return httpx.Response(400, json={"error": error})
        if request.method == "POST":
            return httpx.Response(202, headers={"Operation-Location": status})
        return httpx.Response(200, json=result)

    class MockAzureDIProvider(AzureDIProvider):
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            super().__init__(config, transport=httpx.MockTransport(handle), sleep=lambda _: None)

    monkeypatch.setitem(PROVIDERS, "azure-di", MockAzureDIProvider)
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    args = ["run", "--provider", "azure-di", "--corpus", str(corpus), "--out", str(out)]
    assert main(args) == (2 if outcome == "rejected_key" else 0)
    printed = capsys.readouterr()
    assert key not in printed.out + printed.err
    assert all(key.encode() not in data for data in _snapshot(out).values())
    prediction = out / "plain_continuation_demo.json"
    assert prediction.exists() is (outcome == "success")
    if outcome == "success":
        assert (out / "VERSION.txt").read_text() == (
            "azure-di prebuilt-layout; API 2024-11-30; "
            "features barcodes+formulas+ocrHighResolution+styleFont\n"
        )
        monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        made = len(requests)
        assert main([*args, "--from-raw"]) == 0
        assert len(requests) == made
        assert "Annual Review 2026" in load_prediction(prediction).markdown
    elif outcome == "rejected_key":
        assert "did not accept the key (HTTP 401)" in printed.err
        assert not out.exists()
    else:
        assert "HTTP 400, InvalidRequest" in printed.out
        assert not (out / "raw" / "plain_continuation_demo.json").exists()


def test_a_missing_anthropic_key_preserves_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    manifest = out / RUN_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"provider": "anthropic", "config": AnthropicProvider().config}))
    (out / "plain_continuation_demo.json").write_text('{"markdown":"previous"}')
    before = _snapshot(out)

    def unexpected_fetch(self: AnthropicProvider, document_path: Path) -> RawResult:
        pytest.fail("credential preflight must run before fetch")

    monkeypatch.setattr(AnthropicProvider, "fetch", unexpected_fetch)
    args = ["run", "--provider", "anthropic", "--corpus", str(corpus), "--out", str(out)]
    assert main(args) == 2
    assert "ANTHROPIC_API_KEY is not set or is blank" in capsys.readouterr().err
    assert _snapshot(out) == before


@pytest.mark.parametrize("outcome", ["success", "rejected_key", "service_error"])
def test_anthropic_key_stays_out_of_artifacts_and_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
) -> None:
    key = "test-credential-do-not-persist-42"
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    requests: list[httpx2.Request] = []
    page = (
        '<div data-bbox="[100, 100, 900, 140]" data-label="Title"># Plain page continuation</div>'
    )
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": page}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 20},
        },
        {"type": "message_stop"},
    ]
    stream = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert request.headers["x-api-key"] == key
        if request.url.path.startswith("/v1/models/"):
            if outcome == "rejected_key":
                error = {"type": "authentication_error", "message": f"invalid x-api-key {key}"}
                return httpx2.Response(401, json={"type": "error", "error": error})
            return httpx2.Response(200, json={"id": "claude-opus-5-5", "type": "model"})
        if outcome == "service_error":
            error = {"type": "invalid_request_error", "message": f"bad request with {key}"}
            return httpx2.Response(400, json={"type": "error", "error": error})
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=stream.encode()
        )

    class MockAnthropicProvider(AnthropicProvider):
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            super().__init__(config, transport=httpx2.MockTransport(handle))

    monkeypatch.setitem(PROVIDERS, "anthropic", MockAnthropicProvider)
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    args = ["run", "--provider", "anthropic", "--corpus", str(corpus), "--out", str(out)]
    assert main(args) == (2 if outcome == "rejected_key" else 0)
    printed = capsys.readouterr()
    assert key not in printed.out + printed.err
    assert all(key.encode() not in data for data in _snapshot(out).values())
    prediction = out / "plain_continuation_demo.json"
    assert prediction.exists() is (outcome == "success")
    if outcome == "success":
        assert (
            (out / "VERSION.txt")
            .read_text()
            .startswith("anthropic claude-opus-5-5; effort medium; prompt ")
        )
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        made = len(requests)
        assert main([*args, "--from-raw"]) == 0
        assert len(requests) == made
        markdown = load_prediction(prediction).markdown
        assert markdown.count("# Plain page continuation") == 3
    elif outcome == "rejected_key":
        assert "did not accept the key (HTTP 401, authentication_error)" in printed.err
        assert not out.exists()
    else:
        assert "HTTP 400, invalid_request_error" in printed.out
        assert not (out / "raw" / "plain_continuation_demo.json").exists()


def test_a_missing_openai_key_preserves_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    manifest = out / RUN_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"provider": "openai", "config": OpenAIProvider().config}))
    (out / "plain_continuation_demo.json").write_text('{"markdown":"previous"}')
    before = _snapshot(out)

    def unexpected_fetch(self: OpenAIProvider, document_path: Path) -> RawResult:
        pytest.fail("credential preflight must run before fetch")

    monkeypatch.setattr(OpenAIProvider, "fetch", unexpected_fetch)
    args = ["run", "--provider", "openai", "--corpus", str(corpus), "--out", str(out)]
    assert main(args) == 2
    assert "OPENAI_API_KEY is not set or is blank" in capsys.readouterr().err
    assert _snapshot(out) == before


@pytest.mark.parametrize("outcome", ["success", "rejected_key", "service_error"])
def test_openai_key_stays_out_of_artifacts_and_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
) -> None:
    key = "test-credential-do-not-persist-42"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)
    requests: list[httpx2.Request] = []
    page = (
        '<div data-bbox="[100, 100, 900, 140]" data-label="Title"># Plain page continuation</div>'
    )
    done = {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "model": "gpt-6.1-sol",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": page, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    }
    events = [
        {"type": "response.created", "sequence_number": 0, "response": {**done, "output": []}},
        {"type": "response.completed", "sequence_number": 1, "response": done},
    ]
    stream = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert request.headers["authorization"] == f"Bearer {key}"
        if request.url.path.startswith("/v1/models/"):
            if outcome == "rejected_key":
                error = {"type": "invalid_request_error", "message": f"Incorrect API key {key}"}
                return httpx2.Response(401, json={"error": error})
            return httpx2.Response(200, json={"id": "gpt-6.1-sol", "object": "model"})
        if outcome == "service_error":
            error = {"type": "invalid_request_error", "message": f"bad request with {key}"}
            return httpx2.Response(400, json={"error": error})
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=stream.encode()
        )

    class MockOpenAIProvider(OpenAIProvider):
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            super().__init__(config, transport=httpx2.MockTransport(handle))

    monkeypatch.setitem(PROVIDERS, "openai", MockOpenAIProvider)
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    args = ["run", "--provider", "openai", "--corpus", str(corpus), "--out", str(out)]
    assert main(args) == (2 if outcome == "rejected_key" else 0)
    printed = capsys.readouterr()
    assert key not in printed.out + printed.err
    assert all(key.encode() not in data for data in _snapshot(out).values())
    prediction = out / "plain_continuation_demo.json"
    assert prediction.exists() is (outcome == "success")
    if outcome == "success":
        assert (
            (out / "VERSION.txt")
            .read_text()
            .startswith("openai gpt-6.1-sol; effort medium; prompt ")
        )
        monkeypatch.delenv("OPENAI_API_KEY")
        made = len(requests)
        assert main([*args, "--from-raw"]) == 0
        assert len(requests) == made
        markdown = load_prediction(prediction).markdown
        assert markdown.count("# Plain page continuation") == 3
    elif outcome == "rejected_key":
        assert "did not accept the key (HTTP 401, invalid_request_error)" in printed.err
        assert not out.exists()
    else:
        assert "HTTP 400, invalid_request_error" in printed.out
        assert not (out / "raw" / "plain_continuation_demo.json").exists()


def test_providers_without_options_reject_config() -> None:
    with pytest.raises(ValueError, match="unsupported config"):
        MarkitdownProvider({"mode": "agentic"})
    assert NutrientProvider().config == {"mode": "understand"}


@pytest.mark.parametrize("field", ["api_key", "NUTRIENT_API_KEY"])
def test_nutrient_rejects_credentials_in_config_without_printing_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str
) -> None:
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    key = "credential-that-must-not-appear"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({field: key}))
    assert (
        main(
            [
                "run",
                "--provider",
                "nutrient",
                "--corpus",
                str(corpus),
                "--out",
                str(out),
                "--config",
                str(config),
            ]
        )
        == 2
    )
    printed = capsys.readouterr()
    assert key not in printed.out + printed.err
    assert "unsupported config keys" in printed.err
    assert not out.exists()


@pytest.mark.parametrize("provider", ["nutrient", "nutrient-agentic"])
def test_replay_refuses_legacy_raw_without_config(tmp_path: Path, provider: str) -> None:
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    raw = out / "raw" / "plain_continuation_demo.json"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        json.dumps(
            {
                "provider": provider,
                "document_id": "plain_continuation_demo",
                "payload": {},
                "metadata": {},
            }
        )
    )
    assert (
        main(
            [
                "run",
                "--provider",
                "nutrient",
                "--corpus",
                str(corpus),
                "--out",
                str(out),
                "--from-raw",
            ]
        )
        == 2
    )
    assert not (out / RUN_MANIFEST).exists()


def test_a_docling_run_from_before_chart_extraction_does_not_replay_as_one_with_charts(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    manifest = out / RUN_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"provider": "docling", "config": {}}))
    raw = out / "raw" / "plain_continuation_demo.json"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        json.dumps(
            {
                "provider": "docling",
                "document_id": "plain_continuation_demo",
                "payload": "Text without chart data.\n",
                "metadata": {"config": {}},
            }
        )
    )
    before = _snapshot(out)
    args = ["run", "--provider", "docling", "--corpus", str(corpus), "--out", str(out)]
    assert main([*args, "--from-raw"]) == 2
    assert _snapshot(out) == before


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("existing", ["other.json", "raw/other.json", "RUN.json"])
def test_run_refuses_populated_legacy_folder_even_with_explicit_config(
    tmp_path: Path, existing: str
) -> None:
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    old = out / existing
    old.parent.mkdir(parents=True)
    old.write_text('{"markdown":"untouched legacy document"}')
    config = tmp_path / "config.json"
    config.write_text('{"mode":"agentic"}')
    before = _snapshot(out)
    assert (
        main(
            [
                "run",
                "--provider",
                "nutrient",
                "--config",
                str(config),
                "--corpus",
                str(corpus),
                "--out",
                str(out),
            ]
        )
        == 2
    )
    assert _snapshot(out) == before


def test_invalid_later_corpus_document_preserves_all_run_outputs(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ".pdf")
    invalid = corpus / "zzz_bad" / "groundtruth.json"
    invalid.parent.mkdir()
    invalid.write_text('{"schema_version":"bad"}')
    out = tmp_path / "out"
    manifest = out / RUN_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"provider":"nutrient", "config":{"mode":"agentic"}}')
    (out / "VERSION.txt").write_text("original version\n")
    (out / "plain_continuation_demo.json").write_text('{"markdown":"original"}')
    before = _snapshot(out)
    assert (
        main(
            [
                "run",
                "--provider",
                "nutrient",
                "--corpus",
                str(corpus),
                "--out",
                str(out),
                "--from-raw",
            ]
        )
        == 2
    )
    assert _snapshot(out) == before


@pytest.mark.parametrize(
    "content",
    [
        "{",
        "null",
        "[]",
        "1",
        "{}",
        '{"provider":"nutrient"}',
        '{"config":{}}',
        '{"provider":[],"config":{}}',
        '{"provider":"unknown","config":{}}',
        '{"provider":"nutrient","config":[]}',
        '{"provider":"nutrient","config":{"mode":"bad"}}',
        '{"provider":"nutrient","config":{}}',
    ],
)
@pytest.mark.parametrize("command", ["run", "score"])
def test_invalid_manifest_is_a_cli_error_without_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, command: str
) -> None:
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    manifest = out / RUN_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(content)
    before = _snapshot(out)
    results = tmp_path / "results"
    args = (
        ["--provider", "nutrient", "--out", str(out)]
        if command == "run"
        else ["--predictions", str(out), "--results", str(results)]
    )
    assert main([command, "--corpus", str(corpus), *args]) == 2
    assert "invalid run manifest" in capsys.readouterr().err
    assert _snapshot(out) == before
    assert not results.exists()


@pytest.mark.parametrize("doc_id", ["RUN", "run"])
def test_run_named_document_does_not_collide_with_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, doc_id: str
) -> None:
    corpus = _corpus(tmp_path, ".pdf")
    old = corpus / "plain_continuation_demo"
    data = json.loads((old / "groundtruth.json").read_text())
    data["document"]["id"] = doc_id
    (old / "groundtruth.json").write_text(json.dumps(data))
    old.rename(corpus / doc_id)
    out = tmp_path / "out"

    def fetch(self: MarkitdownProvider, path: Path) -> RawResult:
        return RawResult(self.name, path.parent.name, "# actual prediction")

    monkeypatch.setattr(MarkitdownProvider, "fetch", fetch)
    args = ["run", "--provider", "markitdown", "--corpus", str(corpus), "--out", str(out)]
    assert main(args) == 0
    assert main(args) == 0
    assert main([*args, "--from-raw"]) == 0
    assert load_prediction(out / f"{doc_id}.json").markdown == "# actual prediction"
    results = tmp_path / "results"
    score_args = [
        "score",
        "--corpus",
        str(corpus),
        "--predictions",
        str(out),
        "--results",
        str(results),
    ]
    assert main(score_args) == 0
    assert json.loads((results / "out.json").read_text())["provider"] == "markitdown"
    (out / RUN_MANIFEST).unlink()
    assert main(score_args) == 0
    summary = json.loads((results / "out.json").read_text())
    assert summary["provider"] is None and summary["provider_config"] is None
    assert summary["parsed"] == 1


@pytest.mark.parametrize("populated", [False, True])
def test_missing_corpus_does_not_create_or_change_outputs(tmp_path: Path, populated: bool) -> None:
    out = tmp_path / "out"
    if populated:
        manifest = out / RUN_MANIFEST
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{"provider":"markitdown","config":{}}')
        (out / "VERSION.txt").write_text("original version")
        (out / "old.json").write_text('{"markdown":"original prediction"}')
    before = _snapshot(out)
    assert (
        main(
            [
                "run",
                "--provider",
                "markitdown",
                "--corpus",
                str(tmp_path / "missing"),
                "--out",
                str(out),
            ]
        )
        == 2
    )
    assert _snapshot(out) == before
    assert out.exists() == populated


@pytest.mark.parametrize("name", ["reducto", "extend"])
@pytest.mark.parametrize("key", [None, "", " \t\n", "op://vault/key"])
def test_migrated_provider_preflight_preserves_outputs(tmp_path, monkeypatch, name, key):
    variable = f"{name.upper()}_API_KEY"
    if key is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, key)
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    (out / RUN_MANIFEST).parent.mkdir(parents=True)
    (out / RUN_MANIFEST).write_text(json.dumps({"provider": name, "config": {}}))
    (out / "VERSION.txt").write_text("previous version")
    before = _snapshot(out)
    assert main(["run", "--provider", name, "--corpus", str(corpus), "--out", str(out)]) == 2
    assert _snapshot(out) == before


@pytest.mark.parametrize("name", ["reducto", "extend"])
def test_migrated_provider_cli_replays_without_credentials(tmp_path, monkeypatch, name):
    variable = f"{name.upper()}_API_KEY"
    monkeypatch.setenv(variable, "test-key")
    provider_type = PROVIDERS[name]
    calls = []

    def fetch(self, path):
        calls.append(path)
        field = "result" if name == "reducto" else "output"
        return RawResult(
            name,
            path.parent.name,
            {field: {"chunks": [{"content": "Hello", "blocks": []}]}},
            {"version": self.version(), "raw_cache_key": self.raw_cache_key()},
        )

    monkeypatch.setattr(provider_type, "fetch", fetch)
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    args = ["run", "--provider", name, "--corpus", str(corpus), "--out", str(out)]
    assert main(args) == 0
    raw = out / "raw" / "plain_continuation_demo.json"
    assert json.loads(raw.read_text())["metadata"]["config"] == provider_type().config
    monkeypatch.delenv(variable)
    assert main([*args, "--from-raw"]) == 0
    assert len(calls) == 1
    assert json.loads((out / "plain_continuation_demo.json").read_text())["markdown"] == "Hello"


@pytest.mark.parametrize("name", ["reducto", "extend"])
def test_migrated_provider_rejects_unrecognized_configuration(name):
    with pytest.raises(ValueError, match="unsupported config keys"):
        PROVIDERS[name]({"api_key": "never-persist-this"})


@pytest.mark.parametrize("name", ["extend", "reducto"])
def test_old_empty_rest_config_replays_offline(name, tmp_path, monkeypatch):
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / name
    (out / RUN_MANIFEST).parent.mkdir(parents=True)
    (out / RUN_MANIFEST).write_text(json.dumps({"provider": name, "config": {}}))
    raw_dir = out / "raw"
    raw_dir.mkdir()
    payload = (
        {"output": {"chunks": []}}
        if name == "extend"
        else {"result": {"type": "full", "chunks": []}}
    )
    saved = raw_dir / "plain_continuation_demo.json"
    saved.write_text(
        json.dumps(
            {
                "provider": name,
                "document_id": "plain_continuation_demo",
                "payload": payload,
                "metadata": {"config": {}},
            }
        )
    )
    monkeypatch.delenv(f"{name.upper()}_API_KEY", raising=False)
    args = ["run", "--provider", name, "--corpus", str(corpus), "--out", str(out), "--from-raw"]
    assert main(args) == 0
    assert json.loads((out / RUN_MANIFEST).read_text())["config"] == PROVIDERS[name]().config
    assert main(args) == 0
    config = tmp_path / "changed.json"
    config.write_text(
        json.dumps({"engine": "parse_light"} if name == "extend" else {"mode": "legacy"})
    )
    before = _snapshot(out)
    assert main([*args, "--config", str(config)]) == 2
    assert _snapshot(out) == before
