import json
from pathlib import Path
from typing import Any

import pytest

from wholedoc_bench.corpus.loader import load_groundtruth
from wholedoc_bench.providers import RawResult
from wholedoc_bench.schema import Groundtruth

FIXTURES = Path(__file__).parent / "fixtures"
DEMO = FIXTURES / "plain_continuation_demo"
CHART_PAGE = FIXTURES / "innovation_scoreboard_chart"
TWO_COLUMN_TEXTRACT = FIXTURES / "report_two_column_textract" / "textract_raw.json"
TWO_COLUMN_AZURE = FIXTURES / "report_two_column_azure" / "azure_raw.json"
CHARTS_AZURE = FIXTURES / "charts_report_azure" / "azure_raw.json"
TWO_COLUMN_ANTHROPIC = FIXTURES / "report_two_column_anthropic" / "anthropic_raw.json"
CHARTS_ANTHROPIC = FIXTURES / "charts_report_anthropic" / "anthropic_raw.json"
TWO_COLUMN_OPENAI = FIXTURES / "report_two_column_openai" / "openai_raw.json"
CHARTS_OPENAI = FIXTURES / "charts_report_openai" / "openai_raw.json"
TWO_COLUMN_DATABRICKS = FIXTURES / "report_two_column_databricks" / "databricks_raw.json"
CHARTS_DATABRICKS = FIXTURES / "charts_report_databricks" / "databricks_raw.json"
DEPOT_AZURE = FIXTURES / "depot_notice_azure" / "azure_raw.json"
_AWS_VARIABLES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_SECURITY_TOKEN",
    "AWS_CREDENTIAL_EXPIRATION",
    "AWS_CREDENTIAL_FILE",
)


@pytest.fixture
def isolated_aws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in _AWS_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "aws-credentials"))
    monkeypatch.setenv("BOTO_CONFIG", str(tmp_path / "boto-config"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    return monkeypatch


class StubbedSts:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.stubbers: list[Any] = []

    def assert_checked(self) -> None:
        assert self.stubbers
        for stubber in self.stubbers:
            stubber.assert_no_pending_responses()


@pytest.fixture
def sts(isolated_aws: pytest.MonkeyPatch) -> StubbedSts:
    from boto3.session import Session
    from botocore.stub import Stubber

    stubbed = StubbedSts()
    real_client = Session.client

    def client(self: Session, service_name: str, *args: Any, **kwargs: Any) -> Any:
        made = real_client(self, service_name, *args, **kwargs)  # pyright: ignore[reportArgumentType]
        if service_name == "sts":
            stubber = Stubber(made)
            if stubbed.errors:
                code = stubbed.errors.pop(0)
                stubber.add_client_error("get_caller_identity", code, http_status_code=403)
            else:
                stubber.add_response("get_caller_identity", _IDENTITY)
            stubber.activate()
            stubbed.stubbers.append(stubber)
        return made

    isolated_aws.setattr(Session, "client", client)
    return stubbed


_IDENTITY = {
    "UserId": "AIDAEXAMPLE",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/test",
}


@pytest.fixture
def two_column_textract_raw() -> RawResult:
    payload = json.loads(TWO_COLUMN_TEXTRACT.read_text(encoding="utf-8"))
    return RawResult(provider="textract", document_id="report_two_column_pdf", payload=payload)


@pytest.fixture
def two_column_azure_raw() -> RawResult:
    payload = json.loads(TWO_COLUMN_AZURE.read_text(encoding="utf-8"))
    return RawResult(provider="azure-di", document_id="report_two_column_pdf", payload=payload)


@pytest.fixture
def charts_azure_raw() -> RawResult:
    payload = json.loads(CHARTS_AZURE.read_text(encoding="utf-8"))
    return RawResult(provider="azure-di", document_id="charts_report_pdf", payload=payload)


@pytest.fixture
def two_column_databricks_raw() -> RawResult:
    payload = json.loads(TWO_COLUMN_DATABRICKS.read_text(encoding="utf-8"))
    return RawResult(provider="databricks", document_id="report_two_column_pdf", payload=payload)


@pytest.fixture
def charts_databricks_raw() -> RawResult:
    payload = json.loads(CHARTS_DATABRICKS.read_text(encoding="utf-8"))
    return RawResult(provider="databricks", document_id="charts_report_pdf", payload=payload)


@pytest.fixture
def depot_azure_raw() -> RawResult:
    payload = json.loads(DEPOT_AZURE.read_text(encoding="utf-8"))
    return RawResult(provider="azure-di", document_id="depot_notice", payload=payload)


@pytest.fixture
def demo_groundtruth() -> Groundtruth:
    return load_groundtruth(DEMO / "groundtruth.json")


@pytest.fixture
def demo_nutrient_raw() -> RawResult:
    payload = json.loads((DEMO / "nutrient_raw.json").read_text(encoding="utf-8"))
    return RawResult(provider="nutrient", document_id="plain_continuation_demo", payload=payload)


@pytest.fixture
def chart_nutrient_raw() -> RawResult:
    payload = json.loads((CHART_PAGE / "nutrient_raw.json").read_text(encoding="utf-8"))
    return RawResult(provider="nutrient", document_id=CHART_PAGE.name, payload=payload)
