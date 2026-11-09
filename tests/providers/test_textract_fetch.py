import io
import json
import shutil
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.exceptions import TokenRetrievalError
from botocore.stub import Stubber
from pypdf import PdfReader, PdfWriter

from tests.conftest import DEMO, FIXTURES, TWO_COLUMN_TEXTRACT, StubbedSts
from wholedoc_bench.providers import CredentialError, TextractProvider
from wholedoc_bench.providers.textract import MAX_ATTEMPTS

PAGES = json.loads(TWO_COLUMN_TEXTRACT.read_text(encoding="utf-8"))["pages"]
KEY_ID = "AKIAIOSFODNN7EXAMPLE"
SECRET = "test-secret-do-not-persist-42"


class FakeTextract:
    def __init__(self, *responses: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses = list(responses)

    def analyze_document(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        answer = self.responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "plain_continuation_demo" / "document.pdf"
    path.parent.mkdir()
    shutil.copy(DEMO / "document.pdf", path)
    return path


@pytest.fixture
def aws_keys(isolated_aws: pytest.MonkeyPatch, sts: StubbedSts) -> pytest.MonkeyPatch:
    isolated_aws.setenv("AWS_ACCESS_KEY_ID", KEY_ID)
    isolated_aws.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    isolated_aws.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    return isolated_aws


def test_fetch_sends_each_pdf_page_with_layout_and_tables(pdf: Path) -> None:
    answers = [{**page, "ResponseMetadata": {"RequestId": "r"}} for page in PAGES]
    client = FakeTextract(*answers)
    raw = TextractProvider(client=client).fetch(pdf)

    assert len(client.calls) == 3
    for call in client.calls:
        assert call["FeatureTypes"] == ["LAYOUT", "TABLES"]
        assert len(PdfReader(io.BytesIO(call["Document"]["Bytes"])).pages) == 1
    assert (raw.provider, raw.document_id) == ("textract", "plain_continuation_demo")
    assert raw.payload == {"pages": PAGES}
    assert raw.metadata == {
        "version": "textract AnalyzeDocument LAYOUT+TABLES, model 1.0",
        "config": {},
    }


def test_an_encrypted_pdf_is_split_with_its_empty_user_password(tmp_path: Path) -> None:
    writer = PdfWriter(clone_from=DEMO / "document.pdf")
    writer.encrypt(user_password="", owner_password="owner only", algorithm="AES-256")
    path = tmp_path / "locked" / "document.pdf"
    path.parent.mkdir()
    with path.open("wb") as file:
        writer.write(file)
    client = FakeTextract(*PAGES)
    TextractProvider(client=client).fetch(path)
    pages = [PdfReader(io.BytesIO(c["Document"]["Bytes"])) for c in client.calls]
    assert [(len(p.pages), p.is_encrypted) for p in pages] == [(1, False)] * 3
    assert "Plain page continuation" in pages[0].pages[0].extract_text()


def test_layers_hidden_by_default_stay_hidden_on_each_page(tmp_path: Path) -> None:
    path = tmp_path / "layers" / "document.pdf"
    path.parent.mkdir()
    shutil.copy(FIXTURES / "optional_content" / "document.pdf", path)
    client = FakeTextract(PAGES[0], PAGES[1])
    TextractProvider(client=client).fetch(path)
    for call in client.calls:
        page: Any = PdfReader(io.BytesIO(call["Document"]["Bytes"]))
        used = {ref.idnum for ref in page.pages[0]["/Resources"]["/Properties"].values()}
        off = {ref.idnum for ref in page.root_object["/OCProperties"]["/D"]["/OFF"]}
        assert used and used <= off


def test_an_image_is_sent_as_it_is(tmp_path: Path) -> None:
    image = tmp_path / "receipt" / "document.png"
    image.parent.mkdir()
    image.write_bytes(b"\x89PNG\r\n\x1a\nimage bytes")
    client = FakeTextract(PAGES[0])
    raw = TextractProvider(client=client).fetch(image)
    assert [c["Document"]["Bytes"] for c in client.calls] == [image.read_bytes()]
    assert raw.payload == {"pages": [PAGES[0]]}


def test_the_provider_reads_pdf_and_images_and_takes_no_options() -> None:
    assert TextractProvider.formats == {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    assert TextractProvider().config == {}
    with pytest.raises(ValueError, match="unsupported config keys: aws_secret_access_key"):
        TextractProvider({"aws_secret_access_key": SECRET})


def test_prepare_builds_a_client_from_the_sdk_credential_chain(
    aws_keys: Any, sts: StubbedSts
) -> None:
    provider = TextractProvider()
    provider.prepare()
    sts.assert_checked()
    client = provider._textract()
    assert client.meta.region_name == "eu-central-1"
    assert client.meta.config.retries == {"mode": "standard", "total_max_attempts": MAX_ATTEMPTS}


def test_prepare_without_credentials_fails_before_any_call(isolated_aws: Any) -> None:
    isolated_aws.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    with pytest.raises(CredentialError, match="No AWS credentials found") as error:
        TextractProvider().prepare()
    assert "Do not put credentials in --config" in str(error.value)


def test_an_expired_session_fails_before_any_call(
    aws_keys: pytest.MonkeyPatch,
) -> None:
    class Expired:
        def get_frozen_credentials(self) -> None:
            raise TokenRetrievalError(provider="sso", error_msg=f"token for {SECRET} expired")

    aws_keys.setattr(boto3.Session, "get_credentials", lambda self: Expired())
    with pytest.raises(CredentialError, match="TokenRetrievalError") as error:
        TextractProvider().prepare()
    assert SECRET not in str(error.value)


@pytest.mark.parametrize("code", ["InvalidClientTokenId", "ExpiredToken"])
def test_keys_aws_rejects_fail_before_any_call(
    aws_keys: pytest.MonkeyPatch, sts: StubbedSts, code: str
) -> None:
    sts.errors.append(code)
    with pytest.raises(
        CredentialError, match=f"did not accept the credentials.*\\({code}\\)"
    ) as error:
        TextractProvider().prepare()
    assert KEY_ID not in str(error.value) and SECRET not in str(error.value)


def test_prepare_without_a_region_fails(aws_keys: pytest.MonkeyPatch) -> None:
    aws_keys.delenv("AWS_DEFAULT_REGION")
    with pytest.raises(CredentialError, match="No AWS region"):
        TextractProvider().prepare()


def test_partial_credentials_fail_without_echoing_them(aws_keys: pytest.MonkeyPatch) -> None:
    aws_keys.delenv("AWS_SECRET_ACCESS_KEY")
    with pytest.raises(CredentialError, match="PartialCredentialsError") as error:
        TextractProvider().prepare()
    assert KEY_ID not in str(error.value)


def test_a_service_error_reports_its_code_without_the_message(aws_keys: Any, pdf: Path) -> None:
    provider = TextractProvider()
    provider.prepare()
    client = provider._textract()
    with Stubber(client) as stubber:
        stubber.add_client_error(
            "analyze_document",
            service_error_code="AccessDeniedException",
            service_message=f"denied for {KEY_ID} with {SECRET}",
            http_status_code=403,
        )
        with pytest.raises(RuntimeError, match=r"failed \(AccessDeniedException\)") as error:
            provider.fetch(pdf)
    assert KEY_ID not in str(error.value) and SECRET not in str(error.value)
    assert error.value.__cause__ is None and error.value.__suppress_context__


def test_another_failure_reports_only_its_type(pdf: Path) -> None:
    client = FakeTextract(ConnectionError(f"proxy echoed {SECRET}"))
    with pytest.raises(RuntimeError, match=r"failed \(ConnectionError\)") as error:
        TextractProvider(client=client).fetch(pdf)
    assert SECRET not in str(error.value)
