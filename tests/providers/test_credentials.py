import pytest

from wholedoc_bench.providers import CredentialError, require_env


@pytest.mark.parametrize("value", [None, "", " \t\n"])
def test_require_env_rejects_missing_or_blank_values(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("EXAMPLE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("EXAMPLE_API_KEY", value)
    with pytest.raises(CredentialError, match="EXAMPLE_API_KEY is not set or is blank") as error:
        require_env("EXAMPLE_API_KEY")
    assert isinstance(error.value, ValueError)
    assert "Do not put credentials in --config" in str(error.value)


def test_require_env_preserves_the_supplied_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE_API_KEY", " example-secret ")
    assert require_env("EXAMPLE_API_KEY") == " example-secret "
