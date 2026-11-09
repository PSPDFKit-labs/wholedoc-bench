import os


class CredentialError(ValueError): ...


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise CredentialError(
            f"{name} is not set or is blank; set it in the environment before a live run. "
            "Do not put credentials in --config."
        )
    return value
