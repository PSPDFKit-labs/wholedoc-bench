import re

_INLINE_SYNTAX = re.compile(r"([\\`*_\[\]<>#~|&])")
_BULLET_START = re.compile(r"^([-+=])")
_NUMBER_START = re.compile(r"^(\d+)([.)])")


def escape_markdown(text: str) -> str:
    text = _INLINE_SYNTAX.sub(r"\\\1", text)
    return _NUMBER_START.sub(r"\1\\\2", _BULLET_START.sub(r"\\\1", text))
