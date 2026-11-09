import re
from typing import TypeGuard

FORMATS: dict[str, tuple[int, int]] = {
    "MrzTd1": (3, 30),
    "MrzTd2": (2, 36),
    "MrzTd3": (2, 44),
    "MrzMrva": (2, 44),
    "MrzMrvb": (2, 36),
}

_PREFIX = "Mrz"
_LINE = re.compile(r"[A-Z0-9<]+")
_WEIGHTS = (7, 3, 1)
_DOCUMENT_NUMBER = {"MrzTd1": (0, 5, 14), "MrzTd2": (1, 0, 9)}


def is_mrz(format_: str | None) -> TypeGuard[str]:
    return format_ is not None and format_.startswith(_PREFIX)


def shape_error(format_: str, payload: str) -> str | None:
    if format_ not in FORMATS:
        return f"unknown MRZ format {format_!r}; known: {', '.join(FORMATS)}"
    count, length = FORMATS[format_]
    lines = payload.split("\n")
    if len(lines) != count or any(len(line) != length for line in lines):
        return f"a {format_} zone is {count} lines of {length} characters"
    if not all(_LINE.fullmatch(line) for line in lines):
        return "an MRZ line holds only A to Z, 0 to 9 and <"
    if format_ in _DOCUMENT_NUMBER:
        line, start, digit = _DOCUMENT_NUMBER[format_]
        if lines[line][digit] == "<" and "<" not in lines[line][start:digit]:
            return f"a {format_} document number longer than 9 characters is not defined yet"
    return None


def check_digits_valid(format_: str, payload: str) -> bool:
    lines = payload.split("\n")
    checks = _checks(format_, lines)
    return all(_digit_right(field, digit) for field, digit in checks)


def _checks(format_: str, lines: list[str]) -> list[tuple[str, str]]:
    if format_ == "MrzTd1":
        top, middle = lines[0], lines[1]
        composite = top[5:30] + middle[0:7] + middle[8:15] + middle[18:29]
        return [
            (top[5:14], top[14]),
            (middle[0:6], middle[6]),
            (middle[8:14], middle[14]),
            (composite, middle[29]),
        ]
    line = lines[1]
    checks = [(line[0:9], line[9]), (line[13:19], line[19]), (line[21:27], line[27])]
    if format_ == "MrzTd3":
        composite = line[0:10] + line[13:20] + line[21:43]
        return [*checks, (line[28:42], line[42]), (composite, line[43])]
    if format_ == "MrzTd2":
        return [*checks, (line[0:10] + line[13:20] + line[21:35], line[35])]
    return checks


def _digit_right(field: str, digit: str) -> bool:
    if digit == "<" and set(field) <= {"<"}:
        return True
    return digit == str(_check_digit(field))


def _check_digit(field: str) -> int:
    total = 0
    for index, char in enumerate(field):
        value = 0 if char == "<" else int(char) if char.isdigit() else ord(char) - ord("A") + 10
        total += value * _WEIGHTS[index % 3]
    return total % 10
