import pytest

from tests.builders import ICAO_TD3, LONG_NUMBER_TD1
from wholedoc_bench.scorer.utils.mrz import check_digits_valid, is_mrz, shape_error

ICAO_TD1 = "\n".join(
    [
        "I<UTOD231458907<<<<<<<<<<<<<<<",
        "7408122F1204159UTO<<<<<<<<<<<6",
        "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
    ]
)
ICAO_TD2 = "I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<\nD231458907UTO7408122F1204159<<<<<<<6"
EMPTY_PERSONAL = "\n".join(
    ["P<NLDDE<BRUIJN<<WILLEKE<LISELOTTE<<<<<<<<<<<", "SPECI20212NLD6503101F3108309<<<<<<<<<<<<<<<0"]
)

LONG_NUMBER_TD2 = "I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<\nD23145890<UTO7408122F1204159120<<<<6"


@pytest.mark.parametrize(
    ("format_", "payload"),
    [("MrzTd1", ICAO_TD1), ("MrzTd2", ICAO_TD2), ("MrzTd3", ICAO_TD3)],
)
def test_the_icao_specimens_pass_every_check_digit(format_: str, payload: str) -> None:
    assert shape_error(format_, payload) is None
    assert check_digits_valid(format_, payload)


def test_one_wrong_check_digit_fails_the_zone() -> None:
    assert not check_digits_valid("MrzTd3", ICAO_TD3[:-1] + "1")


def test_an_empty_field_may_have_zero_or_filler_as_its_check_digit() -> None:
    assert check_digits_valid("MrzTd3", EMPTY_PERSONAL)
    assert check_digits_valid("MrzTd3", EMPTY_PERSONAL[:-2] + "<0")


@pytest.mark.parametrize(
    ("format_", "payload", "error"),
    [
        ("MrzTd9", ICAO_TD3, "unknown MRZ format"),
        ("MrzTd3", ICAO_TD3.split("\n")[0], "2 lines of 44"),
        ("MrzTd3", ICAO_TD3.replace("ANNA", "ANNa"), "only A to Z"),
        ("MrzTd1", LONG_NUMBER_TD1, "longer than 9 characters"),
        ("MrzTd2", LONG_NUMBER_TD2, "longer than 9 characters"),
    ],
)
def test_a_payload_that_is_no_zone_of_its_format_is_named(
    format_: str, payload: str, error: str
) -> None:
    assert error in (shape_error(format_, payload) or "")


def test_a_short_document_number_with_a_filler_check_digit_is_a_zone_that_fails() -> None:
    payload = ICAO_TD1.replace("D231458907", "D2314589<<")
    assert shape_error("MrzTd1", payload) is None
    assert not check_digits_valid("MrzTd1", payload)


def test_any_mrz_format_is_an_mrz() -> None:
    assert is_mrz("MrzTd1") and is_mrz("MrzMrvb")
    assert not is_mrz("QrCode") and not is_mrz(None)
