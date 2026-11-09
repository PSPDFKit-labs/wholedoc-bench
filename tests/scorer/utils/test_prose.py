import pytest

from wholedoc_bench.scorer.utils.prose import normalize, sentence_language, sentences, words


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Dr. Smith approved the plan. It passed.", ["Dr. Smith approved the plan.", "It passed."]),
        (
            "The office is in the U.S. The team grew. The answer was no. We left.",
            ["The office is in the U.S.", "The team grew.", "The answer was no.", "We left."],
        ),
        (
            "Some fruit, e.g. apples, is sold. Others, i.e. pears, are not.",
            ["Some fruit, e.g. apples, is sold.", "Others, i.e. pears, are not."],
        ),
        (
            "J. R. R. Tolkien wrote it. See Fig. 3 for details.",
            ["J. R. R. Tolkien wrote it.", "See Fig. 3 for details."],
        ),
        (
            "The total was $1,234.56 on 2024-01-05. It rose 3.5% after No. 5 shipped.",
            ["The total was $1,234.56 on 2024-01-05.", "It rose 3.5% after No. 5 shipped."],
        ),
        (
            "We sell pens, paper, etc. The shop opens at 9 a.m. every day.",
            ["We sell pens, paper, etc.", "The shop opens at 9 a.m. every day."],
        ),
        ('He said "stop." Then it ended! Why?', ['He said "stop."', "Then it ended!", "Why?"]),
    ],
)
def test_english_sentences(text: str, expected: list[str]) -> None:
    assert sentences(text) == expected


def test_german_sentences() -> None:
    assert sentences("Siehe z.B. Nr. 4 im Anhang. Das ist gut.", "de") == [
        "Siehe z.B. Nr. 4 im Anhang.",
        "Das ist gut.",
    ]


def test_sentences_of_empty_text() -> None:
    assert sentences("   ") == []


def test_sentences_see_one_line_for_wrapped_text() -> None:
    assert sentences("The budget was\napproved in March.") == ["The budget was approved in March."]


def test_sentence_language_takes_the_first_supported() -> None:
    assert sentence_language(["xx", "de", "en"]) == "de"
    assert sentence_language(["ro", "en"]) == "ro"
    assert sentence_language(["xx"]) == "en"
    assert sentence_language([]) == "en"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$1,234.56", ["1,234.56"]),
        ("2024-01-05 or 05/01/2024", ["2024-01-05", "or", "05/01/2024"]),
        ("at 10:30, version 1.2.3", ["at", "10:30", "version", "1.2.3"]),
        ("COVID-19 and A4", ["COVID", "19", "and", "A4"]),
        ("item 1. First", ["item", "1", "First"]),
    ],
)
def test_numbers_with_punctuation_are_one_word(text: str, expected: list[str]) -> None:
    assert words(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("हिन्दी भाषा", ["हिन्दी", "भाषा"]),
        ("مُحَمَّد", ["مُحَمَّد"]),
        ("ẹ́kọ́", ["ẹ́kọ́"]),
    ],
)
def test_combining_marks_stay_in_their_word(text: str, expected: list[str]) -> None:
    assert words(text) == expected


def test_nfkc_folds_look_alikes_and_keeps_diacritics() -> None:
    assert normalize("m² ﬁle \uff21") == "m2 file A"
    assert normalize("Müller") != normalize("Muller")
    assert normalize("Mu\u0308ller") == normalize("Müller")


def test_normalize_unifies_quotes_dashes_and_spaces() -> None:
    assert normalize("\u201ca\u201d\u00a0\u2013  b\u00ad") == '"a" - b'


def test_romanian_abbreviations_do_not_end_a_sentence() -> None:
    text = "Dl. Popescu locuiește pe str. Mihai Eminescu nr. 5. A plecat. Dna. Ionescu a venit."
    assert sentences(text, "ro") == [
        "Dl. Popescu locuiește pe str. Mihai Eminescu nr. 5.",
        "A plecat.",
        "Dna. Ionescu a venit.",
    ]


def test_romanian_cedilla_and_comma_below_are_the_same_letters() -> None:
    assert words("şcoală ţară ŞTIRI ŢARĂ") == words("școală țară ȘTIRI ȚARĂ")


def test_romanian_letters_stay_whole_and_count() -> None:
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "Învățământul în țară și știința")
    assert words(decomposed) == ["Învățământul", "în", "țară", "și", "știința"]
    assert words("mână") != words("mana")
