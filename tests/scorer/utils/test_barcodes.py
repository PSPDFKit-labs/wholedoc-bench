from tests.builders import URL, code, invoice, para, truth
from wholedoc_bench.scorer.utils.barcodes import without_payloads


def test_without_payloads_cuts_the_correct_copies_and_drops_an_empty_text() -> None:
    texts = ["Scan to pay.", URL, f"Pay at {URL}", "https://example.com/inv/44170"]
    assert without_payloads(invoice(code("q1"), code("q2")), texts) == [
        "Scan to pay.",
        "Pay at",
        "https://example.com/inv/44170",
    ]


def test_without_payloads_keeps_texts_without_a_payload_as_they_are() -> None:
    texts = ["Scan  to\npay.", "", "  "]
    assert without_payloads(invoice(), texts) == texts
    assert without_payloads(truth(para("p", "Scan to pay.")), texts) == texts


def test_without_payloads_cuts_one_copy_per_code() -> None:
    assert without_payloads(invoice(), [URL, URL]) == [URL]


def test_without_payloads_cuts_a_longer_payload_first() -> None:
    t = invoice(code("q1", payload="INV-4417"), code("q2", payload="INV-4417-A"))
    assert without_payloads(t, ["INV-4417-A", "INV-4417"]) == []


def test_without_payloads_finds_a_payload_over_a_line_break() -> None:
    t = invoice(code("q", payload="Jane Roe ACME Corp"))
    assert without_payloads(t, ["Jane Roe\nACME Corp"]) == []
