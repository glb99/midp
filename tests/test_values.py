import pytest

from midp.values import parse_amount


def test_parse_amount_reads_labels_as_written() -> None:
    assert parse_amount("75,000") == 75000
    assert parse_amount("40,000.") == 40000
    assert parse_amount("-45") == -45


def test_parse_amount_refuses_text_instead_of_guessing() -> None:
    with pytest.raises(ValueError):
        parse_amount("two thousand")
