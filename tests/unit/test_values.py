from decimal import Decimal

import pytest

from midp.values import parse_amount


def test_parse_amount_reads_both_thousands_conventions() -> None:
    assert parse_amount("75,000") == 75000
    assert parse_amount("43.636") == 43636
    assert parse_amount("40,000.") == 40000
    assert parse_amount("Rp 51.000") == 51000
    assert parse_amount("-45") == -45


def test_parse_amount_keeps_decimals() -> None:
    assert parse_amount("1,346,000.50") == Decimal("1346000.50")
    assert parse_amount("99.999,99") == Decimal("99999.99")


def test_parse_amount_refuses_text_instead_of_guessing() -> None:
    with pytest.raises(ValueError):
        parse_amount("two thousand")
