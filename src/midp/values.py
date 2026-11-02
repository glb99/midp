"""Parse the textual values that documents contain into typed values.

Receipts write amounts as "75,000" or "40,000." and quantities as "1 x".
These functions turn them into numbers so that predictions and references
can be compared as numbers, not as strings.
"""

from __future__ import annotations

import re

_AMOUNT = re.compile(r"^-?\d[\d,]*\.?$")
_QUANTITY = re.compile(r"^(\d+)\s*x?$")


def parse_amount(text: str | None) -> int | None:
    """Return the integer amount, or None when the field is empty.

    Thousands separators are commas. A trailing period, which appears in
    some labels, is ignored. Currency is not represented; the document's
    currency is an assumption of the configuration, not of this function.
    """
    if text is None:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if not _AMOUNT.match(cleaned):
        raise ValueError(f"not an amount: {text!r}")
    return int(cleaned.rstrip(".").replace(",", ""))


def parse_quantity(text: str | None) -> int | None:
    """Return the count from a quantity such as "1 x" or "3 x"."""
    if text is None:
        return None
    match = _QUANTITY.match(text.strip())
    if not match:
        raise ValueError(f"not a quantity: {text!r}")
    return int(match.group(1))
