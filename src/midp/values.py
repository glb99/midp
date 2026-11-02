"""Parse the textual values that documents contain into typed values.

Receipts write amounts in different conventions: "75,000" and "75.000"
both mean seventy-five thousand, and some labels add decimals ("99,999.99"
or "99.999,99") or a currency prefix ("Rp 51.000"). Quantities are
written as "1 x".
"""

from __future__ import annotations

import re
from decimal import Decimal

_NUMBER = re.compile(r"^-?\d+([.,]\d+)*$")
_QUANTITY = re.compile(r"^\*?\s*x?\s*(\d+(?:[.,]\d+)?)\s*(?:x|pcs)?\.?$", re.IGNORECASE)


def parse_amount(text: str | None) -> Decimal | None:
    """Return the amount as a Decimal, or None when the field is empty.

    The last separator is a decimal point only when it is followed by one
    or two digits and another separator comes before it; otherwise every
    separator is a thousands separator and must group digits in threes.
    Assumed, not checked: the currency is whatever the configuration says.
    """
    if text is None:
        return None
    cleaned = re.sub(r"^Rp\.?", "", text.strip()).replace(" ", "").rstrip(".")
    if not cleaned or cleaned == "-":  # a lone dash marks an empty field
        return None
    if not _NUMBER.match(cleaned):
        raise ValueError(f"not an amount: {text!r}")

    negative = cleaned.startswith("-")
    digits = cleaned.lstrip("-")
    separators = re.findall(r"[.,]", digits)
    last = max(digits.rfind("."), digits.rfind(","))
    tail = digits[last + 1 :] if last >= 0 else ""

    if separators and len(set(separators)) == 2 and len(tail) in (1, 2) or separators and len(separators) == 1 and len(tail) in (1, 2) and tail != "000":
        integer, decimal = digits[:last], digits[last + 1 :]
    else:
        integer, decimal = digits, ""

    integer_digits = re.sub(r"[.,]", "", integer)
    groups = re.split(r"[.,]", integer)
    if len(groups) > 1 and any(len(group) != 3 for group in groups[1:]):
        raise ValueError(f"thousands groups must have three digits: {text!r}")

    value = Decimal(f"{integer_digits}.{decimal}" if decimal else integer_digits)
    return -value if negative else value


def parse_quantity(text: str | None) -> int | None:
    """Return the count from a quantity such as "1 x", "*2" or "3 Pcs".

    Returns None when the value is not a whole count: a comma or a
    fraction (a weight such as "0.264") or a thousands-style number
    ("1,000"), which points to a misplaced amount.
    """
    if text is None:
        return None
    match = _QUANTITY.match(text.strip())
    if not match:
        raise ValueError(f"not a quantity: {text!r}")
    number = match.group(1)
    if "," in number:
        return None
    whole, _, decimal = number.partition(".")
    if decimal and (len(decimal) > 2 or decimal.strip("0")):
        return None
    return int(whole)
