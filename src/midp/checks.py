"""Checks that a document's extracted fields must pass for its type.

The rules come from the type's config (`checks:`), so each type states what
it requires. A document that fails a check is marked for review: its fields
are still returned, but nothing presents them as validated.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

TOLERANCE = Decimal(1)


def _number(value: Any) -> Decimal:
    return Decimal(str(value))


def check_result(result: dict[str, Any], checks: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for name in checks.get("required", []):
        if result.get(name) in (None, "", []):
            problems.append(f"missing {name}")

    lines = result.get("lines") or []
    if checks.get("lines_not_empty") and not lines:
        problems.append("no lines")

    subtotal, tax, total = result.get("subtotal"), result.get("tax"), result.get("total")
    if (
        checks.get("total_is_subtotal_plus_tax")
        and None not in (subtotal, tax, total)
        and abs(_number(subtotal) + _number(tax) - _number(total)) > TOLERANCE
    ):
        problems.append("subtotal + tax does not equal total")

    if checks.get("lines_sum_to_subtotal") and subtotal is not None and lines:
        amounts = [_number(line["amount"]) for line in lines if line.get("amount") is not None]
        if abs(sum(amounts, Decimal(0)) - _number(subtotal)) > TOLERANCE:
            problems.append("lines do not add up to subtotal")
    return problems
