"""One decision for every method: does this document need a human to look at it?

The field checks (checks.py) are the input. Any classifier's result goes
through here, so a reviewer always sees the same question with its reasons.
"""

from __future__ import annotations

from typing import Any

from midp.checks import check_result

UNKNOWN_TYPE = "the type is unknown or not certain"


def review_document(result: dict[str, Any], checks: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (needs_review, problems). Problems say why, in plain words."""
    problems: list[str] = []
    if result.get("document_type") is None:
        problems.append(UNKNOWN_TYPE)
    else:
        problems.extend(check_result(result, checks))
    return bool(problems), problems
