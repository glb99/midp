from midp.review import UNKNOWN_TYPE, review_document


def test_a_document_with_no_known_type_needs_review() -> None:
    needs, problems = review_document({"document_type": None}, {})

    assert needs is True
    assert problems == [UNKNOWN_TYPE]


def test_a_failed_field_check_needs_review() -> None:
    result = {"document_type": "invoice", "tax": None, "total": 144}

    needs, problems = review_document(result, {"required": ["tax"]})

    assert needs is True
    assert "missing tax" in problems
