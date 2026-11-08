from midp.repository import apply_corrections


def test_a_later_correction_wins_and_only_the_fields_it_names_change() -> None:
    result = {"total": 100, "tax": 21, "lines": [{"description": "A"}]}
    corrections = [{"total": 110}, {"total": 120, "tax": 24}]

    final = apply_corrections(result, corrections)

    assert final == {"total": 120, "tax": 24, "lines": [{"description": "A"}]}


def test_applying_no_corrections_gives_the_answer_unchanged() -> None:
    result = {"total": 100}

    assert apply_corrections(result, []) == {"total": 100}


def test_the_stored_answer_is_not_modified() -> None:
    result = {"total": 100}

    apply_corrections(result, [{"total": 110}])

    assert result == {"total": 100}
