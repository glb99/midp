from evals.metrics import Score, score_lines


def test_lines_are_counted_by_hand() -> None:
    # Reference has A, B and C. Prediction gets A right, gets B's amount
    # wrong, misses C, and adds D.
    reference = [
        {"description": "A", "amount": 100},
        {"description": "B", "amount": 50},
        {"description": "C", "amount": 30},
    ]
    predicted = [
        {"description": "a", "amount": 100},
        {"description": "B", "amount": 40},
        {"description": "D", "amount": 10},
    ]

    score = score_lines(predicted, reference)

    assert score == Score(tp=1, fp=2, fn=2)
    assert score.precision == 1 / 3
    assert score.recall == 1 / 3
