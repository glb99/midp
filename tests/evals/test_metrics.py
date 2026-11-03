from evals.metrics import Score, score_lines


def test_a_longer_name_with_the_same_amount_matches() -> None:
    # The model adds cooking options the label leaves out.
    reference = [{"description": "PEPPER AUS", "amount": 165000}]
    predicted = [{"description": "PEPPER AUS WELL DONE", "amount": 165000}]

    assert score_lines(predicted, reference) == Score(tp=1, fp=0, fn=0)


def test_a_single_word_name_does_not_match_a_longer_one() -> None:
    reference = [{"description": "Tahu", "amount": 10000}]
    predicted = [{"description": "Tahu Goreng", "amount": 10000}]

    assert score_lines(predicted, reference) == Score(tp=0, fp=1, fn=1)


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
