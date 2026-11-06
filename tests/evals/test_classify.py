from evals.classify import class_scores, confusion, review_rate
from evals.metrics import Score

CLASSES = ["receipt", "invoice", "delivery_note"]


def test_a_wrong_type_is_a_false_positive_of_one_and_a_false_negative_of_the_other() -> None:
    # two delivery notes: one right, one read as an invoice
    pairs = [("delivery_note", "delivery_note"), ("delivery_note", "invoice")]

    scores = class_scores(pairs, CLASSES)

    assert scores["delivery_note"] == Score(tp=1, fp=0, fn=1)
    assert scores["invoice"] == Score(tp=0, fp=1, fn=0)


def test_no_type_is_a_false_negative_of_the_true_type_only() -> None:
    pairs = [("invoice", None)]

    scores = class_scores(pairs, CLASSES)

    assert scores["invoice"] == Score(tp=0, fp=0, fn=1)
    assert scores["receipt"] == Score(tp=0, fp=0, fn=0)
    assert review_rate(pairs) == 1.0


def test_the_confusion_matrix_counts_each_pair_once() -> None:
    pairs = [("invoice", "invoice"), ("invoice", "delivery_note"), ("invoice", None)]

    matrix = confusion(pairs, CLASSES)

    assert matrix[("invoice", "invoice")] == 1
    assert matrix[("invoice", "delivery_note")] == 1
    assert matrix[("invoice", "none")] == 1
