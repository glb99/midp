from pathlib import Path

SPLITS = Path(__file__).parents[2] / "evals" / "splits"


def test_the_receipt_examples_are_not_in_the_classification_test_set() -> None:
    test_set = set((SPLITS / "cord_dev.txt").read_text(encoding="utf-8").split()[:40])
    examples = set((SPLITS / "cord_examples.txt").read_text(encoding="utf-8").split())

    assert examples
    assert not (examples & test_set)
