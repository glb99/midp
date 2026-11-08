import pytest

from midp.similarity import vote


def test_the_most_common_type_wins_and_confidence_is_the_best_similarity() -> None:
    assert vote([("a", 0.9), ("b", 0.8), ("a", 0.7)]) == ("a", 0.9)


def test_a_tie_goes_to_the_type_of_the_closest_neighbour() -> None:
    assert vote([("b", 0.9), ("a", 0.8), ("a", 0.7), ("b", 0.6)]) == ("b", 0.9)


def test_no_neighbours_is_an_error() -> None:
    with pytest.raises(ValueError):
        vote([])
