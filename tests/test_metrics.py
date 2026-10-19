import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from region_search.metrics import average_precision, precision_at_k  # noqa: E402


def test_precision_at_k_counts_hits_in_the_top_k():
    ranked = [10, 20, 30, 40, 50]
    assert precision_at_k(ranked, {20, 40, 99}, 1) == 0.0
    assert precision_at_k(ranked, {20, 40, 99}, 2) == 0.5
    assert precision_at_k(ranked, {20, 40, 99}, 4) == 0.5


def test_precision_at_k_rejects_k_outside_the_ranking():
    with pytest.raises(ValueError):
        precision_at_k([1, 2], {1}, 0)
    with pytest.raises(ValueError):
        precision_at_k([1, 2], {1}, 3)


def test_average_precision_perfect_and_worst_ranking():
    assert average_precision([1, 2, 3, 4], {1, 2}) == 1.0
    assert average_precision([3, 4, 1, 2], {1, 2}) == pytest.approx((1 / 3 + 2 / 4) / 2)


def test_average_precision_textbook_example():
    # hits at ranks 1, 3, 6 -> (1/1 + 2/3 + 3/6) / 3
    ranked = ["a", "x", "b", "y", "z", "c"]
    assert average_precision(ranked, {"a", "b", "c"}) == pytest.approx((1 + 2 / 3 + 3 / 6) / 3)


def test_average_precision_counts_relevant_sections_missing_from_the_ranking_as_misses():
    assert average_precision([1, 2], {1, 99}) == 0.5


def test_average_precision_needs_at_least_one_relevant_section():
    with pytest.raises(ValueError):
        average_precision([1, 2], set())
