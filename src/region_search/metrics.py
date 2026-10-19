"""Ranking metrics: how well a ranked list of sections finds the sections that hold a region."""

from collections.abc import Collection, Hashable, Sequence


def precision_at_k(ranked_slices: Sequence[Hashable], labels: Collection[Hashable], k: int) -> float:
    """Fraction of the top-k ranked sections that are in `labels` (the relevant sections)."""
    if not 1 <= k <= len(ranked_slices):
        raise ValueError(f"k must be 1..{len(ranked_slices)}, got {k}")
    relevant = set(labels)
    return sum(s in relevant for s in ranked_slices[:k]) / k


def average_precision(ranked_slices: Sequence[Hashable], labels: Collection[Hashable]) -> float:
    """Mean of precision@k over the ranks k of the relevant sections.

    Divides by all of `labels`, so a relevant section the ranking never returns counts as
    a miss. A random ranking scores about len(labels) / len(ranked_slices).
    """
    relevant = set(labels)
    if not relevant:
        raise ValueError("labels must hold at least one relevant section")
    hits, total = 0, 0.0
    for rank, section in enumerate(ranked_slices, 1):
        if section in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)
