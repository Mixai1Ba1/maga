from typing import Iterable, Sequence


def precision_at_k(retrieved_doc_ids: Sequence[int], relevant_doc_ids: set[int], k: int) -> float:
    top_k = retrieved_doc_ids[:k]
    if k == 0:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / k


def recall_at_k(retrieved_doc_ids: Sequence[int], relevant_doc_ids: set[int], k: int) -> float:
    top_k = retrieved_doc_ids[:k]
    if not relevant_doc_ids:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def reciprocal_rank(retrieved_doc_ids: Sequence[int], relevant_doc_ids: set[int]) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)