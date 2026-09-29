"""Immutable transaction operation descriptions.

A transaction is an ordered list of these operations:

* ``AddBatch(batch_id, triples)``     - register a new batch
* ``ReplaceBatch(batch_id, triples)`` - replace a batch wholesale
* ``RemoveBatch(batch_id)``           - revoke a batch

Each triple is ``(row, column, value)`` with zero-based integer indices.
Duplicate coordinates, inside one batch or across batches, accumulate.
"""

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple, Union

Triple = Tuple[int, int, float]


@dataclass(frozen=True)
class AddBatch:
    batch_id: Any
    triples: Tuple[Triple, ...]


@dataclass(frozen=True)
class ReplaceBatch:
    batch_id: Any
    triples: Tuple[Triple, ...]


@dataclass(frozen=True)
class RemoveBatch:
    batch_id: Any


Operation = Union[AddBatch, ReplaceBatch, RemoveBatch]


def normalize_triples(triples: Iterable[Sequence[Any]]) -> Tuple[Triple, ...]:
    """Convert any iterable of ``(row, col, value)`` into a frozen tuple."""
    normalized: List[Triple] = []
    for index, item in enumerate(triples):
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError(
                f"triple #{index} must be a (row, column, value) sequence"
            )
        row, col, value = item
        normalized.append((row, col, float(value)))
    return tuple(normalized)


def add(batch_id: Any, triples: Iterable[Sequence[Any]]) -> AddBatch:
    """Convenience constructor for an add operation."""
    return AddBatch(batch_id, normalize_triples(triples))


def replace(batch_id: Any, triples: Iterable[Sequence[Any]]) -> ReplaceBatch:
    """Convenience constructor for a replace operation."""
    return ReplaceBatch(batch_id, normalize_triples(triples))


def remove(batch_id: Any) -> RemoveBatch:
    """Convenience constructor for a remove operation."""
    return RemoveBatch(batch_id)
