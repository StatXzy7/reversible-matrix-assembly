"""Sparse matrix assembly engine with batch-scoped, transactional updates."""

import math
import threading
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .errors import (
    BatchExistsError,
    BatchNotFoundError,
    CoordinateOutOfBoundsError,
    DimensionError,
    InvalidOperationError,
    NonFiniteValueError,
)
from .operations import AddBatch, Operation, RemoveBatch, ReplaceBatch
from .snapshot import Coordinate, Snapshot

Key = Tuple[int, int]


def _is_real_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class SparseAssembler:
    """Accumulates sparse matrix contributions from named batches.

    Dimensions are fixed at construction. Every public mutation goes through
    :meth:`submit`, which applies an ordered list of operations atomically:
    either every operation succeeds and the version advances once, or the
    whole transaction is rejected and observable state is untouched.
    """

    def __init__(self, rows: int, cols: int, threshold: float = 0.0) -> None:
        if not _is_real_int(rows) or rows <= 0:
            raise DimensionError("rows must be a positive integer")
        if not _is_real_int(cols) or cols <= 0:
            raise DimensionError("cols must be a positive integer")
        if not _is_real_number(threshold) or not math.isfinite(threshold) or threshold < 0:
            raise DimensionError("threshold must be a finite, non-negative number")

        self._rows = rows
        self._cols = cols
        self._threshold = float(threshold)
        self._version = 0
        self._batches: Dict[Any, Tuple[Tuple[int, int, float], ...]] = {}
        self._entries: Dict[Key, float] = {}
        self._lock = threading.RLock()

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def version(self) -> int:
        return self._version

    @property
    def batch_ids(self) -> Tuple[Any, ...]:
        with self._lock:
            return tuple(self._batches.keys())

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def submit(self, operations: Iterable[Operation]) -> int:
        """Apply ``operations`` as one atomic transaction.

        Returns the version after commit. An empty transaction leaves the
        version unchanged. Any validation failure raises and changes nothing.
        """
        ops = tuple(operations)
        if not ops:
            return self._version

        with self._lock:
            tentative_batches = dict(self._batches)
            tentative_entries = dict(self._entries)

            for position, operation in enumerate(ops):
                self._apply_operation(
                    operation, position, tentative_batches, tentative_entries
                )

            self._batches = tentative_batches
            self._entries = tentative_entries
            self._version += 1
            return self._version

    def _apply_operation(
        self,
        operation: Operation,
        position: int,
        batches: Dict[Any, Tuple[Tuple[int, int, float], ...]],
        entries: Dict[Key, float],
    ) -> None:
        if isinstance(operation, AddBatch):
            if operation.batch_id in batches:
                raise BatchExistsError(
                    f"operation {position}: batch id {operation.batch_id!r} already exists"
                )
            triples = self._validated_triples(operation.triples, position)
            batches[operation.batch_id] = triples
            for row, col, value in triples:
                self._accumulate(entries, row, col, value, position)

        elif isinstance(operation, ReplaceBatch):
            if operation.batch_id not in batches:
                raise BatchNotFoundError(
                    f"operation {position}: batch id {operation.batch_id!r} does not exist"
                )
            triples = self._validated_triples(operation.triples, position)
            for row, col, value in batches[operation.batch_id]:
                self._accumulate(entries, row, col, -value, position)
            batches[operation.batch_id] = triples
            for row, col, value in triples:
                self._accumulate(entries, row, col, value, position)

        elif isinstance(operation, RemoveBatch):
            if operation.batch_id not in batches:
                raise BatchNotFoundError(
                    f"operation {position}: batch id {operation.batch_id!r} does not exist"
                )
            for row, col, value in batches.pop(operation.batch_id):
                self._accumulate(entries, row, col, -value, position)

        else:
            raise InvalidOperationError(
                f"operation {position}: unknown operation type {type(operation).__name__}"
            )

    def _validated_triples(
        self, triples: Any, position: int
    ) -> Tuple[Tuple[int, int, float], ...]:
        if not isinstance(triples, (tuple, list)):
            raise InvalidOperationError(
                f"operation {position}: triples must be a tuple or list"
            )
        checked: List[Tuple[int, int, float]] = []
        for index, triple in enumerate(triples):
            if not isinstance(triple, (tuple, list)) or len(triple) != 3:
                raise InvalidOperationError(
                    f"operation {position}: triple #{index} must be (row, col, value)"
                )
            row, col, value = triple
            if not _is_real_int(row):
                raise InvalidOperationError(
                    f"operation {position}: triple #{index} row must be an integer"
                )
            if not _is_real_int(col):
                raise InvalidOperationError(
                    f"operation {position}: triple #{index} column must be an integer"
                )
            if not _is_real_number(value):
                raise NonFiniteValueError(
                    f"operation {position}: triple #{index} value must be a real number"
                )
            if not (0 <= row < self._rows):
                raise CoordinateOutOfBoundsError(
                    f"operation {position}: row {row} outside 0..{self._rows - 1}"
                )
            if not (0 <= col < self._cols):
                raise CoordinateOutOfBoundsError(
                    f"operation {position}: column {col} outside 0..{self._cols - 1}"
                )
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise NonFiniteValueError(
                    f"operation {position}: triple #{index} value is not finite"
                )
            checked.append((row, col, numeric_value))
        return tuple(checked)

    def _accumulate(
        self,
        entries: Dict[Key, float],
        row: int,
        col: int,
        value: float,
        position: int,
    ) -> None:
        key = (row, col)
        total = entries.get(key, 0.0) + value
        if not math.isfinite(total):
            raise NonFiniteValueError(
                f"operation {position}: accumulation at ({row}, {col}) is not finite"
            )
        if total == 0.0:
            entries.pop(key, None)
        else:
            entries[key] = total

    # ------------------------------------------------------------------ #
    # Readout
    # ------------------------------------------------------------------ #

    def _visible_coordinates(self) -> List[Coordinate]:
        visible = [
            (row, col, value)
            for (row, col), value in self._entries.items()
            if not (abs(value) <= self._threshold)
        ]
        visible.sort(key=lambda item: (item[0], item[1]))
        return visible

    def nonzeros(self) -> List[Coordinate]:
        """Return threshold-filtered nonzeros sorted by (row, column)."""
        with self._lock:
            return self._visible_coordinates()

    def snapshot(self) -> Snapshot:
        """Capture an immutable read-only snapshot of the current state."""
        with self._lock:
            return Snapshot(
                version=self._version,
                rows=self._rows,
                cols=self._cols,
                threshold=self._threshold,
                batch_ids=frozenset(self._batches.keys()),
                coordinates=tuple(self._visible_coordinates()),
            )

    def matvec(self, vector: Sequence[float]) -> List[float]:
        """Multiply the threshold-filtered matrix by ``vector``.

        Raises ``DimensionError`` on length mismatch without changing state.
        """
        if len(vector) != self._cols:
            raise DimensionError(
                f"vector length {len(vector)} does not match column count {self._cols}"
            )
        with self._lock:
            result = [0.0] * self._rows
            for (row, col), value in self._entries.items():
                if abs(value) <= self._threshold:
                    continue
                result[row] += value * vector[col]
            return result
