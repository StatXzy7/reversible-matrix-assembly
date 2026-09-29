"""Local sparse matrix assembly engine.

Multiple computing modules submit sparse matrix contributions keyed by
unique batch ids.  Contributions may be replaced or withdrawn independently,
and several operations can be grouped into one atomic transaction.

The implementation only relies on the Python standard library.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

__all__ = [
    "AssemblyError",
    "Entry",
    "Operation",
    "Add",
    "Replace",
    "Revoke",
    "Snapshot",
    "SparseAssembler",
]

# A (row, column, value) triple supplied by a caller.
Triple = Tuple[int, int, float]
# Raw aggregated storage for a batch / the assembled matrix.
CoordMap = Dict[Tuple[int, int], float]


class AssemblyError(Exception):
    """Raised whenever a transaction or query input is invalid.

    Raising this error never mutates engine state: transactions are applied
    to private working copies and only committed once every operation and
    every triple has been validated.
    """


@dataclass(frozen=True)
class Entry:
    """A single stored (row, column, value) triple."""

    row: int
    col: int
    value: float


# ---------------------------------------------------------------------------
# Transaction operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Operation:
    """Base class for operations accepted by :meth:`SparseAssembler.transaction`."""

    batch_id: str

    def apply(self, batches: Dict[str, CoordMap], assembler: "SparseAssembler") -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class Add(Operation):
    """Add a new batch.  The batch id must not already exist."""

    entries: Sequence[Triple] = ()

    def apply(self, batches: Dict[str, CoordMap], assembler: "SparseAssembler") -> None:
        if self.batch_id in batches:
            raise AssemblyError(f"batch id already exists: {self.batch_id!r}")
        batches[self.batch_id] = assembler._collect(self.entries)


@dataclass(frozen=True)
class Replace(Operation):
    """Replace every triple of an existing batch with new triples."""

    entries: Sequence[Triple] = ()

    def apply(self, batches: Dict[str, CoordMap], assembler: "SparseAssembler") -> None:
        if self.batch_id not in batches:
            raise AssemblyError(f"cannot replace unknown batch id: {self.batch_id!r}")
        batches[self.batch_id] = assembler._collect(self.entries)


@dataclass(frozen=True)
class Revoke(Operation):
    """Remove an existing batch together with all of its contributions."""

    def apply(self, batches: Dict[str, CoordMap], assembler: "SparseAssembler") -> None:
        if self.batch_id not in batches:
            raise AssemblyError(f"cannot revoke unknown batch id: {self.batch_id!r}")
        del batches[self.batch_id]


# ---------------------------------------------------------------------------
# Read-only snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """Immutable read-only view of an assembler state.

    Batches are exposed through the read-only ``batches`` mapping; each batch
    is a tuple of :class:`Entry` sorted by ``(row, col)``.
    """

    version: int
    nrows: int
    ncols: int
    threshold: float
    batches: Mapping[str, Tuple[Entry, ...]]

    @property
    def batch_ids(self) -> Tuple[str, ...]:
        return tuple(self.batches)

    def batch(self, batch_id: str) -> Tuple[Entry, ...]:
        """Return the stored, unfiltered entries of one batch."""
        try:
            return self.batches[batch_id]
        except KeyError:
            raise AssemblyError(f"unknown batch id: {batch_id!r}") from None

    def _effective(self) -> CoordMap:
        """Sum all batches, then drop entries at or below the threshold."""
        combined: CoordMap = {}
        for entries in self.batches.values():
            for entry in entries:
                key = (entry.row, entry.col)
                combined[key] = combined.get(key, 0.0) + entry.value
        result: CoordMap = {}
        for key, value in combined.items():
            if abs(value) > self.threshold:
                result[key] = value
        return result

    def nonzeros(self) -> Tuple[Entry, ...]:
        """Effective non-zero entries sorted by ``(row, col)``.

        Values whose absolute value is no greater than the threshold are
        suppressed here and in :meth:`matvec`, but they are never removed
        from the underlying batches.
        """
        effective = self._effective()
        return tuple(
            Entry(row, col, effective[(row, col)])
            for row, col in sorted(effective)
        )

    def to_dense(self) -> List[List[float]]:
        """Return an ``nrows x ncols`` list applying the threshold (handy for demos)."""
        effective = self._effective()
        return [
            [effective.get((row, col), 0.0) for col in range(self.ncols)]
            for row in range(self.nrows)
        ]

    def matvec(self, vector: Sequence[float]) -> Tuple[float, ...]:
        """Multiply the effective sparse matrix by ``vector``."""
        vector = _check_vector(vector, self.ncols)
        result = [0.0] * self.nrows
        for (row, col), value in self._effective().items():
            result[row] += value * vector[col]
        return tuple(result)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SparseAssembler:
    """Holds a fixed-size sparse matrix assembled from named batches."""

    def __init__(self, nrows: int, ncols: int, threshold: float = 0.0) -> None:
        if not isinstance(nrows, int) or isinstance(nrows, bool) or nrows <= 0:
            raise AssemblyError("nrows must be a positive integer")
        if not isinstance(ncols, int) or isinstance(ncols, bool) or ncols <= 0:
            raise AssemblyError("ncols must be a positive integer")
        threshold = _finite_float(threshold, "threshold")
        if threshold < 0.0:
            raise AssemblyError("threshold must be finite and non-negative")

        self._nrows = nrows
        self._ncols = ncols
        self._threshold = threshold
        self._version = 0
        self._batches: Dict[str, CoordMap] = {}
        self._lock = threading.RLock()

    # -- basic metadata ----------------------------------------------------

    @property
    def nrows(self) -> int:
        return self._nrows

    @property
    def ncols(self) -> int:
        return self._ncols

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    # -- snapshots and queries --------------------------------------------

    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._build_snapshot(self._batches, self._version)

    def nonzeros(self) -> Tuple[Entry, ...]:
        return self.snapshot().nonzeros()

    def matvec(self, vector: Sequence[float]) -> Tuple[float, ...]:
        return self.snapshot().matvec(vector)

    # -- single-operation convenience wrappers -----------------------------

    def add_batch(self, batch_id: str, entries: Iterable[Triple]) -> Snapshot:
        return self.transaction([Add(batch_id, entries)])

    def replace_batch(self, batch_id: str, entries: Iterable[Triple]) -> Snapshot:
        return self.transaction([Replace(batch_id, entries)])

    def revoke_batch(self, batch_id: str) -> Snapshot:
        return self.transaction([Revoke(batch_id)])

    # -- atomic transactions ----------------------------------------------

    def transaction(self, operations: Iterable[Operation]) -> Snapshot:
        ops = list(operations)
        for index, op in enumerate(ops):
            if not isinstance(op, Operation):
                raise AssemblyError(
                    f"operation {index} is not an Operation instance: {op!r}"
                )
            if not isinstance(op.batch_id, str) or not op.batch_id:
                raise AssemblyError(
                    f"operation {index}: batch id must be a non-empty string"
                )

        with self._lock:
            working = dict(self._batches)
            for op in ops:
                op.apply(working, self)
            # Every operation succeeded: publish the working copy atomically.
            non_empty = len(ops) > 0
            new_version = self._version + 1 if non_empty else self._version
            snapshot = self._build_snapshot(working, new_version)
            self._batches = working
            self._version = new_version
            return snapshot

    # -- internals ---------------------------------------------------------

    def _build_snapshot(
        self, batches: Mapping[str, CoordMap], version: int
    ) -> Snapshot:
        immutable_batches = {
            batch_id: tuple(
                Entry(row, col, entries[(row, col)])
                for row, col in sorted(entries)
            )
            for batch_id, entries in batches.items()
        }
        return Snapshot(
            version=version,
            nrows=self._nrows,
            ncols=self._ncols,
            threshold=self._threshold,
            batches=MappingProxyType(immutable_batches),
        )

    def _collect(self, entries: Any) -> CoordMap:
        """Validate caller triples and aggregate duplicate coordinates."""
        if isinstance(entries, (str, bytes)) or not _is_iterable(entries):
            raise AssemblyError("entries must be an iterable of (row, col, value) triples")

        collected: CoordMap = {}
        for index, triple in enumerate(entries):
            row, col, value = _parse_triple(triple, index)
            if not 0 <= row < self._nrows:
                raise AssemblyError(
                    f"triple {index}: row {row} out of range for {self._nrows} rows"
                )
            if not 0 <= col < self._ncols:
                raise AssemblyError(
                    f"triple {index}: column {col} out of range for {self._ncols} columns"
                )
            number = _finite_float(value, f"triple {index} value")
            key = (row, col)
            new_value = collected.get(key, 0.0) + number
            if new_value == 0.0:
                # Exact cancellation leaves no stored contribution.
                collected.pop(key, None)
            else:
                collected[key] = new_value
        return collected


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _is_iterable(value: Any) -> bool:
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _parse_triple(triple: Any, index: int) -> Tuple[int, int, Any]:
    if isinstance(triple, (str, bytes)) or not _is_iterable(triple):
        raise AssemblyError(f"triple {index} must be a (row, col, value) sequence")
    try:
        row, col, value = triple
    except (TypeError, ValueError):
        raise AssemblyError(
            f"triple {index} must contain exactly (row, col, value)"
        ) from None
    if not isinstance(row, int) or isinstance(row, bool) or row < 0:
        raise AssemblyError(f"triple {index}: row must be a non-negative integer")
    if not isinstance(col, int) or isinstance(col, bool) or col < 0:
        raise AssemblyError(f"triple {index}: column must be a non-negative integer")
    # bool is an int subclass in Python; accept it and convert to 0.0/1.0.
    if not isinstance(value, (int, float)):
        raise AssemblyError(f"triple {index}: value must be a finite number")
    return row, col, float(value)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssemblyError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise AssemblyError(f"{label} must be finite, got {number!r}")
    return number


def _check_vector(vector: Any, ncols: int) -> Tuple[float, ...]:
    if isinstance(vector, (str, bytes)) or not _is_iterable(vector):
        raise AssemblyError("vector must be an iterable of numbers")
    values: List[float] = []
    for index, item in enumerate(vector):
        if not isinstance(item, (int, float)):
            raise AssemblyError(f"vector element {index} must be a finite number")
        number = float(item)
        if not math.isfinite(number):
            raise AssemblyError(f"vector element {index} must be finite, got {number!r}")
        values.append(number)
    if len(values) != ncols:
        raise AssemblyError(
            f"vector length {len(values)} does not match matrix column count {ncols}"
        )
    return tuple(values)
