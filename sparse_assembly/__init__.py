"""Reversible sparse matrix assembly engine (pure Python standard library)."""

from .engine import SparseAssembler
from .errors import (
    AssemblyError,
    BatchExistsError,
    BatchNotFoundError,
    CoordinateOutOfBoundsError,
    DimensionError,
    InvalidOperationError,
    NonFiniteValueError,
)
from .operations import AddBatch, RemoveBatch, ReplaceBatch, add, remove, replace
from .snapshot import Snapshot

__all__ = [
    "SparseAssembler",
    "Snapshot",
    "AddBatch",
    "ReplaceBatch",
    "RemoveBatch",
    "add",
    "replace",
    "remove",
    "AssemblyError",
    "BatchExistsError",
    "BatchNotFoundError",
    "CoordinateOutOfBoundsError",
    "DimensionError",
    "InvalidOperationError",
    "NonFiniteValueError",
]
