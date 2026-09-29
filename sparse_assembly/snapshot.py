"""Immutable read-only snapshots of the assembled matrix state."""

from dataclasses import dataclass
from typing import Any, FrozenSet, List, Sequence, Tuple

Coordinate = Tuple[int, int, float]


@dataclass(frozen=True)
class Snapshot:
    """Point-in-time, immutable view of an assembler.

    Attributes:
        version: monotonically increasing version captured at read time.
        rows: number of matrix rows.
        cols: number of matrix columns.
        threshold: active kill threshold.
        batch_ids: frozen set of batch IDs currently registered.
        coordinates: threshold-filtered nonzeros, sorted by (row, col).
    """

    version: int
    rows: int
    cols: int
    threshold: float
    batch_ids: FrozenSet[Any]
    coordinates: Tuple[Coordinate, ...]

    @property
    def nnz(self) -> int:
        return len(self.coordinates)

    def as_list(self) -> List[Coordinate]:
        return list(self.coordinates)

    def to_dense(self) -> List[List[float]]:
        matrix = [[0.0 for _ in range(self.cols)] for _ in range(self.rows)]
        for row, col, value in self.coordinates:
            matrix[row][col] = value
        return matrix

    def matvec(self, vector: Sequence[float]) -> List[float]:
        """Multiply the filtered sparse matrix by ``vector``."""
        if len(vector) != self.cols:
            raise ValueError(
                f"vector length {len(vector)} does not match column count {self.cols}"
            )
        result = [0.0] * self.rows
        for row, col, value in self.coordinates:
            result[row] += value * vector[col]
        return result
