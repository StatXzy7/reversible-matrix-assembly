"""Exception hierarchy for the sparse matrix assembly engine."""


class AssemblyError(Exception):
    """Base class for all engine errors."""


class DimensionError(AssemblyError):
    """Raised when matrix dimensions or vector length are invalid."""


class BatchError(AssemblyError):
    """Base class for batch-lifecycle errors."""


class BatchExistsError(BatchError):
    """Raised when adding a batch whose ID is already registered."""


class BatchNotFoundError(BatchError):
    """Raised when replacing or revoking an unknown batch ID."""


class CoordinateOutOfBoundsError(AssemblyError):
    """Raised when a triple uses a row or column outside the matrix."""


class NonFiniteValueError(AssemblyError):
    """Raised when a numeric contribution is NaN or infinite."""


class InvalidOperationError(AssemblyError):
    """Raised when a transaction operation is malformed."""
