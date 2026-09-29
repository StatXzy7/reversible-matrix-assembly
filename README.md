# Reversible Sparse Matrix Assembly Engine

A local, in-process sparse matrix assembly engine. Multiple compute modules
submit matrix contributions as named **batches**; batches can be added,
replaced wholesale, or revoked, and several operations can be grouped into
one **atomic transaction**. Pure Python 3.11 standard library only — no
third-party dependencies, no network, no services.

## Layout

- `sparse_assembly/engine.py` — `SparseAssembler`: transactions, accumulation, thresholding, queries
- `sparse_assembly/operations.py` — `add` / `replace` / `remove` operation constructors
- `sparse_assembly/snapshot.py` — immutable read-only `Snapshot`
- `sparse_assembly/errors.py` — typed error hierarchy
- `tests/test_assembly.py` — 28 unittest cases
- `demo.py` — self-terminating business demo with printed assertions
- `atk-demo.json` — harness entry point (`kind=terminal`)

## Semantics

- `SparseAssembler(rows, cols, threshold=0.0)`: dimensions must be positive
  integers and are fixed after creation; `threshold` must be a finite,
  non-negative number.
- Batches are keyed by a unique ID. Each batch holds zero-based
  `(row, col, value)` triples; duplicate coordinates inside one batch and
  across batches accumulate.
- `submit([...])` interprets operations **in order**:
  `add(id, triples)` (ID must be new), `replace(id, triples)` (ID must
  exist), `remove(id)` (ID must exist). Either the whole transaction
  commits (and the version advances by exactly one) or it fails and the
  matrix, batch records and version are unchanged.
- Version starts at 0. A non-empty successful transaction increments it
  once; an empty transaction (`submit([])`) leaves it unchanged.
- Failures that abort the transaction include: duplicate add,
  replace/remove of an unknown ID, out-of-bounds coordinates, non-finite
  values (NaN/Inf).
- Queries:
  - `snapshot()` — immutable `Snapshot(version, rows, cols, threshold,
    batch_ids, coordinates)`; later commits never alter old snapshots.
  - `nonzeros()` — visible entries sorted by `(row, col)`.
  - `matvec(x)` — sparse matrix-vector product; a length mismatch raises
    `DimensionError` without changing state.
- Kill threshold: entries with `abs(value) <= threshold` are excluded from
  readout and multiplication, but the underlying batch contributions are
  retained, so replacement/removal restores any remaining contribution.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Run the demo

```bash
python demo.py
```

The demo prints each input batch, transaction, query result and assertion
(23 checks), then exits 0 on success or non-zero on any failure. It runs in
well under 60 seconds and needs no network or port.
