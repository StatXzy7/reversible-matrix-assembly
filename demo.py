#!/usr/bin/env python3
"""Self-terminating business demo for the sparse matrix assembly engine.

Runs a realistic assembly scenario against :class:`SparseAssembler`, prints
every input, operation, result and assertion, and exits with status 0 only
when every assertion passes. No network, no third-party dependencies.
"""

import sys
import traceback

from sparse_assembly import (
    BatchExistsError,
    CoordinateOutOfBoundsError,
    DimensionError,
    SparseAssembler,
    add,
    remove,
    replace,
)

CHECKS = 0


def check(label, actual, expected):
    global CHECKS
    CHECKS += 1
    status = "PASS" if actual == expected else "FAIL"
    print(f"  [assert {CHECKS}] {label}")
    print(f"           expected = {expected}")
    print(f"           actual   = {actual}  -> {status}")
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    section("1. Create assembler: 3 rows x 3 cols, kill threshold 1e-9")
    asm = SparseAssembler(3, 3, threshold=1e-9)
    print(f"  rows={asm.rows} cols={asm.cols} threshold={asm.threshold} version={asm.version}")
    check("initial version is 0", asm.version, 0)
    check("initial nonzeros are empty", asm.nonzeros(), [])

    section("2. Atomic transaction: add two independent module batches")
    tx1 = [
        add("module-stiffness", [(0, 0, 1.0), (0, 0, 2.0), (1, 2, -4.0)]),
        add("module-load", [(0, 0, 5.0), (2, 2, 3.0)]),
    ]
    for op in tx1:
        print(f"  operation: {op}")
    version_after_tx1 = asm.submit(tx1)
    print(f"  result nonzeros (sorted by row, col): {asm.nonzeros()}")
    print(f"  registered batches: {sorted(asm.batch_ids)}")
    check("version advanced once for the whole transaction", version_after_tx1, 1)
    check(
        "duplicate coords accumulate within and across batches",
        asm.nonzeros(),
        [(0, 0, 8.0), (1, 2, -4.0), (2, 2, 3.0)],
    )

    section("3. Matrix-vector product  A * x")
    x = [2.0, 1.0, 4.0]
    print(f"  input vector x = {x}")
    y = asm.matvec(x)
    print(f"  A * x = {y}")
    check("matvec row 0", y[0], 16.0)   # 8*2
    check("matvec row 1", y[1], -16.0)  # -4*4
    check("matvec row 2", y[2], 12.0)   # 3*4

    section("4. Read-only snapshot, then replace one batch and revoke another")
    old_snapshot = asm.snapshot()
    print(f"  snapshot: version={old_snapshot.version} "
          f"batches={sorted(old_snapshot.batch_ids)} coords={old_snapshot.coordinates}")
    tx2 = [
        replace("module-stiffness", [(0, 0, 10.0), (2, 1, 6.0)]),
        remove("module-load"),
    ]
    for op in tx2:
        print(f"  operation: {op}")
    asm.submit(tx2)
    print(f"  result nonzeros: {asm.nonzeros()}")
    print(f"  registered batches: {sorted(asm.batch_ids)}")
    check(
        "old stiffness/load contributions gone, new ones present",
        asm.nonzeros(),
        [(0, 0, 10.0), (2, 1, 6.0)],
    )
    check("version advanced exactly once", asm.version, 2)

    section("5. Invalid atomic transaction is fully rolled back")
    print("  operations: add a valid new batch, then add an EXISTING id")
    bad_tx = [add("module-extra", [(1, 1, 9.0)]), add("module-stiffness", [(0, 1, 1.0)])]
    try:
        asm.submit(bad_tx)
        raise AssertionError("expected BatchExistsError")
    except BatchExistsError as exc:
        print(f"  caught expected error: {type(exc).__name__}: {exc}")
    check("version unchanged after rollback", asm.version, 2)
    check("nonzeros unchanged after rollback",
          asm.nonzeros(), [(0, 0, 10.0), (2, 1, 6.0)])
    check("rolled-back batch is not registered", "module-extra" in asm.batch_ids, False)

    print("  operations: out-of-bounds coordinate (row 9)")
    try:
        asm.submit([add("module-oob", [(9, 0, 1.0)])])
        raise AssertionError("expected CoordinateOutOfBoundsError")
    except CoordinateOutOfBoundsError as exc:
        print(f"  caught expected error: {type(exc).__name__}: {exc}")
    check("version still unchanged", asm.version, 2)

    section("6. Kill threshold: entries hidden from readout but contributions kept")
    asm.submit([add("module-tiny", [(1, 1, 5e-10)])])
    print("  added (1,1)=5e-10 under threshold 1e-9")
    print(f"  nonzeros: {asm.nonzeros()}")
    check("tiny entry suppressed", (1, 1, 5e-10) in asm.nonzeros(), False)
    asm.submit([add("module-tip", [(1, 1, 2.0)])])
    print("  added (1,1)=2.0 in another batch; residual 2.0000000005 reappears")
    coords = asm.nonzeros()
    print(f"  nonzeros: {coords}")
    check("coordinate reappears above threshold",
          [(r, c) for r, c, _ in coords], [(0, 0,), (1, 1,), (2, 1,)])
    check("residual value accumulates both batches", coords[1][2], 2.0000000005)
    asm.submit([remove("module-tip")])
    print("  revoked tipping batch; remaining 5e-10 contribution is hidden again")
    check("revoke recomputes threshold visibility",
          asm.nonzeros(), [(0, 0, 10.0), (2, 1, 6.0)])
    check("but the batch record is preserved", "module-tiny" in asm.batch_ids, True)

    section("7. Snapshot isolation: old snapshot never changes")
    print(f"  old snapshot version={old_snapshot.version} "
          f"coords={old_snapshot.coordinates}")
    check("old snapshot keeps v1 coordinates",
          list(old_snapshot.coordinates),
          [(0, 0, 8.0), (1, 2, -4.0), (2, 2, 3.0)])
    check("old snapshot matvec still works", old_snapshot.matvec([2, 1, 4]), [16.0, -16.0, 12.0])
    print(f"  live engine is at version {asm.version}; snapshots are independent")

    section("8. Empty transaction and vector length validation")
    before = asm.version
    check("empty transaction returns same version", asm.submit([]), before)
    check("empty transaction does not bump version", asm.version, before)
    try:
        asm.matvec([1.0, 2.0])
        raise AssertionError("expected DimensionError")
    except DimensionError as exc:
        print(f"  length-2 vector against 3 cols -> {type(exc).__name__}: {exc}")
    check("state unchanged after bad vector", asm.version, before)

    section(f"ALL DEMO ASSERTIONS PASSED ({CHECKS} checks)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print()
        print("DEMO FAILED")
        traceback.print_exc()
        sys.exit(1)
