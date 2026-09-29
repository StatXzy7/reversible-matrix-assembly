#!/usr/bin/env python3
"""Self-terminating command line demonstration of the sparse assembly engine.

The script prints every input triple, operation, observed result and the
assertion made against it.  It exits with status 0 on success and a non-zero
status (uncaught AssertionError / AssemblyError) on any failure.
"""

from __future__ import annotations

from sparse_assembly import (
    Add,
    AssemblyError,
    Replace,
    Revoke,
    SparseAssembler,
)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def show_matrix(snapshot) -> None:
    dense = snapshot.to_dense()
    print(f"  matrix ({snapshot.nrows}x{snapshot.ncols}, "
          f"threshold={snapshot.threshold}, version={snapshot.version}):")
    for row in dense:
        print("    [" + ", ".join(f"{value:5.1f}" for value in row) + "]")


def check(label, actual, expected):
    status = "OK" if actual == expected else "FAIL"
    print(f"  [assert:{status}] {label}")
    print(f"      expected = {expected}")
    print(f"      actual   = {actual}")
    assert actual == expected, f"{label}: {actual!r} != {expected!r}"


def check_close(label, actual, expected, tol=1e-9):
    ok = (len(actual) == len(expected)
          and all(abs(a - e) <= tol for a, e in zip(actual, expected)))
    print(f"  [assert:{'OK' if ok else 'FAIL'}] {label}")
    print(f"      expected ~= {expected} (tol {tol})")
    print(f"      actual    = {actual}")
    assert ok, f"{label}: {actual!r} !~= {expected!r}"


def expect_error(label, fn, fragment):
    try:
        fn()
    except AssemblyError as exc:
        message = str(exc)
        ok = fragment in message
        print(f"  [assert:{'OK' if ok else 'FAIL'}] {label}")
        print(f"      AssemblyError: {message}")
        assert ok, f"error message {message!r} does not contain {fragment!r}"
        return
    raise AssertionError(f"{label}: expected AssemblyError containing {fragment!r}")


def main() -> None:
    banner("1. Create a 4x4 assembler with zero-elimination threshold 0.5")
    engine = SparseAssembler(nrows=4, ncols=4, threshold=0.5)
    snap0 = engine.snapshot()
    check("initial version", snap0.version, 0)
    check("initial batches", snap0.batch_ids, ())
    show_matrix(snap0)

    banner("2. Submit two batches in one atomic transaction (duplicates accumulate)")
    stiffness = [
        (0, 0, 2.0), (0, 1, -1.0),
        (1, 0, -1.0), (1, 1, 2.0), (1, 2, -1.0),
        (2, 1, -1.0), (2, 2, 2.0), (2, 3, -1.0),
        (3, 2, -1.0), (3, 3, 2.0),
    ]
    load_a = [(0, 0, 1.0), (3, 3, 1.0), (0, 0, 1.0)]  # (0,0) duplicated in batch
    print("  Add 'stiffness':", stiffness)
    print("  Add 'load_a'   :", load_a)
    snap1 = engine.transaction([Add("stiffness", stiffness), Add("load_a", load_a)])
    check("version after non-empty transaction", snap1.version, 1)
    check("batch list", snap1.batch_ids, ("stiffness", "load_a"))
    check("batch 'load_a' stored (in-batch duplicates accumulated)",
          [(e.row, e.col, e.value) for e in snap1.batch("load_a")],
          [(0, 0, 2.0), (3, 3, 1.0)])
    show_matrix(snap1)

    banner("3. Matrix-vector multiplication y = A * [1, 2, 4, 8]")
    vector = [1.0, 2.0, 4.0, 8.0]
    print("  input vector:", vector)
    product = snap1.matvec(vector)
    print("  result y    :", product)
    # Row0=(4,-1,0,0): 4*1-1*2=2; row3=(0,0,-1,3): -3+12=9
    check("matvec values", product, (2.0, -1.0, -2.0, 20.0))

    banner("4. Empty transaction succeeds but does not change the version")
    snap_empty = engine.transaction([])
    check("version unchanged by empty transaction", snap_empty.version, 1)

    banner("5. Snapshot isolation: later commits never mutate old snapshots")
    old_snapshot = engine.snapshot()
    check("old snapshot version before change", old_snapshot.version, 1)
    engine.add_batch("load_b", [(0, 1, 5.0)])
    check("live version after new batch", engine.version, 2)
    check("old snapshot still reports version 1", old_snapshot.version, 1)
    check("old snapshot still sees two batches", old_snapshot.batch_ids,
          ("stiffness", "load_a"))
    check("old snapshot matvec unchanged", old_snapshot.matvec(vector),
          (2.0, -1.0, -2.0, 20.0))

    banner("6. Replace one batch and revoke another; remaining contributions survive")
    print("  Replace 'load_b' with [(1, 1, 4.0)]")
    print("  Revoke  'load_a'")
    snap3 = engine.transaction([
        Replace("load_b", [(1, 1, 4.0)]),
        Revoke("load_a"),
    ])
    check("version after replace+revoke transaction", snap3.version, 3)
    check("remaining batches", snap3.batch_ids, ("stiffness", "load_b"))
    show_matrix(snap3)
    # (0,0) is back to stiffness-only 2.0; (1,1) becomes 6.0; (3,3) back to 2.0.
    check("nonzero coordinates after replace/revoke",
          [(e.row, e.col, e.value) for e in snap3.nonzeros()],
          [(0, 0, 2.0), (0, 1, -1.0),
           (1, 0, -1.0), (1, 1, 6.0), (1, 2, -1.0),
           (2, 1, -1.0), (2, 2, 2.0), (2, 3, -1.0),
           (3, 2, -1.0), (3, 3, 2.0)])

    banner("7. Threshold cancellation: small residual is hidden but never deleted")
    small_engine = SparseAssembler(nrows=2, ncols=2, threshold=1e-9)
    small_engine.add_batch("core", [(0, 0, 1.0), (1, 1, 1e-10)])
    small_engine.add_batch("perturbation", [(1, 1, -0.5e-10)])
    s_small = small_engine.snapshot()
    check("residual 0.5e-10 below threshold is hidden",
          [(e.row, e.col, e.value) for e in s_small.nonzeros()],
          [(0, 0, 1.0)])
    check("batch contribution itself is retained",
          [(e.row, e.col, e.value) for e in s_small.batch("perturbation")],
          [(1, 1, -5e-11)])
    check("hidden residual excluded from matvec", s_small.matvec([7.0, 9.0]),
          (7.0, 0.0))
    small_engine.replace_batch("perturbation", [(1, 1, 3.0)])
    s_after = small_engine.snapshot()
    after_entries = [(e.row, e.col, e.value) for e in s_after.nonzeros()]
    check("visible coordinates after replacement",
          [(r, c) for r, c, _ in after_entries], [(0, 0), (1, 1)])
    check_close("residual becomes a visible ~3.0 contribution",
                [v for _, _, v in after_entries], [1.0, 3.0])
    check_close("matvec reflects remaining contribution again",
                s_after.matvec([7.0, 9.0]), (7.0, 27.0))

    banner("8. Transaction rollback: any failed operation changes nothing")
    before_version = engine.version
    before_batches = engine.snapshot().batch_ids

    def bad_transaction():
        engine.transaction([
            Add("load_c", [(0, 0, 1.0)]),
            Add("load_c", [(0, 0, 9.0)]),  # duplicate id -> whole tx fails
        ])

    expect_error("duplicate batch id rejects the whole transaction",
                 bad_transaction, "already exists")
    check("version unchanged after rollback", engine.version, before_version)
    check("batches unchanged after rollback", engine.snapshot().batch_ids,
          before_batches)
    expect_error("replacing unknown batch fails",
                 lambda: engine.replace_batch("missing", [(0, 0, 1.0)]),
                 "unknown batch id")
    expect_error("revoking unknown batch fails",
                 lambda: engine.revoke_batch("missing"), "unknown batch id")
    expect_error("out-of-range coordinate rejects the transaction",
                 lambda: engine.add_batch("oob", [(4, 0, 1.0)]), "out of range")
    expect_error("negative coordinate rejects the transaction",
                 lambda: engine.add_batch("neg", [(0, -1, 1.0)]),
                 "non-negative integer")
    expect_error("non-finite value rejects the transaction",
                 lambda: engine.add_batch("nan", [(0, 0, float("nan"))]),
                 "must be finite")

    def poison_after_good():
        engine.transaction([
            Add("ghost", [(0, 0, 10.0)]),
            Add("ghost", [(0, 0, 20.0)]),
        ])

    expect_error("first op in failed tx is not visible afterwards",
                 poison_after_good, "already exists")
    check("'ghost' batch was rolled back",
          engine.snapshot().batch_ids, before_batches)
    expected_dense = (
        (2.0, -1.0, 0.0, 0.0),
        (-1.0, 6.0, -1.0, 0.0),
        (0.0, -1.0, 2.0, -1.0),
        (0.0, 0.0, -1.0, 2.0),
    )
    check("matrix values unchanged after all failed attempts",
          tuple(tuple(row) for row in engine.snapshot().to_dense()),
          expected_dense)
    check("version still 3 after every rejected transaction", engine.version, 3)

    banner("9. Vector length mismatch reports an error and leaves state intact")
    expect_error("matvec with wrong vector length",
                 lambda: engine.matvec([1.0, 2.0]), "does not match")
    check("state intact after rejected matvec", engine.version, 3)

    print()
    print("#" * 72)
    print("DEMO PASSED: every operation ran and every assertion held.")
    print("#" * 72)


if __name__ == "__main__":
    main()
