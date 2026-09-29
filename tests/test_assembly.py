import unittest

from sparse_assembly import (
    BatchExistsError,
    BatchNotFoundError,
    CoordinateOutOfBoundsError,
    DimensionError,
    InvalidOperationError,
    NonFiniteValueError,
    SparseAssembler,
    Snapshot,
    AddBatch,
    add,
    remove,
    replace,
)


class AssemblyAndMatvecTests(unittest.TestCase):
    def test_assembly_accumulates_duplicates_within_and_across_batches(self):
        asm = SparseAssembler(3, 3)
        self.assertEqual(asm.version, 0)
        asm.submit([add("a", [(0, 0, 1.0), (0, 0, 2.0), (1, 2, -4.0)])])
        asm.submit([add("b", [(0, 0, 5.0), (2, 2, 3.0)])])

        self.assertEqual(
            asm.nonzeros(),
            [(0, 0, 8.0), (1, 2, -4.0), (2, 2, 3.0)],
        )
        self.assertEqual(asm.version, 2)

    def test_matvec_uses_assembled_values(self):
        asm = SparseAssembler(2, 3)
        asm.submit([add("m", [(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0), (1, 2, -1.0)])])
        # A = [[1,0,2],[0,3,-1]], x = [2, 1, 4]
        result = asm.matvec([2.0, 1.0, 4.0])
        self.assertEqual(result, [10.0, -1.0])

    def test_exact_cancellation_removes_coordinate(self):
        asm = SparseAssembler(2, 2)
        asm.submit([add("a", [(0, 0, 3.0)]), add("b", [(0, 0, -3.0)])])
        self.assertEqual(asm.nonzeros(), [])
        self.assertEqual(asm.matvec([1.0, 1.0]), [0.0, 0.0])


class BatchLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.asm = SparseAssembler(3, 3)
        self.asm.submit(
            [
                add("a", [(0, 0, 2.0), (1, 1, 5.0)]),
                add("b", [(0, 0, 3.0), (2, 2, 7.0)]),
            ]
        )

    def test_replace_batches_old_contribution(self):
        version_before = self.asm.version
        self.asm.submit([replace("a", [(0, 0, 10.0)])])
        self.assertEqual(
            self.asm.nonzeros(),
            [(0, 0, 13.0), (2, 2, 7.0)],
        )
        self.assertEqual(self.asm.version, version_before + 1)

    def test_remove_revokes_contribution(self):
        self.asm.submit([remove("b")])
        self.assertEqual(
            self.asm.nonzeros(),
            [(0, 0, 2.0), (1, 1, 5.0)],
        )
        self.assertEqual(set(self.asm.batch_ids), {"a"})

    def test_replace_then_remove_in_one_transaction(self):
        self.asm.submit([replace("a", [(1, 0, 9.0)]), remove("b")])
        self.assertEqual(self.asm.nonzeros(), [(1, 0, 9.0)])

    def test_remove_then_re_add_same_id_in_one_transaction(self):
        self.asm.submit([remove("a"), add("a", [(2, 1, 4.0)])])
        self.assertEqual(
            self.asm.nonzeros(),
            [(0, 0, 3.0), (2, 1, 4.0), (2, 2, 7.0)],
        )


class TransactionRollbackTests(unittest.TestCase):
    def setUp(self):
        self.asm = SparseAssembler(2, 2)
        self.asm.submit([add("a", [(0, 0, 1.0)])])

    def _assert_unchanged(self, version=1):
        self.assertEqual(self.asm.version, version)
        self.assertEqual(self.asm.nonzeros(), [(0, 0, 1.0)])
        self.assertEqual(set(self.asm.batch_ids), {"a"})

    def test_duplicate_add_fails_and_rolls_back(self):
        with self.assertRaises(BatchExistsError):
            self.asm.submit([add("b", [(1, 1, 2.0)]), add("a", [(0, 1, 1.0)])])
        self._assert_unchanged()

    def test_replace_missing_fails_and_rolls_back(self):
        with self.assertRaises(BatchNotFoundError):
            self.asm.submit([add("b", [(1, 1, 2.0)]), replace("x", [])])
        self._assert_unchanged()

    def test_remove_missing_fails_and_rolls_back(self):
        with self.assertRaises(BatchNotFoundError):
            self.asm.submit([remove("ghost")])
        self._assert_unchanged()

    def test_out_of_bounds_fails_and_rolls_back(self):
        with self.assertRaises(CoordinateOutOfBoundsError):
            self.asm.submit([add("b", [(2, 0, 1.0)])])
        self._assert_unchanged()
        with self.assertRaises(CoordinateOutOfBoundsError):
            self.asm.submit([add("b", [(0, -1, 1.0)])])
        self._assert_unchanged()

    def test_non_finite_value_fails_and_rolls_back(self):
        with self.assertRaises(NonFiniteValueError):
            self.asm.submit([add("b", [(0, 1, float("nan"))])])
        self._assert_unchanged()
        with self.assertRaises(NonFiniteValueError):
            self.asm.submit([add("b", [(0, 1, float("inf"))])])
        self._assert_unchanged()

    def test_later_operation_failure_rolls_back_earlier_effect(self):
        # replace succeeds logically, then the add of the same id fails;
        # the replaced contribution must not leak into committed state.
        with self.assertRaises(BatchExistsError):
            self.asm.submit(
                [
                    replace("a", [(1, 1, 9.0)]),
                    add("a", [(0, 0, 1.0)]),
                ]
            )
        self._assert_unchanged()

    def test_empty_transaction_does_not_advance_version(self):
        version_before = self.asm.version
        self.assertEqual(self.asm.submit([]), version_before)
        self.assertEqual(self.asm.version, version_before)

    def test_version_starts_at_zero_and_increments_once_per_nonempty_commit(self):
        asm = SparseAssembler(2, 2)
        self.assertEqual(asm.version, 0)
        asm.submit([add("a", [(0, 0, 1.0)]), add("b", [(1, 1, 1.0)])])
        self.assertEqual(asm.version, 1)
        asm.submit([])
        self.assertEqual(asm.version, 1)
        asm.submit([replace("a", [(0, 0, 2.0)])])
        self.assertEqual(asm.version, 2)


class ThresholdTests(unittest.TestCase):
    def test_threshold_hides_small_entries_but_keeps_batch_contributions(self):
        asm = SparseAssembler(2, 2, threshold=1e-9)
        asm.submit([add("a", [(0, 0, 5e-10), (1, 1, 2.0)])])
        self.assertEqual(asm.nonzeros(), [(1, 1, 2.0)])
        self.assertEqual(asm.matvec([1.0, 1.0]), [0.0, 2.0])
        self.assertEqual(set(asm.batch_ids), {"a"})

        # A further batch tips the suppressed coordinate back above threshold.
        asm.submit([add("b", [(0, 0, 1.5)])])
        visible = asm.nonzeros()
        self.assertEqual([coord[:2] for coord in visible], [(0, 0), (1, 1)])
        self.assertAlmostEqual(visible[0][2], 1.5000000005)
        self.assertEqual(visible[1][2], 2.0)
        product = asm.matvec([1.0, 1.0])
        self.assertAlmostEqual(product[0], 1.5000000005)
        self.assertEqual(product[1], 2.0)

    def test_threshold_boundary_is_inclusive(self):
        asm = SparseAssembler(1, 1, threshold=0.25)
        asm.submit([add("a", [(0, 0, 0.5)])])
        self.assertEqual(asm.nonzeros(), [(0, 0, 0.5)])
        # 0.5 - 0.25 == 0.25 exactly: inclusive boundary is suppressed.
        asm.submit([add("b", [(0, 0, -0.25)])])
        self.assertEqual(asm.nonzeros(), [])
        # A later contribution brings the coordinate back above threshold.
        asm.submit([add("c", [(0, 0, 1.0)])])
        self.assertEqual(asm.nonzeros(), [(0, 0, 1.25)])

    def test_revoking_restores_residual_above_threshold(self):
        asm = SparseAssembler(1, 2, threshold=1e-6)
        asm.submit([add("big", [(0, 0, 10.0)]), add("cancel", [(0, 0, -9.9999995)])])
        self.assertEqual(asm.nonzeros(), [])  # residual 5e-7 hidden
        asm.submit([remove("cancel")])
        self.assertEqual(asm.nonzeros(), [(0, 0, 10.0)])

    def test_replace_recomputes_threshold_visibility(self):
        asm = SparseAssembler(1, 1, threshold=1e-6)
        asm.submit([add("a", [(0, 0, 1e-7)])])
        self.assertEqual(asm.nonzeros(), [])
        asm.submit([replace("a", [(0, 0, 4.0)])])
        self.assertEqual(asm.nonzeros(), [(0, 0, 4.0)])

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(DimensionError):
            SparseAssembler(2, 2, threshold=-0.1)
        with self.assertRaises(DimensionError):
            SparseAssembler(2, 2, threshold=float("inf"))


class SnapshotIsolationTests(unittest.TestCase):
    def test_snapshot_captures_version_batches_and_coordinates(self):
        asm = SparseAssembler(2, 2)
        asm.submit([add("a", [(0, 0, 1.0)]), add("b", [(1, 1, 2.0)])])
        snap = asm.snapshot()
        self.assertIsInstance(snap, Snapshot)
        self.assertEqual(snap.version, 1)
        self.assertEqual(snap.batch_ids, frozenset({"a", "b"}))
        self.assertEqual(snap.coordinates, ((0, 0, 1.0), (1, 1, 2.0)))

    def test_old_snapshot_is_immutable_after_later_commits(self):
        asm = SparseAssembler(2, 2)
        asm.submit([add("a", [(0, 0, 1.0)])])
        old = asm.snapshot()
        asm.submit([add("b", [(0, 0, 4.0)]), replace("a", [(0, 1, 9.0)])])
        asm.submit([remove("b")])

        self.assertEqual(old.version, 1)
        self.assertEqual(old.batch_ids, frozenset({"a"}))
        self.assertEqual(old.coordinates, ((0, 0, 1.0),))
        self.assertEqual(old.matvec([1.0, 0.0]), [1.0, 0.0])

        # Current engine reflects the later history.
        self.assertEqual(asm.version, 3)
        self.assertEqual(asm.nonzeros(), [(0, 1, 9.0)])

    def test_snapshot_matvec_length_mismatch_raises(self):
        asm = SparseAssembler(2, 3)
        snap = asm.snapshot()
        with self.assertRaises(ValueError):
            snap.matvec([1.0, 1.0])


class InputValidationTests(unittest.TestCase):
    def test_dimensions_must_be_positive_integers(self):
        for bad_rows, bad_cols in [(0, 2), (2, 0), (-1, 2), (2.0, 2), (True, 2)]:
            with self.assertRaises(DimensionError):
                SparseAssembler(bad_rows, bad_cols)

    def test_matvec_length_mismatch_raises_without_state_change(self):
        asm = SparseAssembler(2, 3)
        asm.submit([add("a", [(0, 0, 1.0)])])
        with self.assertRaises(DimensionError):
            asm.matvec([1.0, 1.0])
        self.assertEqual(asm.version, 1)
        self.assertEqual(asm.nonzeros(), [(0, 0, 1.0)])

    def test_non_integer_indices_rejected(self):
        asm = SparseAssembler(2, 2)
        with self.assertRaises(InvalidOperationError):
            asm.submit([add("a", [(0.5, 0, 1.0)])])
        with self.assertRaises(InvalidOperationError):
            asm.submit([add("a", [(0, True, 1.0)])])
        self.assertEqual(asm.version, 0)
        self.assertEqual(asm.batch_ids, ())

    def test_non_numeric_value_rejected(self):
        asm = SparseAssembler(2, 2)
        with self.assertRaises(NonFiniteValueError):
            asm.submit([AddBatch("a", ((0, 0, "1.0"),))])
        with self.assertRaises(InvalidOperationError):
            asm.submit([AddBatch("a", ((0, 0, 1, 2),))])

    def test_unknown_operation_type_rejected(self):
        asm = SparseAssembler(2, 2)
        with self.assertRaises(InvalidOperationError):
            asm.submit([("add", "a", [])])
        self.assertEqual(asm.version, 0)


if __name__ == "__main__":
    unittest.main()
