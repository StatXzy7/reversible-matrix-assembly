import unittest

from sparse_assembly import Add, AssemblyError, Replace, Revoke, SparseAssembler


def entries(snapshot):
    return [(e.row, e.col, e.value) for e in snapshot.nonzeros()]


class TransactionRollbackTest(unittest.TestCase):
    def setUp(self):
        self.engine = SparseAssembler(3, 3)
        self.engine.add_batch("a", [(0, 0, 1.0)])
        self.engine.add_batch("b", [(1, 1, 2.0)])

    def _assert_state_unchanged(self, version=2):
        snap = self.engine.snapshot()
        self.assertEqual(snap.version, version)
        self.assertEqual(snap.batch_ids, ("a", "b"))
        self.assertEqual(entries(snap), [(0, 0, 1.0), (1, 1, 2.0)])

    def test_duplicate_add_fails_entire_transaction(self):
        with self.assertRaises(AssemblyError):
            self.engine.transaction([
                Add("c", [(2, 2, 3.0)]),
                Add("a", [(2, 2, 4.0)]),  # existing id
            ])
        self._assert_state_unchanged()

    def test_replace_unknown_batch_fails(self):
        with self.assertRaises(AssemblyError):
            self.engine.replace_batch("missing", [(0, 0, 1.0)])
        self._assert_state_unchanged()

    def test_revoke_unknown_batch_fails(self):
        with self.assertRaises(AssemblyError):
            self.engine.revoke_batch("missing")
        self._assert_state_unchanged()

    def test_out_of_range_coordinate_rolls_back_all_operations(self):
        with self.assertRaises(AssemblyError):
            self.engine.transaction([
                Add("c", [(2, 2, 3.0)]),
                Add("d", [(3, 0, 1.0)]),  # row out of range
            ])
        self._assert_state_unchanged()

    def test_negative_coordinate_rolls_back(self):
        with self.assertRaises(AssemblyError):
            self.engine.add_batch("bad", [(0, -1, 1.0)])
        self._assert_state_unchanged()

    def test_non_finite_value_rolls_back(self):
        for bad in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises(AssemblyError):
                    self.engine.transaction([
                        Add(f"bad-{bad}", [(0, 0, 1.0)]),
                        Add(f"bad2-{bad}", [(0, 1, bad)]),
                    ])
                self._assert_state_unchanged()

    def test_partial_application_is_never_visible(self):
        with self.assertRaises(AssemblyError):
            self.engine.transaction([
                Add("ghost", [(2, 2, 9.0)]),
                Replace("ghost", [(0, 0, 1.0)]),  # replace allowed? yes if added,
                Revoke("nonexistent"),            # but this one must abort everything
            ])
        self._assert_state_unchanged()

    def test_non_operation_rejected_without_state_change(self):
        with self.assertRaises(AssemblyError):
            self.engine.transaction([("add", "x")])
        self._assert_state_unchanged()

    def test_malformed_triples_roll_back(self):
        for bad in [(0, 0), "nope", [1, 2, "x"], None]:
            with self.subTest(bad=bad):
                with self.assertRaises(AssemblyError):
                    self.engine.add_batch("malformed", [bad])
                self._assert_state_unchanged()

    def test_successful_transaction_after_failed_one(self):
        with self.assertRaises(AssemblyError):
            self.engine.add_batch("a", [(0, 0, 1.0)])
        snap = self.engine.add_batch("c", [(2, 2, 5.0)])
        self.assertEqual(snap.version, 3)
        self.assertEqual(entries(snap),
                         [(0, 0, 1.0), (1, 1, 2.0), (2, 2, 5.0)])


if __name__ == "__main__":
    unittest.main()
