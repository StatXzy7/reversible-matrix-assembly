import unittest

from sparse_assembly import Add, AssemblyError, Replace, Revoke, SparseAssembler


def entries(snapshot):
    return [(e.row, e.col, e.value) for e in snapshot.nonzeros()]


class AssemblyTest(unittest.TestCase):
    def test_initial_state(self):
        engine = SparseAssembler(2, 3)
        snap = engine.snapshot()
        self.assertEqual(snap.version, 0)
        self.assertEqual(snap.nrows, 2)
        self.assertEqual(snap.ncols, 3)
        self.assertEqual(snap.batch_ids, ())
        self.assertEqual(snap.nonzeros(), ())

    def test_add_batch_and_duplicate_accumulation_within_batch(self):
        engine = SparseAssembler(3, 3)
        snap = engine.add_batch("b", [
            (0, 0, 1.0), (0, 0, 2.5),  # duplicate within batch
            (2, 2, -3.0),
        ])
        self.assertEqual(snap.version, 1)
        self.assertEqual(entries(snap), [(0, 0, 3.5), (2, 2, -3.0)])
        self.assertEqual([(e.row, e.col, e.value) for e in snap.batch("b")],
                         [(0, 0, 3.5), (2, 2, -3.0)])

    def test_duplicate_coordinates_accumulate_across_batches(self):
        engine = SparseAssembler(2, 2)
        engine.add_batch("a", [(0, 0, 1.0), (1, 1, 4.0)])
        snap = engine.add_batch("b", [(0, 0, 2.0), (1, 1, -1.5)])
        self.assertEqual(snap.version, 2)
        self.assertEqual(entries(snap), [(0, 0, 3.0), (1, 1, 2.5)])

    def test_nonzeros_sorted_row_then_column(self):
        engine = SparseAssembler(2, 3)
        engine.add_batch("b", [(1, 2, 1.0), (0, 2, 1.0), (1, 0, 1.0),
                               (0, 0, 1.0)])
        coords = [(e.row, e.col) for e in engine.nonzeros()]
        self.assertEqual(coords, [(0, 0), (0, 2), (1, 0), (1, 2)])

    def test_exact_zero_cancellation_is_dropped_from_storage(self):
        engine = SparseAssembler(2, 2)
        snap = engine.add_batch("z", [(0, 0, 2.0), (0, 0, -2.0)])
        self.assertEqual(entries(snap), [])
        self.assertEqual(snap.batch("z"), ())

    def test_replace_batch(self):
        engine = SparseAssembler(2, 2)
        engine.add_batch("a", [(0, 0, 5.0)])
        snap = engine.replace_batch("a", [(1, 1, 7.0)])
        self.assertEqual(snap.version, 2)
        self.assertEqual(entries(snap), [(1, 1, 7.0)])
        self.assertEqual([(e.row, e.col, e.value) for e in snap.batch("a")],
                         [(1, 1, 7.0)])

    def test_replace_releases_old_contribution(self):
        engine = SparseAssembler(2, 2)
        engine.add_batch("a", [(0, 0, 5.0)])
        engine.add_batch("b", [(0, 0, 2.0), (1, 1, 3.0)])
        snap = engine.replace_batch("b", [(1, 1, 9.0)])
        self.assertEqual(entries(snap), [(0, 0, 5.0), (1, 1, 9.0)])

    def test_revoke_batch(self):
        engine = SparseAssembler(2, 2)
        engine.add_batch("a", [(0, 0, 5.0)])
        engine.add_batch("b", [(0, 0, 2.0)])
        snap = engine.revoke_batch("a")
        self.assertEqual(snap.version, 3)
        self.assertEqual(snap.batch_ids, ("b",))
        self.assertEqual(entries(snap), [(0, 0, 2.0)])

    def test_multi_operation_transaction_commits_once(self):
        engine = SparseAssembler(2, 2)
        snap = engine.transaction([
            Add("a", [(0, 0, 1.0)]),
            Add("b", [(0, 0, 2.0)]),
            Replace("a", [(1, 1, 4.0)]),
        ])
        self.assertEqual(snap.version, 1)
        self.assertEqual(snap.batch_ids, ("a", "b"))
        self.assertEqual(entries(snap), [(0, 0, 2.0), (1, 1, 4.0)])

    def test_empty_transaction_does_not_change_version(self):
        engine = SparseAssembler(2, 2)
        engine.add_batch("a", [(0, 0, 1.0)])
        self.assertEqual(engine.version, 1)
        snap = engine.transaction([])
        self.assertEqual(snap.version, 1)
        self.assertEqual(engine.version, 1)
        self.assertEqual(snap.batch_ids, ("a",))

    def test_version_starts_at_zero_and_increments_per_nonempty_transaction(self):
        engine = SparseAssembler(2, 2)
        self.assertEqual(engine.version, 0)
        engine.transaction([])
        self.assertEqual(engine.version, 0)
        engine.add_batch("a", [])
        self.assertEqual(engine.version, 1)
        engine.transaction([Replace("a", [(0, 0, 1.0)])])
        self.assertEqual(engine.version, 2)
        engine.revoke_batch("a")
        self.assertEqual(engine.version, 3)


if __name__ == "__main__":
    unittest.main()
