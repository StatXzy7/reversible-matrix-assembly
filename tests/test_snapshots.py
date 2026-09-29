import unittest

from sparse_assembly import SparseAssembler


def entries(snapshot):
    return [(e.row, e.col, e.value) for e in snapshot.nonzeros()]


class SnapshotIsolationTest(unittest.TestCase):
    def setUp(self):
        self.engine = SparseAssembler(2, 2)
        self.old = self.engine.add_batch("a", [(0, 0, 1.0)])

    def test_old_snapshot_version_never_changes(self):
        self.assertEqual(self.old.version, 1)
        self.engine.add_batch("b", [(1, 1, 2.0)])
        self.assertEqual(self.engine.version, 2)
        self.assertEqual(self.old.version, 1)

    def test_old_snapshot_batches_never_change(self):
        self.assertEqual(self.old.batch_ids, ("a",))
        self.engine.add_batch("b", [(1, 1, 2.0)])
        self.engine.replace_batch("a", [(0, 1, 5.0)])
        self.assertEqual(self.old.batch_ids, ("a",))
        self.assertEqual(entries(self.old), [(0, 0, 1.0)])
        self.assertEqual(self.old.matvec([1.0, 1.0]), (1.0, 0.0))

    def test_old_snapshot_after_revoke_still_queries(self):
        self.engine.revoke_batch("a")
        self.assertEqual(self.engine.snapshot().batch_ids, ())
        self.assertEqual(entries(self.old), [(0, 0, 1.0)])

    def test_snapshot_batches_are_read_only(self):
        with self.assertRaises(TypeError):
            self.old.batches["x"] = ()
        with self.assertRaises(TypeError):
            del self.old.batches["a"]

    def test_batch_entries_tuple_is_immutable(self):
        batch = self.old.batch("a")
        with self.assertRaises(TypeError):
            batch[0] = None  # type: ignore[index]

    def test_new_snapshot_reflects_latest_state(self):
        new = self.engine.add_batch("b", [(1, 1, 2.0)])
        self.assertEqual(new.version, 2)
        self.assertEqual(new.batch_ids, ("a", "b"))
        self.assertEqual(entries(new), [(0, 0, 1.0), (1, 1, 2.0)])

    def test_snapshot_batch_lookup_unknown_id(self):
        from sparse_assembly import AssemblyError
        with self.assertRaises(AssemblyError):
            self.old.batch("nope")


if __name__ == "__main__":
    unittest.main()
