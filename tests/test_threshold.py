import unittest

from sparse_assembly import AssemblyError, SparseAssembler


def entries(snapshot):
    return [(e.row, e.col, e.value) for e in snapshot.nonzeros()]


class ThresholdTest(unittest.TestCase):
    def test_threshold_filters_final_entries(self):
        engine = SparseAssembler(2, 2, threshold=0.5)
        snap = engine.add_batch("a", [
            (0, 0, 0.5),    # abs == threshold -> suppressed
            (0, 1, -0.5),   # equal magnitude -> suppressed
            (1, 0, 0.5001), # just above -> kept
            (1, 1, -2.0),
        ])
        self.assertEqual([(e.row, e.col) for e in snap.nonzeros()],
                         [(1, 0), (1, 1)])

    def test_threshold_applied_after_cross_batch_accumulation(self):
        engine = SparseAssembler(1, 2, threshold=1.0)
        engine.add_batch("a", [(0, 0, 0.6)])
        engine.add_batch("b", [(0, 0, 0.6)])  # total 1.2 > 1.0 -> visible
        snap = engine.add_batch("c", [(0, 1, 0.6)])  # 0.6 alone -> hidden
        self.assertEqual(entries(snap), [(0, 0, 1.2)])

    def test_threshold_does_not_delete_batch_contributions(self):
        engine = SparseAssembler(1, 1, threshold=1e-6)
        engine.add_batch("a", [(0, 0, 5e-7)])
        engine.add_batch("b", [(0, 0, 5e-7)])
        snap = engine.snapshot()
        # 1e-6 == threshold -> hidden
        self.assertEqual(snap.nonzeros(), ())
        # but batches retain raw values
        self.assertEqual([(e.row, e.col, e.value) for e in snap.batch("a")],
                         [(0, 0, 5e-7)])
        self.assertEqual([(e.row, e.col, e.value) for e in snap.batch("b")],
                         [(0, 0, 5e-7)])

    def test_replace_reveals_remaining_contribution(self):
        engine = SparseAssembler(1, 1, threshold=1e-6)
        engine.add_batch("big", [(0, 0, 10.0)])
        engine.add_batch("cancel", [(0, 0, -10.0000005)])
        self.assertEqual(engine.nonzeros(), ())  # residual 5e-7 hidden
        snap = engine.replace_batch("cancel", [(0, 0, -3.0)])
        self.assertEqual(entries(snap), [(0, 0, 7.0)])

    def test_revoke_reveals_remaining_contribution(self):
        engine = SparseAssembler(1, 1, threshold=1e-6)
        engine.add_batch("big", [(0, 0, 10.0)])
        engine.add_batch("cancel", [(0, 0, -9.9999995)])
        self.assertEqual(engine.nonzeros(), ())  # residual 5e-7 hidden
        snap = engine.revoke_batch("cancel")
        self.assertEqual(entries(snap), [(0, 0, 10.0)])

    def test_threshold_excludes_hidden_entries_from_matvec(self):
        engine = SparseAssembler(2, 2, threshold=0.5)
        engine.add_batch("a", [(0, 0, 2.0), (1, 1, 0.4)])
        snap = engine.snapshot()
        self.assertEqual(snap.matvec([1.0, 1.0]), (2.0, 0.0))

    def test_zero_threshold_shows_everything_except_exact_zeros(self):
        engine = SparseAssembler(1, 2, threshold=0.0)
        snap = engine.add_batch("a", [(0, 0, 1e-300), (0, 1, 0.0)])
        self.assertEqual([(e.row, e.col) for e in snap.nonzeros()], [(0, 0)])

    def test_invalid_threshold_rejected(self):
        for bad in (-0.1, float("inf"), float("nan"), "0.1", None):
            with self.subTest(bad=bad):
                with self.assertRaises(AssemblyError):
                    SparseAssembler(2, 2, threshold=bad)


if __name__ == "__main__":
    unittest.main()
