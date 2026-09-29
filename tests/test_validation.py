import unittest

from sparse_assembly import Add, AssemblyError, SparseAssembler


class ValidationTest(unittest.TestCase):
    def test_dimensions_must_be_positive_integers(self):
        for nrows, ncols in [
            (0, 2), (2, 0), (-1, 2), (2, -1),
            (1.5, 2), (2, 1.5), ("2", 2), (2, "2"),
            (True, 2), (None, 2),
        ]:
            with self.subTest(nrows=nrows, ncols=ncols):
                with self.assertRaises(AssemblyError):
                    SparseAssembler(nrows, ncols)

    def test_dimensions_are_fixed_after_creation(self):
        engine = SparseAssembler(3, 4)
        self.assertEqual((engine.nrows, engine.ncols), (3, 4))
        with self.assertRaises(AttributeError):
            engine.nrows = 5  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            engine.ncols = 5  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            engine.threshold = 1.0  # type: ignore[misc]

    def test_indices_must_be_non_negative_ints(self):
        engine = SparseAssembler(2, 2)
        for bad_row in (1.0, True, "1", -1):
            with self.subTest(bad_row=bad_row):
                with self.assertRaises(AssemblyError):
                    engine.add_batch("x", [(bad_row, 0, 1.0)])
        for bad_col in (1.0, True, "1", -1):
            with self.subTest(bad_col=bad_col):
                with self.assertRaises(AssemblyError):
                    engine.add_batch("y", [(0, bad_col, 1.0)])

    def test_value_must_be_finite_number(self):
        engine = SparseAssembler(2, 2)
        for bad in ("1.0", None, [1.0], float("inf"), float("-inf"), float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises(AssemblyError):
                    engine.add_batch(f"v-{bad!r}", [(0, 0, bad)])

    def test_entries_must_be_iterable_of_triples(self):
        engine = SparseAssembler(2, 2)
        for bad in (7, "abc", None, [1, 2, 3], [(0, 0)], [(0, 0, 1, 2)]):
            with self.subTest(bad=bad):
                with self.assertRaises(AssemblyError):
                    engine.add_batch("shape", bad)

    def test_batch_id_validation_does_not_mutate_state(self):
        engine = SparseAssembler(2, 2)
        with self.assertRaises(AssemblyError):
            engine.transaction([Add(123, [(0, 0, 1.0)])])  # type: ignore[arg-type]
        self.assertEqual(engine.version, 0)
        self.assertEqual(engine.snapshot().batch_ids, ())

    def test_boolean_index_but_numeric_value_rules(self):
        engine = SparseAssembler(2, 2)
        with self.assertRaises(AssemblyError):
            engine.add_batch("b", [(True, 0, 1.0)])
        snap = engine.add_batch("ok", [(0, 0, True)])  # bool value accepted as 1.0
        self.assertEqual(
            [(e.row, e.col, e.value) for e in snap.batch("ok")],
            [(0, 0, 1.0)],
        )


if __name__ == "__main__":
    unittest.main()
