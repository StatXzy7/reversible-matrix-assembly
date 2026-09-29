import math
import unittest

from sparse_assembly import Add, AssemblyError, SparseAssembler


def dense_reference(batches, nrows, ncols):
    matrix = [[0.0] * ncols for _ in range(nrows)]
    for triples in batches:
        for row, col, value in triples:
            matrix[row][col] += value
    return matrix


class MatvecTest(unittest.TestCase):
    def test_matvec_matches_dense_reference(self):
        nrows, ncols = 3, 4
        batches = [
            [(0, 0, 2.0), (0, 3, -1.5), (2, 1, 0.25)],
            [(0, 0, 1.0), (1, 2, 3.0), (2, 1, 0.75)],
            [(2, 3, -4.0), (1, 0, 0.5)],
        ]
        engine = SparseAssembler(nrows, ncols)
        for index, triples in enumerate(batches):
            engine.add_batch(f"batch-{index}", triples)

        vector = [1.0, -2.0, 3.0, 0.5]
        reference = dense_reference(batches, nrows, ncols)
        expected = tuple(
            sum(reference[row][col] * vector[col] for col in range(ncols))
            for row in range(nrows)
        )
        result = engine.matvec(vector)
        self.assertEqual(len(result), nrows)
        for got, want in zip(result, expected):
            self.assertTrue(math.isclose(got, want, abs_tol=1e-12), (got, want))
        # Explicit dense cross-check of a couple of cells.
        self.assertEqual(reference[0][0], 3.0)
        self.assertAlmostEqual(result[2], 0.25 * -2 + 0.75 * -2 + -4.0 * 0.5)

    def test_matvec_via_snapshot(self):
        engine = SparseAssembler(2, 2)
        snap = engine.add_batch("a", [(0, 0, 1.0), (0, 1, 2.0),
                                      (1, 0, 3.0), (1, 1, 4.0)])
        self.assertEqual(snap.matvec([1.0, 1.0]), (3.0, 7.0))
        self.assertEqual(engine.matvec([2.0, 0.0]), (2.0, 6.0))

    def test_matvec_zero_matrix(self):
        engine = SparseAssembler(3, 2)
        self.assertEqual(engine.matvec([5.0, -9.0]), (0.0, 0.0, 0.0))

    def test_vector_length_mismatch_raises_and_keeps_state(self):
        engine = SparseAssembler(2, 3)
        engine.add_batch("a", [(0, 0, 1.0)])
        with self.assertRaises(AssemblyError) as ctx:
            engine.matvec([1.0, 2.0])
        self.assertIn("does not match", str(ctx.exception))
        # state untouched
        self.assertEqual(engine.version, 1)
        self.assertEqual(engine.snapshot().batch_ids, ("a",))

    def test_non_finite_vector_raises(self):
        engine = SparseAssembler(1, 2)
        with self.assertRaises(AssemblyError):
            engine.matvec([1.0, float("inf")])
        with self.assertRaises(AssemblyError):
            engine.matvec([1.0, float("nan")])

    def test_invalid_vector_raises(self):
        engine = SparseAssembler(1, 2)
        with self.assertRaises(AssemblyError):
            engine.matvec([1.0, "x"])
        with self.assertRaises(AssemblyError):
            engine.matvec(7)


if __name__ == "__main__":
    unittest.main()
