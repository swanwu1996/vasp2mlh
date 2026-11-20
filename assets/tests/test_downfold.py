import unittest

import numpy as np
from scipy.linalg import eigvalsh

from vasp2mlh.downfold import constrained_matrix


class DownfoldTests(unittest.TestCase):
    def test_metric_subspace_and_energy_reference(self):
        rng = np.random.default_rng(21)
        r = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        s = r @ r.conj().T + np.eye(4)
        l = np.linalg.cholesky(s)
        a = l @ np.diag([0.9, 0.8, 1.0, 1.0])[:, :2]
        e = np.array([-2.0, 2.0])
        h = 6 * s
        out, weight, error = constrained_matrix(h, s, a, e, 0.5)
        self.assertAlmostEqual(weight, 0.64)
        self.assertLess(error, 1e-10)
        np.testing.assert_allclose(eigvalsh(out, s), [-2.0, 2.0, 6.0, 6.0])
        shifted = constrained_matrix(h + 3 * s, s, a, e + 3, 0.5)[0]
        np.testing.assert_allclose(shifted, out + 3 * s)
        with self.assertRaisesRegex(ValueError, "poorly represented"):
            constrained_matrix(h, s, a, e, 0.7)
        with self.assertRaisesRegex(ValueError, "intrudes"):
            constrained_matrix(-4 * s, s, a, e, 0.5)
