import unittest

import numpy as np

from vasp2mlh.band_release import independent_points
from vasp2mlh.model import spin_rotation


class SpinorTests(unittest.TestCase):
    def test_saxis_rotates_pauli_matrices_to_cartesian(self):
        sx = np.array([[0, 1], [1, 0]])
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.diag([1, -1])
        for axis in ([0, 0, 1], [0, 0, -1], [1, 1, 2], [0, -2, 0]):
            u = spin_rotation(axis)
            n = np.asarray(axis) / np.linalg.norm(axis)
            np.testing.assert_allclose(u @ u.conj().T, np.eye(2), atol=1e-14)
            np.testing.assert_allclose(
                u @ sz @ u.conj().T, n[0] * sx + n[1] * sy + n[2] * sz, atol=1e-14
            )
        with self.assertRaisesRegex(ValueError, "SAXIS"):
            spin_rotation([0, 0, 0])

    def test_magnetic_holdout_keeps_distinct_opposite_points(self):
        ref = np.array([[0.23, 0, 0], [-0.23, 0, 0], [1.23, 0, 0], [0.5, 0, 0]])
        source = [[0, 0, 0], [0.5, 0, 0]]
        np.testing.assert_array_equal(
            independent_points(source, ref, time_reversal=False), [0, 1]
        )
        np.testing.assert_array_equal(independent_points(source, ref), [0])
