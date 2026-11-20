import unittest

import numpy as np

from vasp2mlh.locality import complete_subspace, neighbor_faces, retained_bands


class LocalityTests(unittest.TestCase):
    def test_voronoi_weights_for_linear_hexagonal_and_cubic_meshes(self):
        hexagon = np.array([[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [0, 0, 4.0]])
        for cell, mesh, expected in [
            (np.eye(3), [1, 1, 1], 0),
            (np.eye(3), [1, 3, 1], 2),
            (hexagon, [3, 3, 1], 6),
            (np.eye(3), [2, 2, 2], 6),
        ]:
            with self.subTest(mesh=mesh):
                faces = neighbor_faces(cell, mesh)
                self.assertEqual(len(faces), expected)
                for _, weight in faces:
                    self.assertAlmostEqual(weight, 1 / expected)
        skew = np.array([[3.0, -1.0, 2.0], [1.1, 3.0, -0.5], [0.0, 1.0, 6.0]])
        faces = {tuple(s): w for s, w in neighbor_faces(skew, [2, 3, 4])}
        self.assertAlmostEqual(sum(faces.values()), 1)
        for shift, weight in faces.items():
            self.assertGreater(weight, 0)
            self.assertAlmostEqual(weight, faces[tuple(-np.array(shift))])

    def test_polar_eigenpairs_degeneracy_energy_zero_and_cell_representatives(self):
        rng = np.random.default_rng(37)
        mesh = np.array([3, 1, 1])
        k = np.array([[1 / 6, 0, 0], [0.5, 0, 0], [-1 / 6, 0, 0]])
        pos = rng.random((4, 3))
        v = 0.21 * (rng.normal(size=(3, 4, 2)) + 1j * rng.normal(size=(3, 4, 2)))
        e = np.array([[-2.0, -0.4], [-1.0, -1.0], [-2.0, -1.5]])

        def solve(x, energies, prior=12, positions=pos):
            return complete_subspace(k, mesh, np.eye(3), positions, x, energies, prior)[
                0
            ]

        h = solve(v, e)
        u, _, vh = np.linalg.svd(v, full_matrices=False)
        q = u @ vh
        np.testing.assert_allclose(h @ q, q * e[:, None, :], atol=1e-10)
        self.assertTrue(np.all(np.linalg.eigvalsh(h)[:, 2] >= e[:, -1] + 0.1 - 1e-8))
        np.testing.assert_allclose(solve(v, e + 5, 17), h + 5 * np.eye(4), atol=2e-7)
        rotated = v.copy()
        rotated[1] = rotated[1] @ (np.array([[1, 1j], [1j, 1]]) / np.sqrt(2))
        np.testing.assert_allclose(solve(rotated, e), h, atol=2e-7)
        shift = np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0]])
        phase = np.exp(2j * np.pi * (k @ shift.T))
        expected = phase[:, :, None] * h * phase.conj()[:, None, :]
        np.testing.assert_allclose(
            solve(v * phase[:, :, None], e, positions=pos + shift), expected, atol=2e-7
        )

    def test_degenerate_buffer_and_rank_checks(self):
        en = np.array([[-2.0, 0.5, 1.0, 1.0, 5.0], [-1.9, 0.5, 1.0, 1.3, 5.0]])
        for selected, upper in [((1, 2), None), (None, 0.6)]:
            self.assertEqual(retained_bands(en, 4, selected, upper), 4)
            with self.assertRaisesRegex(ValueError, "AO dimension or NBANDS"):
                retained_bands(en, 3, selected, upper)
        with self.assertRaisesRegex(ValueError, "Rank-deficient"):
            complete_subspace(
                np.zeros((1, 3)),
                [1, 1, 1],
                np.eye(3),
                np.zeros((2, 3)),
                np.zeros((1, 2, 1)),
                np.ones((1, 1)),
                10.0,
            )
