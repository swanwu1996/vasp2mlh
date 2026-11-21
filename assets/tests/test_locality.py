import unittest

import numpy as np

from vasp2mlh.locality import complete_subspace, neighbor_faces, retained_bands, retained_counts, WINDOW_FIRST_BUFFER, protected_intervals, window_first_frame


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

    def test_matrix_prior_locks_eigenpairs_and_tracks_energy_origin(self):
        mesh = np.array([2, 1, 1])
        k = np.array([[0, 0, 0], [0.5, 0, 0]])
        pos = np.zeros((3, 3))
        projections = np.array([[[1.0], [0.0], [0.0]]] * 2)
        energies = np.array([[1.0], [2.0]])
        prior = np.array(
            [
                [[0.0, 3.0, 1.0], [3.0, 8.0, 0.5], [1.0, 0.5, 5.0]],
                [[0.0, 0.0, 3.0], [0.0, 9.0, 0.2], [3.0, 0.2, 7.0]],
            ]
        )

        def solve(e, b):
            return complete_subspace(k, mesh, np.eye(3), pos, projections, e, b)[0]

        h = solve(energies, prior)
        np.testing.assert_allclose(
            h[:, :, 0], projections[:, :, 0] * energies, atol=1e-7
        )
        np.testing.assert_allclose(h[:, 1:, 1:], prior[:, 1:, 1:], atol=1e-7)
        np.testing.assert_allclose(
            solve(energies + 7, prior + 7 * np.eye(3)), h + 7 * np.eye(3), atol=1e-7
        )
        np.testing.assert_allclose(
            solve(energies, 10.0),
            solve(energies, np.tile(10 * np.eye(3), (2, 1, 1))),
            atol=1e-7,
        )

    def test_target_polar_frame_and_degenerate_rotation(self):
        rng = np.random.default_rng(51)
        v = rng.normal(size=(1, 7, 5)) + 1j * rng.normal(size=(1, 7, 5))

        def polar(x):
            u, _, vh = np.linalg.svd(x, full_matrices=False)
            return u @ vh

        energies = np.array([[-3.0, 1.0, 1.0, 2.0, 4.0]])
        bounds = protected_intervals(energies, [3, 3], None)
        np.testing.assert_array_equal(bounds, [[1, 3]])
        np.testing.assert_array_equal(
            protected_intervals(energies, None, (0.0, 1.5)), bounds
        )
        np.testing.assert_array_equal(
            protected_intervals(energies, None, (8.0, 9.0)), [[0, 0]]
        )
        q = window_first_frame(v, polar(v), bounds)
        np.testing.assert_allclose(q[:, :, 1:3], polar(v[:, :, 1:3]), atol=1e-12)
        np.testing.assert_allclose(q[0].conj().T @ q[0], np.eye(5), atol=1e-12)
        u = np.eye(5, dtype=complex)
        u[1:3, 1:3] = polar(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        rotated = v @ u
        np.testing.assert_allclose(
            window_first_frame(rotated, polar(rotated), bounds), q @ u, atol=1e-12
        )

    def test_local_buffer_closes_only_the_multiplet_at_its_own_k(self):
        # At k=0 the fourth level is degenerate with level 3. At k=1,
        # levels 4 and 5 form a separate multiplet above the lock.
        energies = np.array(
            [[0.0, 1.0, 2.0, 2.0, 4.0, 6.0], [0.0, 1.0, 2.0, 4.0, 4.0, 6.0]]
        )
        counts = retained_counts(energies, 4, [1, 1], None, WINDOW_FIRST_BUFFER)
        np.testing.assert_array_equal(counts, [4, 3])
        np.testing.assert_array_equal(
            retained_counts(energies, 4, None, 0.5, WINDOW_FIRST_BUFFER), counts
        )
        projections = np.tile(np.eye(4), (2, 1, 1))
        projections[1, :, 3] = 0  # padding outside the retained subspace
        points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        h, weights, _ = complete_subspace(
            points,
            [2, 1, 1],
            np.eye(3),
            np.zeros((4, 3)),
            projections,
            energies,
            10.0,
            counts,
        )
        for i, n in enumerate(counts):
            q = projections[i, :, :n]
            np.testing.assert_allclose(h[i] @ q, q * energies[i, :n], atol=1e-9)
        self.assertGreaterEqual(h[1, 3, 3].real, 2.1)
        np.testing.assert_allclose(weights, 1)
        projections[1, :, 2] = 0
        with self.assertRaisesRegex(ValueError, r"k=\[0.5, 0.0, 0.0\], bands 1-3"):
            complete_subspace(
                points,
                [2, 1, 1],
                np.eye(3),
                np.zeros((4, 3)),
                projections,
                energies,
                10.0,
                counts,
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
