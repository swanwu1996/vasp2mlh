import unittest

import numpy as np
from scipy.special import expit

from vasp2mlh.basis import Shell
from vasp2mlh.calibrate import metric_factors
from vasp2mlh.filtering import filtered_operator
from vasp2mlh.spillage import coverage, diffuse_shells, radial_map


class FilteringTests(unittest.TestCase):
    def test_full_space_ritz_complement_and_energy_zero(self):
        rng = np.random.default_rng(73)
        trial = np.linalg.qr(rng.normal(size=(9, 4)) + 1j * rng.normal(size=(9, 4)))[0]
        energy = np.array([-5, -2, -2, 0, 1, 3, 6, 9, 15])
        stored = 6
        v = trial[:stored].conj().T
        direct = trial.conj().T @ (energy[:, None] * trial)
        centre, width, floor, ceiling, taper = 0.5, 1.3, 0.006, 3, 2

        def project(h, p, e, shift=0):
            return filtered_operator(
                h, p, e, centre + shift, width, floor, ceiling + shift, taper
            )

        result = project(direct, v, energy[:stored])
        x = np.clip((ceiling - energy) / taper, 0, 1)
        psi = (
            np.sqrt(
                floor
                + (1 - floor) * expit((centre - energy) / width) * x**2 * (3 - 2 * x)
            )[:, None]
            * trial
        )
        _, inv = metric_factors(psi.conj().T @ psi)
        q = psi @ inv
        np.testing.assert_allclose(
            result, q.conj().T @ (energy[:, None] * q), atol=1e-12
        )
        shift = 31
        np.testing.assert_allclose(
            project(direct + shift * np.eye(4), v, energy[:stored] + shift, shift),
            result + shift * np.eye(4),
            atol=2e-12,
        )
        rotated = v.copy()
        rotated[:, 1:3] = v[:, 1:3] @ np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
        np.testing.assert_allclose(
            project(direct, rotated, energy[:stored]), result, atol=1e-12
        )
        np.testing.assert_allclose(
            filtered_operator(direct, v, energy[:stored], 0, 1, 1, 3, 2),
            direct,
            atol=1e-12,
        )

    def test_fixed_shell_contraction_and_weakest_direction(self):
        radius = np.linspace(0, 5, 201)
        shells = [
            Shell(0, radius, np.exp(-radius), "s"),
            Shell(1, radius, radius * np.exp(-radius), "p"),
        ]
        extended = diffuse_shells(shells, [1], [1.6], radius, 3.5)
        transform = radial_map(extended, len(shells), [1], [(2, np.pi / 2)])
        np.testing.assert_allclose(transform.T @ transform, np.eye(4), atol=1e-15)
        np.testing.assert_allclose(transform[-3:, 1:], np.eye(3))
        projection = np.zeros((7, 2))
        projection[0, 0] = 1
        projection[-1, 1] = 0.9
        samples = [(np.eye(7), projection, 1, slice(0, 2))]
        self.assertAlmostEqual(coverage(samples, transform, 100)["worst_edge"], 0.81)
        self.assertAlmostEqual(coverage(samples, np.eye(7, 4), 100)["worst_edge"], 0)
        transform[:, -1] = 0
        self.assertIsNone(coverage(samples, transform, 100))
