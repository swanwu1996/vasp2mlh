import tempfile
from pathlib import Path
import unittest

import numpy as np
from ase import Atoms
from scipy.linalg import block_diag, eigvalsh

from vasp2mlh.basis import Shell, harmonics
from vasp2mlh.model import project, localized_transform, mesh_indices, bloch, mesh_fourier
from vasp2mlh.vasp import Wavecar, plane_waves, HSQDTM


class ProjectionTests(unittest.TestCase):
    def test_tiled_fourier_matches_direct_sum_on_shifted_shuffled_mesh(self):
        mesh = [3, 4, 2]
        residues = np.indices(mesh).reshape(3, -1).T
        rng = np.random.default_rng(929)
        k = ((residues + [0.2, 0.5, 0.0]) / mesh)[rng.permutation(len(residues))]
        a = rng.normal(size=(2, len(k), 5, 7)) + 1j * rng.normal(size=(2, len(k), 5, 7))
        before = a.copy()
        direct = np.einsum(
            "rk,skij->srij", np.exp(-2j * np.pi * residues @ k.T), a
        ) / len(k)
        np.testing.assert_allclose(
            mesh_fourier(k, mesh, residues, a, work_bytes=1), direct, atol=1e-13
        )
        np.testing.assert_array_equal(a, before)

    def test_metric_and_energy_gauge(self):
        rng = np.random.default_rng(19)
        p = rng.normal(size=(7, 2)) + 1j * rng.normal(size=(7, 2))
        delta = np.diag([0.02, -0.01])
        metric = np.eye(7) + p @ delta @ p.conj().T
        val, vec = np.linalg.eigh(metric)
        c = (vec / np.sqrt(val)) @ vec.conj().T
        phi = rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))
        energies = np.arange(7.0)
        h, s, a, weights, err = project(phi, p, delta, c, energies, False, 0)
        np.testing.assert_allclose(s, phi.conj().T @ metric @ phi)
        np.testing.assert_allclose(a, phi.conj().T @ metric @ c)
        np.testing.assert_allclose(h, a @ np.diag(energies) @ a.conj().T)
        self.assertLess(err, 1e-12)
        self.assertAlmostEqual(weights.sum(), 3)
        h2 = project(phi, p, delta, c[:, :4], energies[:4], False, 9)[0]
        shifted = project(phi, p, delta, c[:, :4], energies[:4] + 2, False, 11)[0]
        np.testing.assert_allclose(shifted, h2 + 2 * s)

    def test_spinor_cross_block(self):
        phi = np.eye(2)
        c = np.array([[1, 1j], [0, 0], [0, 0], [1j, 1]]) / np.sqrt(2)
        h, s, _, _, err = project(
            phi, np.zeros((2, 1)), np.zeros((1, 1)), c, np.array([1.0, 3.0]), True, 4.0
        )
        self.assertLess(err, 1e-12)
        self.assertAlmostEqual(abs(h[0, 3]), 1.0)
        np.testing.assert_allclose(eigvalsh(h, block_diag(s, s)), [1, 3, 4, 4])

    def test_shifted_mesh_and_pair_images(self):
        rng = np.random.default_rng(743)
        mesh = [2, 3, 2]
        k = (np.indices(mesh).reshape(3, -1).T + 0.5) / mesh
        rng.shuffle(k)
        cell = [[2, 1, 0], [0.2, 1.7, 0], [0, 0, 5.0]]
        atoms = Atoms(
            "SiC",
            cell=cell,
            scaled_positions=[[0.12, 0.2, 0], [0.8, 0.6, 0.5]],
            pbc=True,
        )
        a = rng.normal(size=(len(k), 3, 3)) + 1j * rng.normal(size=(len(k), 3, 3))
        s = a.conj().transpose(0, 2, 1) @ a
        h = np.array([block_diag(v, v * 2) for v in s])[None]
        r, hr, sr = localized_transform(k, mesh, atoms, [1, 2], h, s)
        for mr, mk in [(hr, h), (sr, s)]:
            np.testing.assert_allclose(bloch(k, r, mr), mk, atol=1e-12)
            reverse = {tuple(t): i for i, t in enumerate(r)}
            peer = np.take(mr, [reverse[tuple(-t)] for t in r], axis=-3)
            np.testing.assert_allclose(mr, peer.conj().swapaxes(-1, -2), atol=1e-12)
        k[0] = k[1]
        with self.assertRaises(ValueError):
            mesh_indices(k, mesh)

    def test_real_harmonics_and_radial_gaussian(self):
        np.testing.assert_allclose(
            harmonics(1, np.eye(3)),
            np.eye(3)[:, [1, 2, 0]] * np.sqrt(3 / (4 * np.pi)),
            atol=1e-14,
        )
        r = np.linspace(0, 12, 500)
        radial = Shell(0, r, np.exp(-r * r / 2), "Gaussian")
        q = np.array([0, 0.5, 2, 3])
        actual = radial.transform(q)
        np.testing.assert_allclose(actual / actual[0], np.exp(-q * q / 2), rtol=1e-9)

    def test_wavecar_both_precisions_and_truncation(self):
        for tag, dtype in [(45200, "<c8"), (45210, "<c16")]:
            with self.subTest(tag=tag), tempfile.TemporaryDirectory() as d:
                path = Path(d) / "WAVECAR"
                recl = 128
                with path.open("wb") as file:
                    for record, data in enumerate(
                        (
                            [recl, 1, tag],
                            [1, 1, 10, *np.eye(3).ravel(), 2],
                            [2, 0, 0, 0, 1.25, 0, 0.75],
                            [0.2 + 1j, 0.4 - 0.7j],
                        )
                    ):
                        file.seek(record * recl)
                        np.asarray(data, dtype=dtype if record == 3 else "<f8").tofile(
                            file
                        )
                    file.truncate(4 * recl)
                reader = Wavecar(path)
                _, en, c = reader.read(0, 0)
                np.testing.assert_allclose(c[:, 0], [0.2 + 1j, 0.4 - 0.7j])
                np.testing.assert_allclose(en, [[1.25, 0.75]])
                with path.open("r+b") as file:
                    file.truncate(3 * recl)
                with self.assertRaises(ValueError):
                    Wavecar(path)

    def test_legacy_wavecar_header_cannot_extend_into_the_next_record(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "WAVECAR"
            with path.open("wb") as out:
                np.array([128, 1, 45200], dtype="<f8").tofile(out)
                out.seek(128)
                np.array([1, 8, 400, *np.eye(3).ravel(), 0], dtype="<f8").tofile(out)
            with self.assertRaisesRegex(ValueError, "224 bytes in a 128-byte record"):
                Wavecar(path)

    def test_skew_cell_gvector_order(self):
        cell = np.array([[4, 0, 0], [3.9, 0.7, 0], [1.3, 0.6, 3.7]])
        k = np.array([-0.41, 0.42, 0.25])
        # Independent brute-force FFT storage ordering for an oversized grid.
        freq = np.fft.fftfreq(33) * 33
        b = 2 * np.pi * np.linalg.inv(cell).T
        expected = [
            (np.array([x, y, z]) + k) @ b for z in freq for y in freq for x in freq
        ]
        expected = np.array([q for q in expected if q @ q * HSQDTM < 200])
        np.testing.assert_allclose(plane_waves(k, cell, 200), expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
