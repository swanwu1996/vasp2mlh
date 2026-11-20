import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from ase import Atoms
from ase.io import write

from vasp2mlh.basis import Shell
from vasp2mlh.calibrate import align_ritz, calibrate, write_ordinary
from vasp2mlh.filtering import filter_model
from vasp2mlh.operator import ScalarOperator
from vasp2mlh.overlap import overlap, quadrature_blocks
from vasp2mlh.realspace import matrices_k
from vasp2mlh.sparse import sparsify
from vasp2mlh.vasp import HSQDTM


class RefinementTests(unittest.TestCase):
    def test_two_centre_gaussian_shells(self):
        radius = np.linspace(0, 7, 501)
        radial = np.exp(-(radius**2))
        shells = [Shell(0, radius, radial, "s"), Shell(1, radius, radius * radial, "p")]
        displacement = np.array([0.6, -0.3, 0.7])
        atoms = Atoms(
            "H2", positions=[[0, 0, 0], displacement], cell=[35] * 3, pbc=True
        )
        pairs = dict(quadrature_blocks(atoms, {"H": shells}, 48))
        overlap = np.exp(-np.dot(displacement, displacement) / 2)
        forward = pairs[0, 0, 0, 0, 1]
        self.assertAlmostEqual(forward[0, 0], overlap, places=7)
        np.testing.assert_allclose(
            forward[1:, 0], displacement[[1, 2, 0]] * overlap, atol=1e-8
        )
        np.testing.assert_allclose(forward[0, 1:], -forward[1:, 0], atol=1e-8)
        np.testing.assert_allclose(pairs[0, 0, 0, 0, 0], np.eye(4), atol=1e-8)
        np.testing.assert_array_equal(pairs[0, 0, 0, 1, 0], forward.T)

    def test_metric_transport_and_native_overlap_output(self):
        with tempfile.TemporaryDirectory() as dirname:
            folder = Path(dirname)
            atoms = Atoms("He", cell=[20] * 3, pbc=True)
            write(folder / "POSCAR", atoms, format="vasp")
            radius = np.linspace(0, 4, 101)
            np.savetxt(
                folder / "radial", np.column_stack((radius, np.exp(-(radius**2))))
            )
            spec = {
                "He": [
                    {"source": "table", "file": "radial", "unit": "angstrom", "l": 0}
                ]
            }
            config = folder / "basis.json"
            config.write_text(json.dumps(spec))
            ov = folder / "overlap.h5"
            overlap(folder / "POSCAR", config, ov)
            raw = folder / "raw.h5"
            with h5py.File(ov) as a, h5py.File(raw, "w") as f:
                self.assertEqual(a["entries"].id, a["pairs/S"].id)
                np.testing.assert_array_equal(a["chunk_shapes"][:], [[1, 1]])
                for name in ["structure", "basis"]:
                    a.copy(name, f)
                f["structure/projection_cell"] = atoms.cell.array
                f.attrs.update(
                    hamiltonian_method="direct_scalar_paw",
                    format="vasp2mlh/1",
                    shells=a.attrs["shells"],
                    nelect=2,
                    fermi_eV=0.0,
                )
                f["k/mesh"] = [1, 1, 1]
                f["k/points"] = [[0.0, 0.0, 0.0]]
                f["k/H"] = [[[[8.0 + 0j]]]]
                f["k/S"] = [[[2.0 + 0j]]]
                f["k/eigenvalues"] = [[[3.0]]]
                f["k/projection"] = [[[[np.sqrt(2) + 0j]]]]
            result = folder / "effective.h5"
            calibrate(raw, ov, result)
            with h5py.File(result) as f:
                h, s = matrices_k(f, [[0.15, 0, 0]])
                np.testing.assert_allclose(h[0], 3 * s, atol=1e-12)
                np.testing.assert_allclose(
                    f["calibration/edge_subspace_fraction"][:], 1
                )
                self.assertEqual(f.attrs["metric"], "ordinary")
                self.assertAlmostEqual(s.item().real, 1.0, places=5)

            filtered = folder / "filtered.h5"
            filter_model(raw, ov, filtered, 1.0, 1.0, 1.0, 2.0)
            with h5py.File(filtered) as f:
                h, s = matrices_k(f, [[0.17, 0, 0]])
                np.testing.assert_allclose(h[0], 4 * s, atol=1e-12)
                np.testing.assert_allclose(f["filter/min_edge_weight"][:], 1.0)
                self.assertEqual(f.attrs["hamiltonian_method"], "filtered_paw_ritz")

    def test_energy_shift_on_overlap_support_outside_the_mesh(self):
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
        r = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
        sr = np.array([0.12, 1.0, 0.12])[:, None, None]
        k = np.zeros((1, 3))
        sk = sr.sum(axis=0)[None]
        probe = [[0.37, 0, 0]]
        with tempfile.TemporaryDirectory() as folder:
            stored = []
            for shift in (0, 10):
                prefix = Path(folder) / str(shift)
                dense = prefix.with_suffix(".h5")
                sparse = prefix.with_suffix(".sparse")
                with h5py.File(dense, "w") as f:
                    f.attrs.update(
                        format="vasp2mlh/1",
                        shells=json.dumps({"H": [0]}),
                        spinor=False,
                        spin_channels=1,
                        metric="ordinary",
                        fermi_eV=2.0 + shift,
                    )
                    f["structure/Z"] = [1]
                    f["structure/cell"] = atoms.cell.array
                    f["structure/fractional"] = k
                    f.create_group("basis")
                    f["k/mesh"] = [1, 1, 1]
                    f["k/points"] = k
                    f["k/weights"] = [1.0]
                    for key in (
                        "eigenvalues",
                        "occupation",
                        "gram_error",
                        "projectability",
                    ):
                        f["k/" + key] = np.ones((1, 1, 1))
                    h = (3.7 + shift * sk)[None].astype(complex)
                    f["k/H"] = h
                    f["k/S"] = sk
                    write_ordinary(f, f, atoms, [0, 1], r, sr, k, sk, h)
                    np.testing.assert_allclose(matrices_k(f, k)[0], h)
                    dense_at_probe = matrices_k(f, probe)[0]
                sparsify(dense, sparse, cutoff=0.01, relative_pruning=True)
                with h5py.File(sparse) as f:
                    hp, sp = matrices_k(f, probe)
                    stored.append([dense_at_probe, hp])
            for initial, moved in zip(*stored):
                np.testing.assert_allclose(moved - initial, 10 * sp[None], atol=1e-12)

    def test_scalar_potential_convolution(self):
        op = ScalarOperator.__new__(ScalarOperator)
        op.shape = (9, 3, 3)
        x = 2 * np.pi * np.arange(9) / 9
        op.potential = (2 + np.cos(x))[:, None, None] * np.ones(op.shape)
        op.blocks = {"dij": np.array([[1.75]])}
        q = np.array([[-1.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]])
        p = np.array([[1j], [0.2], [-0.4j]])
        vectors = np.diag([1.0, 1j, -2])
        expected = (
            np.diag([2 + HSQDTM, 2, 2 + HSQDTM])
            + 0.5 * (np.eye(3, k=1) + np.eye(3, k=-1))
            + 1.75 * p @ p.conj().T
        )
        actual = op.apply(q, np.zeros(3), np.eye(3) * 2 * np.pi, p, vectors)
        np.testing.assert_allclose(actual, expected @ vectors, atol=1e-12)

    def test_alignment_covariance_and_reference_energy(self):
        ritz = np.array([1.0, 1.0, 3.0])
        target = np.array([0.0, 0.0, 2.5])
        gen = np.random.default_rng(81)
        overlap = gen.normal(size=(3, 3)) + 1j * gen.normal(size=(3, 3))
        change = np.eye(3, dtype=complex)
        change[:2, :2] = np.linalg.qr(overlap[:2, :2])[0]
        energies, polar, _ = align_ritz(ritz, target, overlap, 3, 1e-6, 1)
        moved, other, _ = align_ritz(ritz + 7, target + 7, overlap @ change, 3, 1e-6, 1)
        np.testing.assert_allclose(other, polar @ change, atol=1e-12)
        np.testing.assert_allclose(
            (other * moved) @ other.conj().T,
            (polar * energies) @ polar.conj().T + 7 * np.eye(3),
            atol=1e-12,
        )
        _, singular, _ = align_ritz(ritz, target, np.zeros_like(overlap), 3, 1e-6, 1)
        self.assertEqual(len(singular), 0)

    def test_sparse_pruning_retains_geometry_overlap(self):
        with tempfile.TemporaryDirectory() as dirname:
            src, dst = (Path(dirname) / s for s in ["dense.h5", "sparse.h5"])
            points = np.array([[0, 0, 0], [0.25, 0, 0]])
            with h5py.File(src, "w") as f:
                f.attrs.update(
                    format="vasp2mlh/1",
                    shells=json.dumps({"H": [0]}),
                    spinor=False,
                    spin_channels=1,
                    metric="ordinary",
                )
                f["structure/Z"] = [1]
                f["structure/cell"] = np.eye(3)
                f["structure/fractional"] = [[0.0, 0.0, 0.0]]
                f.create_group("basis")
                f["real/R"] = [[-1, 0, 0], [0, 0, 0], [1, 0, 0]]
                f["real/H"] = np.array([0.002, 1.0, 0.002]).reshape(1, 3, 1, 1)
                f["real/S"] = np.array([0.1, 1.0, 0.1]).reshape(3, 1, 1)
                f["k/points"] = points
                f["k/mesh"] = [2, 1, 1]
                f["k/weights"] = [0.5, 0.5]
                for key in [
                    "occupation",
                    "eigenvalues",
                    "gram_error",
                    "projectability",
                ]:
                    f["k/" + key] = np.zeros((1, 2, 1))
                h, s = matrices_k(f, points)
                f["k/H"] = h
                f["k/S"] = s
            rep = sparsify(src, dst, h_tol=0.01, max_band_change=0.01)
            self.assertEqual(rep["stored_nonzero_H_blocks"], 1)
            with h5py.File(dst) as f:
                h2, s2 = matrices_k(f, points)
                np.testing.assert_allclose(s2, s, atol=1e-14)
                np.testing.assert_allclose(h2, 1, atol=1e-14)
