import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from vasp2mlh.realspace import matrices_k
from vasp2mlh.sparse import sparsify
from vasp2mlh.vasp import Wavecar


class SparseTests(unittest.TestCase):
    def test_pair_roundtrip_and_rejected_truncation(self):
        rng = np.random.default_rng(14)
        for spinor, channels in ((False, 1), (False, 2), (True, 1)):
            with (
                self.subTest(spinor=spinor, channels=channels),
                tempfile.TemporaryDirectory() as folder,
            ):
                src, out = (Path(folder) / p for p in ("full", "sparse"))
                with h5py.File(src, "w") as f:
                    f.attrs.update(
                        format="vasp2mlh/1",
                        shells=json.dumps({"H": [0], "He": [1]}),
                        spinor=spinor,
                        spin_channels=channels,
                    )
                    f["structure/Z"] = [1, 2]
                    f["structure/cell"] = np.eye(3) * 3
                    f["structure/fractional"] = [[0, 0, 0], [0.2, 0, 0]]
                    f.create_group("basis")
                    k = np.array([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [-0.25, 0, 0]])
                    f["k/points"] = k
                    f["k/mesh"] = [4, 1, 1]
                    f["k/weights"] = np.ones(4) / 4
                    dim = 4 * (1 + spinor)
                    h = rng.normal(size=(channels, dim, dim)) + 1j * rng.normal(
                        size=(channels, dim, dim)
                    )
                    f["real/R"] = [[-1, 0, 0], [0, 0, 0], [1, 0, 0]]
                    f["real/H"] = np.stack(
                        (h.conj().swapaxes(-1, -2), h + h.conj().swapaxes(-1, -2), h),
                        axis=1,
                    )
                    s = np.eye(4) * 0.01
                    f["real/S"] = np.array([s, np.eye(4), s])
                    hh, ss = matrices_k(f, k)
                    f["k/H"], f["k/S"] = hh, ss
                    for name in [
                        "occupation",
                        "eigenvalues",
                        "gram_error",
                        "projectability",
                    ]:
                        f["k/" + name] = np.zeros((channels, len(k), 1))
                report = sparsify(
                    src, out, max_band_change=1e-11, extra_points=[[0.31, 0.1, 0.2]]
                )
                self.assertLess(report["mesh"]["max_delta_H_eV"], 1e-12)
                before = out.read_bytes()
                with self.assertRaisesRegex(ValueError, "band change"):
                    sparsify(src, out, cutoff=1.0, max_band_change=1e-6)
                self.assertEqual(before, out.read_bytes())

    def test_spanning_wavecar(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "WAVECAR"
            for tag, dtype in [(53300, "<c8"), (53310, "<c16")]:
                with p.open("wb") as stream:

                    def record(n, arr, dtype="<f8"):
                        stream.seek(n * 128)
                        np.asarray(arr, dtype=dtype).tofile(stream)

                    record(0, [128, 1, tag])
                    record(1, [2, 8, 10, *np.eye(3).ravel(), 0])
                    for ik in range(2):
                        base = 2 + ik * 10
                        record(base, [2, ik * 0.25, 0, 0, *np.tile([1.5, 0, 0.8], 8)])
                        for n in range(8):
                            record(base + 2 + n, [ik + 1j, n - 0.25j], dtype)
                    stream.truncate(22 * 128)
                k, e, c = Wavecar(p).read(0, 1)
                np.testing.assert_allclose(k, [0.25, 0, 0])
                np.testing.assert_allclose(e[:, 1], 0.8)
                np.testing.assert_allclose(c[:, -1], [1 + 1j, 7 - 0.25j])
