import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from ase import Atoms

from vasp2mlh.comparison import compare_path


class ComparisonTests(unittest.TestCase):
    def test_shared_points_energy_offsets_and_unique_holdout(self):
        atoms = Atoms("He", positions=[[0, 0, 0]], cell=np.eye(3), pbc=True)
        k = np.array([[0, 0, 0], [0.25, 0, 0], [1.25, 0, 0], [0.5, 0, 0]])
        native = np.array([[0.0, 1.0], [0.1, 1.2], [0.1, 1.2], [0.3, 1.0]])
        ref = {
            "atoms": atoms,
            "icharg": 11,
            "spinor": False,
            "nspin": 1,
            "encut": 100.0,
            "nelect": 2.0,
            "kpoints": k,
            "eig": native[None],
        }
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "model.h5"
            with h5py.File(model, "w") as f:
                f["structure/Z"] = atoms.numbers
                f["structure/cell"] = atoms.cell
                f["structure/fractional"] = [[0, 0, 0]]
                f["k/points"] = [[0.0, 0.0, 0.0]]
                f["k/eigenvalues"] = native[None, :1]
                f.attrs.update(
                    nelect=2.0, encut_eV=100.0, spinor=False, spin_channels=1
                )
            with (
                patch("vasp2mlh.comparison.metadata", return_value=ref),
                patch(
                    "vasp2mlh.comparison.openmx_path",
                    return_value=(k, native + 4, [[0, "A"], [3, "B"]]),
                ),
                patch("vasp2mlh.comparison.bands", return_value=native[None] + 0.5),
            ):
                d = compare_path(model, Path(folder) / "path.xml", "unused", 1)
                self.assertEqual(d["scored"], [False, True, False, True])
                self.assertAlmostEqual(d["errors"]["model_vasp"]["mae_eV"], 0.5)
                self.assertAlmostEqual(d["errors"]["openmx_vasp"]["mae_eV"], 0)
                self.assertAlmostEqual(d["shared_native_drift_eV"], 0)
