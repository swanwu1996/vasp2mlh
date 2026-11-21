import importlib.util
import json
import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import numpy as np
from vasp2mlh.basis import Shell
from vasp2mlh.cli import main
from vasp2mlh.export import deeph, hamgnn, hamgnn_layout
from vasp2mlh.pipeline import file_sha256
from vasp2mlh.readback import load_graph, verify_export
from vasp2mlh.realspace import SPARSE_FORMAT
from vasp2mlh.registry import audit, write_registry


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / "model.h5"
        self.lock = self.root / "basis.h5"
        self.ks = np.array([[0.27, 0, 0], [0, 0, 0]])
        r = np.linspace(0, 3, 51)
        shells = [
            Shell(l, r, r**l * np.exp(-z * r), str(z))
            for l, z in [(0, 1), (0, 2), (1, 1)]
        ]
        paw = {"H": "a" * 64}
        write_registry({"H": shells}, self.lock, "test", paw)
        with h5py.File(self.lock) as lib, h5py.File(self.model, "w") as f:
            lib.copy("basis", f)
            f.attrs.update(
                format=SPARSE_FORMAT,
                shells=json.dumps({"H": [0, 0, 1]}),
                spinor=False,
                soc=False,
                spin_channels=1,
                metric="ordinary",
                hamiltonian_method="filtered_paw_ritz",
                filter_floor=0.005,
                filter_width_eV=1.0,
                filter_taper_eV=2.0,
                fermi_eV=1.0,
                filter_centre_eV=7.0,
                nelect=1.0,
                doping_charge=0.0,
                total_energy_eV=-7.0,
                xc_settings=json.dumps({"GGA": "PE"}),
                paw_sha256_by_element=json.dumps(paw),
                truncation_report=json.dumps(
                    dict(cutoff_A=5.0, h_tol_eV=0.0, s_tol=0.0, relative_pruning=False)
                ),
            )
            f["structure/Z"] = [1]
            f["structure/cell"] = np.eye(3) * 3
            f["structure/fractional"] = [[0.0, 0.0, 0.0]]
            f["pairs/indices"] = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
            f["pairs/offsets"] = [0, 25, 50]
            f["pairs/H"] = np.r_[np.diag(np.arange(5)).ravel(), np.arange(25) * 0.01][
                None, :
            ].astype(complex)
            f["pairs/S"] = np.r_[np.eye(5).ravel(), np.eye(5).ravel() * 0.1].astype(
                complex
            )

    def check(self, fmt):
        with h5py.File(self.model) as model:
            return verify_export(self.root, fmt, model, self.ks, 14)

    def test_hamgnn_selects_common_mask_and_preserves_frozen_shells(self):
        masks = {14: {8: [0], 12: [0, 1]}, 19: {8: [0], 12: [0, 1, 2]}}
        with patch("vasp2mlh.export.upstream_masks", return_value=masks):
            self.assertEqual(hamgnn_layout({"O": [0], "Mg": [0, 0]})[0], 14)
            self.assertEqual(
                hamgnn_layout({"O": [0], "Mg": [0]}, extend=True),
                (14, {"O": [0], "Mg": [0, 0]}),
            )
            with self.assertRaisesRegex(ValueError, "compatible"):
                hamgnn_layout({"Mg": [0]})
            with self.assertRaisesRegex(ValueError, "compatible"):
                hamgnn_layout({"O": [1]}, extend=True)

    def test_hamgnn_output_aliases_preserve_inputs(self):
        for original in (self.model, self.lock):
            before = file_sha256(original)
            alias = self.root / "alias"
            os.link(original, alias)
            for dst in (original, alias):
                with self.assertRaisesRegex(ValueError, "collides"):
                    hamgnn([self.model], dst, registry=self.lock)
                self.assertEqual(file_sha256(dst), before)
            alias.unlink()
            alias.symlink_to(original)
            with self.assertRaisesRegex(ValueError, "collides"):
                hamgnn([self.model], alias, registry=self.lock)
            alias.unlink()
        control = self.root / "control.json"
        control.write_text(
            json.dumps(
                dict(
                    command="hamgnn",
                    parameters=dict(models=["model.h5"], output="config-alias"),
                )
            )
        )
        os.link(control, self.root / "config-alias")
        before = file_sha256(control)
        with self.assertRaises(SystemExit):
            main([str(control)])
        self.assertEqual(file_sha256(control), before)

    def test_label_audit_compares_recipes_and_provenance(self):
        other = self.root / "other.h5"
        shutil.copyfile(self.model, other)
        with h5py.File(other, "r+") as f:
            f.attrs.update(fermi_eV=4.0, filter_centre_eV=10.0)
        self.assertEqual(len(audit([self.model, other], self.lock, labels=True)), 2)
        for name, value, reason in [
            ("filter_floor", 0.05, "filter_floor"),
            ("subspace_bands", 2, "subspace_bands"),
            (
                "truncation_report",
                json.dumps(
                    dict(cutoff_A=6, h_tol_eV=0.0, s_tol=0.0, relative_pruning=False)
                ),
                "cutoff_A",
            ),
            ("paw_sha256_by_element", None, "PAW provenance"),
            ("xc_settings", None, "XC provenance"),
        ]:
            with self.subTest(name=name):
                shutil.copyfile(self.model, other)
                with h5py.File(other, "r+") as f:
                    if value is None:
                        del f.attrs[name]
                    else:
                        f.attrs[name] = value
                with self.assertRaisesRegex(ValueError, reason):
                    audit([self.model, other], self.lock, labels=True)
        with h5py.File(self.model, "r+") as f:
            f.attrs["subspace_bands"] = 3
        shutil.copyfile(self.model, other)
        with h5py.File(other, "r+") as f:
            f.attrs["subspace_bands"] = 2
        with self.assertRaisesRegex(ValueError, "subspace_bands"):
            audit([self.model, other], self.lock, labels=True)

    def test_hubbard_dataset_checks_element_and_missing_identity(self):
        other = self.root / "other.h5"
        settings = dict(
            GGA="PE",
            LDAU="T",
            LDAUTYPE=2,
            HUBBARD_BY_ELEMENT={"H": {"LDAUL": 0, "LDAUU": 2.0, "LDAUJ": 0.0}},
        )
        with h5py.File(self.model, "r+") as f:
            f.attrs["xc_settings"] = json.dumps(settings)
        shutil.copyfile(self.model, other)
        audit([self.model, other], self.lock, labels=True)
        settings["HUBBARD_BY_ELEMENT"]["H"]["LDAUU"] = 1.0
        with h5py.File(other, "r+") as f:
            f.attrs["xc_settings"] = json.dumps(settings)
        with self.assertRaisesRegex(ValueError, "Hubbard"):
            audit([self.model, other], self.lock, labels=True)
        settings.pop("HUBBARD_BY_ELEMENT")
        with h5py.File(other, "r+") as f:
            f.attrs["xc_settings"] = json.dumps(settings)
        with self.assertRaisesRegex(ValueError, "provenance"):
            audit([other], self.lock, labels=True)

    def test_deeph_readback_checks_units_order_and_missing_files(self):
        folder = self.root / "deeph/0"
        deeph(self.model, folder, self.lock)
        self.assertEqual(self.check("deeph")["max_delta_Hk_eV"], 0.0)
        with h5py.File(folder / "hamiltonian.h5", "r+") as f:
            data = f["entries"][:]
            f["entries"][:] = data / 27.2
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            self.check("deeph")
        with h5py.File(folder / "hamiltonian.h5", "r+") as f:
            f["entries"][:] = data
            f["atom_pairs"][1] = f["atom_pairs"][0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.check("deeph")
        (folder / "overlap.h5").unlink()
        (folder / "hamiltonian.h5").unlink()
        with self.assertRaises(FileNotFoundError):
            self.check("deeph")

    def make_complex_spinor(self):
        rng = np.random.default_rng(73)
        z = rng.normal(size=(2, 10, 10)) + 1j * rng.normal(size=(2, 10, 10))
        z[0] = (z[0] + z[0].conj().T) / 2
        with h5py.File(self.model, "r+") as f:
            f.attrs.update(spinor=True, saxis=[0.0, 0.0, 1.0])
            del f["pairs/H"]
            f["pairs/H"] = z.reshape(1, -1)

    def test_spinor_deeph_preserves_complex_spin_flip_blocks(self):
        self.make_complex_spinor()
        deeph(self.model, self.root / "deeph/0", self.lock)
        self.assertEqual(self.check("deeph")["max_delta_Hk_eV"], 0.0)

    @unittest.skipUnless(
        importlib.util.find_spec("DFT_interfaces"), "HamGNN dependencies unavailable"
    )
    def test_spinor_hamgnn_preserves_and_checks_imaginary_spin_flip(self):
        self.make_complex_spinor()
        archive = self.root / "graph_data.npz"
        hamgnn([self.model], archive, 14, self.lock)
        self.assertLess(self.check("hamgnn")["max_delta_Hk_eV"], 1e-6)
        graph = load_graph(archive)
        self.assertGreater(graph.iHoff.abs().max().item(), 0.0)
        graph.iHoff.zero_()
        np.savez_compressed(archive, graph={0: graph})
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            self.check("hamgnn")

    @unittest.skipUnless(
        importlib.util.find_spec("DFT_interfaces"), "HamGNN dependencies unavailable"
    )
    def test_hamgnn_roundtrip_detects_reordered_orbitals_and_reverse_edges(self):
        archive = self.root / "graph_data.npz"
        hamgnn([self.model], archive, 14, self.lock)
        self.assertLess(self.check("hamgnn")["max_delta_Hk_eV"], 1e-6)
        g = load_graph(archive)
        slots = np.arange(14)
        slots[[3, 4]] = slots[[4, 3]]
        h = g.hamiltonian.numpy().reshape(-1, 14, 14)[:, slots][:, :, slots]
        g.hamiltonian = g.hamiltonian.new_tensor(h.reshape(-1, 196))
        g.Hon, g.Hoff = g.hamiltonian[:1], g.hamiltonian[1:]
        np.savez_compressed(archive, graph={0: g})
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            self.check("hamgnn")
        g.inv_edge_idx[0] = 0
        np.savez_compressed(archive, graph={0: g})
        with self.assertRaisesRegex(ValueError, "reverse edges"):
            self.check("hamgnn")
