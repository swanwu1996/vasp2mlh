import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from ase import Atoms
from ase.io import write
from scipy.integrate import simpson

from vasp2mlh.atomic import generate
from vasp2mlh.cli import genbasis_main
from vasp2mlh.basis import Shell, load_basis
from vasp2mlh.overlap import overlap, quadrature_blocks
from vasp2mlh.registry import audit, fingerprint, freeze, read_registry


class RegistryTests(unittest.TestCase):
    def test_frozen_basis_is_portable_and_identifies_radial_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            config = d / "basis.json"
            lock = d / "basis.h5"
            pos = d / "POSCAR"
            r = np.linspace(0, 4, 151)
            np.savetxt(d / "ao.dat", np.c_[r, np.exp(-(r**2)), r * np.exp(-(r**2))])
            shell = {"l": 0, "source": "table", "file": "ao.dat", "unit": "angstrom"}
            config.write_text(
                json.dumps({"H": [shell], "He": [shell, dict(shell, l=1, zeta=1)]})
            )
            freeze(config, lock)
            write(
                pos,
                Atoms("He", positions=[[0, 0, 0]], cell=[12] * 3, pbc=True),
                format="vasp",
            )
            sample = d / "sample.h5"
            overlap(pos, lock, sample)
            info = audit([sample], lock)[0]
            self.assertEqual(info["spatial_dimension"], 4)
            standard = load_basis(config, {})["He"]
            self.assertEqual(info["sha256_by_element"]["He"], fingerprint(standard))
            # The same dimensions and element names can conceal a different radial function.
            np.savetxt(d / "ao.dat", np.c_[r, np.exp(-2 * r**2), r * np.exp(-(r**2))])
            other = d / "other.h5"
            freeze(config, other)
            with self.assertRaisesRegex(ValueError, "frozen basis for He"):
                audit([sample], other)
            (d / "ao.dat").unlink()
            config.unlink()
            np.testing.assert_array_equal(
                load_basis(lock, {}, ["He"])["He"][0].radial, standard[0].radial
            )
            with self.assertRaisesRegex(ValueError, "no definition for Li"):
                load_basis(lock, {}, ["Li"])
            with h5py.File(sample, "a") as f:
                f["pairs/offsets"][-1] -= 1
            with self.assertRaisesRegex(ValueError, "Pair dimensions"):
                audit([sample], lock)
            with h5py.File(lock, "a") as f:
                f["basis/He"].attrs["sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_basis(lock, {}, ["He"])

    def test_genbasis_defaults_and_explicit_configuration(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("vasp2mlh.atomic.generate") as generate,
        ):
            old = Path.cwd()
            self.addCleanup(os.chdir, old)
            os.chdir(temp)
            cfg = Path("basis.toml")
            cfg.write_text('command="convert"')
            genbasis_main([])
            generate.assert_called_once_with("basis.h5", ["POTCAR"], None)
            with self.assertRaises(SystemExit):
                genbasis_main(["basis.toml"])
            with self.assertRaises(SystemExit):
                genbasis_main(["missing.toml"])
            folder = Path("subdir")
            folder.mkdir()
            cfg = folder / "basis.toml"
            calls = []
            generate.side_effect = lambda *args: calls.append((Path.cwd(), args))
            cfg.write_text('[parameters]\nlayout="layout.toml"')
            genbasis_main([str(cfg)])
            self.assertEqual(
                calls, [(folder.resolve(), ("basis.h5", ["POTCAR"], "layout.toml"))]
            )
            cfg.write_text('[parameters]\noutput="basis.toml"')
            with self.assertRaises(SystemExit):
                genbasis_main([str(cfg)])
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(Path.cwd(), Path(temp))

    def test_matched_paw_libraries_and_polarization(self):
        r = np.linspace(0.001, 4, 81)
        pot = SimpleNamespace(
            element="He",
            ls=[1, 0],
            grid=r,
            sha256="reference",
            waves=[r**2 * np.exp(-r), r * np.exp(-2 * r)],
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("vasp2mlh.atomic.read_potcar", return_value={"He": pot}),
        ):
            d = Path(temp)
            raw = d / "native.h5"
            standard = d / "compact.h5"
            recipe = d / "layout.json"
            recipe.write_text(
                json.dumps({"He": {"cutoff_A": 3.0, "multiplicities": [2, 1, 1]}})
            )
            generate(raw, ["pot"])
            generate(standard, ["pot"], recipe)
            self.assertEqual([s.l for s in read_registry(raw)["He"]], [0, 1])
            actual = load_basis(standard, {"He": pot}, ["He"])["He"]
            self.assertEqual([s.l for s in actual], [0, 0, 1, 2])
            self.assertTrue(all(s.radial[-1] == 0 for s in actual))
            r = actual[0].radius
            gram = np.array(
                [
                    [simpson(r * r * a.radial * b.radial, x=r) for a in actual[:2]]
                    for b in actual[:2]
                ]
            )
            np.testing.assert_allclose(gram, np.eye(2), atol=1e-10)
            # Identity of the AO includes radial signs and shell order.
            self.assertNotEqual(
                fingerprint(actual), fingerprint([actual[1], actual[0], *actual[2:]])
            )
            generate(d / "again.h5", ["pot"], recipe)
            self.assertEqual(
                fingerprint(actual), fingerprint(read_registry(d / "again.h5")["He"])
            )
            with self.assertRaisesRegex(ValueError, "PAW reference differs"):
                load_basis(standard, {"He": SimpleNamespace(sha256="other")}, ["He"])

    def test_onsite_resolves_sharp_paw_radial_feature(self):
        radius = np.linspace(0, 3, 601)
        radial = np.exp(-(((radius - 1.123) / 0.018) ** 2))
        shell = Shell(0, radius, radial, "narrow")
        atoms = Atoms("H", cell=[10] * 3, pbc=True)
        pairs = dict(quadrature_blocks(atoms, {"H": [shell]}, 12))
        np.testing.assert_allclose(pairs[(0, 0, 0, 0, 0)], [[1]], atol=1e-12)
