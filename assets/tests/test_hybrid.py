import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from xml.etree import ElementTree as ET

import numpy as np

from vasp2mlh.band_release import check_convergence, check_hybrid, reference_points, resolve_pbe_run, check_pbe_exchange
from vasp2mlh.filtering import filtered_operator, spectral_baseline
from vasp2mlh.model import mesh_indices
from vasp2mlh.vasp import effective_hybrid_flag, screened_hybrid_maturity, source_fermi, split_mesh, xc_parameters


class HybridTests(unittest.TestCase):
    def test_hubbard_species_echo_and_effective_flag(self):
        from vasp2mlh.vasp import hubbard_parameters, check_hubbard

        root = ET.fromstring("""<modeling><incar><v name="LDAUL">-1 2</v>
          <v name="LDAUU">0 5</v><v name="LDAUJ">0 1</v></incar><parameters>
          <i name="LDAU">T</i><v name="LDAUTYPE">2</v><v name="LDAUL">-1</v>
          <v name="LDAUU">0</v><v name="LDAUJ">0</v></parameters></modeling>""")
        params = root.find("parameters")

        def read():
            return hubbard_parameters(root, params, ["O", "Ni"])

        settings = read()
        self.assertEqual(
            settings["HUBBARD_BY_ELEMENT"]["Ni"],
            {"LDAUL": 2, "LDAUU": 5.0, "LDAUJ": 1.0},
        )
        with self.assertRaisesRegex(ValueError, "LMAXMIX"):
            check_hubbard(dict(operator_parameters={**settings, "LMAXMIX": "2"}))
        with self.assertRaisesRegex(ValueError, "ground-state"):
            check_hubbard(dict(operator_parameters={**settings, "LDAUTYPE": 3}))
        params.find("v[@name='LDAUU']").text = "4"
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            read()
        root.remove(root.find("incar"))
        with self.assertRaisesRegex(ValueError, "input echo"):
            read()
        params.find("i[@name='LDAU']").text = "F"
        self.assertEqual(read(), {"LDAU": "F"})

    def test_screened_hybrid_maturity(self):
        xc = {"LHFCALC": "T", "HFSCREEN": "0.2"}
        for spinor in (True, False):
            self.assertEqual(
                screened_hybrid_maturity(dict(operator_parameters=xc, spinor=spinor)),
                {"screened_hybrid_maturity": "beta"},
            )
        for settings in ({}, {**xc, "LHFCALC": "F"}, {**xc, "HFSCREEN": "0"}):
            self.assertEqual(screened_hybrid_maturity(dict(operator_parameters=settings)), {})

    def test_only_resolved_exchange_flag_identifies_source(self):
        root = ET.fromstring(
            '<root><incar><i name="LHFCALC">T</i></incar>'
            '<parameters><i name="LHFCALC">.false.</i></parameters></root>'
        )
        params = root.find("parameters")
        self.assertEqual(effective_hybrid_flag(params), "F")
        extra = ET.SubElement(params, "i", name="LHFCALC")
        extra.text = "T"
        with self.assertRaisesRegex(ValueError, "LHFCALC"):
            effective_hybrid_flag(params)
        params.clear()
        with self.assertRaisesRegex(ValueError, "LHFCALC"):
            effective_hybrid_flag(params)

    def test_anchor_cannot_override_functional_detection(self):
        with TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "pbe").mkdir()
            (run / "pbe/vasprun.xml").touch()
            hse = dict(
                operator_parameters={"LHFCALC": "T", "HFSCREEN": "0.2"},
                spinor=False,
                soc=False,
            )
            for flag, screen in (("F", "0.2"), ("T", "0")):
                other = {
                    **hse,
                    "operator_parameters": {"LHFCALC": flag, "HFSCREEN": screen},
                }
                self.assertIsNone(resolve_pbe_run(other, run, None, None))
                with self.assertRaisesRegex(ValueError, "require"):
                    resolve_pbe_run(other, run, run / "pbe", None)
            self.assertEqual(resolve_pbe_run(hse, run, None, None), run / "pbe")
            for legacy in ("locality-constrained", "spectral-ritz"):
                with self.assertRaisesRegex(ValueError, "retired"):
                    resolve_pbe_run(hse, run, None, legacy)
            with self.assertRaisesRegex(ValueError, "spinors"):
                resolve_pbe_run({**hse, "spinor": True}, run, None, None)
            with self.assertRaisesRegex(ValueError, "pbe_run"):
                resolve_pbe_run(hse, run, run / "missing", None)

    def test_auto_reference_requires_one_compatible_candidate(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "hse"
            run.mkdir()
            candidates = (run / "pbe", root / "pbe-dense")
            for folder in candidates:
                folder.mkdir()
                (folder / "vasprun.xml").touch()
            src = dict(
                operator_parameters={"LHFCALC": "T", "HFSCREEN": "0.2"},
                spinor=False,
                soc=False,
            )

            def check(src, run, candidate):
                if candidate == candidates[0]:
                    raise ValueError("coarse mesh")

            with patch("vasp2mlh.band_release.check_pbe_pair", side_effect=check):
                self.assertEqual(resolve_pbe_run(src, run, None, "auto"), candidates[1])
            with patch("vasp2mlh.band_release.check_pbe_pair"):
                with self.assertRaisesRegex(ValueError, "Several compatible"):
                    resolve_pbe_run(src, run, None, None)
                self.assertEqual(
                    resolve_pbe_run(src, run, candidates[1], None), candidates[1]
                )

    def test_pbe_reference_identity(self):
        with TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "POTCAR").write_text(" LEXCH = PE\n LEXCH = PE\n")
            check_pbe_exchange({"operator_parameters": {}}, run, screened=False)
            for settings in (
                {"GGA": "PS"},
                {"GGA": "CA"},
                {"GGA": "PE", "AGGAX": "0.75"},
                {"GGA": "PE", "AEXX": "0.25"},
                {"GGA": "PE", "LRHFCALC": "T"},
            ):
                with self.assertRaisesRegex(ValueError, "PBE"):
                    check_pbe_exchange(
                        {"operator_parameters": settings}, run, screened=False
                    )
            (run / "POTCAR").write_text(" LEXCH = PE\n LEXCH = CA\n")
            with self.assertRaises(ValueError):
                check_pbe_exchange({"operator_parameters":{}},run,screened=False)

    def test_exchange_settings_and_scf(self):
        params = dict(LHFCALC="T", AEXX="0.25", HFSCREEN="0.2", HFRCUT="-1", GGA="PE")
        meta = dict(
            nspin=1,
            spinor=False,
            soc=False,
            lreal="F",
            icharg=2,
            operator_parameters=params,
        )
        check_hybrid(meta)
        self.assertEqual(xc_parameters(params), params)
        self.assertNotEqual(
            xc_parameters(params), xc_parameters({**params, "HFSCREEN": "0.0"})
        )
        with self.assertRaisesRegex(ValueError, "self-consistent"):
            check_hybrid({**meta, "icharg": 11})
        with self.assertRaisesRegex(ValueError, "HFSCREEN"):
            check_hybrid({**meta, "operator_parameters": {**params, "HFSCREEN": "nan"}})

    def test_zero_weight_interleaved_reference_exclusion_and_identity(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "vasprun.xml").write_text("hybrid run A")
            src = dict(
                kpoints=np.array([[0, 0, 0], [0.17, 0, 0], [0.5, 0, 0], [-0.17, 0, 0]]),
                weights=np.array([0.5, 0, 0.5, 0]),
                eig=np.array(
                    [[[1.0, 2.0], [200.0, 300.0], [3.0, 4.0], [205.0, 303.0]]]
                ),
                mesh=None,
                operator_parameters={"LHFCALC": "T"},
                xml_path=root / "vasprun.xml",
            )
            split_mesh(src, None)
            self.assertEqual(src["mesh"], [2, 1, 1])
            np.testing.assert_array_equal(src["wave_indices"], [0, 2])
            np.testing.assert_array_equal(src["eig"], [[[1.0, 2.0], [3.0, 4.0]]])
            mesh_indices(src["kpoints"], src["mesh"])
            ref, selected = reference_points(src, root)
            np.testing.assert_array_equal(selected, [0])
            self.assertEqual(ref["source"], "zero-weight")
            other = root / "other"
            other.mkdir()
            (other / "vasprun.xml").write_text("hybrid run B")
            with self.assertRaisesRegex(ValueError, "same SCF"):
                reference_points(src, other)

    def test_fermi_uses_the_integration_grid_in_either_xml_order(self):
        scf = "<dos><i name='efermi'>-3.8</i></dos>"
        holdout = "<dos comment='kpoints_opt'><i name='efermi'>-4.1</i></dos>"
        for sections in (holdout + scf, scf + holdout, scf):
            self.assertEqual(
                source_fermi(
                    ET.fromstring("<calculation>" + sections + "</calculation>")
                ),
                -3.8,
            )
        for invalid in (holdout, scf + scf):
            with self.assertRaisesRegex(ValueError, "source-grid Fermi"):
                source_fermi(
                    ET.fromstring("<calculation>" + invalid + "</calculation>")
                )

    def test_opt_points_are_only_used_for_validation(self):
        optional = ET.fromstring(
            "<eigenvalues_kpoints_opt><kpoints><varray name='kpointlist'><v>.2 .3 .0</v></varray></kpoints><eigenvalues><array><set><set><set><r>8.5</r><r>9.6</r></set></set></set></array></eigenvalues></eigenvalues_kpoints_opt>"
        )
        src = dict(
            weights=np.array([0.0, 1.0]),
            kpoints=np.array([[0.41, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            eig=np.array([[[5.1, 7.2], [1.0, 3.0]]]),
            mesh=[1, 1, 1],
        )
        split_mesh(src, optional)
        np.testing.assert_array_equal(src["eig"], [[[1.0, 3.0]]])
        np.testing.assert_array_equal(src["holdout"]["eig"], [[[8.5, 9.6]]])
        np.testing.assert_array_equal(src["zero_weight"]["eig"], [[[5.1, 7.2]]])
        np.testing.assert_array_equal(src["zero_weight"]["wave_indices"], [0])
        np.testing.assert_array_equal(src["wave_indices"], [1])

    def test_exact_hybrid_eigenpairs_and_energy_origin(self):
        rng = np.random.default_rng(612)
        v = np.linalg.qr(rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4)))[0]
        e = np.array([-4.0, -3.0, 1.0, 4.0])

        def reconstruct(energies):
            ceiling = energies[-1] + 2
            eta = ceiling + 10
            h = spectral_baseline(v, energies, ceiling, 2.0, eta)
            return filtered_operator(
                h, v, energies, energies[1] + 6, 1.0, 0.0001, ceiling, 2.0
            )

        h = reconstruct(e)
        np.testing.assert_allclose(np.linalg.eigvalsh(h), [*e, 16.0], atol=1e-10)
        np.testing.assert_allclose(reconstruct(e + 8.2), h + 8.2 * np.eye(5), atol=1e-9)
        delta = np.array([0.0, 0.0, 0.7, 0.0])
        np.testing.assert_allclose(
            reconstruct(e + delta) - h, (v * delta) @ v.conj().T, atol=1e-9
        )

    def test_high_bands_do_not_change_hybrid_completion(self):
        e = np.array([-2.0, 1.0, 13.0])
        v = np.array([[0.9, 0.2, 0.1], [0.1, 0.7, 0.1]])
        a = spectral_baseline(v[:, :2], e[:2], 10.0, 2.0, 20.0)
        np.testing.assert_allclose(spectral_baseline(v, e, 10.0, 2.0, 20.0), a)

    def test_degenerate_bands_are_invariant_under_complex_rotation(self):
        rng = np.random.default_rng(418)
        v = 0.1 * (rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4)))
        energies = np.array([-2.0, 1.0, 1.0, 12.0])
        rotation = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[
            0
        ]
        rotated = v.copy()
        rotated[:, 1:3] = v[:, 1:3] @ rotation

        def h(p):
            return filtered_operator(
                spectral_baseline(p, energies, 10.0, 2.0, 20.0),
                p,
                energies,
                6.0,
                1.0,
                0.0001,
                10.0,
                2.0,
            )

        np.testing.assert_allclose(h(v), h(rotated), atol=1e-10)

    def test_stopcar_cannot_certify_an_unconverged_exchange_state(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            meta = dict(
                nelm=100,
                final_scf_steps=6,
                operator_parameters={"LHFCALC": "T"},
                ediff_eV=1e-7,
                final_energy_change_eV=0.05,
            )
            with self.assertRaisesRegex(ValueError, "unconfirmed"):
                check_convergence(meta, root)
            for name, value in (
                ("ediff_eV", np.nan),
                ("final_energy_change_eV", np.nan),
            ):
                with self.assertRaisesRegex(ValueError, "unconfirmed"):
                    check_convergence({**meta, name: value}, root)
            (root / "OUTCAR").write_text("aborting loop because hard stop was set")
            with self.assertRaisesRegex(ValueError, "stopped"):
                check_convergence(meta, root)
            (root / "OUTCAR").write_text("aborting loop because EDIFF is reached")
            check_convergence(meta, root)
