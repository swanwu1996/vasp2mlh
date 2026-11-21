import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from vasp2mlh.band_release import (
    check_convergence,
    check_scalar,
    independent_points,
    reconstruct,
    score_window,
    window_mask,
)


class ScalarBandReleaseTests(unittest.TestCase):
    def test_independent_failure_in_one_spin_channel(self):
        points = np.array([[0.1, 0.2, 0.3]])
        native = np.array([[[-1.0, 1.0]], [[-1.0, 1.0]]])
        levels = native.copy()
        levels[1] += 0.013
        with patch("vasp2mlh.band_release.bands", return_value=levels):
            with self.assertRaisesRegex(ValueError, "Independent"):
                score_window(
                    None, points, native, (1, 2), None, 0.010, 0.050, independent=True
                )
            score, rows = score_window(
                None, points, native, (1, 2), None, 1.0, 1.0, independent=True
            )
        self.assertLess(score["rmse_eV"], 0.010)
        self.assertAlmostEqual(score["worst_channel_rmse_eV"], 0.013)
        self.assertEqual(rows["spin_1based"].tolist(), [1, 1, 2, 2])

    def test_convergence_on_last_allowed_iteration(self):
        with TemporaryDirectory() as d:
            p = Path(d)
            cap = {"nelm": 4, "final_scf_steps": 4}
            log = p / "OUTCAR"
            log.write_text(
                "---- Iteration 1(4) ----\naborting loop because EDIFF is reached\n"
            )
            check_convergence(cap, p)
            with log.open("a") as f:
                f.write("---- Iteration 2(4) ----\n")
            with self.assertRaisesRegex(ValueError, "convergence"):
                check_convergence(cap, p)

    def test_scope(self):
        meta = {"nspin": 1, "spinor": False, "soc": False, "lreal": "F"}
        for functional in ("PE", "CA"):
            check_scalar({**meta, "operator_parameters": {"GGA": functional}})
        for change in ({"nspin": 2}, {"soc": True}, {"spinor": True}, {"lreal": "A"}):
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "scalar"),
            ):
                check_scalar({**meta, "operator_parameters": {}, **change})
        for tag, value in (
            ("LDAU", "T"),
            ("METAGGA", "SCAN"),
            ("LHFCALC", "T"),
            ("XC", "SCAN PBE"),
        ):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                check_scalar({**meta, "operator_parameters": {tag: value}})

    def test_independence_under_translation_and_time_reversal(self):
        mesh = [[0.25, 0, 0]]
        ref = [[1.25, 0, 0], [0.75, 0, 0], [0, 0.12, 0], [0, 0.88, 0], [0, 0, 0.12]]
        np.testing.assert_array_equal(independent_points(mesh, ref), [2, 4])
        with self.assertRaisesRegex(ValueError, "No independent"):
            independent_points(mesh, ref[:2])

    def test_count_both_entry_and_exit_at_window_boundary(self):
        native = np.array([[-1.0, 0.2, 0.8, 2.0]])
        model = np.array([[-0.2, 0.8, 0.9, 2.5, 3.0]])
        np.testing.assert_array_equal(
            window_mask(native, model, None, (-0.3, 0.3)), [[True, True, False, False]]
        )
        np.testing.assert_array_equal(
            window_mask(native, model, (3, 4), None), [[False, False, True, True]]
        )
        with self.assertRaisesRegex(ValueError, "above"):
            window_mask(native, model, None, (0.0, 2.0))
        with self.assertRaisesRegex(ValueError, "counterpart"):
            window_mask(
                np.array([[-2.0, 3.0]]),
                np.array([[-3.0, -2.0, 0.5, 3.0]]),
                None,
                (0.0, 1.0),
            )
        with self.assertRaisesRegex(ValueError, "range"):
            window_mask(native, model, (4, 5), None)

    @patch("vasp2mlh.band_release.bands")
    def test_absolute_energy_gate_and_empty_window(self, calculate):
        points = np.array([[0.1, 0.2, 0.3], [0.2, 0, 0]])
        native = np.array([[-1.0, 1.0, 5.0], [-0.8, 1.0, 5.0]])
        calculate.return_value = (native + [[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]])[None]
        for independent in (True, False):
            with (
                self.subTest(independent=independent),
                self.assertRaisesRegex(ValueError, "maximum"),
            ):
                score_window(
                    None,
                    points,
                    native,
                    (1, 2),
                    None,
                    1.0,
                    0.015,
                    independent=independent,
                )
        summary, pairs = score_window(
            None, points, native, (1, 2), None, 0.011, 0.021, independent=True
        )
        self.assertAlmostEqual(summary["rmse_eV"], 0.01)
        self.assertEqual(summary["worst_band_1based"], 2)
        np.testing.assert_array_equal(pairs["band_1based"], [1, 2, 1, 2])
        with self.assertRaisesRegex(ValueError, "No states"):
            score_window(
                None, points, native, None, (2.0, 3.0), 1.0, 1.0, independent=True
            )

    def test_reject_colliding_output_before_reading_inputs(self):
        with TemporaryDirectory() as d:
            p = Path(d)
            basis = p / "basis.h5"
            basis.write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "collides"):
                reconstruct(
                    p,
                    basis,
                    p,
                    basis,
                    cutoff=1.0,
                    max_rmse=0.01,
                    max_error=0.1,
                    energy_window=(-1.0, 1.0),
                )
            self.assertEqual(basis.read_bytes(), b"original")
