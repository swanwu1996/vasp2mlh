import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from vasp2mlh.pipeline import file_sha256
from vasp2mlh.cli import main, valid_main
from vasp2mlh.configuration import load_config


class SingleConversionTests(unittest.TestCase):
    def test_v2h_default_config_and_subcommands(self):
        with TemporaryDirectory() as folder:
            previous = Path.cwd()
            try:
                os.chdir(folder)
                Path("WAVECAR").touch()
                with patch("vasp2mlh.cli.execute") as execute:
                    main([])
                    self.assertEqual(execute.call_args.args[1].run, ".")
                    Path("mlh.toml").write_text('command="convert"\n[parameters]\nrun="mesh"')
                    main([])
                    self.assertEqual(execute.call_args.args[1].run, "mesh")
                with patch("vasp2mlh.validation.validate_sample") as validate:
                    main(["valid"])
                    self.assertEqual(validate.call_args.args[0].run, "mesh")
                    self.assertEqual(validate.call_args.args[1], Path("mlh.toml").resolve())
                with patch("vasp2mlh.atomic.generate") as generate:
                    main(["genbasis"])
                    generate.assert_called_once_with("basis.h5", ["POTCAR"], None, "paw-compact-v1")
            finally:
                os.chdir(previous)

    def test_empty_wavecar_reports_vasp_setting(self):
        with TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "WAVECAR").touch()
            with patch("sys.stderr") as stderr, self.assertRaises(SystemExit):
                main(["convert", str(run)])
            self.assertIn("LWAVE=T", str(stderr.write.call_args))
            self.assertFalse((run / "mlh").exists())

    def test_rejected_export_leaves_no_partial_dataset_and_can_retry(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "source"
            run.mkdir()
            for name in ("vasprun.xml", "POTCAR", "CHGCAR"):
                (run / name).write_text("fixture")
            for name in ("WAVECAR", "vaspout.h5"):
                (run / name).write_text("fixture")
            cfg = root / "mlh.json"
            cfg.write_text(
                json.dumps({"command": "convert", "parameters": {"run": "source"}})
            )
            report = {"status": "passed", "rmse_eV": 0.001, "max_error_eV": 0.002}
            with (
                patch("vasp2mlh.conversion.generate") as generate,
                patch(
                    "vasp2mlh.conversion.reconstruct", return_value=report
                ) as reconstruct,
                patch(
                    "vasp2mlh.conversion.deeph", side_effect=ValueError("export failed")
                ) as export,
            ):
                with self.assertRaises(SystemExit):
                    main([str(cfg)])
                self.assertFalse((run / "mlh").exists())
                generate.assert_called_once()
                (run / "basis.h5").write_text("project basis")
                export.side_effect = None
                main([str(cfg)])
                generate.assert_called_once()
                args, kwargs = reconstruct.call_args
                self.assertEqual(args[1:3], (run / "basis.h5", None))
                self.assertEqual(kwargs["energy_window"], [-1.0, 2.0])
                self.assertEqual(
                    load_config(run / "mlh/source_validation.json"), report
                )
                self.assertFalse((run / "mlh/validation.json").exists())
                with self.assertRaises(SystemExit):
                    main([str(cfg)])
                self.assertEqual(reconstruct.call_count, 2)

    def test_validator_scores_saved_model_and_replaces_stale_pass_on_failure(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "ref").mkdir()
            (root / "vasprun.xml").write_text("source")
            (root / "ref/vasprun.xml").write_text("reference")
            (root / "results").mkdir()
            path = root / "results/model.h5"
            with h5py.File(path, "w") as f:
                f.create_group("source_validation").attrs["report"] = json.dumps(
                    {
                        "source_xml_sha256": file_sha256(root / "vasprun.xml"),
                        "input_sha256": {},
                    }
                )
            before = file_sha256(path)
            config = root / "mlh.toml"
            config.write_text(
                'command="convert"\n[parameters]\nrun="."\nreference="ref"\noutput_dir="results"\n'
            )
            meta = {
                "kpoints": np.zeros((1, 3)),
                "eig": np.array([[[0.0, 4.0]]]),
                "efermi": 0.0,
            }
            with (
                patch(
                    "vasp2mlh.validation.verify_export",
                    return_value={"status": "passed"},
                ) as verify_export,
                patch("vasp2mlh.validation.metadata", return_value=meta),
                patch(
                    "vasp2mlh.validation.reference_data",
                    return_value=(meta, np.array([0])),
                ),
                patch("vasp2mlh.band_release.bands", return_value=meta["eig"] + 0.001),
            ):
                valid_main([str(config)])
                report = root / "results/validation.json"
                self.assertEqual(load_config(report)["status"], "passed")
                config.write_text(config.read_text() + "max_rmse=0.0\n")
                with self.assertRaises(SystemExit):
                    valid_main([str(config)])
                failed = load_config(report)
                self.assertEqual(failed["status"], "failed")
                self.assertAlmostEqual(failed["rmse_eV"], 0.001)
                self.assertEqual(file_sha256(path), before)
                config.write_text(
                    config.read_text().replace("max_rmse=0.0", "max_rmse=0.01")
                )
                valid_main([str(config)])
                verify_export.side_effect = FileNotFoundError("missing Hamiltonian")
                with self.assertRaises(SystemExit):
                    valid_main([str(config)])
                failed = load_config(report)
                self.assertEqual(failed["native_band_status"], "passed")
                self.assertEqual(failed["export_validation"]["status"], "failed")
                self.assertEqual(failed["status"], "failed")
                verify_export.side_effect = None
                valid_main([str(config)])
                path.unlink()
                with self.assertRaises(SystemExit):
                    valid_main([str(config)])
                self.assertEqual(load_config(report)["status"], "failed")
                self.assertIn("Missing", load_config(report)["error"])

    def test_output_alias_and_conflict(self):
        with TemporaryDirectory() as temp, patch("vasp2mlh.cli.execute") as execute:
            cfg = Path(temp) / "mlh.toml"
            cfg.write_text(
                'command="convert"\n[parameters]\nrun="."\noutput="old-name"\n'
            )
            main([str(cfg)])
            self.assertEqual(execute.call_args.args[1].output_dir, "old-name")
            cfg.write_text(cfg.read_text() + 'output_dir="another-name"\n')
            with self.assertRaises(SystemExit):
                main([str(cfg)])
