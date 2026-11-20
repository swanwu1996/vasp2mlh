import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from vasp2mlh.cli import main
from vasp2mlh.configuration import load_config


class SingleConversionTests(unittest.TestCase):
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
            reference = run / "reference"
            reference.mkdir(parents=True)
            for name in ("vasprun.xml", "POTCAR", "CHGCAR"):
                (run / name).write_text("fixture")
                (reference / name).write_text("fixture")
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
                self.assertEqual(args[1:3], (run / "basis.h5", reference))
                self.assertEqual(kwargs["energy_window"], [-1.0, 2.0])
                self.assertEqual(load_config(run / "mlh/validation.json"), report)
                with self.assertRaises(SystemExit):
                    main([str(cfg)])
                self.assertEqual(reconstruct.call_count, 2)
