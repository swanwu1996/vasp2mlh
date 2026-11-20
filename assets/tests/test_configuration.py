import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py
from threadpoolctl import threadpool_info

from vasp2mlh.cli import main, make_parser
from vasp2mlh.batch import read_jobs
from vasp2mlh.configuration import load_config, operation_args, runtime_controls
from vasp2mlh.registry import FORMAT


class ConfigurationTests(unittest.TestCase):
    def test_toml_and_json_operations_use_config_directory(self):
        with TemporaryDirectory() as d:
            p = Path(d)
            with h5py.File(p / "model.h5", "w") as f:
                f.create_group("band_validation").attrs["report"] = (
                    '{"status":"passed"}'
                )
            options = {"model": "model.h5", "output": "result.json"}
            (p / "op.toml").write_text(
                'command="band-report"\n[parameters]\nmodel="model.h5"\noutput="result.json"\n'
            )
            (p / "op.json").write_text(
                json.dumps({"command": "band-report", "parameters": options})
            )
            cwd = Path.cwd()
            for name in ("op.toml", "op.json"):
                main([str(p / name)])
                self.assertEqual(Path.cwd(), cwd)
                self.assertEqual(load_config(p / "result.json"), {"status": "passed"})

    def test_batch_preflight_from_one_file(self):
        with TemporaryDirectory() as d:
            p = Path(d)
            with h5py.File(p / "basis.h5", "w") as f:
                f.attrs["format"] = FORMAT
            for name in ("WAVECAR", "POTCAR", "vasprun.xml"):
                (p / name).write_bytes(b"present")
            path = p / "batch.toml"
            path.write_text(
                'mode="projection"\noutput="out"\nregistry="basis.h5"\n[runtime]\nworkers=2\nthreads=1\ncheck_only=true\n[[jobs]]\nid="a"\nrun="."\n'
            )
            with self.assertRaises(SystemExit):
                main([str(path)])
            self.assertEqual(
                load_config(p / "out/preflight.json")["jobs"][0]["status"], "failed"
            )
            self.assertFalse((p / "out/a.h5").exists())
            path.write_text(
                path.read_text().replace(
                    'run="."', 'run="."\n[jobs.options]\ncutoff="12"'
                )
            )
            with self.assertRaisesRegex(ValueError, "numeric"):
                read_jobs(path)

    def test_typed_flags_windows_and_lists(self):
        _, parsers = make_parser()
        with self.assertRaisesRegex(ValueError, "true or false"):
            operation_args(
                "bands",
                parsers["bands"],
                {
                    "model": "m",
                    "points": "k",
                    "output": "o",
                    "validated_window": "false",
                },
            )
        args = operation_args(
            "hamgnn",
            parsers["hamgnn"],
            {"models": ["a", "b"], "output": "o", "nao_max": 19},
        )
        self.assertEqual(args.models, ["a", "b"])
        base = {
            "run": "r",
            "registry": "b",
            "reference": "v",
            "output": "o",
            "cutoff": 12,
            "max_rmse": 0.01,
            "max_error": 0.05,
        }
        for options in ({}, {"energy_window": [-1, 2], "band_range": [1, 3]}):
            with self.assertRaisesRegex(ValueError, "Choose one"):
                operation_args(
                    "reconstruct-bands", parsers["reconstruct-bands"], base | options
                )
        with self.assertRaisesRegex(ValueError, "progress must be true or false"):
            runtime_controls(
                {"command": "convert", "runtime": {"progress": "false"}}, False
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            runtime_controls({"runtime": {"threads": True}}, True)
        with self.assertRaisesRegex(ValueError, "Allowed runtime"):
            runtime_controls({"runtime": {"check_only": True}}, False)

    def test_convert_applies_thread_limit_from_toml_and_cli(self):
        def limit(parser, args):
            return {
                pool["num_threads"]
                for pool in threadpool_info()
                if pool["user_api"] == "blas"
            }

        with (
            TemporaryDirectory() as d,
            patch("vasp2mlh.cli.execute", side_effect=limit),
        ):
            config = Path(d) / "mlh.toml"
            config.write_text('command="convert"\n[parameters]\nrun="."\nthreads=2\n')
            self.assertEqual(main([str(config)]), {2})
            self.assertEqual(main(["convert", ".", "--threads", "3"]), {3})
            self.assertEqual(main(["convert", "."]), {1})
            with self.assertRaises(SystemExit):
                main(["convert", ".", "--threads", "0"])

    def test_legacy_subcommand_and_new_backend_scope(self):
        with (
            TemporaryDirectory() as d,
            patch.dict(os.environ, VASP2MLH_RADIAL_BACKEND="scipy"),
        ):
            p = Path(d) / "control.json"
            p.write_text(
                json.dumps(
                    {
                        "command": "validate",
                        "parameters": {"model": "input.h5"},
                        "runtime": {"backend": "auto"},
                    }
                )
            )
            seen = []
            with patch(
                "vasp2mlh.cli.execute",
                side_effect=lambda parser, args: seen.append(
                    (args.model, os.environ["VASP2MLH_RADIAL_BACKEND"])
                ),
            ):
                main([str(p)])
                main(["validate", "original.h5"])
            self.assertEqual(seen, [("input.h5", "auto"), ("original.h5", "scipy")])
