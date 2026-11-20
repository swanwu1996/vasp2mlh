from contextlib import redirect_stderr
from io import StringIO
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from scipy.integrate import simpson
from ase import Atoms
from types import SimpleNamespace

from vasp2mlh.batch import convert, read_jobs, run_batch
from vasp2mlh.basis import Shell, compiled_radial, simpson_weights
from vasp2mlh.model import RadialTable
from vasp2mlh.realspace import SPARSE_FORMAT
from vasp2mlh.registry import FORMAT


class BatchTests(unittest.TestCase):
    def setUp(self):
        env = tempfile.TemporaryDirectory()
        self.addCleanup(env.cleanup)
        self.root = Path(env.name)
        self.registry = self.root / "basis.h5"
        with h5py.File(self.registry, "w") as f:
            f.attrs["format"] = FORMAT
        self.source = self.root / "run"
        self.source.mkdir()
        for name in ("WAVECAR", "vasprun.xml", "POTCAR"):
            (self.source / name).write_text(name)
        self.config = {
            "mode": "projection",
            "registry": "basis.h5",
            "jobs": [{"id": "a", "run": "run"}],
        }
        self.manifest = self.root / "jobs.json"
        self.out = self.root / "out"

    def write(self):
        self.manifest.write_text(json.dumps(self.config))
        return read_jobs(self.manifest)

    def test_preflight_and_names(self):
        self.config["jobs"].append({"id": "b", "run": "absent"})
        self.write()
        log = StringIO()
        with (
            patch(
                "vasp2mlh.batch.metadata",
                return_value={"atoms": Atoms(), "spinor": False},
            ),
            patch(
                "vasp2mlh.batch.projection_header",
                return_value=(SimpleNamespace(nk=1, ns=1), None),
            ),
            patch("vasp2mlh.batch.read_registry", return_value={}),
            redirect_stderr(log),
        ):
            summary = run_batch(self.manifest, self.out, check_only=True)
        self.assertIn("2/2", log.getvalue())
        self.assertIn("ready=1", log.getvalue())
        self.assertIn("failed=1", log.getvalue())
        self.assertNotIn("\r", log.getvalue())
        self.config["runtime"] = {"progress": False}
        self.write()
        legacy = StringIO()
        with redirect_stderr(legacy):
            run_batch(self.manifest, self.out, check_only=True)
        self.assertIn("2/2", legacy.getvalue())
        self.assertIn("b failed", legacy.getvalue())
        self.assertEqual([r["status"] for r in summary["jobs"]], ["ready", "failed"])
        self.assertFalse((self.out / "summary.json").exists())
        for name in ("a", "../not-a-job"):
            self.config["jobs"][1]["id"] = name
            with self.assertRaisesRegex(ValueError, "job id"):
                self.write()

    def test_preflight_rejects_source_mode_and_missing_paw_records(self):
        from vasp2mlh.batch import preflight

        job = self.write()[0]
        job.update(mode="validated", reference=self.source, potential_run=self.source)
        (self.source / "CHGCAR").write_text("nonempty")
        with h5py.File(self.source / "vaspout.h5", "w") as f:
            f["results/potential/total"] = [0]
        with patch(
            "vasp2mlh.batch.metadata",
            return_value={"operator_parameters": {}, "lreal": "T"},
        ):
            with self.assertRaisesRegex(ValueError, "LREAL=T"):
                preflight(job)
        with (
            patch(
                "vasp2mlh.batch.metadata",
                return_value={"atoms": Atoms(), "spinor": False},
            ),
            patch("vasp2mlh.batch.check_source"),
            patch(
                "vasp2mlh.batch.projection_header",
                return_value=(SimpleNamespace(nk=1, ns=1), [1, 1, 1]),
            ),
            patch("vasp2mlh.batch.read_registry", return_value={}),
            patch("vasp2mlh.batch.check_potential_source"),
        ):
            with self.assertRaisesRegex(ValueError, "paw/dij"):
                preflight(job)

    @patch("vasp2mlh.batch.sparsify")
    @patch("vasp2mlh.batch.build")
    def test_restart_fingerprints_corruption_and_failed_rebuild(self, build, sparse):
        job = self.write()[0]
        (self.out / ".batch").mkdir(parents=True)

        def store(dense, dest, **options):
            with h5py.File(dest, "w") as f:
                f.attrs["format"] = SPARSE_FORMAT

        sparse.side_effect = store

        def go():
            return convert(job, self.out, 1)

        self.assertEqual(go()["status"], "converted")
        self.assertEqual(go()["status"], "skipped")
        self.assertEqual(build.call_count, 1)
        target = self.out / "a.h5"
        target.write_bytes(b"incomplete")
        self.assertEqual(go()["status"], "converted")
        job["options"]["cutoff"] = 4.0
        self.assertEqual(go()["status"], "converted")
        old = target.read_bytes()
        (self.source / "WAVECAR").write_text("new wavefunction")
        sparse.side_effect = ValueError("calculation failed")
        self.assertEqual(go()["status"], "failed")
        self.assertEqual(target.read_bytes(), old)
        sparse.side_effect = store
        self.assertEqual(go()["status"], "converted")
        self.assertEqual(go()["status"], "skipped")
        with h5py.File(target, "w") as f:
            f.attrs["user"] = "other"
        self.assertIn("not owned", go()["error"])

    def test_validated_requires_reference_and_defaults_to_fermi_window(self):
        self.config["mode"] = "validated"
        with self.assertRaisesRegex(ValueError, "window"):
            self.write()
        self.config["defaults"] = {
            "band_range": [1, 2],
            "cutoff": 5,
            "max_rmse": 0.02,
            "max_error": 0.1,
        }
        with self.assertRaises(KeyError):
            self.write()
        self.config["jobs"][0]["reference"] = "blind"
        self.assertEqual(self.write()[0]["reference"], self.root / "blind")
        del self.config["defaults"]["band_range"]
        self.assertEqual(self.write()[0]["options"]["energy_window"], [-1.0, 2.0])


class RadialTests(unittest.TestCase):
    def test_quadrature_even_odd_irregular(self):
        rng = np.random.default_rng(182)
        for n in (3, 4, 11, 12, 513):
            r = np.cumsum(rng.uniform(0.001, 0.25, n))
            y = rng.normal(size=n)
            self.assertAlmostEqual(simpson_weights(r) @ y, simpson(y, x=r), places=11)

    def test_table_on_shared_and_new_momenta(self):
        r = np.linspace(0, 4, 100)
        shells = [Shell(l, r, r**l * np.exp(-r), "trial") for l in (0, 1, 4)]
        table = RadialTable({"X": shells})
        table.capacity = 4
        for q in ([0.1, 1.0, 3.0], [1.0, 0.1, 1.0, 2.0, 3.0, 5.0], [1.0, 0.5]):
            result = table.evaluate(q)
            for sh in shells:
                np.testing.assert_allclose(result[id(sh)], sh.transform(q), atol=1e-13)

    @unittest.skipIf(compiled_radial is None, "optional Cython extension absent")
    def test_native_matches_scipy(self):
        q = np.r_[0.0, 1e-12, 0.9999999, 1.0, 1.0000001, np.linspace(0.02, 30, 147)]
        for n in (100, 101):
            r = np.geomspace(1e-6, 4, n)
            for l in range(5):
                sh = Shell(l, r, r**l * np.exp(-r), "trial")
                with patch.dict(os.environ, VASP2MLH_RADIAL_BACKEND="scipy"):
                    ref = sh.transform(q)
                with patch.dict(os.environ, VASP2MLH_RADIAL_BACKEND="cython"):
                    np.testing.assert_allclose(
                        sh.transform(q), ref, atol=1e-12, rtol=1e-12
                    )
