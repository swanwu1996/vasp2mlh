"""Test the action of an ordinary-AO Hamiltonian on independent bare PAW projections."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from vasp2mlh.basis import read_potcar
from vasp2mlh.calibrate import metric_factors
from vasp2mlh.comparison import periodic_match
from vasp2mlh.model import adjoint, ao_and_metric
from vasp2mlh.pipeline import file_sha256
from vasp2mlh.realspace import matrices_k, structure
from vasp2mlh.registry import read_shells
from vasp2mlh.vasp import Wavecar, metadata, plane_waves


def audit(run, models):
    run = Path(run)
    ref = metadata(run / "vasprun.xml")
    wave = Wavecar(run / "WAVECAR")
    pots = read_potcar(run / "POTCAR")
    n = int(ref["nelect"] / 2)
    edge = slice(n - 2, n + 2)
    if ref["spinor"] or ref["icharg"] != 11 or wave.ns != 1 or n < 2 or n + 2 > wave.nb:
        raise ValueError(
            "Expected a fixed-density scalar reference with a four-band edge"
        )
    results = {}
    for name in models:
        with h5py.File(name) as f:
            a = structure(f)
            if (
                f.attrs["metric"] != "ordinary"
                or not np.isclose(ref["encut"], f.attrs["encut_eV"])
                or not np.isclose(ref["nelect"], f.attrs["nelect"])
                or periodic_match(ref["kpoints"][:, None], f["k/points"][:][None]).any()
                or not np.array_equal(a.numbers, ref["atoms"].numbers)
                or not np.allclose(
                    a.positions, ref["atoms"].positions, rtol=0, atol=1e-7
                )
                or not np.allclose(a.cell, wave.cell, rtol=0, atol=1e-7)
                or f.attrs["potcar_sha256"] != file_sha256(run / "POTCAR")
                or f.attrs["charge_density_sha256"] != file_sha256(run / "CHGCAR")
            ):
                raise ValueError("Geometry, metric, PAW or fixed density differs")
            basis = {s: read_shells(g) for s, g in f["basis"].items()}
            h, s = matrices_k(f, ref["kpoints"])
            residual = []
            within = []
            for i in range(wave.nk):
                k, e, c = wave.read(0, i)
                if not np.allclose(k, ref["kpoints"][i], atol=1e-7) or not np.allclose(
                    e[:, 0], ref["eig"][0, i], atol=1e-4, rtol=0
                ):
                    raise ValueError(
                        "Independent eigenpairs differ between XML and WAVECAR"
                    )
                phi, p, d = ao_and_metric(
                    plane_waves(k, wave.cell, wave.encut),
                    ref["atoms"],
                    basis,
                    pots,
                    ref["spline"],
                )
                sp = phi + p @ (d @ (adjoint(p) @ phi))
                _, ip = metric_factors(adjoint(phi) @ sp)
                v = ip @ (adjoint(sp) @ c[:, edge])
                # Closest orthonormal frame to the four projected reference columns.
                left, singular, right = np.linalg.svd(v, full_matrices=False)
                if singular[-1] < 1e-6:
                    raise ValueError("Independent projected edge is rank deficient")
                q = left @ right
                _, ia = metric_factors(s[i])
                delta = ia @ h[0, i] @ ia @ q - q * e[edge, 0]
                residual.extend(np.linalg.norm(delta, axis=0))
                within.append(np.linalg.norm(adjoint(q) @ delta, ord=2))
            results[str(name)] = {
                "rms_action_eV": float(np.mean(np.square(residual)) ** 0.5),
                "max_action_eV": float(max(residual)),
                "mean_window_operator_norm_eV": float(np.mean(within)),
                "max_window_operator_norm_eV": float(max(within)),
            }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.models), indent=2))
