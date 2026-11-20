"""Test the action of an ordinary-AO Hamiltonian on independent bare PAW projections."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import eigvalsh

from vasp2mlh.band_release import independent_points, window_mask

from vasp2mlh.basis import read_potcar
from vasp2mlh.calibrate import metric_factors
from vasp2mlh.model import adjoint, ao_and_metric
from vasp2mlh.pipeline import file_sha256
from vasp2mlh.realspace import matrices_k, structure
from vasp2mlh.registry import read_shells
from vasp2mlh.vasp import Wavecar, metadata, plane_waves


def audit(run, models, energy_window=None):
    run = Path(run)
    ref = metadata(run / "vasprun.xml")
    wave = Wavecar(run / "WAVECAR")
    pots = read_potcar(run / "POTCAR")
    n = int(ref["nelect"] / 2)
    edge = slice(n - 2, n + 2)
    if energy_window is not None and (
        not np.isfinite(energy_window).all() or energy_window[0] >= energy_window[1]
    ):
        raise ValueError("Energy window must have finite, increasing bounds")
    if (
        ref["spinor"]
        or ref["icharg"] != 11
        or wave.ns != 1
        or (energy_window is None and (n < 2 or n + 2 > wave.nb))
    ):
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
                or f.attrs["spinor"]
                or f.attrs["spin_channels"] != 1
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
            if wave.nk != len(ref["kpoints"]):
                raise ValueError("Different WAVECAR and XML k-point counts")
            indices = independent_points(f["k/points"][:], ref["kpoints"])
            h, s = matrices_k(f, ref["kpoints"][indices])
            residual = []
            within = []
            projected_norms, leakage_norms = [], []
            weights = []
            scored_points = 0
            window = (
                None
                if energy_window is None
                else np.asarray(energy_window) + f.attrs["fermi_eV"]
            )
            for i, record in enumerate(indices):
                k, e, c = wave.read(0, record)
                if not np.allclose(
                    k, ref["kpoints"][record], atol=1e-7
                ) or not np.allclose(e[:, 0], ref["eig"][0, record], atol=1e-4, rtol=0):
                    raise ValueError(
                        "Independent eigenpairs differ between XML and WAVECAR"
                    )
                selected = edge
                if window is not None:
                    levels = eigvalsh(h[0, i], s[i])
                    mask = window_mask(e[:, 0][None], levels[None], None, window)
                    selected = np.flatnonzero(mask[0])
                    if not len(selected):
                        continue
                scored_points += 1
                phi, p, d = ao_and_metric(
                    plane_waves(k, wave.cell, wave.encut),
                    ref["atoms"],
                    basis,
                    pots,
                    ref["spline"],
                )
                sp = phi + p @ (d @ (adjoint(p) @ phi))
                _, ip = metric_factors(adjoint(phi) @ sp)
                v = ip @ (adjoint(sp) @ c[:, selected])
                # Closest orthonormal frame to the selected projected reference columns.
                left, singular, right = np.linalg.svd(v, full_matrices=False)
                if singular[-1] < 1e-6:
                    raise ValueError("Independent projected edge is rank deficient")
                weights.append(float(singular[-1] ** 2))
                q = left @ right
                _, ia = metric_factors(s[i])
                delta = ia @ h[0, i] @ ia @ q - q * e[selected, 0]
                residual.extend(np.linalg.norm(delta, axis=0))
                projected = adjoint(q) @ delta
                within.append(np.linalg.norm(projected, ord=2))
                projected_norms.extend(np.linalg.norm(projected, axis=0))
                leakage_norms.extend(np.linalg.norm(delta - q @ projected, axis=0))
            if not residual:
                raise ValueError(
                    "No independently sampled states enter the requested window"
                )
            results[str(name)] = {
                "kpoints_scored": scored_points,
                "kpoints_total": wave.nk,
                "independent_k": len(indices),
                "states_scored": len(residual),
                "energy_window_from_source_fermi_eV": energy_window,
                "min_subspace_weight": min(weights),
                "rms_action_eV": float(np.mean(np.square(residual)) ** 0.5),
                "max_action_eV": float(max(residual)),
                "rms_in_subspace_action_eV": float(
                    np.mean(np.square(projected_norms)) ** 0.5
                ),
                "rms_outside_subspace_action_eV": float(
                    np.mean(np.square(leakage_norms)) ** 0.5
                ),
                "mean_window_operator_norm_eV": float(np.mean(within)),
                "max_window_operator_norm_eV": float(max(within)),
            }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("models", nargs="+")
    parser.add_argument(
        "--energy-window",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="relative to the model source Fermi energy, eV",
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.models, args.energy_window), indent=2))
