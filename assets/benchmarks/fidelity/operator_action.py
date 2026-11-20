"""Test the action of an ordinary-AO Hamiltonian on independent bare PAW projections."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import block_diag, eigvalsh

from vasp2mlh.band_release import check_source, independent_points, window_mask

from vasp2mlh.basis import read_potcar
from vasp2mlh.calibrate import metric_factors
from vasp2mlh.model import adjoint, ao_and_metric, spin_rotation
from vasp2mlh.pipeline import file_sha256
from vasp2mlh.realspace import matrices_k, structure
from vasp2mlh.registry import read_shells
from vasp2mlh.vasp import Wavecar, metadata, plane_waves, spectral_mode


def reference_records(meta, wave):
    if spectral_mode(meta):
        holdout = meta.get("zero_weight")
        if holdout is None:
            raise ValueError(
                "A spectral action check requires independent zero-weight WAVECAR records in the same SCF"
            )
        if wave.nk != meta["wave_nk"]:
            raise ValueError("Different WAVECAR and XML k-point counts")
        return holdout, holdout["wave_indices"]
    if meta["icharg"] != 11 or wave.nk != len(meta["kpoints"]):
        raise ValueError("Expected a matching scalar fixed-density WAVECAR and XML")
    return meta, np.arange(wave.nk)


def audit(run, models, energy_window=None, band_range=None):
    run = Path(run)
    ref = metadata(run / "vasprun.xml")
    wave = Wavecar(run / "WAVECAR")
    pots = read_potcar(run / "POTCAR")
    spinor = ref["spinor"]
    n = int(ref["nelect"] / (1 if spinor else 2))
    if band_range is None:
        band_range = (n - 1, n + 2) if energy_window is None else None
    if band_range is not None and (
        energy_window is not None or not 1 <= band_range[0] <= band_range[1] <= wave.nb
    ):
        raise ValueError("Choose a valid 1-based band range or an energy window")
    if energy_window is not None and (
        not np.isfinite(energy_window).all() or energy_window[0] >= energy_window[1]
    ):
        raise ValueError("Energy window must have finite, increasing bounds")
    if wave.ns != 1:
        raise ValueError("Expected a single scalar or spinor WAVECAR channel")
    spectral = spectral_mode(ref)
    if spectral:
        check_source(ref, run)
    samples, records = reference_records(ref, wave)
    pot_hash = file_sha256(run / "POTCAR")
    reference_hash = file_sha256(run / ("WAVECAR" if spectral else "CHGCAR"))
    xml_hash = file_sha256(run / "vasprun.xml") if spectral else None
    results = {}
    for name in models:
        with h5py.File(name) as f:
            a = structure(f)
            if (
                f.attrs["metric"] != "ordinary"
                or not np.isclose(ref["encut"], f.attrs["encut_eV"])
                or not np.isclose(ref["nelect"], f.attrs["nelect"])
                or bool(f.attrs["spinor"]) != spinor
                or (spinor and not np.allclose(f.attrs["saxis"], [0, 0, 1]))
                or f.attrs["spin_channels"] != 1
                or not np.array_equal(a.numbers, ref["atoms"].numbers)
                or not np.allclose(
                    a.positions, ref["atoms"].positions, rtol=0, atol=1e-7
                )
                or not np.allclose(a.cell, wave.cell, rtol=0, atol=1e-7)
                or f.attrs["potcar_sha256"] != pot_hash
            ):
                raise ValueError("Geometry, metric or PAW differs")
            if spectral:
                provenance = json.loads(f["source_validation"].attrs["report"])
                if (
                    provenance["source_xml_sha256"] != xml_hash
                    or provenance["input_sha256"]["WAVECAR"] != reference_hash
                ):
                    raise ValueError(
                        "Spectral operator action requires the model's exact source XML and WAVECAR"
                    )
            elif f.attrs["charge_density_sha256"] != reference_hash:
                raise ValueError("Fixed reference density differs from model")
            basis = {s: read_shells(g) for s, g in f["basis"].items()}
            indices = independent_points(
                f["k/points"][:], samples["kpoints"], time_reversal=not spinor
            )
            h, s = matrices_k(f, samples["kpoints"][indices])
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
            for i, point in enumerate(indices):
                k, e, c = wave.read(0, records[point])
                if not np.allclose(
                    k, samples["kpoints"][point], atol=1e-7, rtol=0
                ) or not np.allclose(
                    e[:, 0], samples["eig"][0, point], atol=1e-4, rtol=0
                ):
                    raise ValueError(
                        "Independent eigenpairs differ between XML and WAVECAR"
                    )
                ordinary = block_diag(s[i], s[i]) if spinor else s[i]
                levels = eigvalsh(h[0, i], ordinary)
                mask = window_mask(e[:, 0][None], levels[None], band_range, window)
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
                components = np.split(c[:, selected], 2 if spinor else 1)
                v = np.vstack([ip @ (adjoint(sp) @ part) for part in components])
                if spinor:
                    v = (spin_rotation(ref["saxis"]) @ v.reshape(2, -1)).reshape(
                        v.shape
                    )
                # Closest orthonormal frame to the selected projected reference columns.
                left, singular, right = np.linalg.svd(v, full_matrices=False)
                if singular[-1] < 1e-6:
                    raise ValueError("Independent projected edge is rank deficient")
                weights.append(float(singular[-1] ** 2))
                q = left @ right
                _, ia = metric_factors(s[i])
                if spinor:
                    ia = block_diag(ia, ia)
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
                "kpoints_total": len(records),
                "independent_k": len(indices),
                "states_scored": len(residual),
                "reference_kind": samples.get("source", "fixed-density"),
                "spinor": bool(spinor),
                "band_range_1based": band_range,
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
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--bands",
        type=int,
        nargs=2,
        help="inclusive ordered band indices, starting at 1",
    )
    target.add_argument(
        "--energy-window",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="relative to the model source Fermi energy, eV",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.run, args.models, args.energy_window, args.bands), indent=2
        )
    )
