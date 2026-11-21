"""Diagnostics in the original VASP energy gauge and the PAW metric."""

import json
from pathlib import Path
import re

import h5py
import numpy as np
from scipy.linalg import block_diag

from .export import blocks, structure
from .realspace import SPARSE_FORMAT, matrices_k
from .pipeline import bands
from .vasp import metadata


def held_out(model, xml):
    ref = metadata(xml)
    if "zero_weight" in ref:
        points = ref["zero_weight"]["kpoints"]
        levels = ref["zero_weight"]["eig"]
    else:
        selected = ref["weights"] == 0
        points = ref["kpoints"][selected]
        levels = ref["eig"][:, selected]
    if not len(points):
        raise ValueError(
            "Expected zero-weight validation points in a VASP mesh + path calculation"
        )
    with h5py.File(model) as f:
        atoms = structure(f)
        displacement = (
            atoms.get_scaled_positions() - ref["atoms"].get_scaled_positions()
        )
        if (
            not np.array_equal(atoms.numbers, ref["atoms"].numbers)
            or not np.allclose(atoms.cell, ref["atoms"].cell)
            or not np.allclose(displacement, np.rint(displacement), atol=1e-7)
            or f.attrs["spinor"] != ref["spinor"]
            or f.attrs["soc"] != ref["soc"]
            or f.attrs["spin_channels"] != levels.shape[0]
        ):
            raise ValueError("Reference cell, atoms or spin mode differ from model")
        result = []
        for i, k in enumerate(points):
            if np.any(
                np.all(np.abs((f["k/points"][:] - k + 0.5) % 1 - 0.5) < 1e-7, axis=1)
            ):
                raise ValueError(
                    f"Validation point {k} already belongs to the source mesh"
                )
            s = matrices_k(f, k)[1][0]
            smin = np.linalg.eigvalsh(s)[0]
            row = dict(k=k.tolist(), min_overlap_eigenvalue=float(smin))
            if smin > 0:
                eig = bands(f, k)[:, 0]
                n = min(eig.shape[-1], levels.shape[-1])
                errors = eig[:, :n] - levels[:, i, :n]
                occupied = levels[:, i, :n] <= ref["efermi"]
                row["lowest_bands_rmse_eV"] = float(np.sqrt(np.mean(errors**2)))
                if occupied.any():
                    row["below_fermi_rmse_eV"] = float(
                        np.sqrt(np.mean(errors[occupied] ** 2))
                    )
            result.append(row)
    return result


def spectra(model, lo, hi, step, sigma):
    if not hi > lo or min(step, sigma) <= 0:
        raise ValueError("DOS requires emin < emax, positive step and broadening")
    axis = np.arange(lo, hi + step / 2, step)
    with h5py.File(model) as f:
        if f.attrs["format"] == SPARSE_FORMAT:
            raise ValueError("Projected DOS needs the projection file containing A(k)")
        points = f["k/points"][:]
        reference = f["k/eigenvalues"][:] - f.attrs["fermi_eV"]
        eig = bands(f, points) - f.attrs["fermi_eV"]
        shells = json.loads(f.attrs["shells"])
        labels = [
            f"{s}{i + 1}_shell{n}_l{l}_m{m}"
            for i, s in enumerate(structure(f).get_chemical_symbols())
            for n, l in enumerate(shells[s])
            for m in range(-l, l + 1)
        ]
        deg = 2 if len(eig) == 1 and not f.attrs["spinor"] else 1
        dos = np.zeros((len(axis), len(labels) + 2))
        populations = np.zeros(len(labels))
        for ik, wk in enumerate(f["k/weights"][:]):
            vals, vecs = np.linalg.eigh(f["k/S"][ik])
            whitening = (vecs * vals ** (-0.5)) @ vecs.conj().T
            if f.attrs["spinor"]:
                whitening = block_diag(whitening, whitening)
            for spin in range(len(eig)):
                psi = whitening @ f["k/projection"][spin, ik]
                weights = np.abs(psi) ** 2
                if f.attrs["spinor"]:
                    weights = sum(np.split(weights, 2))
                populations += wk * deg * (weights @ f["k/occupation"][spin, ik])

                def gaussian(energy):
                    return (
                        wk
                        * deg
                        * np.exp(-0.5 * ((axis[:, None] - energy) / sigma) ** 2)
                        / (sigma * np.sqrt(2 * np.pi))
                    )

                peaks = gaussian(reference[spin, ik])
                dos[:, 0] += peaks.sum(axis=1)
                dos[:, 1] += gaussian(eig[spin, ik]).sum(axis=1)
                dos[:, 2:] += peaks @ weights.T
    return axis, dos, labels, populations


def locality(model):
    with h5py.File(model) as f:
        atoms = structure(f)
        rows = []
        for (x, y, z, i, j), h, s in blocks(f):
            dr = (
                atoms.positions[j]
                - atoms.positions[i]
                + np.array([x, y, z]) @ atoms.cell
            )
            rows.append(
                (
                    x,
                    y,
                    z,
                    i,
                    j,
                    np.linalg.norm(dr),
                    np.linalg.norm(h),
                    np.linalg.norm(s),
                )
            )
    return sorted(rows, key=lambda r: r[5])


def lobster_summary(log):
    text = Path(log).read_text()
    if "finished in" not in text:
        raise ValueError("Incomplete lobsterout")
    matches = re.findall(
        r"(abs\. charge spilling|abs\. total spilling|total spilling|charge spilling):\s*([\d.]+)%",
        text,
        re.I,
    )
    if not matches:
        raise ValueError("No spilling values found in lobsterout")
    return {
        "version": text.splitlines()[0],
        "spilling_percent": [(name, float(v)) for name, v in matches],
        "warnings": [line for line in text.splitlines() if line.startswith("WARNING:")],
    }
