"""Use one atom-pair Wigner-Seitz interpolator for external scalar H(k), S(k)."""

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import eigvalsh

from vasp2mlh.comparison import periodic_match
from vasp2mlh.model import bloch, localized_transform
from vasp2mlh.pipeline import file_sha256
from vasp2mlh.realspace import structure
from vasp2mlh.vasp import metadata


def matrices(path):
    rows = iter(Path(path).read_text().splitlines())
    points, parts = [], []
    for head in rows:
        if not head.strip():
            continue
        point = np.array(head.split(" at ")[1].split(), float)
        part = next(rows).strip()
        labels = next(rows).split()[1:]
        values = []
        for label in labels:
            name, *v = next(rows).split()
            if name != label or len(v) != len(labels):
                raise ValueError("Unexpected LOBSTER matrix row")
            values.append(v)
        x = np.array(values, float)
        if part == "Real parts":
            points.append(point)
            parts.append(x.astype(complex))
        elif part == "Imag parts" and np.array_equal(point, points[-1]):
            parts[-1] += 1j * x
        else:
            raise ValueError("Unexpected LOBSTER complex matrix layout")
    return np.array(points), np.array(parts), labels


def score(f, h, s, counts, reference, retained):
    k = f["k/points"][:]
    atoms = structure(f)
    atoms.set_cell(f["structure/projection_cell"][:], scale_atoms=True)
    r, hr, sr = localized_transform(k, f["k/mesh"][:], atoms, counts, h[None], s)
    ref = metadata(reference)
    for n, key in [("POTCAR", "potcar_sha256"), ("CHGCAR", "charge_density_sha256")]:
        if file_sha256(Path(reference).parent / n) != f.attrs[key]:
            raise ValueError("Fixed-density reference identity mismatch")
    if (
        ref["icharg"] != 11
        or ref["spinor"]
        or ref["nspin"] != 1
        or not np.array_equal(atoms.numbers, ref["atoms"].numbers)
        or not np.isclose(ref["nelect"], f.attrs["nelect"])
        or not np.isclose(ref["encut"], f.attrs["encut_eV"])
        or not np.allclose(atoms.positions, ref["atoms"].positions)
        or not np.allclose(atoms.cell, ref["atoms"].cell)
    ):
        raise ValueError("Expected matching fixed-density structure")
    kt, idx = np.unique(np.round(ref["kpoints"], 8), axis=0, return_index=True)
    independent = ~periodic_match(kt[:, None], k[None]).any(axis=1)
    kt, idx = kt[independent], idx[independent]
    if not len(kt):
        raise ValueError("Reference has no independent k points")
    ev = np.array(
        [eigvalsh(a, b) for a, b in zip(bloch(kt, r, hr)[0], bloch(kt, r, sr))]
    )
    start = int(f.attrs["nelect"] / 2) - 2
    stop = min(start + 4, retained)
    if start < 0 or stop <= start or stop > ref["eig"].shape[-1]:
        raise ValueError("Retained spectrum does not reach target window")
    err = abs(ev[:, start:stop] - ref["eig"][0, idx, start:stop])
    return {
        "nk": len(kt),
        "bands": [start + 1, stop],
        "mae_meV": 1000 * err.mean(),
        "max_meV": 1000 * err.max(),
        "per_band_mae_meV": (1000 * err.mean(axis=0)).tolist(),
    }


def lobster(folder, model, refs):
    d = Path(folder)
    k, h, labels = matrices(d / "hamiltonMatrices.lobster")
    ks, _s, sl = matrices(d / "overlapMatrices.lobster")
    with h5py.File(model) as f:
        if (
            labels != sl
            or not np.allclose(k, ks)
            or not periodic_match(k, f["k/points"][:], tol=1e-6).all()
        ):
            raise ValueError("LOBSTER matrices or k ordering differ")
        source = metadata(d / "vasprun.xml")
        if (
            not np.allclose(source["eig"][0], f["k/eigenvalues"][0], rtol=0, atol=1e-4)
            or file_sha256(d / "POTCAR") != f.attrs["potcar_sha256"]
        ):
            raise ValueError("LOBSTER and model source calculations differ")
        ref = f["k/eigenvalues"][0, :, : len(labels)]
        # LOBSTER reconstructs H after symmetric orthonormalization of the basis.
        # Its separately printed S belongs to the original overlapping orbitals.
        identity = np.broadcast_to(np.eye(len(labels)), h.shape)
        ev = np.linalg.eigvalsh(h)
        print(
            "mesh errors by band",
            np.round(abs(ev - ref).mean(axis=0), 5),
            "mean signed",
            np.round((ev - ref).mean(axis=0), 5),
            flush=True,
        )
        names = [name.split("_")[0] for name in labels]
        atoms = structure(f)
        counts = [
            names.count(f"{s}{i + 1}")
            for i, s in enumerate(atoms.get_chemical_symbols())
        ]
        if any(n < 1 for n in counts) or sum(counts) != len(labels):
            raise ValueError("Unexpected atom grouping")
        for ref in refs:
            print(ref, json.dumps(score(f, h, identity, counts, ref, len(labels))))


if __name__ == "__main__":
    import sys

    lobster(sys.argv[1], sys.argv[2], sys.argv[3:])
