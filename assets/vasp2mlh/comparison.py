"""Matched scalar band paths from a VASP model, fixed-density VASP and OpenMX."""

import json
import re
from pathlib import Path

import h5py
import numpy as np
from ase.units import Bohr, Hartree

from .pipeline import bands, file_sha256, output_file
from .realspace import structure
from .vasp import metadata


def periodic_match(a, b, tol=2e-7):
    difference = np.asarray(a) - np.asarray(b)
    return np.all(abs(difference - np.rint(difference)) < tol, axis=-1)


def openmx_path(filename, atoms, electrons):
    filename = Path(filename)
    out = filename.with_suffix(".out").read_text()
    rows = iter(filename.read_text().splitlines())
    nb, spin, mu = next(rows).split()
    if int(spin) or not re.search(
        r"^\s*scf\.SpinPolarization\s+off\b", out, re.IGNORECASE | re.MULTILINE
    ):
        raise ValueError("OpenMX comparison requires a scalar, spin-unpolarized run")
    states = re.findall(r"Number of States\s*=\s*(\S+)", out)
    fermi = re.findall(r"Chemical potential \(Hartree\)\s+([-+.\deE]+)", out)
    if not states or not np.isclose(float(states[-1]), electrons, rtol=0, atol=1e-5):
        raise ValueError("OpenMX and VASP valence electron counts differ")
    if not fermi or abs(float(fermi[-1]) - float(mu)) > 1e-8:
        raise ValueError("OpenMX .Band and .out chemical potentials differ")
    recip = np.array(next(rows).split(), float).reshape(3, 3) / Bohr
    if not np.allclose(
        recip, 2 * np.pi * atoms.cell.reciprocal(), atol=2e-6, rtol=1e-6
    ):
        raise ValueError("OpenMX reciprocal lattice differs from model")
    coords = out.rpartition("Fractional coordinates of the final structure")[2]
    if not coords:
        raise ValueError("Missing final coordinates in OpenMX output")
    coords = re.findall(
        r"^\s*\d+\s+(\S+)\s+([-+.\deE]+)\s+([-+.\deE]+)\s+([-+.\deE]+)\s*$",
        coords,
        re.MULTILINE,
    )[: len(atoms)]
    if [r[0] for r in coords] != atoms.get_chemical_symbols() or not periodic_match(
        np.array([r[1:] for r in coords], float), atoms.get_scaled_positions()
    ).all():
        raise ValueError("OpenMX atom order or coordinates differ from model")
    segments = [next(rows).split() for _ in range(int(next(rows)))]
    k, energies, ticks = [], [], []
    for seg in segments:
        n = int(seg[0])
        expected = np.linspace(np.array(seg[1:4], float), np.array(seg[4:7], float), n)
        ticks.append([len(k), seg[-2]])
        for point in expected:
            count, *xyz = next(rows).split()
            xyz = np.array(xyz, float)
            values = np.array(next(rows).split(), float)
            if (
                count != nb
                or values.shape != (int(nb),)
                or not np.isfinite(values).all()
                or xyz.shape != (3,)
                or not periodic_match(xyz, point)
            ):
                raise ValueError("Invalid OpenMX band row or path segment")
            k.append(xyz)
            energies.append(values * Hartree)
        ticks.append([len(k) - 1, seg[-1]])
    if any(line.strip() for line in rows) or not k:
        raise ValueError("Unexpected OpenMX band path length")
    return np.array(k), np.array(energies), ticks


def compare_path(model, vasp_xml, openmx_band, edge_bands=2):
    ref = metadata(vasp_xml)
    if ref["icharg"] != 11:
        raise ValueError(
            "Use an ICHARG=11 VASP path with the source SCF charge density"
        )
    with h5py.File(model) as f:
        ref_dir = Path(vasp_xml).parent
        provenance = {}
        for filename, attribute in [
            ("POTCAR", "potcar_sha256"),
            ("CHGCAR", "charge_density_sha256"),
        ]:
            verified = False
            if attribute in f.attrs and (ref_dir / filename).is_file():
                verified = file_sha256(ref_dir / filename) == f.attrs[attribute]
                if not verified:
                    raise ValueError(
                        f"VASP path {filename} checksum differs from the source model"
                    )
            provenance[filename] = verified
        atoms = structure(f)
        other = ref["atoms"]
        ne = float(f.attrs["nelect"])
        if (
            ref["spinor"]
            or ref["nspin"] != 1
            or f.attrs["spinor"]
            or f.attrs["spin_channels"] != 1
            or abs(ne / 2 - round(ne / 2)) > 1e-7
        ):
            raise ValueError(
                "Comparison requires scalar spin-degenerate bands and even integer filling"
            )
        if (
            atoms.get_chemical_symbols() != other.get_chemical_symbols()
            or not np.allclose(atoms.cell, other.cell, rtol=0, atol=1e-6)
            or not periodic_match(
                atoms.get_scaled_positions(), other.get_scaled_positions()
            ).all()
            or not np.isclose(ne, ref["nelect"])
            or not np.isclose(ref["encut"], f.attrs["encut_eV"])
        ):
            raise ValueError(
                "VASP path geometry, electron count or cutoff differs from model"
            )
        try:
            k, om, ticks = openmx_path(openmx_band, atoms, ne)
        except StopIteration as exc:
            raise ValueError("Incomplete OpenMX band path") from exc
        if (
            k.shape != ref["kpoints"].shape
            or not periodic_match(k, ref["kpoints"]).all()
        ):
            raise ValueError(
                "VASP and OpenMX must use the same ordered fractional path"
            )
        model_e = bands(f, k)[0]
        mesh = f["k/points"][:]
        mesh_matches = periodic_match(k[:, None], mesh[None])
        # A shared native mesh point checks the fixed-density path energy reference.
        mesh_ref = f["k/eigenvalues"][0]
        truncation = json.loads(f.attrs.get("truncation_report", "null"))
    native = ref["eig"][0]
    filled = round(ne / 2)
    start, stop = filled - edge_bands, filled + edge_bands
    if (
        edge_bands < 1
        or start < 0
        or stop > min(arr.shape[1] for arr in [model_e, native, om])
    ):
        raise ValueError("Edge window exceeds available valence or conduction bands")
    ref_zero, om_zero = native[:, filled - 1].max(), om[:, filled - 1].max()
    arrays = {
        "vasp": native[:, start:stop] - ref_zero,
        "model": model_e[:, start:stop] - ref_zero,
        "openmx": om[:, start:stop] - om_zero,
    }
    mask = np.array(
        [
            not mesh_matches[i].any() and not periodic_match(k[:i], point).any()
            for i, point in enumerate(k)
        ]
    )
    if not np.any(mask):
        raise ValueError("Path has no independent off-mesh points")
    metrics = {}
    for left, right in [("model", "vasp"), ("openmx", "vasp"), ("model", "openmx")]:
        err = (arrays[left] - arrays[right])[mask]
        metrics[f"{left}_{right}"] = {
            "mae_eV": float(np.abs(err).mean()),
            "rmse_eV": float(np.sqrt(np.mean(err**2))),
            "max_eV": float(np.abs(err).max()),
        }
    near = mask[:, None] & (np.abs(arrays["vasp"]) <= 1.0)
    near_mae = (
        {
            f"{left}_{right}": float(np.abs(arrays[left] - arrays[right])[near].mean())
            for left, right in [
                ("model", "vasp"),
                ("openmx", "vasp"),
                ("model", "openmx"),
            ]
        }
        if near.any()
        else {}
    )
    ki, mi = np.nonzero(mesh_matches)
    overlap_drift = (
        float(np.abs(native[ki, start:stop] - mesh_ref[mi, start:stop]).max())
        if len(ki)
        else None
    )
    distance = np.r_[
        0.0,
        np.cumsum(
            np.linalg.norm(
                np.diff(k, axis=0) @ (2 * np.pi * atoms.cell.reciprocal()), axis=1
            )
        ),
    ]
    # Do not count the jump between independent line segments as a sampled interval.
    for i, _ in ticks[2::2]:
        distance[i:] -= distance[i] - distance[i - 1]
    return {
        "formula": atoms.get_chemical_formula(),
        "atoms": len(atoms),
        "band_numbers": list(range(start + 1, stop + 1)),
        "occupied_bands": filled,
        "fractional_k": k.tolist(),
        "distance_inv_A": distance.tolist(),
        "ticks": ticks,
        "mesh_point": mesh_matches.any(axis=1).tolist(),
        "scored": mask.tolist(),
        "unique_off_mesh": int(mask.sum()),
        "vasp_zero_eV": float(ref_zero),
        "openmx_zero_eV": float(om_zero),
        "shared_native_drift_eV": overlap_drift,
        "bands_eV": {key: val.tolist() for key, val in arrays.items()},
        "errors": metrics,
        "sampled_gap_eV": {
            key: float(val[:, edge_bands].min() - val[:, edge_bands - 1].max())
            for key, val in arrays.items()
        },
        "truncation": truncation,
        "input_checksums_verified": provenance,
        "near_edge_state_count": int(near.sum()),
        "vasp_window_1eV_mae": near_mae,
    }


def write_comparison(result, destination):
    with output_file(destination) as tmp:
        tmp.write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        )
