"""Calibrate scalar Ritz energies on the source mesh in a fixed ordinary-AO metric."""

import itertools
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import eigh, svd

from .model import adjoint, bloch, localized_transform
from .overlap import overlap_translations
from .pipeline import file_sha256, output_file
from .realspace import orbital_bounds, structure


def metric_factors(metric):
    vals, vec = eigh(metric)
    if vals[0] <= max(1e-12, vals[-1] * 1e-10):
        raise ValueError("Numerically dependent AO overlap")
    return (vec * np.sqrt(vals)) @ adjoint(vec), (vec / np.sqrt(vals)) @ adjoint(vec)


def corrected_levels(ritz, reference, plateau, degeneracy):
    count = min(len(ritz), len(reference))
    if not 1 <= plateau <= count:
        raise ValueError("Requested calibrated window exceeds available states")
    delta = np.zeros_like(ritz)
    weight = np.minimum(1.0, (count - np.arange(count)) / max(1, count - plateau))
    delta[:count] = (reference[:count] - ritz[:count]) * weight
    # Each unresolved Ritz multiplet gets one scalar correction, including
    # multiplets crossing the edge of the available VASP spectrum.
    starts = np.r_[0, np.where(np.diff(ritz) > degeneracy)[0] + 1, len(ritz)]
    for start, stop in itertools.pairwise(starts):
        if stop - start > 1:
            delta[start:stop] = delta[start:stop].mean()
    return ritz + delta


def align_ritz(ritz, reference, projected, plateau, degeneracy, width):
    levels = corrected_levels(ritz, reference, plateau, degeneracy)
    count = min(len(ritz), len(reference))
    stop = count

    def split(v, i):
        return 0 < i < len(v) and v[i] - v[i - 1] <= degeneracy

    while width > 0 and stop > 0:
        if split(ritz, stop) or split(reference, stop):
            stop -= 1
            continue
        # Use a common kernel centre for each input multiplet. The polar
        # factor then transforms covariantly under changes of degenerate basis.
        centres = ritz[:stop].copy()
        edges = np.r_[0, np.where(np.diff(reference[:stop]) > degeneracy)[0] + 1, stop]
        for first, last in itertools.pairwise(edges):
            centres[first:last] = centres[first:last].mean()
        kernel = np.exp(-0.5 * ((ritz[:stop, None] - centres[None, :]) / width) ** 2)
        u, sig, vh = svd(projected[:stop, :stop] * kernel, full_matrices=False)
        if sig[-1] > 1e-6:
            taper = np.minimum(1.0, (count - np.arange(stop)) / max(1, count - plateau))
            levels[:stop] = ritz[:stop] + taper * (reference[:stop] - ritz[:stop])
            return levels, u @ vh, float(sig[-1])
        stop -= 1
    return levels, np.empty((0, 0), complex), 0.0


def ordinary_geometry(src, ov):
    atoms = structure(src)
    geom = structure(ov)
    shells = json.loads(src.attrs["shells"])
    if (
        list(atoms.numbers) != list(geom.numbers)
        or not np.allclose(atoms.cell, geom.cell, atol=1e-7, rtol=0)
        or not np.allclose(
            src["structure/fractional"],
            ov["structure/fractional"],
            atol=1e-7,
            rtol=0,
        )
        or shells != json.loads(ov.attrs["shells"])
    ):
        raise ValueError(
            "Overlap and source must have the same structure, cell representatives and shells"
        )
    for sym, spec in shells.items():
        for i in range(len(spec)):
            for field in ["r", "R"]:
                key = f"basis/{sym}/{i}/{field}"
                if not np.array_equal(src[key][:], ov[key][:]):
                    raise ValueError("Radial basis mismatch")
    bounds = orbital_bounds(src)
    r_ao, s_ao = overlap_translations(ov, bounds)
    k = src["k/points"][:]
    s_k = bloch(k, r_ao, s_ao)
    return atoms, bounds, r_ao, s_ao, k, s_k


def overlap_mesh_alias(mesh_r, mesh_s, exact_r, exact_s):
    """Relative Brillouin-zone RMS error of interpolating the known AO overlap."""
    mesh = {tuple(r): s for r, s in zip(mesh_r, mesh_s)}
    error_sq = 0.0
    norm_sq = 0.0
    for r, block in zip(exact_r, exact_s):
        delta = mesh.pop(tuple(r), 0.0) - block
        error_sq += np.vdot(delta, delta).real
        norm_sq += np.vdot(block, block).real
    for block in mesh.values():
        error_sq += np.vdot(block, block).real
    return float(np.sqrt(error_sq / norm_sq))


def write_ordinary(out, src, atoms, bounds, r_ao, s_ao, k, s_k, hnew):
    # S(R) is known beyond the mesh image cell. Carry the energy origin on
    # that exact support so an absolute energy shift gives c*S(k) at every k.
    origin = (
        float(src.attrs["fermi_eV"])
        if "fermi_eV" in src.attrs
        else float(src["k/eigenvalues"][0, 0, 0])
    )
    rh, hr, s_mesh = localized_transform(
        k, src["k/mesh"][:], atoms, np.diff(bounds), hnew - origin * s_k, s_k
    )
    out.attrs["interpolation_reference_eV"] = origin
    out.attrs["overlap_mesh_alias_relative_rms"] = overlap_mesh_alias(
        rh, s_mesh, r_ao, s_ao
    )
    del s_mesh
    translations = np.unique(np.vstack([rh, r_ao]), axis=0)
    mapping = {tuple(r): i for i, r in enumerate(translations)}
    out["real/R"] = translations
    hds = out.create_dataset(
        "real/H",
        shape=(1, len(translations), bounds[-1], bounds[-1]),
        dtype="c16",
    )
    sds = out.create_dataset(
        "real/S", shape=(len(translations), bounds[-1], bounds[-1]), dtype="f8"
    )
    for i, r in enumerate(rh):
        hds[0, mapping[tuple(r)]] = hr[0, i]
    for i, r in enumerate(r_ao):
        row = mapping[tuple(r)]
        sds[row] = s_ao[i]
        hds[0, row] = hds[0, row] + origin * s_ao[i]
    out["k/H"][:] = hnew
    out["k/S"][:] = s_k


def calibrate(model, overlap_file, output, plateau=None, degeneracy=1e-5, width=1.0):
    if Path(output).resolve() in {Path(model).resolve(), Path(overlap_file).resolve()}:
        raise ValueError("Choose a separate output")
    if not np.isfinite(width) or width < 0:
        raise ValueError("Invalid alignment width")
    if not np.isfinite(degeneracy) or degeneracy < 0:
        raise ValueError("Invalid degeneracy tolerance")
    with h5py.File(model) as src, h5py.File(overlap_file) as ov:
        if (
            src.attrs.get("hamiltonian_method") != "direct_scalar_paw"
            or "subspace" in src
            or src.attrs["format"] != "vasp2mlh/1"
            or ov.attrs.get("format") != "vasp2mlh/overlap/1"
        ):
            raise ValueError(
                "Expected a direct scalar PAW archive and independent AO overlap file"
            )
        atoms, bounds, r_ao, s_ao, k, s_k = ordinary_geometry(src, ov)
        hnew = np.empty(src["k/H"].shape, complex)
        nocc = int(np.ceil(src.attrs["nelect"] / 2))
        count = min(src["k/projection"].shape[-2:])
        if not 0 < nocc <= count:
            raise ValueError(
                "The AO basis and source bands must accommodate the occupied manifold"
            )
        plateau = min(count, nocc + 8) if plateau is None else plateau
        if not 1 <= plateau <= count:
            raise ValueError("Invalid calibrated band count")
        atoms.set_cell(src["structure/projection_cell"][:], scale_atoms=True)
        with output_file(output) as tmp, h5py.File(tmp, "w") as out:
            for key in src:
                if key != "real":
                    src.copy(key, out)
            out.attrs.update(src.attrs)
            fidelity = []
            errors = []
            aligned = []
            minimum = []
            for ik, (h, s, ordinary) in enumerate(zip(src["k/H"][0], src["k/S"], s_k)):
                half, inverse = metric_factors(s)
                root, _ = metric_factors(ordinary)
                ritz, c = eigh(h, s)
                a = src["k/projection"][0, ik]
                adjusted, rot, smallest = align_ritz(
                    ritz,
                    src["k/eigenvalues"][0, ik],
                    adjoint(c) @ a,
                    plateau,
                    degeneracy,
                    width,
                )
                u = half @ c
                stop = len(rot)
                if stop:
                    u[:, :stop] = u[:, :stop] @ rot
                aligned.append(stop)
                minimum.append(smallest)
                b = root @ u
                hnew[0, ik] = (b * adjusted) @ adjoint(b)
                out["k/projection"][0, ik] = root @ (inverse @ a)
                edge = slice(max(0, nocc - 2), min(count, nocc + 2))
                # Fraction of the raw VASP edge-space weight in the four Ritz directions.
                edge_projection = inverse @ a[:, edge]
                fidelity.append(
                    float(
                        np.linalg.norm(adjoint(u[:, edge]) @ edge_projection) ** 2
                        / np.linalg.norm(edge_projection) ** 2
                    )
                )
                errors.append(
                    float(
                        np.max(
                            np.abs(
                                np.sort(adjusted)[:plateau]
                                - src["k/eigenvalues"][0, ik, :plateau]
                            )
                        )
                    )
                )
            write_ordinary(out, src, atoms, bounds, r_ao, s_ao, k, s_k, hnew)
            out["calibration/aligned_bands"] = aligned
            out["calibration/min_filtered_singular_value"] = minimum
            out["calibration/edge_subspace_fraction"] = fidelity
            out["calibration/mesh_ordered_error_eV"] = errors
            out.attrs.update(
                hamiltonian_method="ritz_calibrated",
                metric="ordinary",
                calibration_plateau=plateau,
                degeneracy_eV=degeneracy,
                alignment_width_eV=width,
                overlap_sha256=file_sha256(overlap_file),
            )
