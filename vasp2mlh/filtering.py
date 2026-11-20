"""Variational projection on spectrally filtered PAW trial orbitals."""

from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import eigh
from scipy.special import expit

from .calibrate import metric_factors, ordinary_geometry, write_ordinary
from .model import adjoint
from .pipeline import file_sha256, output_file


def filtered_operator(h, projections, energies, centre, width, floor, ceiling, taper):
    """All inputs use the orthonormal PAW trial frame and one absolute energy zero."""
    x = np.clip((ceiling - energies) / taper, 0, 1)
    weight = (1 - floor) * expit((centre - energies) / width) * x**2 * (3 - 2 * x)
    weighted = projections * weight
    gram = floor * np.eye(len(h)) + weighted @ adjoint(projections)
    moment = floor * h + (weighted * energies) @ adjoint(projections)
    _, inverse = metric_factors(gram)
    effective = inverse @ moment @ inverse
    return (effective + adjoint(effective)) / 2


def filter_model(model, overlap, output, centre_offset, width, floor, taper):
    if Path(output).resolve() in {Path(p).resolve() for p in (model, overlap)}:
        raise ValueError("Choose a separate output")
    if not np.isfinite([centre_offset, width, floor, taper]).all() or not (
        width > 0 and taper > 0 and 0 < floor <= 1
    ):
        raise ValueError("Require finite energies, positive widths and 0 < floor <= 1")
    with h5py.File(model) as src, h5py.File(overlap) as ov:
        if (
            src.attrs.get("format") != "vasp2mlh/1"
            or src.attrs.get("hamiltonian_method") != "direct_scalar_paw"
            or ov.attrs.get("format") != "vasp2mlh/overlap/1"
            or src["k/H"].shape[0] != 1
            or "subspace" in src
        ):
            raise ValueError(
                "Expected a scalar direct PAW model and ordinary AO overlap"
            )
        atoms, bounds, r, s_real, k, ordinary = ordinary_geometry(src, ov)
        atoms.set_cell(src["structure/projection_cell"][:], scale_atoms=True)
        energies = src["k/eigenvalues"][0]
        ceiling = float(energies[:, -1].min())
        centre = float(src.attrs["fermi_eV"] + centre_offset)
        if centre >= ceiling:
            raise ValueError("Filter centre reaches the available VASP band ceiling")
        nocc = int(np.ceil(src.attrs["nelect"] / 2))
        edge = slice(max(0, nocc - 2), min(nocc + 2, len(energies[0]), bounds[-1]))
        if (
            not 1 <= nocc <= min(len(energies[0]), bounds[-1])
            or edge.stop <= edge.start
        ):
            raise ValueError("AO or VASP spectrum does not cover the occupied edge")
        hnew = np.empty(src["k/H"].shape, complex)
        with output_file(output) as tmp, h5py.File(tmp, "w") as out:
            for key in src:
                if key != "real":
                    src.copy(key, out)
            out.attrs.update(src.attrs)
            coverage, errors = [], []
            for i, energy in enumerate(energies):
                _, inverse = metric_factors(src["k/S"][i])
                root, _ = metric_factors(ordinary[i])
                h = inverse @ src["k/H"][0, i] @ inverse
                v = inverse @ src["k/projection"][0, i]
                effective = filtered_operator(
                    h, v, energy, centre, width, floor, ceiling, taper
                )
                hnew[0, i] = root @ effective @ root
                # Preserve the bare AO projection convention of ordinary-metric archives.
                out["k/projection"][0, i] = root @ v
                coverage.append(np.linalg.svd(v[:, edge], compute_uv=False)[-1] ** 2)
                errors.append(eigh(effective, eigvals_only=True)[edge] - energy[edge])
            write_ordinary(out, src, atoms, bounds, r, s_real, k, ordinary, hnew)
            out["filter/min_edge_weight"] = coverage
            out["filter/mesh_edge_errors_eV"] = errors
            out.attrs.update(
                hamiltonian_method="filtered_paw_ritz",
                metric="ordinary",
                filter_centre_eV=centre,
                filter_width_eV=width,
                filter_floor=floor,
                filter_ceiling_eV=ceiling,
                filter_taper_eV=taper,
                overlap_sha256=file_sha256(overlap),
            )
