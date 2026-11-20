"""Variational projection on spectrally filtered PAW trial orbitals."""

from pathlib import Path
import json

import h5py
import numpy as np
from scipy.linalg import block_diag, eigh
from scipy.special import expit

from .calibrate import metric_factors, ordinary_geometry, write_ordinary
from .model import adjoint, spin_rotation
from .pipeline import file_sha256, output_file
from .vasp import is_hybrid


def upper_taper(energies, ceiling, taper):
    x = np.clip((ceiling - energies) / taper, 0, 1)
    return x**2 * (3 - 2 * x)


def spectral_baseline(v, energies, ceiling, taper, complement):
    """PAW spectral moment with a common upper cutoff and a flat complement."""
    moment = (energies - complement) * upper_taper(energies, ceiling, taper)
    return (v * moment) @ adjoint(v) + complement * np.eye(v.shape[0])


def filtered_operator(
    h,
    projections,
    energies,
    centre,
    width,
    floor,
    ceiling,
    taper,
    *,
    return_projections=False,
):
    """All inputs use the orthonormal PAW trial frame and one absolute energy zero."""
    weight = (
        (1 - floor)
        * expit((centre - energies) / width)
        * upper_taper(energies, ceiling, taper)
    )
    weighted = projections * weight
    gram = floor * np.eye(len(h)) + weighted @ adjoint(projections)
    moment = floor * h + (weighted * energies) @ adjoint(projections)
    _, inverse = metric_factors(gram)
    effective = inverse @ moment @ inverse
    effective = (effective + adjoint(effective)) / 2
    if return_projections:
        # <orthonormal filtered trial | native PAW eigenstate>.
        return effective, (inverse @ projections) * np.sqrt(floor + weight)
    return effective


def filter_model(
    model, overlap, output, centre_offset, width, floor, taper, *, spectral=False
):
    if Path(output).resolve() in {Path(p).resolve() for p in (model, overlap)}:
        raise ValueError("Choose a separate output")
    if not np.isfinite([centre_offset, width, floor, taper]).all() or not (
        width > 0 and taper > 0 and 0 < floor <= 1
    ):
        raise ValueError("Require finite energies, positive widths and 0 < floor <= 1")
    with h5py.File(model) as src, h5py.File(overlap) as ov:
        if (
            src.attrs.get("format") != "vasp2mlh/1"
            or (
                src.attrs.get("hamiltonian_method", "spectral")
                != ("spectral" if spectral else "direct_scalar_paw")
            )
            or ov.attrs.get("format") != "vasp2mlh/overlap/1"
            or src["k/H"].shape[0] != 1
            or "subspace" in src
        ):
            raise ValueError(
                "Model method, spin channel count or AO overlap is incompatible with spectral filtering"
            )
        spinor = bool(src.attrs.get("spinor", False))
        if (
            spectral
            and not spinor
            and not is_hybrid(
                {"operator_parameters": json.loads(src.attrs["xc_settings"])}
            )
        ):
            raise ValueError("A hybrid spectral model requires hybrid eigenpairs")
        atoms, bounds, r, s_real, k, ordinary = ordinary_geometry(src, ov)
        atoms.set_cell(src["structure/projection_cell"][:], scale_atoms=True)
        energies = src["k/eigenvalues"][0]
        ceiling = float(energies[:, -1].min())
        centre = float(src.attrs["fermi_eV"] + centre_offset)
        if spectral:
            required = centre + 2 * taper
            if ceiling < required:
                raise ValueError(
                    f"WAVECAR spectrum ends at {ceiling:.4f} eV; "
                    f"increase NBANDS to cover {required:.4f} eV across the source mesh"
                )
            ceiling = required
        if centre >= ceiling:
            raise ValueError("Filter centre reaches the available VASP band ceiling")
        nocc = int(np.ceil(src.attrs["nelect"] / (1 if spinor else 2)))
        edge = slice(
            max(0, nocc - 2), min(nocc + 2, len(energies[0]), src["k/H"].shape[-1])
        )
        if (
            not 1 <= nocc <= min(len(energies[0]), src["k/H"].shape[-1])
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
                if spinor:
                    inverse, root = block_diag(inverse, inverse), block_diag(root, root)
                v = inverse @ src["k/projection"][0, i]
                if spinor:
                    v = (spin_rotation(src.attrs["saxis"]) @ v.reshape(2, -1)).reshape(
                        v.shape
                    )
                if spectral:
                    h = spectral_baseline(
                        v, energy, ceiling, taper, src.attrs["complement_eV"]
                    )
                else:
                    h = inverse @ src["k/H"][0, i] @ inverse
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
                hamiltonian_method=(
                    "spinor_spectral_ritz" if spinor else "hybrid_spectral_ritz"
                )
                if spectral
                else "filtered_paw_ritz",
                metric="ordinary",
                filter_centre_eV=centre,
                filter_width_eV=width,
                filter_floor=floor,
                filter_ceiling_eV=ceiling,
                filter_taper_eV=taper,
                overlap_sha256=file_sha256(overlap),
            )

            if spinor:
                out.attrs["source_saxis"] = src.attrs["saxis"]
                out.attrs["saxis"] = [0.0, 0.0, 1.0]
            if spectral:
                out.attrs["complement_policy"] = (
                    "filter_centre_plus_two_tapers_plus_10_eV"
                )
