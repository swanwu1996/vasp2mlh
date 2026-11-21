"""Variational projection on spectrally filtered PAW trial orbitals."""

from pathlib import Path
import json

import h5py
import numpy as np
from scipy.linalg import block_diag, eigh
from scipy.special import expit

from .configuration import BOUNDED_COLLINEAR, SCALAR_POLAR
from .polar import MIN_POLAR_SINGULAR_VALUE, anchor_polar_window  # re-export the scalar API
from .calibrate import metric_factors, ordinary_geometry, write_ordinary
from .model import adjoint, spin_rotation
from .pipeline import file_sha256, output_file
from .vasp import is_hybrid


def bound_hubbard_spectrum(h, recipe):
    """Smoothly bound only unresolved high Ritz levels in an orthonormal frame."""
    edge = recipe["linear_through_eV"]
    width = recipe["saturation_width_eV"]
    values, vectors = np.linalg.eigh(h)
    high = values > edge
    values[high] = edge + width * np.tanh((values[high] - edge) / width)
    return (vectors * values[..., None, :]) @ adjoint(vectors)


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
    model, overlap, output, centre_offset, width, floor, taper, *, spectral=False, hubbard_recipe=None, collinear_recipe=None, polar_recipe=None
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
            or src.attrs.get("hamiltonian_method", "spectral") not in (("spectral",) if spectral else ("direct_scalar_paw", "direct_hubbard_paw"))
            or ov.attrs.get("format") != "vasp2mlh/overlap/1"
            or src["k/H"].shape[0] not in ((1,) if spectral else (1, 2))
            or "subspace" in src
        ):
            raise ValueError(
                "Model method, spin channel count or AO overlap is incompatible with spectral filtering"
            )
        if (
            src.attrs.get("hamiltonian_method") == "direct_hubbard_paw"
            and hubbard_recipe is None
        ):
            raise ValueError("Use v2h with the target energy window for a DFT+U reconstruction")
        if collinear_recipe is not None and (
            src.attrs.get("hamiltonian_method") != "direct_scalar_paw"
            or src["k/H"].shape[0] != 2 or spectral or hubbard_recipe is not None
        ):
            raise ValueError("A collinear window requires a two-channel direct semilocal operator")
        if polar_recipe is not None and (
            src.attrs.get("hamiltonian_method") != "direct_scalar_paw"
            or src["k/H"].shape[0] != 1 or src.attrs.get("spinor", False)
            or spectral or hubbard_recipe is not None or collinear_recipe is not None
        ):
            raise ValueError("A scalar polar window requires a nonmagnetic semilocal operator")
        bound = polar_recipe or collinear_recipe or hubbard_recipe
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
        energies = src["k/eigenvalues"][:]
        ceiling = float(energies[:, :, -1].min())
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
            max(0, nocc - 2), min(nocc + 2, energies.shape[-1], src["k/H"].shape[-1])
        )
        if (
            not 1 <= nocc <= min(energies.shape[-1], src["k/H"].shape[-1])
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
            for spin, channel in enumerate(energies):
                for i, energy in enumerate(channel):
                    _, inverse = metric_factors(src["k/S"][i])
                    root, _ = metric_factors(ordinary[i])
                    if spinor:
                        inverse, root = (
                            block_diag(inverse, inverse),
                            block_diag(root, root),
                        )
                    v = inverse @ src["k/projection"][spin, i]
                    if spinor:
                        v = (
                            spin_rotation(src.attrs["saxis"]) @ v.reshape(2, -1)
                        ).reshape(v.shape)
                    if spectral:
                        h = spectral_baseline(
                            v, energy, ceiling, taper, src.attrs["complement_eV"]
                        )
                    else:
                        h = inverse @ src["k/H"][spin, i] @ inverse
                    effective = filtered_operator(
                        h, v, energy, centre, width, floor, ceiling, taper
                    )
                    if bound is not None:
                        effective = bound_hubbard_spectrum(effective, bound)
                    if polar_recipe is not None:
                        try:
                            effective = anchor_polar_window(effective, v, energy, polar_recipe["band_range_1based"])
                        except ValueError as exc:
                            raise ValueError(f"{exc} at k={k[i].tolist()}") from exc
                    hnew[spin, i] = root @ effective @ root
                    # Preserve the bare AO projection convention of ordinary-metric archives.
                    out["k/projection"][spin, i] = root @ v
                    coverage.append(
                        np.linalg.svd(v[:, edge], compute_uv=False)[-1] ** 2
                    )
                    errors.append(eigh(effective, eigvals_only=True)[edge] - energy[edge])
            write_ordinary(out, src, atoms, bounds, r, s_real, k, ordinary, hnew)
            out["filter/min_edge_weight"] = coverage
            out["filter/mesh_edge_errors_eV"] = errors
            out.attrs.update(
                hamiltonian_method=(
                    "spinor_spectral_ritz" if spinor else "hybrid_spectral_ritz"
                )
                if spectral
                else ("hubbard-paw-ritz-v1" if src.attrs.get("hamiltonian_method") == "direct_hubbard_paw" else "filtered_paw_ritz"),
                metric="ordinary",
                filter_centre_eV=centre,
                filter_width_eV=width,
                filter_floor=floor,
                filter_ceiling_eV=ceiling,
                filter_taper_eV=taper,
                overlap_sha256=file_sha256(overlap),
            )

            if polar_recipe is not None:
                out.attrs["hamiltonian_method"] = SCALAR_POLAR
                out.attrs["polar_recipe"] = json.dumps(polar_recipe)
            elif collinear_recipe is not None:
                out.attrs["hamiltonian_method"] = BOUNDED_COLLINEAR
                out.attrs["collinear_recipe"] = json.dumps(collinear_recipe)
            elif hubbard_recipe is not None:
                out.attrs["hubbard_recipe"] = json.dumps(hubbard_recipe)
            if spinor:
                out.attrs["source_saxis"] = src.attrs["saxis"]
                out.attrs["saxis"] = [0.0, 0.0, 1.0]
            if spectral:
                out.attrs["complement_policy"] = (
                    "filter_centre_plus_two_tapers_plus_10_eV"
                )
