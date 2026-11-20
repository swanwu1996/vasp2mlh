"""Impose a selected VASP eigenspace in the nonorthogonal AO Hamiltonian."""

from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import block_diag, cholesky, eigvalsh, solve_triangular, svd

from .model import adjoint, localized_transform
from .pipeline import output_file
from .realspace import orbital_bounds, structure


def constrained_matrix(h, s, projection, energies, min_weight):
    # In the orthonormal AO frame, the polar factor is the closest isometry
    # to the columns of the projected Bloch states (symmetric orthogonalization).
    lower = cholesky(s, lower=True)
    w = solve_triangular(lower, projection, lower=True)
    u, singular, vh = svd(w, full_matrices=False)
    weight = float(singular[-1] ** 2)
    if len(singular) < len(energies) or weight < min_weight:
        raise ValueError(
            f"Projected subspace is rank deficient or poorly represented (minimum weight {weight:.5g})"
        )
    q = u @ vh
    rest = lower @ (np.eye(len(s)) - q @ adjoint(q))
    orth_h = solve_triangular(lower, h, lower=True)
    orth_h = solve_triangular(lower, adjoint(orth_h), lower=True)
    b = lower @ q
    constrained = (b * energies) @ adjoint(b) + rest @ orth_h @ adjoint(rest)
    constrained = (constrained + adjoint(constrained)) / 2
    spectrum = eigvalsh(constrained, s)
    error = float(np.abs(spectrum[: len(energies)] - energies).max())
    if error > 1e-6:
        raise ValueError(
            f"Complement intrudes into the constrained bands (ordered spectrum error {error:.5g} eV)"
        )
    return constrained, weight, error


def downfold(model, output, band_count, min_weight=0.5):
    if Path(model).resolve() == Path(output).resolve():
        raise ValueError("Choose a separate output for the constrained Hamiltonian")
    if not 0 < min_weight <= 1 or band_count < 1:
        raise ValueError("Require positive band count and minimum weight in (0, 1]")
    with h5py.File(model) as src:
        if src.attrs["format"] != "vasp2mlh/1" or "subspace" in src:
            raise ValueError("Use a dense, unconstrained projection archive")
        h, s, a, e = (
            src["k/" + key] for key in ["H", "S", "projection", "eigenvalues"]
        )
        if band_count > min(a.shape[-2:]):
            raise ValueError(
                "Requested band count exceeds available VASP states or AO dimension"
            )
        with output_file(output) as tmp, h5py.File(tmp, "w") as out:
            for key in src:
                src.copy(key, out)
            out.attrs.update(src.attrs)
            new_h = np.empty(h.shape, dtype=complex)
            weights = np.empty(h.shape[:2])
            errors = np.empty_like(weights)
            for channel, ik in np.ndindex(h.shape[:2]):
                metric = s[ik]
                if src.attrs["spinor"]:
                    metric = block_diag(metric, metric)
                try:
                    new_h[channel, ik], weights[channel, ik], errors[channel, ik] = (
                        constrained_matrix(
                            h[channel, ik],
                            metric,
                            a[channel, ik, :, :band_count],
                            e[channel, ik, :band_count],
                            min_weight,
                        )
                    )
                except ValueError as exc:
                    raise ValueError(f"Spin {channel}, mesh point {ik}: {exc}") from exc
            atoms = structure(src)
            # Older archives stored the WAVECAR cell. VASP XML writes the
            # projection geometry to eight decimals; image ties can resolve differently.
            cells = (
                [src["structure/projection_cell"][:]]
                if "structure/projection_cell" in src
                else [atoms.cell.array, np.round(atoms.cell.array, 8)]
            )
            for cell in cells:
                atoms.set_cell(cell, scale_atoms=True)
                r, hr, sr = localized_transform(
                    src["k/points"][:],
                    src["k/mesh"][:],
                    atoms,
                    np.diff(orbital_bounds(src)),
                    new_h,
                    s[:],
                )
                if np.array_equal(r, src["real/R"][:]) and np.allclose(
                    sr, src["real/S"][:], atol=1e-11, rtol=1e-10
                ):
                    break
            else:
                raise ValueError(
                    "Cannot reproduce the stored real-space overlap image convention"
                )
            out["k/H"][:] = new_h
            out["real/H"][:] = hr
            out["subspace/min_weight"] = weights
            out["subspace/max_ordered_error_eV"] = errors
            out.attrs.update(
                subspace_bands=band_count,
                subspace_weight_threshold=min_weight,
                subspace_min_weight=float(weights.min()),
                subspace_mesh_max_error_eV=float(errors.max()),
            )
