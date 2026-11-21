"""Target-window anchoring in an orthonormal PAW projection frame."""

import json

import h5py
import numpy as np

from .calibrate import aligned_pbe_reference, metric_factors, ordinary_geometry, write_ordinary
from .configuration import HSE_POLAR
from .model import adjoint
from .pipeline import output_file, paths_collide
from .spillage import DEGENERACY_EV, complete_multiplets
from .vasp import screened_hybrid_maturity


MIN_POLAR_SINGULAR_VALUE = 1e-6


def polar_window_frame(projections, energies, band_range):
    """Close the target multiplet and take its full-rank polar factor."""
    first, last = band_range
    if not 1 <= first <= last <= min(projections.shape[0], len(energies)):
        raise ValueError("Polar window exceeds the ordered AO or VASP spectrum")
    edge = complete_multiplets(energies, first - 1, last)
    if edge.stop == len(energies):
        raise ValueError("Increase NBANDS to resolve the upper polar-window multiplet")
    columns = projections[:, edge]
    left, singular, right = np.linalg.svd(columns, full_matrices=False)
    if len(singular) < columns.shape[1] or singular[-1] < MIN_POLAR_SINGULAR_VALUE:
        raise ValueError("Polar-window PAW projection is rank deficient")
    return left @ right, edge


def complete_polar_frame(prior, frame, levels):
    """Retain the prior on the orthogonal complement of locked target eigenpairs."""
    complement = np.eye(len(prior)) - frame @ adjoint(frame)
    anchored = (frame * levels) @ adjoint(frame) + complement @ prior @ complement
    return (anchored + adjoint(anchored)) / 2


def anchor_polar_window(prior, projections, energies, band_range):
    """Impose native eigenpairs on the closest orthonormal bare PAW target frame."""
    frame, edge = polar_window_frame(projections, energies, band_range)
    return complete_polar_frame(prior, frame, energies[edge])


def hse_polar_model(model, overlap_file, pbe_model, output, band_range):
    """Anchor HSE eigenpairs and interpolate the correction to the dense PBE prior."""
    if paths_collide(output, (model, overlap_file, pbe_model)):
        raise ValueError("Choose a separate HSE polar model destination")
    with h5py.File(model) as src, h5py.File(overlap_file) as ov, h5py.File(pbe_model) as pbe:
        xc = json.loads(src.attrs["xc_settings"])
        if (
            src.attrs["metric"] != "PAW" or src.attrs["spinor"] or src.attrs["soc"]
            or src["k/H"].shape[0] != 1 or not screened_hybrid_maturity({"operator_parameters": xc})
        ):
            raise ValueError("HSE polar requires a scalar screened-hybrid PAW projection")
        if (
            pbe.attrs["metric"] != "ordinary"
            or pbe.attrs.get("hamiltonian_method") != "locality_constrained_paw"
            or pbe.attrs["potcar_sha256"] != src.attrs["potcar_sha256"]
        ):
            raise ValueError("Expected the matched ordinary-AO PBE reference model")
        atoms, bounds, translations, s_real, k, sk = ordinary_geometry(src, ov)
        atoms.set_cell(src["structure/projection_cell"][:], scale_atoms=True)
        offset, reference = aligned_pbe_reference(pbe, ov, k, src.attrs["fermi_eV"])
        ref_k = reference[0]
        hnew = np.empty(src["k/H"].shape, complex)
        with output_file(output) as tmp, h5py.File(tmp, "w") as out:
            out.attrs.update(src.attrs)
            for key in src:
                if key != "real":
                    src.copy(key, out)
            worst_action = 0.0
            ordered_errors = []
            intrusion_points = 0
            first, last = band_range
            for i, ordinary in enumerate(sk):
                root, inverse = metric_factors(ordinary)
                paw_inverse = metric_factors(src["k/S"][i])[1]
                projected = paw_inverse @ src["k/projection"][0, i]
                prior = inverse @ ref_k[i] @ inverse
                energies = src["k/eigenvalues"][0, i]
                frame, edge = polar_window_frame(projected, energies, band_range)
                effective = complete_polar_frame(prior, frame, energies[edge])
                residual = effective @ frame - frame * energies[edge]
                worst_action = max(worst_action, float(np.linalg.norm(residual, axis=0).max()))
                spectrum = np.linalg.eigvalsh(effective)
                ordered_errors.extend(spectrum[first - 1:last] - energies[first - 1:last])
                z = np.linalg.qr(frame, mode="complete")[0][:, frame.shape[1]:]
                complement_levels = np.linalg.eigvalsh(adjoint(z) @ prior @ z)
                lo = energies[edge.start] + DEGENERACY_EV
                hi = energies[edge.stop - 1] - DEGENERACY_EV
                intrusion_points += int(np.any((complement_levels > lo) & (complement_levels < hi)))
                hnew[0, i] = root @ effective @ root
                out["k/projection"][0, i] = root @ projected
            write_ordinary(out, src, atoms, bounds, translations, s_real, k, sk, hnew, reference=reference)
            del out.attrs["complement_eV"]
            diagnostics = dict(
                kpoints=len(k),
                complement_intrusion_k=intrusion_points,
                max_target_action_eV=worst_action,
                ordered_rms_eV=float(np.mean(np.square(ordered_errors)) ** 0.5),
                ordered_max_eV=float(np.max(np.abs(ordered_errors))),
            )
            recipe = dict(
                method=HSE_POLAR,
                band_range_1based=list(band_range),
                degeneracy_eV=DEGENERACY_EV,
                min_singular_value=MIN_POLAR_SINGULAR_VALUE,
                target="HSE_eigenpairs",
                alignment="source_fermi_difference",
                prior_method=str(pbe.attrs["hamiltonian_method"]),
                prior_recipe=json.loads(pbe.attrs["locality_recipe"]),
            )
            out.attrs.update(
                hamiltonian_method=HSE_POLAR,
                metric="ordinary",
                complement_policy="fermi_aligned_pbe_matrix",
                pbe_alignment_eV=offset,
                polar_recipe=json.dumps(recipe),
                polar_source_diagnostics=json.dumps(diagnostics),
            )
