"""Hermitian atom-pair sparsification and comparison to the untruncated AO model."""

from pathlib import Path
import json

import h5py
import numpy as np
from scipy.linalg import eigvalsh, block_diag

from .model import adjoint
from .pipeline import output_file
from .realspace import SPARSE_FORMAT, matrices_k, pair_blocks, reverse, structure


def compare(source, sparse, points, original_h, original_s, band_range=None):
    h, s = matrices_k(sparse, points)
    smin = float(np.linalg.eigvalsh(s).min())
    report = dict(
        nk=len(points),
        min_S=smin,
        max_delta_H_eV=float(np.max(abs(h - original_h))),
        max_delta_S=float(np.max(abs(s - original_s))),
    )
    if smin <= 0:
        raise ValueError(
            f"Truncated S is not positive definite on validation points (min={smin:.6g})"
        )

    def energies(h, s):
        if source.attrs["spinor"]:
            s = [block_diag(x, x) for x in s]
        return np.array([[eigvalsh(a, b) for a, b in zip(channel, s)] for channel in h])

    ref = energies(original_h, original_s)
    diff = energies(h, s) - ref
    report["max_band_change_eV"] = float(np.max(abs(diff)))
    report["band_change_rmse_eV"] = float(np.sqrt(np.mean(diff**2)))
    if band_range is not None:
        first, last = band_range
        if first < 1 or last < first or last > diff.shape[-1]:
            raise ValueError(
                "Band range must be within the 1-based ordered model spectrum"
            )
        selected = diff[:, :, first - 1 : last]
        report["window_max_band_change_eV"] = float(np.max(np.abs(selected)))
        report["window_band_change_rmse_eV"] = float(np.sqrt(np.mean(selected**2)))
    return report


def sparsify(
    model,
    output,
    cutoff=None,
    h_tol=0.0,
    s_tol=0.0,
    max_band_change=None,
    extra_points=None,
    band_range=None,
    spinless_time_reversal=False,
):
    if Path(model).resolve() == Path(output).resolve():
        raise ValueError("Use a separate destination for the sparse model")
    for value in (cutoff, h_tol, s_tol, max_band_change):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError("Cutoffs and tolerances must be finite and nonnegative")
    with h5py.File(model) as src:
        if src.attrs["format"] != "vasp2mlh/1":
            raise ValueError("Expected an untruncated VASP2MLH projection file")
        if spinless_time_reversal and (
            src.attrs["spinor"]
            or src.attrs["spin_channels"] != 1
            or src.attrs.get("soc", False)
        ):
            raise ValueError(
                "Spinless time reversal requires a nonmagnetic scalar model in a real AO basis"
            )
        imag_h = imag_s = 0.0
        geometry_overlap = src.attrs.get("metric") == "ordinary"
        atoms = structure(src)
        pairs = {key: (h, s) for key, h, s in pair_blocks(src)}
        keys, hs, ss = [], [], []
        asym_h = asym_s = 0.0
        for key, (h, s) in pairs.items():
            peer = reverse(key)
            if peer not in pairs:
                raise ValueError(f"Missing conjugate atom pair for {key}")
            if key > peer:
                continue
            rh, rs = pairs[peer]
            asym_h = max(asym_h, float(np.max(abs(h - adjoint(rh)))))
            asym_s = max(asym_s, float(np.max(abs(s - adjoint(rs)))))
            h = (h + adjoint(rh)) / 2
            s = (s + adjoint(rs)) / 2
            if spinless_time_reversal:
                # In the real tesseral AO gauge, (M(k) + M(-k)*)/2 is Re M(R).
                imag_h = max(imag_h, float(np.max(np.abs(h.imag))))
                imag_s = max(imag_s, float(np.max(np.abs(s.imag))))
                h, s = h.real, s.real
            x, y, z, i, j = key
            dr = (
                atoms.positions[j]
                - atoms.positions[i]
                + np.array([x, y, z]) @ atoms.cell
            )
            # Apply the same pruning decision to a bond and its Hermitian reverse.
            if key != peer and (
                (cutoff is not None and np.linalg.norm(dr) > cutoff + 1e-10)
                or (
                    max(np.linalg.norm(channel) for channel in h) <= h_tol
                    and (geometry_overlap or np.linalg.norm(s) <= s_tol)
                )
            ):
                if geometry_overlap and np.any(s):
                    # Geometry determines S even when the H block is truncated.
                    h = np.zeros_like(h)
                else:
                    continue
            keys.append(key)
            hs.append(h.reshape(len(h), -1))
            ss.append(s.ravel())
        if asym_h > 1e-7 or asym_s > 1e-9:
            raise ValueError(
                f"Input H/S violate Hermiticity ({asym_h:.4g} eV, {asym_s:.4g})"
            )
        with output_file(output) as tmp, h5py.File(tmp, "w") as out:
            out.attrs.update(src.attrs)
            out.attrs["format"] = SPARSE_FORMAT
            for name in (
                "structure",
                "basis",
                "k/points",
                "k/weights",
                "k/mesh",
                "k/eigenvalues",
                "k/occupation",
                "k/gram_error",
                "k/projectability",
            ):
                src.copy(name, out.require_group("k") if name.startswith("k/") else out)
            if "calibration" in src:
                src.copy("calibration", out)
            for diagnostic in (
                "k/operator_edge_rayleigh_error_eV",
                "k/operator_edge_residual_norm_eV",
            ):
                if diagnostic in src:
                    src.copy(diagnostic, out["k"])
            out["pairs/indices"] = keys
            out["pairs/offsets"] = np.r_[0, np.cumsum([len(s) for s in ss])]
            out["pairs/H"] = np.concatenate(hs, axis=1)
            out["pairs/S"] = np.concatenate(ss)
            report = dict(
                band_range_1based=band_range,
                source_directed_pairs=len(pairs),
                stored_unique_pairs=len(keys),
                stored_nonzero_H_blocks=sum(bool(np.any(h)) for h in hs),
                retained_directed_pairs=2 * len(keys) - len(atoms),
                source_pair_hermiticity_H_eV=asym_h,
                source_pair_hermiticity_S=asym_s,
                cutoff_A=cutoff,
                h_tol_eV=h_tol,
                s_tol=s_tol,
                max_allowed_band_change_eV=max_band_change,
            )
            if spinless_time_reversal:
                report.update(
                    spinless_time_reversal=True,
                    removed_imag_H_max_eV=imag_h,
                    removed_imag_S_max=imag_s,
                )
            grid = src["k/points"][:]
            report["mesh"] = compare(
                src, out, grid, src["k/H"][:], src["k/S"][:], band_range
            )
            if extra_points is not None:
                k = np.atleast_2d(extra_points)
                if k.shape[1] != 3 or not len(k) or not np.isfinite(k).all():
                    raise ValueError("Expected finite fractional validation points")
                report["extra_points"] = compare(
                    src, out, k, *matrices_k(src, k), band_range=band_range
                )
                report["extra_points"]["points"] = k.tolist()
            error_key = (
                "max_band_change_eV"
                if band_range is None
                else "window_max_band_change_eV"
            )
            for group in ("mesh", "extra_points"):
                if (
                    group in report
                    and max_band_change is not None
                    and report[group][error_key] > max_band_change
                ):
                    raise ValueError(
                        f"{group} band change {report[group][error_key]:.6g} eV exceeds {max_band_change} eV"
                    )
            out.attrs["truncation_report"] = json.dumps(report)
    return report
