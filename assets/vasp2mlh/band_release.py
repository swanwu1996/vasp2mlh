"""Build and qualify a fixed-basis scalar band model against independent VASP energies."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from ase.io import write

from . import __version__
from .comparison import periodic_match
from .filtering import filter_model
from .locality import locality_model, PRIOR_OFFSET_EV
from .overlap import overlap
from .pipeline import bands, build, file_sha256, output_file
from .registry import fingerprint, inspect_basis, read_registry
from .sparse import sparsify
from .vasp import is_hybrid, spectral_mode, metadata, screened_hybrid_maturity


def check_scalar(meta):
    settings = meta["operator_parameters"]
    if meta["lreal"] not in ("F", ".FALSE."):
        raise ValueError(
            f"VASP used LREAL={meta['lreal']}. For scalar reconstruction, rerun "
            "with LREAL=.FALSE., ISYM=-1 and WRT_POTENTIAL=total paw on a full mesh."
        )
    if (
        meta["nspin"] != 1
        or meta["spinor"]
        or meta["soc"]
        or any(
            settings.get(tag, "F") not in ("F", ".FALSE.", "FALSE")
            for tag in ("LHFCALC", "LDAU", "LUSE_VDW", "LSPIRAL")
        )
        or settings.get("METAGGA", "") not in ("", "--", "F", "NONE", ".FALSE.")
        or settings.get("XC", "") not in ("", "--", "NONE")
    ):
        raise ValueError(
            "Band release requires scalar nonmagnetic semilocal LDA/GGA with reciprocal PAW projectors; custom XC combinations are unsupported"
        )
    if settings.get("GGA") in ("LI", "LIBXC") and not settings.get("LIBXC1"):
        raise ValueError("Missing Libxc functional identity in VASP XML")


def check_hybrid(meta):
    settings = meta["operator_parameters"]
    # The generalized-KS spectrum is the source of the nonlocal exchange contribution.
    scalar_settings = {**settings, "LHFCALC": "F"}
    check_scalar({**meta, "operator_parameters": scalar_settings})
    if meta["icharg"] not in (0, 1, 2):
        raise ValueError(
            "Hybrid bands require self-consistent occupied orbitals (ICHARG=0, 1 or 2)"
        )
    fraction = float(settings.get("AEXX", "nan"))
    screening = float(settings.get("HFSCREEN", "nan"))
    if not 0 < fraction <= 1 or not np.isfinite(screening) or screening < 0:
        raise ValueError("Missing or invalid AEXX or HFSCREEN in hybrid XML")


def independent_points(source, reference, *, time_reversal=True):
    """Keep one reference sample per time-reversal pair, modulo reciprocal translations."""
    selected = []
    src = np.asarray(source)
    ref = np.asarray(reference)
    if not np.isfinite(ref).all() or not np.isfinite(src).all():
        raise ValueError("Non-finite k coordinates")
    for i, point in enumerate(ref):
        if periodic_match(src, point).any() or (
            time_reversal and periodic_match(src, -point).any()
        ):
            continue
        previous = ref[selected]
        if not periodic_match(previous, point).any() and (
            not time_reversal or not periodic_match(previous, -point).any()
        ):
            selected.append(i)
    if not selected:
        raise ValueError(
            "No independent k points remain after mesh, periodic and time-reversal matching"
        )
    return np.asarray(selected)


def check_convergence(meta, directory):
    at_limit = 0 < meta["nelm"] <= meta["final_scf_steps"]
    # STOPCAR may end a run before NELM and still produce a complete XML.
    outcar = directory / "OUTCAR"
    reason = None
    if outcar.is_file():
        with outcar.open("rb") as log:
            for line in log:
                if b"----" in line and b" Iteration " in line:
                    reason = None
                if b"aborting loop because" in line:
                    reason = line.strip()
    if reason and b"EDIFF is reached" in reason:
        return
    if reason and b"stop" in reason.lower():
        raise ValueError("VASP stopped before final electronic convergence")
    if at_limit:
        raise ValueError(
            "Electronic iteration limit reached without confirmed final convergence"
        )
    if spectral_mode(meta):
        delta = meta.get("final_energy_change_eV")
        ediff = meta["ediff_eV"]
        if (
            delta is None
            or not np.isfinite([delta, ediff]).all()
            or ediff <= 0
            or delta > ediff + 1.01e-8
        ):
            raise ValueError(
                "SCF convergence is unconfirmed by the final energy change or OUTCAR"
            )


def window_mask(native, model, band_range, window):
    """Score every ordered level entering the declared window from either spectrum."""
    if not np.isfinite(native).all() or not np.isfinite(model).all():
        raise ValueError("Non-finite band energies")
    count = min(native.shape[1], model.shape[1])
    indices = np.arange(1, count + 1)
    if band_range is not None:
        first, last = band_range
        if not 1 <= first <= last <= count:
            raise ValueError("Band range exceeds available VASP or AO levels")
        return np.broadcast_to(
            (indices >= first) & (indices <= last), (len(native), count)
        )
    lower, upper = window
    if np.any(native[:, -1] <= upper) or np.any(model[:, -1] <= upper):
        raise ValueError(
            "VASP and AO spectra must extend above the energy window at each validation point"
        )

    def inside(x):
        return (x >= lower) & (x <= upper)

    if inside(native[:, count:]).any() or inside(model[:, count:]).any():
        raise ValueError("An in-window band has no counterpart in the other spectrum")
    return inside(native[:, :count]) | inside(model[:, :count])


def score_window(
    model, points, native, band_range, window, max_rmse, max_error, *, independent
):
    levels = bands(model, points)[0]
    ki, bi = np.nonzero(window_mask(native, levels, band_range, window))
    pairs = {
        "k_index": ki,
        "band_1based": bi + 1,
        "vasp_eV": native[ki, bi],
        "reconstructed_eV": levels[ki, bi],
    }
    summary = {"scored_k": len(np.unique(ki)), "observations": len(ki)}
    if not len(ki):
        if independent:
            raise ValueError(
                "No states in the declared window at independent validation points"
            )
        return summary, pairs
    error = pairs["reconstructed_eV"] - pairs["vasp_eV"]
    peak = int(np.argmax(abs(error)))
    maximum = float(abs(error[peak]))
    rmse = float(np.sqrt(np.mean(error**2)))
    summary.update(
        mae_eV=float(abs(error).mean()),
        rmse_eV=rmse,
        max_error_eV=maximum,
        worst_band_1based=int(bi[peak] + 1),
        worst_k=points[ki[peak]].tolist(),
    )
    if rmse > max_rmse or maximum > max_error:
        sample = "Independent" if independent else "Source mesh"
        raise ValueError(
            f"{sample} band validation failed: RMS {rmse:.8g} eV (limit {max_rmse:g}), maximum {maximum:.8g} eV (limit {max_error:g}), band {bi[peak] + 1}, k={points[ki[peak]].tolist()}"
        )
    return summary, pairs


def reference_points(src, reference):
    if spectral_mode(src):
        xml = reference / "vasprun.xml"
        if xml.resolve() != src["xml_path"] and file_sha256(xml) != file_sha256(
            src["xml_path"]
        ):
            raise ValueError(
                "Spectral validation requires the same SCF XML as the projected orbitals"
            )
        if "holdout" not in src:
            raise ValueError(
                "Spectral validation needs KPOINTS_OPT or zero-weight off-grid points in the same SCF"
            )
        ref = src["holdout"]
        return ref, independent_points(
            src["kpoints"], ref["kpoints"], time_reversal=not src.get("spinor")
        )
    ref = metadata(reference / "vasprun.xml")
    check_scalar(ref)
    check_convergence(ref, reference)
    if ref["icharg"] != 11:
        raise ValueError("Use an ICHARG=11 fixed-density validation run")
    a, b = src["atoms"], ref["atoms"]
    if (
        not np.array_equal(a.numbers, b.numbers)
        or not np.allclose(a.cell, b.cell, atol=1e-7, rtol=0)
        or not periodic_match(a.get_scaled_positions(), b.get_scaled_positions()).all()
        or not np.isclose(src["nelect"], ref["nelect"], atol=1e-7, rtol=0)
        or not np.isclose(src["encut"], ref["encut"], atol=1e-7, rtol=0)
        or src["operator_parameters"] != ref["operator_parameters"]
        or src["spline"] != ref["spline"]
        or src["titles"] != ref["titles"]
    ):
        raise ValueError(
            "Validation structure, PAW, electron count, cutoff or operator settings differ from source"
        )
    return ref, independent_points(
        src["kpoints"], ref["kpoints"], time_reversal=not src.get("spinor")
    )


def reference_data(src, reference, hashes):
    ref, selected = reference_points(src, reference)
    for name in ("POTCAR", "WAVECAR") if spectral_mode(src) else ("POTCAR", "CHGCAR"):
        if (
            file_sha256(
                (src["xml_path"].parent if spectral_mode(src) else reference) / name
            )
            != hashes[name]
        ):
            raise ValueError(f"Validation {name} differs from source")
    return ref, selected


def check_source(src, run):
    if src.get("spinor"):
        if src["nspin"] != 1 or src["icharg"] not in (0, 1, 2):
            raise ValueError(
                "Spinor bands require a self-consistent vasp_ncl spin density"
            )
        scalar = {**src, "spinor": False, "soc": False}
        if is_hybrid(src):
            check_hybrid(scalar)
        else:
            check_scalar(scalar)
        check_convergence(src, run)
        return
    if is_hybrid(src):
        check_hybrid(src)
        check_convergence(src, run)
        return
    check_scalar(src)
    check_convergence(src, run)
    if src["icharg"] != 11 and not (
        src["icharg"] in (0, 1, 2) and src["writes_charge"]
    ):
        raise ValueError("Source must use ICHARG=11 or write its converged SCF density")


def reconstruct(
    run,
    registry,
    reference,
    output,
    *,
    cutoff,
    max_rmse,
    max_error,
    band_range=None,
    energy_window=None,
    mesh=None,
    potential_run=None,
    relative_pruning=False,
    completion_method=None,
    centre_offset=6.0,
    width=1.0,
    floor=None,
    taper=2.0,
    on_stage=None,
    on_kpoint=None,
):
    if on_stage is not None:
        on_stage("Check VASP inputs")
    if (band_range is None) == (energy_window is None):
        raise ValueError("Choose either an ordered band range or an energy window")
    if energy_window is not None and (
        not np.isfinite(energy_window).all() or not energy_window[0] < energy_window[1]
    ):
        raise ValueError("Energy window bounds must be finite and increasing")
    if (
        not np.isfinite([cutoff, max_rmse, max_error]).all()
        or cutoff <= 0
        or min(max_rmse, max_error) < 0
    ):
        raise ValueError(
            "Use a positive finite cutoff and finite nonnegative release tolerances"
        )

    run, registry, output = map(Path, (run, registry, output))
    reference = Path(reference) if reference is not None else None
    potential = Path(potential_run) if potential_run else run
    inputs = {registry.resolve()}
    for directory in {run, potential, reference} - {None}:
        inputs.update(
            (directory / n).resolve()
            for n in (
                "vasprun.xml",
                "vaspout.h5",
                "WAVECAR",
                "CHGCAR",
                "POTCAR",
                "POSCAR",
                "KPOINTS",
                "INCAR",
                "OUTCAR",
                "CONTCAR",
            )
        )
    if output.resolve() in inputs:
        raise ValueError("Model output collides with an input")
    src = metadata(run / "vasprun.xml")
    check_source(src, run)
    mode = spectral_mode(src)
    if mode and potential.resolve() != run.resolve():
        raise ValueError(
            "Spectral projection uses the source WAVECAR; omit potential_run"
        )
    if completion_method not in (None, "spectral-ritz", "locality-constrained"):
        raise ValueError(f"Unknown hybrid method {completion_method}")
    localized = mode is not None and completion_method != "spectral-ritz"
    if completion_method is not None and mode is None:
        raise ValueError(
            "Choosing a completion method requires a hybrid or noncollinear source"
        )
    if localized and (
        floor is not None or (centre_offset, width, taper) != (6.0, 1.0, 2.0)
    ):
        raise ValueError("Set completion_method='spectral-ritz' to use Ritz filter controls")
    if floor is None:
        floor = 0.0001 if mode else 0.005
    xml_hash = file_sha256(run / "vasprun.xml")
    inputs_for_hash = ("POTCAR", "WAVECAR") if mode else ("POTCAR", "CHGCAR")
    hashes = {name: file_sha256(run / name) for name in inputs_for_hash}
    if reference is not None:
        ref, selected = reference_data(src, reference, hashes)
        points = ref["kpoints"][selected]
        ref_levels = ref["eig"][0, selected]
    a = src["atoms"]
    fixed = read_registry(registry, a.get_chemical_symbols())
    expected = {s: fingerprint(v) for s, v in fixed.items()}
    source_bands = src["eig"].shape[-1]
    if reference is not None:
        source_bands = min(source_bands, ref_levels.shape[1])
    if (
        band_range is not None
        and not 1 <= band_range[0] <= band_range[1] <= source_bands
    ):
        raise ValueError("Band range exceeds the available VASP spectrum")
    abs_window = (
        None
        if energy_window is None
        else (np.asarray(energy_window) + src["efermi"]).tolist()
    )
    if abs_window is not None and (
        not np.isfinite(abs_window).all()
        or np.any(src["eig"][0, :, -1] <= abs_window[1])
    ):
        raise ValueError("Source VASP spectrum must extend above the energy window")
    if mode and not localized:
        ceiling = src["efermi"] + centre_offset + 2 * taper
        if np.any(src["eig"][0, :, -1] < ceiling):
            raise ValueError(
                "Increase NBANDS to cover the filter ceiling at every mesh point"
            )
        requested_top = (
            abs_window[1]
            if abs_window is not None
            else np.max(src["eig"][0, :, band_range[1] - 1])
        )
        if requested_top >= ceiling:
            raise ValueError("Target window extends outside the retained spectrum")
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(dir=output.parent) as scratch,
        output_file(output) as candidate,
    ):
        d = Path(scratch)
        write(d / "geometry.vasp", a, format="vasp", direct=True)
        if on_stage is not None:
            on_stage("Project WAVECAR")
        build(
            run,
            registry,
            d / "direct.h5",
            mesh=mesh,
            potential_run=None if mode else potential,
            complement=src["efermi"]
            + (PRIOR_OFFSET_EV if localized else centre_offset + 2 * taper + 10.0)
            if mode
            else 0.0,
            on_kpoint=on_kpoint,
        )
        if on_stage is not None:
            on_stage("Compute AO overlap")
        overlap(d / "geometry.vasp", registry, d / "overlap.h5", order=96)
        if on_stage is not None:
            on_stage("Filter Hamiltonian")
        if localized:
            locality_model(
                d / "direct.h5",
                d / "overlap.h5",
                d / "filtered.h5",
                band_range,
                energy_window,
            )
        else:
            filter_model(
                d / "direct.h5",
                d / "overlap.h5",
                d / "filtered.h5",
                centre_offset,
                width,
                floor,
                taper,
                spectral=bool(mode),
            )
        if on_stage is not None:
            on_stage("Sparsify H(R)")
        sparsify(
            d / "filtered.h5",
            candidate,
            cutoff=cutoff,
            spinless_time_reversal=not src["spinor"],
            band_range=band_range,
            relative_pruning=relative_pruning,
        )
        with h5py.File(candidate, "r+") as model:
            if (
                model.attrs["potcar_sha256"] != hashes["POTCAR"]
                or (
                    not mode
                    and model.attrs["charge_density_sha256"] != hashes["CHGCAR"]
                )
                or (mode and file_sha256(run / "WAVECAR") != hashes["WAVECAR"])
                or file_sha256(run / "vasprun.xml") != xml_hash
            ):
                raise ValueError(
                    "Source electronic inputs changed during reconstruction"
                )
            basis = inspect_basis(model, expected)
            if on_stage is not None:
                on_stage("Validate source mesh")
            mesh_score, mesh_pairs = score_window(
                model,
                model["k/points"][:],
                src["eig"][0],
                band_range,
                abs_window,
                max_rmse,
                max_error,
                independent=False,
            )
            report = dict(
                profile=f"{mode}-spectral-reconstruction/1"
                if mode
                else "scalar-band-reconstruction/1",
                version=__version__,
                status="source_mesh_passed"
                if mesh_score["observations"]
                else "source_mesh_no_states",
                band_range_1based=band_range,
                energy_window_from_source_fermi_eV=energy_window,
                energy_window_absolute_eV=abs_window,
                source_fermi_eV=src["efermi"],
                cutoff_A=cutoff,
                cutoff_applies_to="H-mu*S" if relative_pruning else "H",
                mesh=model["k/mesh"][:].tolist(),
                overlap_mesh_alias_relative_rms=float(
                    model.attrs["overlap_mesh_alias_relative_rms"]
                ),
                basis=basis,
                cutoff_reference_eV=float(model.attrs["interpolation_reference_eV"]),
                source_mesh_score=mesh_score,
                rmse_limit_eV=max_rmse,
                max_limit_eV=max_error,
                registry_sha256=json.loads(model.attrs["basis_config"])[
                    "registry_sha256"
                ],
                source_xml_sha256=xml_hash,
                operator_settings=src["operator_parameters"],
                filter_parameters={
                    name: float(model.attrs[name])
                    for name in (
                        "filter_centre_eV",
                        "filter_width_eV",
                        "filter_floor",
                        "filter_taper_eV",
                        "filter_ceiling_eV",
                    )
                    if name in model.attrs
                },
                input_sha256=hashes,
            )
            report.update(screened_hybrid_maturity(src))
            if localized:
                del report["filter_parameters"]
                report["profile"] = f"{mode}-locality-reconstruction/1"
                report["locality_completion"] = dict(
                    recipe=json.loads(model.attrs["locality_recipe"]),
                    retained_bands=int(model["locality"].attrs["retained_bands"]),
                    iterations=int(model["locality"].attrs["iterations"]),
                    min_retained_subspace_weight=float(
                        model["locality/min_retained_weight"][:].min()
                    ),
                )
            elif mode:
                report["spectral_completion"] = {
                    "policy": str(model.attrs["complement_policy"]),
                    "complement_eV": float(model.attrs["complement_eV"]),
                    "source_bands": int(src["eig"].shape[-1]),
                    "min_edge_subspace_weight": float(
                        model["filter/min_edge_weight"][:].min()
                    ),
                }
            if reference is None:
                group = model.create_group("source_validation")
            else:
                if on_stage is not None:
                    on_stage("Validate off-grid bands")
                independent, observations = score_window(
                    model,
                    points,
                    ref_levels,
                    band_range,
                    abs_window,
                    max_rmse,
                    max_error,
                    independent=True,
                )
                report.update(
                    status="passed",
                    independent_k=len(points),
                    reference_k_total=len(ref["kpoints"]),
                    reference_kind=ref.get("source", "fixed-density"),
                    reference_xml_sha256=file_sha256(reference / "vasprun.xml"),
                    **independent,
                )
                group = model.create_group("band_validation")
                group["kpoints"] = points
                group["reference_k_index"] = selected
                for name, values in observations.items():
                    group[name] = values
            group.attrs["report"] = json.dumps(report, allow_nan=False)
            mesh_group = group.create_group("source_mesh")
            for name, values in mesh_pairs.items():
                mesh_group[name] = values
    return report
