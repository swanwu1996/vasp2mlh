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
from .overlap import overlap
from .pipeline import bands, build, file_sha256, output_file
from .registry import fingerprint, inspect_basis, read_registry
from .sparse import sparsify
from .vasp import metadata


def check_scalar(meta):
    settings = meta["operator_parameters"]
    if (
        meta["nspin"] != 1
        or meta["spinor"]
        or meta["soc"]
        or meta["lreal"] not in ("F", ".FALSE.")
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


def independent_points(source, reference):
    """Keep one reference sample per time-reversal pair, modulo reciprocal translations."""
    selected = []
    src = np.asarray(source)
    ref = np.asarray(reference)
    if not np.isfinite(ref).all() or not np.isfinite(src).all():
        raise ValueError("Non-finite k coordinates")
    for i, point in enumerate(ref):
        if periodic_match(src, point).any() or periodic_match(src, -point).any():
            continue
        previous = ref[selected]
        if (
            not periodic_match(previous, point).any()
            and not periodic_match(previous, -point).any()
        ):
            selected.append(i)
    if not selected:
        raise ValueError(
            "No independent k points remain after mesh, periodic and time-reversal matching"
        )
    return np.asarray(selected)


def check_convergence(meta, directory):
    if meta["nelm"] <= 0 or meta["final_scf_steps"] < meta["nelm"]:
        return
    # A converged final iteration can coincide with NELM.
    outcar = directory / "OUTCAR"
    reached_ediff = False
    if outcar.is_file():
        with outcar.open("rb") as log:
            for line in log:
                if b"----" in line and b" Iteration " in line:
                    reached_ediff = False
                if b"aborting loop because EDIFF is reached" in line:
                    reached_ediff = True
    if not reached_ediff:
        raise ValueError(
            "Electronic iteration limit reached without confirmed final convergence"
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
    centre_offset=6.0,
    width=1.0,
    floor=0.005,
    taper=2.0,
):
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

    run, registry, reference, output = map(Path, (run, registry, reference, output))
    potential = Path(potential_run) if potential_run else run
    inputs = {registry.resolve()}
    for directory in (run, reference, potential):
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
    ref = metadata(reference / "vasprun.xml")
    for meta, directory in ((src, run), (ref, reference)):
        check_scalar(meta)
        check_convergence(meta, directory)
    if src["icharg"] != 11 and not (
        src["icharg"] in (0, 1, 2) and src["writes_charge"]
    ):
        raise ValueError("Source must use ICHARG=11 or write its converged SCF density")
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
    hashes = {}
    for name in ("POTCAR", "CHGCAR"):
        hashes[name] = file_sha256(run / name)
        if file_sha256(reference / name) != hashes[name]:
            raise ValueError(f"Validation {name} differs from source")
    selected = independent_points(src["kpoints"], ref["kpoints"])
    points = ref["kpoints"][selected]
    ref_levels = ref["eig"][0, selected]
    fixed = read_registry(registry, a.get_chemical_symbols())
    expected = {s: fingerprint(v) for s, v in fixed.items()}
    source_bands = src["eig"].shape[-1]
    if band_range is not None and not 1 <= band_range[0] <= band_range[1] <= min(
        source_bands, ref_levels.shape[1]
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
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(dir=output.parent) as scratch,
        output_file(output) as candidate,
    ):
        d = Path(scratch)
        write(d / "geometry.vasp", a, format="vasp", direct=True)
        build(run, registry, d / "direct.h5", mesh=mesh, potential_run=potential)
        overlap(d / "geometry.vasp", registry, d / "overlap.h5", order=96)
        filter_model(
            d / "direct.h5",
            d / "overlap.h5",
            d / "filtered.h5",
            centre_offset,
            width,
            floor,
            taper,
        )
        sparsify(
            d / "filtered.h5",
            candidate,
            cutoff=cutoff,
            spinless_time_reversal=True,
            band_range=band_range,
            relative_pruning=relative_pruning,
        )
        with h5py.File(candidate, "r+") as model:
            if (
                model.attrs["potcar_sha256"] != hashes["POTCAR"]
                or model.attrs["charge_density_sha256"] != hashes["CHGCAR"]
            ):
                raise ValueError("Source PAW or density changed during reconstruction")
            basis = inspect_basis(model, expected)
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
                profile="scalar-band-reconstruction/1",
                version=__version__,
                status="passed",
                band_range_1based=band_range,
                energy_window_from_source_fermi_eV=energy_window,
                energy_window_absolute_eV=abs_window,
                source_fermi_eV=src["efermi"],
                cutoff_A=cutoff,
                cutoff_applies_to="H-mu*S" if relative_pruning else "H",
                mesh=model["k/mesh"][:].tolist(),
                basis=basis,
                cutoff_reference_eV=float(model.attrs["interpolation_reference_eV"]),
                independent_k=len(points),
                reference_k_total=len(ref["kpoints"]),
                source_mesh_score=mesh_score,
                **independent,
                rmse_limit_eV=max_rmse,
                max_limit_eV=max_error,
                registry_sha256=json.loads(model.attrs["basis_config"])[
                    "registry_sha256"
                ],
                source_xml_sha256=file_sha256(run / "vasprun.xml"),
                reference_xml_sha256=file_sha256(reference / "vasprun.xml"),
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
                },
                input_sha256=hashes,
            )
            group = model.create_group("band_validation")
            group.attrs["report"] = json.dumps(report, allow_nan=False)
            group["kpoints"] = points
            group["reference_k_index"] = selected
            for name, values in observations.items():
                group[name] = values
            mesh_group = group.create_group("source_mesh")
            for name, values in mesh_pairs.items():
                mesh_group[name] = values
    return report
