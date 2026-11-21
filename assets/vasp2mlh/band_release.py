"""Build fixed-basis band models and qualify their spectra against independent VASP states."""

import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree.ElementTree import ParseError

import h5py
import numpy as np
from ase.io import read, write

from . import __version__
from .comparison import periodic_match
from .configuration import COLLINEAR_POLAR, BOUNDED_COLLINEAR, HSE_POLAR, SCALAR_POLAR, SPINOR_POLAR, HUBBARD_POLAR
from .filtering import filter_model
from .polar import MIN_POLAR_SINGULAR_VALUE, hse_polar_model
from .locality import locality_model, PBE_ANCHORED, PRIOR_OFFSET_EV, WINDOW_FIRST_BUFFER, COMMON_WINDOW_BUFFER
from .overlap import overlap
from .pipeline import bands, build, file_sha256, output_file, projection_header
from .registry import fingerprint, inspect_basis, read_registry
from .sparse import sparsify
from .spillage import DEGENERACY_EV, basis_quality
from .vasp import check_hubbard, is_hubbard, is_hybrid, spectral_mode, metadata, screened_hybrid_maturity, split_mesh


def check_scalar(meta, *, collinear=False):
    settings = meta["operator_parameters"]
    if meta["lreal"] not in ("F", ".FALSE."):
        raise ValueError(
            f"VASP used LREAL={meta['lreal']}. For scalar reconstruction, rerun "
            "with LREAL=.FALSE., ISYM=-1 and WRT_POTENTIAL=total paw on a full mesh."
        )
    hubbard = is_hubbard(meta)
    if hubbard:
        check_hubbard(meta)
    if (
        meta["nspin"] not in ((1, 2) if collinear or hubbard else (1,))
        or meta["spinor"]
        or meta["soc"]
        or any(
            settings.get(tag, "F") not in ("F", ".FALSE.", "FALSE")
            for tag in ("LHFCALC", "LUSE_VDW", "LSPIRAL")
        )
        or settings.get("METAGGA", "") not in ("", "--", "F", "NONE", ".FALSE.")
        or settings.get("XC", "") not in ("", "--", "NONE")
    ):
        raise ValueError(
            "Band release requires scalar or collinear semilocal LDA/GGA with reciprocal PAW projectors; custom XC combinations are unsupported"
        )
    if settings.get("GGA") in ("LI", "LIBXC") and not settings.get("LIBXC1"):
        raise ValueError("Missing Libxc functional identity in VASP XML")


def check_hybrid(meta):
    settings = meta["operator_parameters"]
    # The generalized-KS spectrum is the source of the nonlocal exchange contribution.
    if is_hubbard(meta):
        raise ValueError("Combined hybrid and DFT+U is unsupported")
    scalar_settings = {**settings, "LHFCALC": "F"}
    check_scalar({**meta, "operator_parameters": scalar_settings}, collinear=False)
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
    if spectral_mode(meta) or is_hubbard(meta):
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
    levels = bands(model, points)
    native = np.asarray(native)
    if native.ndim == 2:
        native = native[None]
    if native.ndim != 3 or levels.shape[:2] != native.shape[:2]:
        raise ValueError("Reference and model spin or k-point counts differ")
    channels, point_ids, band_ids = [], [], []
    for channel, (target, spectrum) in enumerate(zip(native, levels)):
        ki, bi = np.nonzero(window_mask(target, spectrum, band_range, window))
        channels.append(np.full(len(ki), channel, dtype=int))
        point_ids.append(ki)
        band_ids.append(bi)
    spins, ki, bi = map(np.concatenate, (channels, point_ids, band_ids))
    pairs = {
        "k_index": ki,
        "band_1based": bi + 1,
        "vasp_eV": native[spins, ki, bi],
        "reconstructed_eV": levels[spins, ki, bi],
    }
    summary = {"scored_k": len(np.unique(ki)), "observations": len(ki)}
    if not len(ki):
        if independent:
            raise ValueError("No states in the declared window at independent validation points")
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
    worst_channel_rmse = rmse
    if len(native) > 1:
        pairs["spin_1based"] = spins + 1
        per_spin = []
        for i in range(len(native)):
            errors = error[spins == i]
            entry = {"spin_1based": i + 1, "observations": len(errors)}
            if len(errors):
                entry.update(
                    rmse_eV=float(np.sqrt(np.mean(errors**2))),
                    max_error_eV=float(np.max(abs(errors))),
                )
                worst_channel_rmse = max(worst_channel_rmse, entry["rmse_eV"])
            per_spin.append(entry)
        summary.update(
            spin_scores=per_spin,
            worst_channel_rmse_eV=worst_channel_rmse,
            worst_spin_1based=int(spins[peak] + 1),
        )
    if worst_channel_rmse > max_rmse or maximum > max_error:
        sample = "Independent" if independent else "Source mesh"
        raise ValueError(
            f"{sample} band validation failed: RMS {worst_channel_rmse:.8g} eV (limit {max_rmse:g}), maximum {maximum:.8g} eV (limit {max_error:g}), band {bi[peak] + 1}, k={points[ki[peak]].tolist()}"
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
    check_scalar(ref, collinear=True)
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
        or src["nspin"] != ref["nspin"]
        or src["spline"] != ref["spline"]
        or src["titles"] != ref["titles"]
    ):
        raise ValueError(
            "Validation structure, PAW, electron count, cutoff or operator settings differ from source"
        )
    return ref, independent_points(
        src["kpoints"], ref["kpoints"], time_reversal=not src.get("spinor")
    )


def default_reference(src, run):
    if src is not None and spectral_mode(src):
        return run
    candidates = sorted(
        {
            folder.resolve()
            for parent in (run, run.parent)
            for name in ("reference", "blind")
            if ((folder := parent / name) / "vasprun.xml").is_file()
            and folder.resolve() != run.resolve()
        }
    )
    if len(candidates) > 1:
        if src is None:
            src = metadata(run / "vasprun.xml")
        hashes = {name: file_sha256(run / name) for name in ("POTCAR", "CHGCAR")}
        matches = []
        for folder in candidates:
            try:
                reference_data(src, folder, hashes)
            except (OSError, ValueError, ParseError, KeyError, IndexError):
                continue
            matches.append(folder)
        if len(matches) != 1:
            raise ValueError(
                "Cannot select a unique compatible validation calculation; set reference"
            )
        return matches[0]
    return candidates[0] if candidates else run / "reference"


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
    if is_hubbard(src):
        check_hubbard(src)
    if src.get("spinor"):
        if src["nspin"] != 1 or src["icharg"] not in (0, 1, 2):
            raise ValueError(
                "Spinor bands require a self-consistent vasp_ncl spin density"
            )
        scalar = {**src, "spinor": False, "soc": False}
        if is_hubbard(src):
            # Same-SCF spinor eigenpairs include U and do not read a frozen CHGCAR.
            scalar["operator_parameters"] = {**src["operator_parameters"], "LDAU": "F"}
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
    check_scalar(src, collinear=True)
    check_convergence(src, run)
    if src["icharg"] != 11 and not (
        src["icharg"] in (0, 1, 2) and src["writes_charge"]
    ):
        raise ValueError("Source must use ICHARG=11 or write its converged SCF density")


def prepare_full_mesh(run, destination):
    """Stage a nonmagnetic semilocal fixed-density VASP run on the original full grid."""
    run, dest = Path(run), Path(destination)
    if dest.exists() or dest.is_symlink():
        raise ValueError("Choose a new directory for full-mesh inputs")
    src = metadata(run / "vasprun.xml")
    check_scalar(src, collinear=False)
    check_convergence(src, run)
    if is_hubbard(src) or src["icharg"] not in (0, 1, 2) or not src["writes_charge"]:
        raise ValueError("Full-mesh preparation requires a converged nonmagnetic LDA/GGA density")
    from .basis import read_potcar

    geometry = read(run / "CHGCAR", format="vasp")
    atoms = src["atoms"]
    pots = read_potcar(run / "POTCAR")
    if (not np.array_equal(geometry.numbers, atoms.numbers)
            or not np.allclose(geometry.cell, atoms.cell, rtol=0, atol=1e-7)
            or not periodic_match(geometry.get_scaled_positions(), atoms.get_scaled_positions()).all()
            or set(pots) != set(src["titles"])
            or any(pot.title.split() != src["titles"][sym].split() for sym, pot in pots.items())):
        raise ValueError("Charge geometry or PAW identities differ from the source XML")
    xml = ET.parse(run / "vasprun.xml").getroot()
    generation = xml.find("./kpoints/generation")
    mesh = src["mesh"]
    if generation is None or mesh is None or generation.get("param") not in ("Gamma", "Monkhorst-Pack"):
        raise ValueError("Expected a regular Gamma or Monkhorst-Pack grid in the source XML")
    shift = generation.findtext("v[@name='usershift']", "0 0 0").split()
    if len(shift) != 3 or not np.isfinite(np.asarray(shift, dtype=float)).all():
        raise ValueError("Invalid k-mesh shift")
    incar = xml.find("incar")
    if incar is None:
        raise ValueError("Missing INCAR echo in source XML")
    controls = {node.attrib["name"]: " ".join(node.text.split()) for node in incar
                if node.tag in ("i", "v") and node.text and node.text.strip()}
    controls.update(ICHARG="11", ISTART="0", ISYM="-1", NSW="0", IBRION="-1",
                    LREAL="F", LWAVE="T", LCHARG="F", WRT_POTENTIAL="total paw",
                    NBANDS=str(src["eig"].shape[-1]))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / "input"
        stage.mkdir()
        for name in ("CHGCAR", "POTCAR"):
            if not (run / name).stat().st_size:
                raise ValueError(f"Empty source {name}")
            shutil.copyfile(run / name, stage / name)
        write(stage / "POSCAR", src["atoms"], format="vasp", direct=True)
        (stage / "INCAR").write_text("".join(f"{key} = {value}\n" for key, value in controls.items()))
        (stage / "KPOINTS").write_text(
            "Full source mesh\n0\n" + generation.get("param") + "\n" +
            " ".join(map(str, mesh)) + "\n" + " ".join(shift) + "\n"
        )
        stage.rename(dest)


def resolve_pbe_run(src, run, pbe_run, method, retained_policy=None):
    """Route from the effective VASP parameters, never from a folder name."""
    screened = bool(screened_hybrid_maturity(src))
    if retained_policy not in (None, "auto") and (
        not screened or retained_policy not in (WINDOW_FIRST_BUFFER, COMMON_WINDOW_BUFFER, "global-buffer-1")
    ):
        raise ValueError(f"{retained_policy} requires a scalar screened PBE hybrid source")
    if not screened:
        if pbe_run is not None or method in (PBE_ANCHORED, HSE_POLAR):
            raise ValueError(
                "PBE-referenced HSE methods and pbe_run require a scalar screened PBE hybrid source"
            )
        return None
    if src["spinor"] or src["soc"]:
        raise ValueError(
            "The HSE beta method requires a scalar source; hybrid spinors are unsupported"
        )
    if method not in (None, "auto", PBE_ANCHORED, HSE_POLAR):
        raise ValueError(
            "Legacy HSE completion retired. Choose pbe-anchored-v1 or hse-polar-window-v1"
        )
    directory = Path(pbe_run) if pbe_run is not None else run / "pbe"
    if pbe_run is None:
        neighbours = sorted(
            {
                child.resolve()
                for parent in (run, run.parent)
                for child in parent.iterdir()
                if child.name.lower().startswith("pbe")
                and (child / "vasprun.xml").is_file()
            }
        )
        if len(neighbours) == 1:
            return neighbours[0]
        if len(neighbours) > 1:
            matches = []
            for candidate in neighbours:
                try:
                    check_pbe_pair(src, run, candidate)
                except (ValueError, OSError, ParseError, KeyError, IndexError):
                    continue
                matches.append(candidate)
            if len(matches) == 1:
                return matches[0]
            if matches:
                raise ValueError(
                    "Several compatible PBE references; set pbe_run: "
                    + ", ".join(map(str, matches))
                )
            raise ValueError("No compatible dense PBE reference among " + ", ".join(map(str, neighbours)))
    if not (directory / "vasprun.xml").is_file():
        raise ValueError(
            f"HSE requires a matching dense PBE calculation at {directory}; set pbe_run in mlh.toml"
        )
    return directory


def check_pbe_exchange(meta, directory, *, screened):
    settings = meta["operator_parameters"]
    gga = settings.get("GGA", "--")
    if gga in ("--", ""):
        with (directory / "POTCAR").open(errors="replace") as stream:
            codes = {
                line.split("=", 1)[1].split()[0].upper()
                for line in stream
                if line.lstrip().startswith("LEXCH") and "=" in line
            }
        pbe = codes == {"PE"}
    else:
        pbe = gga == "PE"
    fraction = float(settings.get("AEXX", 0))
    exchange = 1 - fraction if screened else 1.0
    switches = ("LRHFCALC", "LTHOMAS", "LMODELHF", "LHFONE")
    if (
        not pbe
        or any(
            settings.get(tag, "F") not in ("F", "FALSE", ".FALSE.") for tag in switches
        )
        or float(settings.get("EXXOEP", 0)) != 0
        or float(settings.get("AEXX_LR", 0)) != 0
        or (not screened and fraction != 0)
        or any(
            not np.isclose(
                float(settings.get(tag, expected)), expected, atol=1e-7, rtol=0
            )
            for tag, expected in (
                ("ALDAX", exchange),
                ("AGGAX", exchange),
                ("ALDAC", 1.0),
                ("AGGAC", 1.0),
            )
        )
    ):
        raise ValueError(
            "Expected PBE-based screened exchange for HSE and unmodified PBE for pbe_run"
        )


def check_pbe_pair(src, run, directory, mesh=None):
    pbe = metadata(directory / "vasprun.xml")
    if is_hubbard(pbe):
        raise ValueError("A DFT+U calculation cannot serve as the pure PBE anchor")
    if is_hybrid(pbe):
        raise ValueError(
            "pbe_run contains a hybrid calculation (LHFCALC=T); supply semilocal PBE"
        )
    check_scalar(pbe, collinear=False)
    if pbe["icharg"] not in (0, 1, 2):
        raise ValueError("pbe_run requires a self-consistent scalar PBE calculation")
    check_convergence(pbe, directory)
    check_pbe_exchange(src, run, screened=True)
    check_pbe_exchange(pbe, directory, screened=False)
    a, b = src["atoms"], pbe["atoms"]
    if (
        not np.array_equal(a.numbers, b.numbers)
        or not np.allclose(a.cell, b.cell, atol=1e-7, rtol=0)
        or not np.allclose(
            a.get_scaled_positions(wrap=False),
            b.get_scaled_positions(wrap=False),
            atol=1e-7,
            rtol=0,
        )
        or src["titles"] != pbe["titles"]
        or src["spline"] != pbe["spline"]
        or not np.allclose(
            [src["nelect"], src["encut"]],
            [pbe["nelect"], pbe["encut"]],
            atol=1e-7,
            rtol=0,
        )
        or any(
            src["operator_parameters"].get(tag) != pbe["operator_parameters"].get(tag)
            for tag in (
                "LASPH",
                "GGA_COMPAT",
                "VOSKOWN",
                "LDIPOL",
                "IDIPOL",
                "EFIELD",
                "PARAM1",
                "PARAM2",
            )
        )
        or file_sha256(run / "POTCAR") != file_sha256(directory / "POTCAR")
    ):
        raise ValueError(
            "HSE and PBE geometry, PAW datasets or electronic settings differ"
        )
    split_mesh(pbe, None)
    _, dense = projection_header(directory, pbe, None)
    coarse = np.asarray(mesh if mesh is not None else src["mesh"])
    if np.any(dense < coarse) or not np.any(dense > coarse):
        raise ValueError(
            "pbe_run requires a finer full mesh, at least as dense along every reciprocal axis"
        )
    return pbe


def pbe_input_hashes(directory):
    return {
        name: file_sha256(directory / name)
        for name in ("vasprun.xml", "POTCAR", "WAVECAR")
    }


def source_window_bands(levels, window):
    """Smallest ordered envelope of source states touching the energy interval."""
    inside = np.any((levels >= window[0]) & (levels <= window[1]), axis=(0, 1))
    indices = np.flatnonzero(inside)
    if not indices.size:
        raise ValueError("No source states in the polar energy window. Supply an ordered band range.")
    return [int(indices[0]) + 1, int(indices[-1]) + 1]


def release_recipe(src, method, policy):
    """Choose a physical reconstruction for the installed release line."""
    major = __version__.split(".")[0]
    screened = bool(screened_hybrid_maturity(src))
    auto = method in (None, "auto")
    if major == "1":
        if method in (SCALAR_POLAR, COLLINEAR_POLAR, HUBBARD_POLAR, SPINOR_POLAR, HSE_POLAR):
            raise ValueError(f"{method} requires the 2.x polar release")
        if screened:
            if policy not in (None, "auto", "global-buffer-1"):
                raise ValueError("The target-first HSE buffer policy requires the 2.x polar release")
            policy = "global-buffer-1"
        return method, policy
    if major != "2":
        raise ValueError(f"Unknown reconstruction release line {major}")
    if screened:
        if method not in (None, "auto", PBE_ANCHORED, HSE_POLAR) or policy not in (None, "auto", WINDOW_FIRST_BUFFER, COMMON_WINDOW_BUFFER):
            raise ValueError("2.x screened HSE requires target-first PBE-referenced completion")
        selected_policy = WINDOW_FIRST_BUFFER if policy in (None, "auto") else policy
        return (PBE_ANCHORED if auto else method), selected_policy
    if is_hybrid(src):
        raise ValueError("2.x supports scalar screened PBE hybrids for polar reconstruction")
    if src["spinor"]:
        selected = SPINOR_POLAR
    elif is_hubbard(src):
        selected = HUBBARD_POLAR
    else:
        selected = COLLINEAR_POLAR if src["nspin"] == 2 else SCALAR_POLAR
    if not auto and method != selected:
        raise ValueError(f"2.x requires {selected} for this source")
    return selected, policy


def reconstruct(
    run,
    registry,
    reference,
    output,
    *,
    cutoff=None,
    max_rmse,
    max_error,
    band_range=None,
    energy_window=None,
    mesh=None,
    potential_run=None,
    pbe_run=None,
    relative_pruning=False,
    completion_method=None,
    retained_policy=None,
    centre_offset=None,
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
        not np.isfinite([max_rmse, max_error]).all()
        or (cutoff is not None and (not np.isfinite(cutoff) or cutoff <= 0))
        or min(max_rmse, max_error) < 0
    ):
        raise ValueError(
            "Use a positive finite cutoff and finite nonnegative release tolerances"
        )

    run, registry, output = map(Path, (run, registry, output))
    reference = Path(reference) if reference is not None else None
    potential = Path(potential_run) if potential_run else run
    if output.resolve() == registry.resolve():
        raise ValueError("Model output collides with the AO registry")
    xml_hash = file_sha256(run / "vasprun.xml")
    src = metadata(run / "vasprun.xml")
    check_source(src, run)
    completion_method, retained_policy = release_recipe(src, completion_method, retained_policy)
    hse_polar = completion_method == HSE_POLAR
    if hse_polar and band_range is None:
        raise ValueError(f"{HSE_POLAR} requires an ordered band range")
    spinor_polar = completion_method == SPINOR_POLAR
    hubbard_polar = completion_method == HUBBARD_POLAR
    polar_window = completion_method in (SCALAR_POLAR, COLLINEAR_POLAR, SPINOR_POLAR, HUBBARD_POLAR)
    if cutoff is None:
        cutoff = 18.0 if is_hubbard(src) or polar_window or completion_method == BOUNDED_COLLINEAR else 12.0
    pbe_dir = resolve_pbe_run(src, run, pbe_run, completion_method, retained_policy)
    if completion_method == "auto":
        completion_method = None
    if pbe_dir is None or retained_policy == "global-buffer-1":
        retained_policy = None
    inputs = {registry.resolve()}
    for directory in {run, potential, reference, pbe_dir} - {None}:
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
    mode = spectral_mode(src)
    if mode and potential.resolve() != run.resolve():
        raise ValueError(
            "Spectral projection uses the source WAVECAR; omit potential_run"
        )
    bounded_collinear = completion_method == BOUNDED_COLLINEAR
    if completion_method in (SCALAR_POLAR, COLLINEAR_POLAR):
        channels = 2 if completion_method == COLLINEAR_POLAR else 1
        if src["nspin"] != channels or mode or is_hubbard(src):
            raise ValueError(f"{completion_method} requires {channels} semilocal PAW channel(s) without U")
    if hubbard_polar and (mode or not is_hubbard(src)):
        raise ValueError(f"{HUBBARD_POLAR} requires scalar or collinear DFT+U")
    if spinor_polar and (not src["spinor"] or is_hybrid(src)):
        raise ValueError(f"{SPINOR_POLAR} requires semilocal spinor eigenpairs")
    if bounded_collinear and (src["nspin"] != 2 or mode or is_hubbard(src) or energy_window is None):
        raise ValueError(f"{BOUNDED_COLLINEAR} requires two collinear semilocal channels without U and an energy window")
    if completion_method not in (None, PBE_ANCHORED, HSE_POLAR, BOUNDED_COLLINEAR, SCALAR_POLAR, COLLINEAR_POLAR, SPINOR_POLAR, HUBBARD_POLAR, "spectral-ritz", "locality-constrained"):
        raise ValueError(f"Unknown completion method {completion_method}")
    auto_centre = centre_offset is None
    if auto_centre:
        centre_offset = 6.0
    localized = mode is not None and completion_method not in ("spectral-ritz", SPINOR_POLAR)
    if completion_method is not None and mode is None and not (bounded_collinear or polar_window):
        raise ValueError(
            "Choosing a completion method requires a hybrid or noncollinear source"
        )
    if localized and (
        floor is not None or (centre_offset, width, taper) != (6.0, 1.0, 2.0)
    ):
        raise ValueError("Set completion_method='spectral-ritz' to use Ritz filter controls")
    if floor is None:
        small_floor = (mode is not None and not spinor_polar) or bounded_collinear or completion_method in (SCALAR_POLAR, COLLINEAR_POLAR)
        floor = 0.0001 if small_floor else 0.005
    if pbe_dir is not None:
        pbe_hashes = pbe_input_hashes(pbe_dir)
        pbe = check_pbe_pair(src, run, pbe_dir, mesh)
    inputs_for_hash = ("POTCAR", "WAVECAR") if mode else ("POTCAR", "CHGCAR")
    hashes = {name: file_sha256(run / name) for name in inputs_for_hash}
    if not mode and (is_hubbard(src) or src["nspin"] == 2 or polar_window):
        hashes.update(WAVECAR=file_sha256(run / "WAVECAR"), potential_h5=file_sha256(potential / "vaspout.h5"))
    if reference is not None:
        ref, selected = reference_data(src, reference, hashes)
        points = ref["kpoints"][selected]
        ref_levels = ref["eig"][:, selected]
    a = src["atoms"]
    fixed = read_registry(registry, a.get_chemical_symbols())
    expected = {s: fingerprint(v) for s, v in fixed.items()}
    source_bands = src["eig"].shape[-1]
    if reference is not None:
        source_bands = min(source_bands, ref_levels.shape[-1])
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
        or np.any(src["eig"][:, :, -1] <= abs_window[1])
    ):
        raise ValueError("Source VASP spectrum must extend above the energy window")
    anchor_range = band_range
    if polar_window and anchor_range is None:
        anchor_range = source_window_bands(src["eig"], abs_window)
    window_recipe = None
    if is_hubbard(src) or bounded_collinear or polar_window:
        upper = (
            abs_window[1]
            if abs_window is not None
            else float(np.max(src["eig"][:, :, band_range[1] - 1]))
        )
        window_recipe = dict(
            method="window-bounded-paw-v1",
            band_range_1based=band_range,
            energy_window_from_fermi_eV=energy_window,
            linear_through_eV=upper + 2.0,
            upper_buffer_eV=2.0,
            saturation_width_eV=2.0,
        )
        if polar_window:
            upper = max(upper, float(src["eig"][:, :, anchor_range[1] - 1].max()))
            window_recipe.update(band_range_1based=anchor_range, linear_through_eV=upper + 2.0)
            if energy_window is not None:
                window_recipe["anchor_policy"] = "ordered-source-envelope"
            window_recipe.update(method=completion_method, degeneracy_eV=DEGENERACY_EV, min_singular_value=MIN_POLAR_SINGULAR_VALUE)
            if spinor_polar:
                window_recipe.update(prior="bounded_spinor_spectral_ritz", prior_offset_eV=PRIOR_OFFSET_EV)
                if auto_centre:
                    window_recipe["filter_centre_policy"] = "spinor-window-and-available-spectrum"
        if auto_centre and mode is None and (not polar_window or hubbard_polar):
            centre_offset = max(6.0, upper + 2.0 - src["efermi"])
            window_recipe["filter_centre_policy"] = "max-fermi-plus-6-or-target-plus-2"
    if not mode and src["efermi"] + centre_offset > float(src["eig"][:, :, -1].min()) + 1e-4:
        raise ValueError("Filter centre exceeds the available VASP band ceiling. Increase NBANDS or lower centre_offset.")
    if mode and not localized and not spinor_polar:
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
        TemporaryDirectory(dir=os.environ.get("TMPDIR") or output.parent) as scratch,
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
            + (PRIOR_OFFSET_EV if localized or spinor_polar else centre_offset + 2 * taper + 10.0)
            if mode
            else 0.0,
            on_kpoint=on_kpoint,
        )
        with h5py.File(d / "direct.h5") as trial:
            coverage = basis_quality(trial, band_range, abs_window)
        if on_stage is not None:
            on_stage("Compute AO overlap")
        overlap(d / "geometry.vasp", registry, d / "overlap.h5", order=96)
        if on_stage is not None:
            on_stage(
                "Project PBE reference and complete HSE"
                if pbe_dir is not None
                else "Filter Hamiltonian"
            )
        pbe_model = None
        if pbe_dir is not None:
            pbe_model = d / "pbe-locality.h5"
            build(
                pbe_dir,
                registry,
                d / "pbe-projection.h5",
                integration_only=True,
                complement=pbe["efermi"] + PRIOR_OFFSET_EV,
                on_kpoint=on_kpoint,
            )
            try:
                locality_model(
                    d / "pbe-projection.h5",
                    d / "overlap.h5",
                    pbe_model,
                    band_range,
                    energy_window,
                    retained_policy=retained_policy,
                )
            except ValueError as exc:
                raise ValueError(f"PBE reference completion failed: {exc}") from exc
        if hse_polar:
            hse_polar_model(d / "direct.h5", d / "overlap.h5", pbe_model, d / "filtered.h5", band_range)
        elif localized:
            locality_model(
                d / "direct.h5",
                d / "overlap.h5",
                d / "filtered.h5",
                band_range,
                energy_window,
                hubbard_recipe=window_recipe,
                pbe_model=pbe_model,
                retained_policy=retained_policy,
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
                hubbard_recipe=window_recipe if is_hubbard(src) else None,
                collinear_recipe=window_recipe if bounded_collinear else None,
                polar_recipe=window_recipe if polar_window else None,
            )
        if on_stage is not None:
            on_stage("Sparsify H(R)")
        sparsify(
            d / "filtered.h5",
            candidate,
            cutoff=cutoff,
            spinless_time_reversal=not src["spinor"] and src["nspin"] == 1,
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
                or (
                    "WAVECAR" in hashes
                    and file_sha256(run / "WAVECAR") != hashes["WAVECAR"]
                )
                or (
                    "potential_h5" in hashes
                    and (
                        model.attrs["potential_sha256"] != hashes["potential_h5"]
                        or file_sha256(potential / "vaspout.h5")
                        != hashes["potential_h5"]
                    )
                )
                or file_sha256(run / "vasprun.xml") != xml_hash
            ):
                raise ValueError(
                    "Source electronic inputs changed during reconstruction"
                )
            if pbe_dir is not None and pbe_input_hashes(pbe_dir) != pbe_hashes:
                raise ValueError("PBE reference inputs changed during reconstruction")
            if spinor_polar:
                window_recipe = json.loads(model.attrs["polar_recipe"])
            basis = inspect_basis(model, expected)
            if on_stage is not None:
                on_stage("Validate source mesh")
            try:
                mesh_score, mesh_pairs = score_window(
                    model,
                    model["k/points"][:],
                    src["eig"],
                    band_range,
                    abs_window,
                    max_rmse,
                    max_error,
                    independent=False,
                )
            except ValueError as exc:
                worst = coverage["worst_target_subspace"]
                if worst is None:
                    raise
                diagnosis = ""
                if hse_polar:
                    local = json.loads(model.attrs["polar_source_diagnostics"])
                    diagnosis = (
                        f" Before truncation, ordered RMS {local['ordered_rms_eV']:.5g} eV, "
                        f"complement levels inside target at {local['complement_intrusion_k']}/{local['kpoints']} k points."
                    )
                raise ValueError(
                    f"{exc}.{diagnosis} Weakest source PAW target-subspace weight {worst['weight']:.5g} "
                    f"at k={worst['k']}, spin {worst['spin_1based']}, bands {worst['band_range_1based']}"
                ) from exc
            report = dict(
                profile=f"{mode}-spectral-reconstruction/1"
                if mode
                else "scalar-band-reconstruction/1",
                version=__version__,
                algorithm_family="original" if __version__.split(".")[0] == "1" else "polar",
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
                source_basis_quality=coverage,
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
            report["source_functional"] = (
                "HSE"
                if pbe_dir is not None
                else "hybrid"
                if is_hybrid(src)
                else "semilocal"
            )
            if is_hubbard(src):
                report.update(
                    source_functional="DFT+U",
                    hubbard_maturity="experimental" if src["spinor"] or hubbard_polar else "beta",
                    hubbard_recipe=window_recipe,
                    hubbard_by_element=src["operator_parameters"]["HUBBARD_BY_ELEMENT"],
                    spin_channels=src["nspin"],
                )
                if src["spinor"]:
                    report["operator_source"] = (
                        "self-consistent DFT+U spinor eigenpairs"
                    )
                else:
                    report.update(
                        profile="dft-u-paw-reconstruction/1",
                        operator_source="vaspout.h5:results/paw/dij",
                        paw_rayleigh_max_eV=float(
                            np.max(model["k/operator_edge_rayleigh_error_eV"])
                        ),
                        paw_residual_max_eV=float(
                            np.max(model["k/operator_edge_residual_norm_eV"])
                        ),
                    )
            if src["nspin"] == 2 and not is_hubbard(src):
                report.update(
                    profile="collinear-paw-reconstruction/1",
                    collinear_maturity="experimental",
                    spin_channels=2,
                    operator_source="vaspout.h5:results/paw/dij",
                    paw_rayleigh_max_eV=float(np.max(model["k/operator_edge_rayleigh_error_eV"])),
                    paw_residual_max_eV=float(np.max(model["k/operator_edge_residual_norm_eV"])),
                )
            if polar_window:
                report.update(polar_recipe=window_recipe, polar_maturity="experimental")
            if spinor_polar:
                report["profile"] = "spinor-polar-window/1"
            if bounded_collinear:
                report["collinear_recipe"] = window_recipe
            report["completion_method"] = str(model.attrs["hamiltonian_method"])
            report.update(screened_hybrid_maturity(src))
            if hse_polar:
                del report["filter_parameters"]
                report.update(
                    polar_recipe=json.loads(model.attrs["polar_recipe"]),
                    polar_source_diagnostics=json.loads(model.attrs["polar_source_diagnostics"]),
                    polar_maturity="experimental",
                )
            elif localized:
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
            if pbe_dir is not None:
                report.update(
                    profile="hse-polar-window/1" if hse_polar else "hse-pbe-anchored/1",
                    pbe_reference=dict(
                        input_sha256=pbe_hashes,
                        mesh=list(pbe["mesh"]),
                        operator_settings=pbe["operator_parameters"],
                        alignment_eV=float(model.attrs["pbe_alignment_eV"]),
                    ),
                )
            if pbe_dir is not None and retained_policy is None:
                report["retained_policy"] = "global-buffer-1"
            if retained_policy is not None:
                report["retained_policy"] = retained_policy
                if hse_polar:
                    report["retained_policy_scope"] = "PBE reference"
                with h5py.File(pbe_model) as baseline:
                    stages = [(baseline, "pbe_reference")]
                    if not hse_polar:
                        stages.append((model, "locality_completion"))
                    for target, stage in stages:
                        counts, frequencies = np.unique(
                            target["locality/retained_by_k"][:], return_counts=True
                        )
                        report[stage]["retained_band_histogram"] = dict(
                            zip(map(str, counts), map(int, frequencies))
                        )
                        report[stage]["min_retained_singular_value"] = float(np.sqrt(np.min(target["locality/min_retained_weight"][:])))
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
        if reference is not None:
            from .operator_action import independent_action

            report["operator_action"] = independent_action(
                src, run, reference, candidate, energy_window, band_range
            )
            with h5py.File(candidate, "r+") as model:
                model["band_validation"].attrs["report"] = json.dumps(report, allow_nan=False)
    return report
