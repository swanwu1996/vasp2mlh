"""Score an already exported native model without modifying its matrices."""

import json
from pathlib import Path
from .readback import verify_export
from .operator_action import independent_action

import h5py
import numpy as np

from .band_release import default_reference, reference_data, score_window, pbe_input_hashes, resolve_pbe_run
from .configuration import COLLINEAR_POLAR, BOUNDED_COLLINEAR, DEFAULT_ENERGY_WINDOW, HSE_POLAR, SCALAR_POLAR, SPINOR_POLAR, HUBBARD_POLAR
from .pipeline import file_sha256, output_file, paths_collide
from .vasp import metadata, screened_hybrid_maturity


def independent_report(model, *, output=None):
    """Read the batch report or the independent report for this single model."""
    if "band_validation" in model:
        return json.loads(model["band_validation"].attrs["report"])
    if "source_validation" not in model:
        raise ValueError("This model has no independent band validation")
    path = Path(model.filename).with_name("validation.json")
    if output is not None and paths_collide(output, [path]):
        raise ValueError(f"Output collides with the independent validation report: {path}")
    if not path.is_file():
        raise ValueError(f"Missing {path}. Run v2h_valid for this model first.")
    report = json.loads(path.read_text())
    if not isinstance(report, dict) or report.get("status") not in ("passed", "failed"):
        raise ValueError(f"Invalid independent validation report: {path}")
    if report.get("model_sha256") != file_sha256(model.filename):
        raise ValueError(f"Validation report does not identify this model. Run v2h_valid again: {path}")
    return report


def validate_sample(args, config):
    run = Path(args.run).resolve()
    reference = Path(args.reference) if args.reference else run / "reference"
    output = Path(args.output_dir) if args.output_dir else run / "mlh"
    model_path = output / "model.h5"
    target = output / "validation.json"
    if target.is_symlink() or paths_collide(target, [config, model_path]):
        raise ValueError(
            "Validation report destination collides with a symlink or configuration"
        )
    report = {
        "status": "failed",
        "native_band_status": "failed",
        "export_validation": {"format": args.format, "status": "not_checked"},
    }
    try:
        if not model_path.is_file():
            raise ValueError(
                f"Missing {model_path}. Run v2h with this configuration first."
            )
        if args.format == "auto":
            formats = [
                fmt
                for fmt, item in (
                    ("deeph", output / "deeph/0"),
                    ("hamgnn", output / "graph_data.npz"),
                )
                if item.exists()
            ]
            if len(formats) > 1:
                raise ValueError("Both export formats exist; specify format")
            args.format = formats[0] if formats else "deeph"
            report["export_validation"]["format"] = args.format
        with h5py.File(model_path, "r") as model:
            # Older combined conversions kept source provenance in band_validation.
            group = (
                "source_validation"
                if "source_validation" in model
                else "band_validation"
            )
            provenance = json.loads(model[group].attrs["report"])
            if file_sha256(run / "vasprun.xml") != provenance["source_xml_sha256"]:
                raise ValueError("Source XML differs from the model's recorded source")
            src = metadata(run / "vasprun.xml")
            maturity = screened_hybrid_maturity(src)
            report.update(maturity)
            provenance.update(maturity)
            pbe_run = getattr(args, "pbe_run", None)
            method = getattr(args, "completion_method", None)
            if method in ("locality-constrained", "spectral-ritz", "pbe-anchored-v1", HSE_POLAR, BOUNDED_COLLINEAR, SCALAR_POLAR, COLLINEAR_POLAR, SPINOR_POLAR, HUBBARD_POLAR):
                recorded_method = provenance.get(
                    "completion_method", model.attrs.get("hamiltonian_method")
                )
                accepted = {
                    SCALAR_POLAR: {SCALAR_POLAR},
                    COLLINEAR_POLAR: {COLLINEAR_POLAR},
                    SPINOR_POLAR: {SPINOR_POLAR},
                    HUBBARD_POLAR: {HUBBARD_POLAR},
                    HSE_POLAR: {HSE_POLAR},
                    "pbe-anchored-v1": {"pbe-anchored-v1"},
                    BOUNDED_COLLINEAR: {BOUNDED_COLLINEAR},
                    "locality-constrained": {"locality_constrained_paw"},
                    "spectral-ritz": {"spinor_spectral_ritz", "hybrid_spectral_ritz"},
                }
                if recorded_method not in accepted[method]:
                    raise ValueError("Completion method differs from the recorded conversion")
            policy = getattr(args, "retained_policy", None)
            recorded = provenance.get(
                "retained_policy",
                "global-buffer-1" if "pbe_reference" in provenance else None,
            )
            if policy not in (None, "auto", recorded):
                raise ValueError("Retained-band policy differs from the recorded conversion")
            if "pbe_reference" in provenance:
                directory = resolve_pbe_run(src, run, pbe_run, method, policy)
                if (
                    directory is None
                    or pbe_input_hashes(directory)
                    != provenance["pbe_reference"]["input_sha256"]
                ):
                    raise ValueError(
                        "PBE reference inputs differ from the model's recorded PBE source"
                    )
            elif pbe_run is not None or method in ("pbe-anchored-v1", HSE_POLAR):
                raise ValueError("This model was not built with the specified PBE anchor")
            hashes = provenance["input_sha256"]
            if "potential_h5" in hashes:
                potential = Path(args.potential_run) if args.potential_run else run
                if (
                    file_sha256(potential / "vaspout.h5") != hashes["potential_h5"]
                    or file_sha256(run / "WAVECAR") != hashes["WAVECAR"]
                ):
                    raise ValueError("PAW potential or source WAVECAR differs from the recorded conversion")
            if not args.reference:
                reference = default_reference(src, run)
            ref, selected = reference_data(src, reference, provenance["input_sha256"])
            limits = (args.max_rmse, args.max_error)
            if not np.isfinite(limits).all() or min(limits) < 0:
                raise ValueError("Use finite nonnegative band error limits")
            window = args.energy_window
            if args.band_range is None and window is None:
                args.band_range = provenance.get("band_range_1based")
                if args.band_range is None:
                    window = provenance.get("energy_window_from_source_fermi_eV") or list(DEFAULT_ENERGY_WINDOW)
            absolute = None
            if window is not None:
                if not np.isfinite(window).all() or window[0] >= window[1]:
                    raise ValueError(
                        "Energy window bounds must be finite and increasing"
                    )
                absolute = (np.asarray(window) + src["efermi"]).tolist()
            bound = provenance.get("polar_recipe") or provenance.get("collinear_recipe") or provenance.get("hubbard_recipe")
            if bound is not None and "linear_through_eV" in bound:
                if absolute is not None:
                    top = absolute[1]
                else:
                    first, last = args.band_range
                    if (
                        not 1
                        <= first
                        <= last
                        <= min(src["eig"].shape[-1], ref["eig"].shape[-1])
                    ):
                        raise ValueError(
                            "Band range exceeds the source or reference spectrum"
                        )
                    top = max(
                        float(np.max(src["eig"][:, :, last - 1])),
                        float(np.max(ref["eig"][:, selected, last - 1])),
                    )
                if top > bound["linear_through_eV"] + 1e-4:
                    raise ValueError("Requested bands exceed the stored linear energy domain; reconvert for this target")
            mesh, _ = score_window(
                model,
                src["kpoints"],
                src["eig"],
                args.band_range,
                absolute,
                np.inf,
                np.inf,
                independent=False,
            )
            blind, _ = score_window(
                model,
                ref["kpoints"][selected],
                ref["eig"][:, selected],
                args.band_range,
                absolute,
                np.inf,
                np.inf,
                independent=True,
            )
            failed = [
                name
                for name, score in (("source mesh", mesh), ("independent", blind))
                if score.get("observations")
                and (score.get("worst_channel_rmse_eV", score["rmse_eV"]) > limits[0] or score["max_error_eV"] > limits[1])
            ]
            report = dict(
                provenance,
                status="failed" if failed else "passed",
                source_mesh_score=mesh,
                model_sha256=file_sha256(model_path),
                failed_checks=failed,
                reference_xml_sha256=file_sha256(reference / "vasprun.xml"),
                independent_k=len(selected),
                reference_kind=ref.get("source", "fixed-density"),
                reference_k_total=len(ref["kpoints"]),
                band_range_1based=args.band_range,
                energy_window_from_source_fermi_eV=window,
                energy_window_absolute_eV=absolute,
                rmse_limit_eV=limits[0],
                max_limit_eV=limits[1],
                **blind,
            )
            report["native_band_status"] = report["status"]
            report["export_validation"] = {"format": args.format, "status": "failed"}
            report["export_validation"] = verify_export(
                output, args.format, model, ref["kpoints"][selected], args.nao_max
            )
            report["operator_action"] = independent_action(src, run, reference, model_path, window, args.band_range)
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        if output.is_dir():
            with output_file(target) as tmp:
                tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise ValueError(str(exc)) from exc
    with output_file(target) as tmp:
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Independent band error: RMS {1000 * blind['rmse_eV']:.3f} meV, maximum {1000 * blind['max_error_eV']:.3f} meV; {blind['observations']} states on {blind['scored_k']}/{len(selected)} independent k points"
    )
    print(f"Validation report: {target.absolute()}")
    for entry in blind.get("spin_scores", []):
        if entry["observations"]:
            print(
                f"Spin {entry['spin_1based']}: RMS {1000 * entry['rmse_eV']:.3f} meV, maximum {1000 * entry['max_error_eV']:.3f} meV ({entry['observations']} states)"
            )
        else:
            print(f"Spin {entry['spin_1based']}: no independent states in the target window")
    if failed:
        raise ValueError(f"Band error limits exceeded on {', '.join(failed)}")
