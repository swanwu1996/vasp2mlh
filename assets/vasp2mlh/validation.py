"""Score an already exported native model without modifying its matrices."""

import json
from pathlib import Path
from .readback import verify_export

import h5py
import numpy as np

from .band_release import reference_data, score_window
from .configuration import DEFAULT_ENERGY_WINDOW
from .pipeline import file_sha256, output_file, paths_collide
from .vasp import spectral_mode, metadata


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
            if spectral_mode(src) and not args.reference:
                reference = run
            ref, selected = reference_data(src, reference, provenance["input_sha256"])
            limits = (args.max_rmse, args.max_error)
            if not np.isfinite(limits).all() or min(limits) < 0:
                raise ValueError("Use finite nonnegative band error limits")
            window = args.energy_window
            if args.band_range is None and window is None:
                window = list(DEFAULT_ENERGY_WINDOW)
            absolute = None
            if window is not None:
                if not np.isfinite(window).all() or window[0] >= window[1]:
                    raise ValueError(
                        "Energy window bounds must be finite and increasing"
                    )
                absolute = (np.asarray(window) + src["efermi"]).tolist()
            mesh, _ = score_window(
                model,
                src["kpoints"],
                src["eig"][0],
                args.band_range,
                absolute,
                np.inf,
                np.inf,
                independent=False,
            )
            blind, _ = score_window(
                model,
                ref["kpoints"][selected],
                ref["eig"][0, selected],
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
                and (score["rmse_eV"] > limits[0] or score["max_error_eV"] > limits[1])
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
    if failed:
        raise ValueError(f"Band error limits exceeded on {', '.join(failed)}")
