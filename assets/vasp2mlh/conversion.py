"""Single-structure handoff from VASP to a validated MLH sample."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .atomic import generate
from .band_release import reconstruct
from .configuration import DEFAULT_ENERGY_WINDOW
from .export import deeph, hamgnn
from .progress import ConversionProgress, progress_bar


def require_file(path, hint):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Missing or empty {path}. {hint}")


def convert_sample(args):
    run = Path(args.run).resolve()
    potential = Path(args.potential_run) if args.potential_run else run
    output = Path(args.output_dir) if args.output_dir else run / "mlh"
    if output.exists() or output.is_symlink():
        raise ValueError(
            f"Output already exists: {output}. Choose a new output directory with --output-dir."
        )
    if args.nao_max is not None and args.format != "hamgnn":
        raise ValueError("nao_max applies to HamGNN export")
    require_file(
        run / "WAVECAR",
        "Generate a full uniform complex WAVECAR with LWAVE=T and LREAL=F.",
    )
    for name in ("vasprun.xml", "POTCAR"):
        require_file(run / name, "Use the matching source calculation.")
    # reconstruct checks the functional-specific operator inputs after parsing the XML.
    registry = Path(args.registry) if args.registry else run / "basis.h5"
    needs_basis = (
        not args.registry and not registry.exists() and not registry.is_symlink()
    )
    if not needs_basis:
        require_file(registry, "Supply a frozen AO registry with --registry.")

    output.parent.mkdir(parents=True, exist_ok=True)
    # AO preparation, six reconstruction stages and the final sample export.
    with (
        TemporaryDirectory(dir=output.parent) as staging,
        progress_bar("Prepare AO basis", 8, "stage") as bar,
    ):
        progress = ConversionProgress(bar)
        result = Path(staging) / "result"
        result.mkdir()
        if needs_basis:
            registry = result / "basis.h5"
            generate(registry, [run / "POTCAR"])
        window = args.energy_window
        if window is None and args.band_range is None:
            window = list(DEFAULT_ENERGY_WINDOW)
        model = result / "model.h5"
        report = reconstruct(
            run,
            registry,
            None,
            model,
            potential_run=potential,
            pbe_run=args.pbe_run,
            mesh=args.mesh,
            band_range=args.band_range,
            energy_window=window,
            cutoff=args.cutoff,
            max_rmse=args.max_rmse,
            max_error=args.max_error,
            completion_method=args.completion_method,
            centre_offset=args.centre_offset,
            width=args.width,
            floor=args.floor,
            taper=args.taper,
            on_stage=progress.stage,
            on_kpoint=progress.projected,
        )
        progress.stage(f"Export {args.format}")
        if args.format == "deeph":
            sample = Path("deeph/0")
            deeph(model, result / sample, registry)
        else:
            sample = Path("graph_data.npz")
            hamgnn([model], result / sample, args.nao_max or 14, registry)
        (result / "source_validation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        result.rename(output)
        bar.update(1)
    if report.get("screened_hybrid_maturity") == "beta":
        print(f"HSE reconstruction: {report.get('completion_method', 'screened hybrid')} (beta)")
    print(f"{args.format} sample: {(output / sample).absolute()}")
