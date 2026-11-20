import argparse
import json
import os
import sys
from pathlib import Path
from xml.etree.ElementTree import ParseError

import h5py
import numpy as np

from .comparison import compare_path, write_comparison
from .analysis import held_out, locality, lobster_summary, spectra
from .sparse import sparsify
from .export import deeph, deeph_batch, hamgnn
from .pipeline import bands, build, validate, output_file


def make_parser():
    parser = argparse.ArgumentParser(
        prog="vasp2mlh",
        epilog="Run a JSON/TOML control file with: vasp2mlh path/to/config.toml",
        description="VASP WAVECAR to PAW-metric AO Hamiltonians",
    )
    from . import __version__

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subs = parser.add_subparsers(dest="command", required=True)
    single = subs.add_parser(
        "convert", help="convert one VASP run to validated MLH training labels"
    )
    single.add_argument("run", help="VASP directory with a full uniform mesh")
    single.add_argument(
        "--reference",
        help="fixed-density validation directory (default: RUN/reference)",
    )
    single.add_argument(
        "--potential-run", help="matching PAW potential directory (default: RUN)"
    )
    single.add_argument(
        "--registry",
        help="frozen AO basis (default: RUN/basis.h5 if present, otherwise POTCAR native orbitals)",
    )
    single.add_argument(
        "-o", "--output", help="new result directory (default: RUN/mlh)"
    )
    single.add_argument("--format", choices=["deeph", "hamgnn"], default="deeph")
    single.add_argument(
        "--nao-max",
        type=int,
        choices=[14, 19, 26],
        help="HamGNN orbital mask size (default: 14)",
    )
    window = single.add_mutually_exclusive_group()
    window.add_argument(
        "--band-range",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        help="inclusive 1-based band numbers",
    )
    window.add_argument(
        "--energy-window",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="eV relative to the source Fermi level (default: -1 to +2)",
    )
    single.add_argument(
        "--cutoff", type=float, default=12.0, help="H cutoff in angstrom (default: 12)"
    )
    single.add_argument(
        "--max-rmse",
        type=float,
        default=0.010,
        help="RMS error limit in eV (default: 0.010)",
    )
    single.add_argument(
        "--max-error",
        type=float,
        default=0.050,
        help="maximum absolute error in eV (default: 0.050)",
    )
    single.add_argument(
        "--mesh",
        type=int,
        nargs=3,
        help="dimensions for an explicitly listed uniform grid",
    )
    bulk = subs.add_parser(
        "convert-batch",
        help="convert VASP directories in a JSON manifest to sparse HDF5 models",
    )
    bulk.add_argument(
        "manifest", help="JSON job list, with paths relative to the manifest"
    )
    bulk.add_argument("output", nargs="?", help="batch output directory")
    bulk.add_argument("--workers", type=int, help="number of concurrent structures")
    bulk.add_argument("--threads", type=int, help="BLAS/OpenMP threads per worker")
    bulk.add_argument(
        "--check-only",
        action="store_true",
        default=None,
        help="check required file presence and size without conversion",
    )
    release = subs.add_parser(
        "reconstruct-bands",
        help="build a fixed-basis scalar LDA/GGA model and validate its cutoff spectrum on independent VASP points",
    )
    release.add_argument("run", help="full uniform VASP source mesh")
    release.add_argument("registry", help="frozen AO registry HDF5")
    release.add_argument(
        "reference", help="matched ICHARG=11 VASP validation directory"
    )
    release.add_argument("output", help="validated sparse HDF5 archive")
    scope = release.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--band-range",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        help="inclusive ordered band numbers, starting at 1",
    )
    scope.add_argument(
        "--energy-window",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="energy interval in eV relative to the source Fermi energy",
    )
    release.add_argument(
        "--cutoff", type=float, required=True, help="hard H block cutoff in angstrom"
    )
    release.add_argument(
        "--relative-pruning",
        action="store_true",
        help="instead cut H minus the source energy reference times S, retaining the full ordinary S",
    )
    release.add_argument(
        "--max-rmse",
        type=float,
        required=True,
        help="maximum accepted RMS band error in eV for each of the source and validation samples",
    )
    release.add_argument(
        "--max-error",
        type=float,
        required=True,
        help="maximum accepted single-band error in eV on either sample",
    )
    release.add_argument(
        "--potential-run",
        help="matching WRT_POTENTIAL=total paw directory (default: source run)",
    )
    release.add_argument(
        "--mesh",
        type=int,
        nargs=3,
        help="dimensions for explicitly listed uniform source points",
    )
    release.add_argument(
        "--centre-offset",
        type=float,
        default=6.0,
        help="filter centre above the source Fermi energy in eV (default 6)",
    )
    release.add_argument(
        "--width", type=float, default=1.0, help="filter width in eV (default 1)"
    )
    release.add_argument(
        "--floor",
        type=float,
        default=0.005,
        help="squared filter amplitude floor (default .005)",
    )
    release.add_argument(
        "--taper",
        type=float,
        default=2.0,
        help="filter cutoff taper width in eV (default 2)",
    )
    report = subs.add_parser(
        "band-report",
        help="show the independent validation report embedded in a reconstructed band model",
    )
    report.add_argument("model")
    report.add_argument("--output", help="write the embedded validation report to JSON")
    gen = subs.add_parser(
        "basis-generate",
        help="generate a raw PAW reference or a compact multi-zeta AO registry",
    )
    gen.add_argument("output")
    gen.add_argument("--potcar", nargs="+", required=True)
    gen.add_argument(
        "--layout",
        help="JSON element cutoffs in angstrom and shell multiplicities; omit for the raw PAW baseline",
    )
    opt = subs.add_parser(
        "basis-optimize",
        help="search fixed-layout radial mixtures on a construction WAVECAR",
    )
    for arg in ("run", "registry", "config", "output"):
        opt.add_argument(arg)
    freeze = subs.add_parser(
        "basis-freeze",
        help="freeze normalized radial functions and shell order for a model family",
    )
    freeze.add_argument("config")
    freeze.add_argument("output")
    freeze.add_argument("--potcar")
    audit = subs.add_parser(
        "basis-check",
        help="check AO identity and matrix dimensions against a frozen registry",
    )
    audit.add_argument("registry")
    audit.add_argument("models", nargs="+")
    batch = subs.add_parser(
        "deeph-batch", help="export a checked cohort of DeepH samples"
    )
    batch.add_argument("models", nargs="+")
    batch.add_argument("-o", "--output", required=True)
    batch.add_argument("--basis-lock", required=True)
    make = subs.add_parser(
        "build", help="project WAVECAR using the matching POTCAR and vasprun.xml"
    )
    make.add_argument("run")
    make.add_argument("basis", help="shell configuration JSON or frozen basis registry")
    make.add_argument("output")
    make.add_argument("--mesh", type=int, nargs=3)
    make.add_argument(
        "--complement-eV",
        type=float,
        default=0.0,
        help="energy assigned to unresolved spectral weight",
    )
    make.add_argument(
        "--potential-run",
        help="matching scalar VASP run with WRT_POTENTIAL = total paw",
    )
    fit = subs.add_parser(
        "calibrate",
        help="calibrate direct-operator Ritz energies to the source VASP mesh with geometry-only AO overlap",
    )
    fit.add_argument("model")
    fit.add_argument("overlap")
    fit.add_argument("output")
    fit.add_argument(
        "--bands",
        type=int,
        help="full correction through this band, taper above (default: occupied + 8)",
    )
    fit.add_argument(
        "--alignment-width",
        type=float,
        default=1.0,
        help="Gaussian Ritz alignment width in eV; zero preserves Ritz eigenvectors",
    )
    filt = subs.add_parser(
        "filter", help="build a spectrally filtered scalar PAW Ritz model"
    )
    filt.add_argument("model")
    filt.add_argument("overlap")
    filt.add_argument("output")
    filt.add_argument(
        "--centre-offset",
        type=float,
        required=True,
        help="filter centre relative to source Fermi energy in eV",
    )
    filt.add_argument("--width", type=float, required=True, help="logistic width in eV")
    filt.add_argument(
        "--floor",
        type=float,
        required=True,
        help="squared filter amplitude above the band ceiling",
    )
    filt.add_argument(
        "--taper",
        type=float,
        default=2.0,
        help="width of smooth cutoff below the common band ceiling in eV",
    )
    ov = subs.add_parser(
        "overlap",
        help="compute ordinary AO overlap blocks from structure and radial functions alone",
    )
    ov.add_argument("structure")
    ov.add_argument("basis")
    ov.add_argument("output")
    ov.add_argument(
        "--potcar", help="only for basis specifications using POTCAR radial shells"
    )
    ov.add_argument(
        "--order",
        type=int,
        help="two-centre quadrature order (96 for generated PAW libraries, otherwise 48)",
    )
    sparse = subs.add_parser(
        "sparsify", help="write a validated Hermitian block-sparse H/S pair archive"
    )
    sparse.add_argument("model")
    sparse.add_argument("output")
    sparse.add_argument(
        "--cutoff", type=float, help="maximum pair distance in angstrom"
    )
    sparse.add_argument(
        "--h-tol",
        type=float,
        default=0.0,
        help="Hamiltonian block Frobenius threshold (eV)",
    )
    sparse.add_argument(
        "--s-tol", type=float, default=0.0, help="overlap block Frobenius threshold"
    )
    sparse.add_argument(
        "--relative-pruning",
        action="store_true",
        help="prune H minus the recorded energy origin times S, preserving energy-shift covariance",
    )
    sparse.add_argument(
        "--max-band-change",
        type=float,
        help="maximum accepted truncation error (eV; defaults to all bands)",
    )
    sparse.add_argument(
        "--check-points", help="additional fractional k triples for validation"
    )
    sparse.add_argument(
        "--band-range",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        help="enforce error limit over these 1-based bands; also report full-spectrum change",
    )
    sparse.add_argument(
        "--spinless-time-reversal",
        action="store_true",
        help="enforce H(k)=H(-k)* and S(k)=S(-k)* for an explicitly nonmagnetic scalar calculation; validate the resulting spectral change",
    )
    check = subs.add_parser(
        "validate", help="summarize PAW, Fourier, band and spilling diagnostics"
    )
    check.add_argument("model")
    check.add_argument("--lobsterout")
    check.add_argument(
        "--reference-xml",
        help="independent VASP calculation containing extra zero-weight k points",
    )
    band = subs.add_parser(
        "bands", help="solve H(k)c=ES(k)c on fractional reciprocal points"
    )
    band.add_argument("model")
    band.add_argument("points", help="three fractional k coordinates per line")
    band.add_argument(
        "output", help="text, kx ky kz spin(0-based) then ordered eigenvalues in eV"
    )
    band.add_argument(
        "--validated-window",
        action="store_true",
        help="emit only the declared model window as kx ky kz band_1based energy_eV rows",
    )
    dos = subs.add_parser(
        "dos", help="Gaussian DOS, projected DOS and Lowdin orbital populations"
    )
    dos.add_argument("model")
    dos.add_argument("output")
    dos.add_argument("--window", nargs=2, type=float, default=[-15, 15])
    dos.add_argument("--step", type=float, default=0.05)
    dos.add_argument("--sigma", type=float, default=0.2)
    loc = subs.add_parser(
        "locality", help="tabulate atom-pair distances and matrix norms"
    )
    loc.add_argument("model")
    loc.add_argument("output")
    dh = subs.add_parser("deeph", help="export one DeepH-pack 1.x sample")
    dh.add_argument("model")
    dh.add_argument("destination")
    dh.add_argument("--basis-lock")
    ham = subs.add_parser("hamgnn", help="export a HamGNN OpenMX graph archive")
    ham.add_argument("models", nargs="+")
    ham.add_argument("--basis-lock")
    ham.add_argument("-o", "--output", required=True)
    ham.add_argument("--nao-max", type=int, choices=(14, 19, 26), default=14)
    subspace = subs.add_parser(
        "downfold",
        help="constrain the lowest VASP bands using metric-aware polar orthogonalization",
    )
    subspace.add_argument("model")
    subspace.add_argument("output")
    subspace.add_argument("--bands", type=int, required=True)
    subspace.add_argument("--min-weight", type=float, default=0.5)
    cmp = subs.add_parser(
        "compare",
        help="benchmark a model on matched fixed-density VASP and OpenMX band paths",
    )
    cmp.add_argument("model")
    cmp.add_argument("vasp_xml")
    cmp.add_argument(
        "openmx_band", help="OpenMX .Band with its corresponding .out alongside"
    )
    cmp.add_argument("output", help="local comparison JSON")
    cmp.add_argument("--title", help="display name for the local viewer")
    cmp.add_argument(
        "--edge-bands",
        type=int,
        default=2,
        help="number of ordered bands on each side of filling",
    )
    view = subs.add_parser(
        "view", help="create a self-contained interactive comparison report"
    )
    view.add_argument("comparisons", nargs="+")
    view.add_argument("-o", "--output", required=True)
    return parser, subs.choices


def main(argv=None):
    from contextlib import ExitStack
    from threadpoolctl import threadpool_limits
    from .configuration import (
        backend_setting,
        load_config,
        operation_args,
        runtime_controls,
    )

    parser, commands = make_parser()
    argv = sys.argv[1:] if argv is None else argv
    settings = {}
    try:
        with ExitStack() as stack:
            if len(argv) == 1 and Path(argv[0]).suffix.lower() in (".json", ".toml"):
                path = Path(argv[0]).resolve()
                config = load_config(path)
                command = config.get("command", "convert-batch")
                if command == "convert-batch":
                    args = parser.parse_args(["convert-batch", str(path)])
                else:
                    if (
                        not isinstance(command, str)
                        or command not in commands
                        or config.keys() - {"command", "parameters", "runtime"}
                    ):
                        raise ValueError("Unknown command or configuration field")
                    settings = runtime_controls(config, batch=False)
                    args = operation_args(
                        command, commands[command], config.get("parameters", {})
                    )
                    stack.callback(os.chdir, Path.cwd())
                    os.chdir(path.parent)
                    for name in ("output", "destination"):
                        dest = getattr(args, name, None)
                        if dest is not None and Path(dest).resolve() == path:
                            raise ValueError("Output collides with control file")
                    stack.enter_context(backend_setting(settings.get("backend")))
                    if "threads" in settings:
                        stack.enter_context(
                            threadpool_limits(limits=settings["threads"])
                        )
            else:
                args = parser.parse_args(argv)
            if args.command == "convert":
                args.progress = settings.get("progress", True)
            if args.command == "convert" and "threads" not in settings:
                stack.enter_context(threadpool_limits(limits=1))
            return execute(parser, args)
    except (
        ValueError,
        ParseError,
        KeyError,
        IndexError,
        OSError,
        ImportError,
        np.linalg.LinAlgError,
    ) as exc:
        parser.exit(2, f"vasp2mlh: {exc}\n")


def execute(parser, args):
    if args.command == "convert":
        from .conversion import convert_sample

        convert_sample(args)
    elif args.command == "convert-batch":
        from .batch import run_batch

        summary = run_batch(
            args.manifest, args.output, args.workers, args.threads, args.check_only
        )
        if summary["status"] == "failed":
            parser.exit(1, "Some jobs failed; see the batch report.\n")
    elif args.command == "reconstruct-bands":
        from .band_release import reconstruct

        options = vars(args).copy()
        del options["command"]
        print(json.dumps(reconstruct(**options), indent=2))
    elif args.command == "band-report":
        with h5py.File(args.model) as f:
            report = json.loads(f["band_validation"].attrs["report"])
        if args.output:
            if Path(args.output).resolve() == Path(args.model).resolve():
                raise ValueError("Report output collides with model")
            with output_file(args.output) as tmp:
                tmp.write_text(json.dumps(report, indent=2) + "\n")
        else:
            print(json.dumps(report, indent=2))
    elif args.command == "basis-generate":
        from .atomic import generate

        generate(args.output, args.potcar, args.layout)
    elif args.command == "basis-optimize":
        from .spillage import optimize

        print(
            json.dumps(
                optimize(args.run, args.registry, args.config, args.output),
                indent=2,
            )
        )
    elif args.command == "basis-freeze":
        from .registry import freeze

        freeze(args.config, args.output, args.potcar)
    elif args.command == "basis-check":
        from .registry import audit

        print(json.dumps(audit(args.models, args.registry), indent=2))
    elif args.command == "deeph-batch":
        deeph_batch(args.models, args.output, args.basis_lock)
    elif args.command == "calibrate":
        from .calibrate import calibrate

        calibrate(
            args.model,
            args.overlap,
            args.output,
            args.bands,
            width=args.alignment_width,
        )
    elif args.command == "filter":
        from .filtering import filter_model

        filter_model(
            args.model,
            args.overlap,
            args.output,
            args.centre_offset,
            args.width,
            args.floor,
            args.taper,
        )
    elif args.command == "overlap":
        from .overlap import overlap

        overlap(args.structure, args.basis, args.output, args.potcar, args.order)
    elif args.command == "compare":
        result = compare_path(
            args.model, args.vasp_xml, args.openmx_band, args.edge_bands
        )
        result["title"] = args.title or result["formula"]
        write_comparison(result, args.output)
        print(json.dumps(result["errors"], indent=2))
    elif args.command == "view":
        from .viewer import render

        render(args.comparisons, args.output)
    elif args.command == "downfold":
        from .downfold import downfold

        downfold(args.model, args.output, args.bands, args.min_weight)
    elif args.command == "build":
        build(
            args.run,
            args.basis,
            args.output,
            args.mesh,
            args.complement_eV,
            args.potential_run,
        )
    elif args.command == "sparsify":
        extra = np.loadtxt(args.check_points, ndmin=2) if args.check_points else None
        print(
            json.dumps(
                sparsify(
                    args.model,
                    args.output,
                    args.cutoff,
                    args.h_tol,
                    args.s_tol,
                    args.max_band_change,
                    extra,
                    args.band_range,
                    args.spinless_time_reversal,
                    relative_pruning=args.relative_pruning,
                ),
                indent=2,
            )
        )
    elif args.command == "validate":
        report = validate(args.model)
        if args.lobsterout:
            report["lobster"] = lobster_summary(args.lobsterout)
        if args.reference_xml:
            report["held_out"] = held_out(args.model, args.reference_xml)
        print(json.dumps(report, indent=2))
    elif args.command == "deeph":
        deeph(args.model, args.destination, args.basis_lock)
    elif args.command == "hamgnn":
        hamgnn(args.models, args.output, args.nao_max, args.basis_lock)
    elif args.command == "bands":
        if Path(args.output).resolve() in {
            Path(args.model).resolve(),
            Path(args.points).resolve(),
        }:
            raise ValueError("Band output collides with the model or k-point input")
        k = np.loadtxt(args.points, ndmin=2)
        if k.shape[1] != 3 or not np.isfinite(k).all():
            raise ValueError("Expected finite fractional k-point triples")
        with h5py.File(args.model) as f:
            e = bands(f, k)
            if args.validated_window:
                profile = json.loads(f["band_validation"].attrs["report"])
                values = e[0]
                br = profile["band_range_1based"]
                if br is None:
                    lo, hi = profile["energy_window_absolute_eV"]
                    if np.any(values[:, -1] <= hi):
                        raise ValueError(
                            "AO spectrum does not extend above the declared window at all query points"
                        )
                    mask = (values >= lo) & (values <= hi)
                else:
                    mask = np.zeros(values.shape, dtype=bool)
                    mask[:, br[0] - 1 : br[1]] = True
                ki, bi = np.nonzero(mask)
                with output_file(args.output) as tmp:
                    np.savetxt(
                        tmp,
                        np.column_stack((k[ki], bi + 1, values[ki, bi])),
                        header="kx ky kz band_1based energy_eV",
                    )
                return
        with output_file(args.output) as tmp:
            np.savetxt(
                tmp,
                np.vstack(
                    [
                        np.column_stack((k, np.full(len(k), s), v))
                        for s, v in enumerate(e)
                    ]
                ),
                header="kx ky kz spin band_energies_eV",
            )
    elif args.command == "dos":
        axis, dos, labels, pop = spectra(
            args.model, *args.window, args.step, args.sigma
        )
        with output_file(args.output) as tmp:
            np.savetxt(
                tmp,
                np.column_stack((axis, dos)),
                header="E_minus_Ef_eV VASP_DOS AO_model_DOS " + " ".join(labels),
            )
        print(json.dumps(dict(zip(labels, pop)), indent=2))
    elif args.command == "locality":
        with output_file(args.output) as tmp:
            np.savetxt(
                tmp,
                locality(args.model),
                header="Rx Ry Rz i j distance_A H_frobenius_eV S_frobenius",
                fmt=["%d"] * 5 + ["%.10g"] * 3,
            )
