import argparse
import json
from xml.etree.ElementTree import ParseError

import h5py
import numpy as np

from .comparison import compare_path, write_comparison
from .analysis import held_out, locality, lobster_summary, spectra
from .sparse import sparsify
from .export import deeph, hamgnn
from .pipeline import bands, build, validate, output_file


def main():
    parser = argparse.ArgumentParser(
        prog="vasp2mlh", description="VASP WAVECAR to PAW-metric AO Hamiltonians"
    )
    subs = parser.add_subparsers(dest="command", required=True)
    make = subs.add_parser(
        "build", help="project WAVECAR using the matching POTCAR and vasprun.xml"
    )
    make.add_argument("run")
    make.add_argument("basis", help="JSON shell configuration")
    make.add_argument("output")
    make.add_argument("--mesh", type=int, nargs=3)
    make.add_argument(
        "--complement-eV",
        type=float,
        default=0.0,
        help="energy assigned to unresolved spectral weight",
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
    ham = subs.add_parser("hamgnn", help="export a HamGNN OpenMX graph archive")
    ham.add_argument("models", nargs="+")
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
    args = parser.parse_args()
    try:
        if args.command == "compare":
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
            build(args.run, args.basis, args.output, args.mesh, args.complement_eV)
        elif args.command == "sparsify":
            extra = (
                np.loadtxt(args.check_points, ndmin=2) if args.check_points else None
            )
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
            deeph(args.model, args.destination)
        elif args.command == "hamgnn":
            hamgnn(args.models, args.output, args.nao_max)
        elif args.command == "bands":
            k = np.loadtxt(args.points, ndmin=2)
            if k.shape[1] != 3 or not np.isfinite(k).all():
                raise ValueError("Expected finite fractional k-point triples")
            with h5py.File(args.model) as f:
                e = bands(f, k)
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
