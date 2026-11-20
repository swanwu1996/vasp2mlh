import argparse
import json
from xml.etree.ElementTree import ParseError

import h5py
import numpy as np

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
        help="maximum accepted truncation error (all bands, eV)",
    )
    sparse.add_argument(
        "--check-points", help="additional fractional k triples for validation"
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
    args = parser.parse_args()
    try:
        if args.command == "build":
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
