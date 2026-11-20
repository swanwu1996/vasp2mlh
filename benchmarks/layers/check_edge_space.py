"""Compare sparse model and independent native VASP edge subspaces at matched k points."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import eigh, svd

from vasp2mlh.basis import Shell, read_potcar
from vasp2mlh.calibrate import metric_factors
from vasp2mlh.model import adjoint, ao_and_metric
from vasp2mlh.pipeline import file_sha256, output_file
from vasp2mlh.realspace import matrices_k
from vasp2mlh.vasp import Wavecar, metadata, plane_waves


def native_spaces(meta, wave, basis, potentials, edge):
    for ik, point in enumerate(meta["kpoints"]):
        k, bands, coeff = wave.read(0, ik)
        if not np.allclose(k, point, atol=1e-7) or not np.allclose(
            bands[:, 0], meta["eig"][0, ik], atol=1e-4
        ):
            raise ValueError("XML/WAVECAR mismatch")
        ao, p, delta = ao_and_metric(
            plane_waves(k, wave.cell, wave.encut),
            meta["atoms"],
            basis,
            potentials,
            meta["spline"],
        )
        sphi = ao + p @ (delta @ (adjoint(p) @ ao))
        _, inv = metric_factors(adjoint(ao) @ sphi)
        outer = slice(edge.start - 1, edge.stop + 1)
        projection = inv @ (adjoint(sphi) @ coeff[:, outer])
        target, strengths, _ = svd(projection[:, 1:-1], full_matrices=False)
        extended, ext_strengths, _ = svd(projection, full_matrices=False)
        if min(strengths[-1], ext_strengths[-1]) < 1e-6:
            raise ValueError("Rank-deficient native reference subspace")
        gap = min(
            bands[edge.start, 0] - bands[edge.start - 1, 0],
            bands[edge.stop, 0] - bands[edge.stop - 1, 0],
        )
        yield k, target, extended, strengths[-1] ** 2, gap


def cached_spaces(cache, identity, compute):
    if cache is None:
        yield from compute()
        return
    if not cache.exists():
        with output_file(cache) as path, h5py.File(path, "w") as f:
            f.attrs["identity"] = json.dumps(identity, sort_keys=True)
            for i, (k, target, outer, weight, gap) in enumerate(compute()):
                row = f.create_group(str(i))
                row["k"], row["target"], row["outer"] = k, target, outer
                row.attrs.update(weight=weight, gap=gap)
    with h5py.File(cache) as f:
        if json.loads(f.attrs["identity"]) != identity:
            raise ValueError(
                "Subspace cache does not match the supplied basis or reference"
            )
        for i in range(len(f)):
            row = f[str(i)]
            yield (
                row["k"][:],
                row["target"][:],
                row["outer"][:],
                row.attrs["weight"],
                row.attrs["gap"],
            )


def evaluate(run, models, cache=None):
    meta = metadata(run / "vasprun.xml")
    wave = Wavecar(run / "WAVECAR")
    potentials = read_potcar(run / "POTCAR")
    density_hash = file_sha256(run / "CHGCAR")
    paw_hash = file_sha256(run / "POTCAR")
    nocc = int(meta["nelect"] / 2)
    edge = slice(nocc - 2, nocc + 2)
    if meta["spinor"] or meta["nspin"] != 1 or meta["icharg"] != 11:
        raise ValueError("Expected fixed-density scalar reference")
    results = {}
    with ExitStack() as stack:
        files = [stack.enter_context(h5py.File(path)) for path in models]
        first = files[0]
        basis = {}
        for symbol, angular in json.loads(first.attrs["shells"]).items():
            group = first["basis"][symbol]
            basis[symbol] = [
                Shell(l, group[str(i)]["r"][:], group[str(i)]["R"][:], symbol)
                for i, l in enumerate(angular)
            ]
        for f in files:
            if (
                f.attrs["metric"] != "ordinary"
                or f.attrs["overlap_sha256"] != first.attrs["overlap_sha256"]
                or f.attrs["potcar_sha256"] != paw_hash
                or f.attrs["charge_density_sha256"] != density_hash
                or not np.array_equal(f["structure/Z"][:], meta["atoms"].numbers)
                or not np.allclose(
                    f["structure/fractional"][:],
                    meta["atoms"].get_scaled_positions(wrap=False),
                    rtol=0,
                    atol=1e-7,
                )
                or not np.allclose(f["structure/cell"][:], wave.cell, rtol=0, atol=1e-7)
            ):
                raise ValueError(
                    "Reference and labels must share geometry, basis, PAWs and density"
                )
            results[f.filename] = []
        identity = {
            "potcar": paw_hash,
            "density": density_hash,
            "basis": first.attrs["overlap_sha256"],
        }
        if cache is not None:
            identity.update(
                wave=file_sha256(run / "WAVECAR"), xml=file_sha256(run / "vasprun.xml")
            )
        for k, target, outer, weight, gap in cached_spaces(
            cache, identity, lambda: native_spaces(meta, wave, basis, potentials, edge)
        ):
            for f in files:
                h, s = matrices_k(f, [k])
                root, _ = metric_factors(s[0])
                _, c = eigh(h[0, 0], s[0], subset_by_index=(edge.start, edge.stop - 1))
                cos2 = svd(adjoint(root @ c) @ target, compute_uv=False) ** 2
                results[f.filename].append(
                    dict(
                        k=k.tolist(),
                        principal_cos2=cos2.tolist(),
                        mean_cos2=float(cos2.mean()),
                        cos2_with_neighbor_bands=(
                            svd(adjoint(root @ c) @ outer, compute_uv=False) ** 2
                        ).tolist(),
                        min_reference_projectability=float(weight),
                        external_gap_eV=float(gap),
                    )
                )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("models", nargs="+")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        help="Private cache of independent projected reference subspaces",
    )
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(evaluate(args.reference, args.models, args.cache), indent=2) + "\n"
    )
