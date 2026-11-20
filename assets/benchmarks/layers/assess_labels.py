"""Compress a scalar AO model and compare its four ordered edge bands with fixed-density VASP."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from vasp2mlh.comparison import compare_path, write_comparison
from vasp2mlh.model import bloch
from vasp2mlh.overlap import overlap_translations
from vasp2mlh.pipeline import bands, file_sha256
from vasp2mlh.realspace import matrices_k, orbital_bounds
from vasp2mlh.sparse import sparsify
from vasp2mlh.vasp import metadata


def check_interior(source, sparse, overlap, reference):
    ref = metadata(reference)
    k = ref["kpoints"]
    nocc = int(ref["nelect"] / 2)
    edge = slice(nocc - 2, nocc + 2)
    with h5py.File(sparse) as f, h5py.File(source) as dense, h5py.File(overlap) as ov:
        for name, attr in (
            ("POTCAR", "potcar_sha256"),
            ("CHGCAR", "charge_density_sha256"),
        ):
            if file_sha256(reference.parent / name) != f.attrs[attr]:
                raise ValueError(f"Interior and source {name} differ")
        if (
            ref["spinor"]
            or ref["nspin"] != 1
            or ref["icharg"] != 11
            or not np.array_equal(f["structure/Z"][:], ref["atoms"].numbers)
            or not np.allclose(
                f["structure/cell"][:], ref["atoms"].cell, rtol=0, atol=1e-7
            )
            or not np.allclose(
                f["structure/fractional"][:],
                ref["atoms"].get_scaled_positions(wrap=False),
                rtol=0,
                atol=1e-7,
            )
            or f.attrs["nelect"] != ref["nelect"]
            or not np.isclose(f.attrs["encut_eV"], ref["encut"])
        ):
            raise ValueError(
                "Incompatible fixed-density reference geometry or electronic settings"
            )
        diff_k = k[:, None] - f["k/points"][:]
        if np.any(np.all(abs(diff_k - np.rint(diff_k)) < 1e-7, axis=2)):
            raise ValueError("Interior reference intersects the source mesh")
        ev = bands(f, k)[0, :, edge]
        delta = ev - ref["eig"][0, :, edge]
        r, s = overlap_translations(ov, orbital_bounds(f))
        return dict(
            nk=len(k),
            mae_eV=float(abs(delta).mean()),
            max_eV=float(abs(delta).max()),
            rmse_eV=float(np.mean(delta**2) ** 0.5),
            compression_max_eV=float(abs(ev - bands(dense, k)[0, :, edge]).max()),
            independent_S_max=float(abs(matrices_k(f, k)[1] - bloch(k, r, s)).max()),
        )


def assess(
    model,
    overlap,
    path,
    openmx,
    prefix,
    interior=None,
    title=None,
    thresholds=(0.03, 0.01, 0.003, 0.001, 0),
):
    ref = metadata(path)
    nocc = int(ref["nelect"] / 2)
    target = Path(str(prefix) + "-labels.h5")
    for tol in thresholds:
        try:
            sparsify(
                model,
                target,
                h_tol=tol,
                band_range=[nocc - 1, nocc + 2],
                max_band_change=0.010,
                extra_points=ref["kpoints"],
                spinless_time_reversal=True,
            )
            break
        except ValueError as error:
            if "band change" not in str(error):
                raise
            print(f"Rejected H threshold {tol} eV: {error}", flush=True)
    else:
        raise ValueError("No threshold passed the compression test")
    result = compare_path(target, path, openmx)
    result["title"] = title or prefix.name
    write_comparison(result, Path(str(prefix) + "-labels.json"))
    if interior is not None:
        gauges = check_interior(model, target, overlap, interior)
        write_comparison(gauges, Path(str(prefix) + "-interior.json"))
        print(gauges, flush=True)
    print(result["errors"]["model_vasp"], result["sampled_gap_eV"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "overlap", "path", "openmx", "prefix"]:
        parser.add_argument(name, type=Path)
    parser.add_argument("--interior", type=Path)
    parser.add_argument("--title")
    assess(**vars(parser.parse_args()))
