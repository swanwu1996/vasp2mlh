"""Decompose ordered band and operator-action errors at independent VASP points."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import eigvalsh

from vasp2mlh.basis import read_potcar
from vasp2mlh.calibrate import metric_factors
from vasp2mlh.comparison import periodic_match
from vasp2mlh.filtering import filtered_operator
from vasp2mlh.model import adjoint, ao_and_metric
from vasp2mlh.operator import ScalarOperator
from vasp2mlh.pipeline import file_sha256
from vasp2mlh.realspace import matrices_k, structure
from vasp2mlh.registry import read_shells
from vasp2mlh.vasp import Wavecar, metadata, plane_waves


def stats(x):
    a = np.asarray(x)
    return {
        "mae": float(abs(a).mean()),
        "rms": float(np.mean(abs(a) ** 2) ** 0.5),
        "max": float(abs(a).max()),
    }


def measure(run, potential, model, sparse):
    run, potential = Path(run), Path(potential)
    meta = metadata(run / "vasprun.xml")
    wave = Wavecar(run / "WAVECAR")
    pots = read_potcar(run / "POTCAR")
    if (
        meta["icharg"] != 11
        or meta["spinor"]
        or wave.ns != 1
        or wave.nk != len(meta["kpoints"])
        or not np.allclose(wave.cell, meta["atoms"].cell, rtol=0, atol=1e-7)
        or not np.isclose(wave.encut, meta["encut"])
        or meta["lreal"] not in ("F", ".FALSE.")
    ):
        raise ValueError(
            "Expected matching scalar fixed-density reciprocal-projector references"
        )
    op = ScalarOperator(potential, meta, pots)
    local_op = None
    if (run / "vaspout.h5").is_file():
        with h5py.File(run / "vaspout.h5") as f:
            has_total = "results/potential/total" in f and "results/paw/dij" in f
        if has_total:
            local_op = ScalarOperator(run, meta, pots)
    with h5py.File(model) as src, h5py.File(sparse) as out:
        if (
            src.attrs.get("hamiltonian_method") != "filtered_paw_ritz"
            or src.attrs["format"] != "vasp2mlh/1"
            or out.attrs["format"] != "vasp2mlh/block-sparse/1"
        ):
            raise ValueError("Expected a dense filtered model and its sparse label")
        for key in [
            "filter_centre_eV",
            "filter_width_eV",
            "filter_floor",
            "filter_ceiling_eV",
            "filter_taper_eV",
            "overlap_sha256",
            "potential_sha256",
            "potcar_sha256",
            "charge_density_sha256",
            "nelect",
        ]:
            if src.attrs[key] != out.attrs[key]:
                raise ValueError("Dense and sparse source settings differ")
        if src.attrs.get("interpolation_reference_eV") != out.attrs.get(
            "interpolation_reference_eV"
        ):
            raise ValueError(
                "Dense and sparse labels use different interpolation references"
            )
        if src.attrs["potential_sha256"] != file_sha256(potential / "vaspout.h5"):
            raise ValueError("Source potential differs from supplied operator")
        for n, attr in [
            ("POTCAR", "potcar_sha256"),
            ("CHGCAR", "charge_density_sha256"),
        ]:
            if any(file_sha256(d / n) != src.attrs[attr] for d in [run, potential]):
                raise ValueError("PAW or density identity mismatch")
        k = meta["kpoints"]
        atoms = structure(src)
        if (
            not np.isclose(meta["nelect"], src.attrs["nelect"])
            or not np.array_equal(atoms.numbers, meta["atoms"].numbers)
            or not np.allclose(atoms.cell, wave.cell, rtol=0, atol=1e-7)
            or not np.allclose(
                atoms.positions, meta["atoms"].positions, rtol=0, atol=2e-7
            )
            or periodic_match(k[:, None], src["k/points"][:][None]).any()
            or not np.array_equal(src["k/points"][:], out["k/points"][:])
            or not np.array_equal(src["k/eigenvalues"][:], out["k/eigenvalues"][:])
        ):
            raise ValueError(
                "Require disjoint k points and matching source, reference and label"
            )
        n = round(src.attrs["nelect"] / 2)
        edge = slice(n - 2, n + 2)
        nb = src["k/eigenvalues"].shape[-1]
        if not np.isclose(2 * n, meta["nelect"]) or n < 2 or n + 2 > nb or wave.nb < nb:
            raise ValueError("Incompatible electron count or reference band window")
        basis = {s: read_shells(g) for s, g in src["basis"].items()}
        dense_h, sd = matrices_k(src, k)
        sparse_h, ss = matrices_k(out, k)
        if not np.allclose(ss, sd, rtol=0, atol=1e-10):
            raise ValueError("Compression changed the ordinary overlap")
        centre, width, floor, ceiling, taper = [
            src.attrs["filter_" + v]
            for v in ("centre_eV", "width_eV", "floor", "ceiling_eV", "taper_eV")
        ]
        rows = []
        for i, point in enumerate(k):
            where, en, c = wave.read(0, i)
            en = en[:nb, 0]
            c = c[:, :nb]
            if not np.allclose(where, point, atol=1e-7, rtol=0) or not np.allclose(
                en, meta["eig"][0, i, :nb], atol=1e-4, rtol=0
            ):
                raise ValueError("WAVECAR/XML reference mismatch")
            q = plane_waves(where, wave.cell, wave.encut)
            phi, p, d = ao_and_metric(q, meta["atoms"], basis, pots, meta["spline"])
            if not np.allclose(d, op.blocks["qij"], rtol=0, atol=2e-5):
                raise ValueError("PAW metric augmentation mismatch")
            sp = phi + p @ (d @ (adjoint(p) @ phi))
            _, ip = metric_factors(adjoint(phi) @ sp)
            v = ip @ (adjoint(sp) @ c)
            ritz = ip @ (adjoint(phi) @ op.apply(q, where, wave.cell, p, phi)) @ ip
            ritz = (ritz + adjoint(ritz)) / 2
            loc, vf = filtered_operator(
                ritz,
                v,
                en,
                centre,
                width,
                floor,
                ceiling,
                taper,
                return_projections=True,
            )
            weaker_floor = filtered_operator(
                ritz, v, en, centre, width, floor / 5, ceiling, taper
            )
            _, ia = metric_factors(sd[i])
            hd = ia @ dense_h[0, i] @ ia
            hs = ia @ sparse_h[0, i] @ ia
            left, sing, right = np.linalg.svd(v[:, edge], full_matrices=False)
            target = left @ right
            uf, sf, vhf = np.linalg.svd(vf[:, edge], full_matrices=False)
            if min(sing[-1], sf[-1]) < 1e-6:
                raise ValueError("Rank-deficient edge projection")
            filtered_target = uf @ vhf
            e = en[edge]
            native = c[:, edge]
            sn = native + p @ (d @ (adjoint(p) @ native))
            rn = op.apply(q, where, wave.cell, p, native) - sn * e
            local_rn = (
                local_op.apply(q, where, wave.cell, p, native) - sn * e
                if local_op
                else None
            )

            def levels(h):
                return eigvalsh(h, subset_by_index=(n - 2, n + 1))

            el, ed, es = levels(loc), levels(hd), levels(hs)
            terms = {
                "same_k": loc @ target - target * e,
                "interpolation": (hd - loc) @ target,
                "compression": (hs - hd) @ target,
                "total": hs @ target - target * e,
            }
            rows.append(
                {
                    "k": where.tolist(),
                    "min_weight": float(sing[-1] ** 2),
                    "filtered_min_weight": float(sf[-1] ** 2),
                    "min_S": float(eigvalsh(sd[i], subset_by_index=(0, 0))[0]),
                    "ref_gram_error": float(
                        abs(adjoint(native) @ sn - np.eye(4)).max()
                    ),
                    "native_rayleigh_eV": np.abs(
                        np.sum(native.conj() * rn, axis=0)
                    ).tolist(),
                    "native_residual_eV": np.linalg.norm(rn, axis=0).tolist(),
                    "local_potential_residual_eV": np.linalg.norm(
                        local_rn, axis=0
                    ).tolist()
                    if local_op
                    else None,
                    "local_potential_rayleigh_eV": abs(
                        np.sum(native.conj() * local_rn, axis=0)
                    ).tolist()
                    if local_op
                    else None,
                    "filtered_frame_action_eV": {
                        "same_k": np.linalg.norm(
                            loc @ filtered_target - filtered_target * e, axis=0
                        ).tolist(),
                        "total": np.linalg.norm(
                            hs @ filtered_target - filtered_target * e, axis=0
                        ).tolist(),
                    },
                    "energy_eV": {
                        key: value.tolist()
                        for key, value in {
                            "direct_ritz": levels(ritz) - e,
                            "same_k": el - e,
                            "interpolation": ed - el,
                            "compression": es - ed,
                            "total": es - e,
                            "floor_div5_same_k": levels(weaker_floor) - e,
                        }.items()
                    },
                    "action_eV": {
                        key: np.linalg.norm(value, axis=0).tolist()
                        for key, value in terms.items()
                    },
                    "window_action_eV": {
                        key: np.linalg.norm(adjoint(target) @ value, axis=0).tolist()
                        for key, value in terms.items()
                    },
                    "closure_action_eV": float(
                        abs(
                            terms["same_k"]
                            + terms["interpolation"]
                            + terms["compression"]
                            - terms["total"]
                        ).max()
                    ),
                }
            )
        summary = {
            name: {key: stats([r[name][key] for r in rows]) for key in rows[0][name]}
            for name in (
                "energy_eV",
                "action_eV",
                "window_action_eV",
                "filtered_frame_action_eV",
            )
        }
        summary.update(
            min_weight=min(r["min_weight"] for r in rows),
            min_S=min(r["min_S"] for r in rows),
            max_gram_error=max(r["ref_gram_error"] for r in rows),
            native_rayleigh_eV=stats([r["native_rayleigh_eV"] for r in rows]),
            native_residual_eV=stats([r["native_residual_eV"] for r in rows]),
            closure_action_eV=max(r["closure_action_eV"] for r in rows),
        )
        if local_op:
            for key in ("local_potential_residual_eV", "local_potential_rayleigh_eV"):
                summary[key] = stats([r[key] for r in rows])
        return {
            "formula": atoms.get_chemical_formula(),
            "mesh": src["k/mesh"][:].tolist(),
            "interpolation_reference_eV": src.attrs.get("interpolation_reference_eV"),
            "filter_centre_eV": float(centre),
            "filter_ceiling_eV": float(ceiling),
            "filter_floor": float(floor),
            "filter_width_eV": float(width),
            "filter_taper_eV": float(taper),
            "bands": [n - 1, n + 2],
            "nk": len(k),
            "summary": summary,
            "points": rows,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "potential", "model", "sparse"):
        parser.add_argument(key)
    args = parser.parse_args()
    print(json.dumps(measure(**vars(args)), indent=2))
