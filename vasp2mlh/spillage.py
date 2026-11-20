"""Select radial mixtures on construction WAVECARs with a fixed orbital layout."""

import json
from itertools import product
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import block_diag, eigvalsh, solve

from .basis import Shell, read_potcar
from .model import adjoint, ao_and_metric
from .pipeline import file_sha256
from .registry import read_registry, write_registry
from .vasp import Wavecar, metadata, plane_waves


def diffuse_shells(shells, selected, widths, radius, taper_start):
    x = np.clip((radius - taper_start) / (radius[-1] - taper_start), 0, 1)
    taper = 1 - x**3 * (10 - 15 * x + 6 * x * x)
    extended = list(shells)
    for i in selected:
        l = shells[i].l
        for width in widths:
            radial = radius**l * np.exp(-0.5 * (radius / width) ** 2) * taper
            extended.append(Shell(l, radius, radial, f"gaussian-l{l}-sigma{width:g}"))
    return extended


def radial_map(extended, original_count, selected, options):
    bounds = np.cumsum([0] + [2 * sh.l + 1 for sh in extended])
    dim = bounds[original_count]
    transform = np.eye(bounds[-1], dim)
    for index, (candidate, angle) in zip(selected, options):
        cols = slice(bounds[index], bounds[index + 1])
        rows = slice(bounds[candidate], bounds[candidate + 1])
        transform[cols, cols] *= np.cos(angle)
        transform[rows, cols] = np.sin(angle) * np.eye(2 * extended[index].l + 1)
    return transform


def coverage(samples, transform, condition_limit):
    weights, occupied, conditions = [], [], []
    for s, a, nocc, edge in samples:
        trial = transform.T @ s @ transform
        vals = eigvalsh(trial)
        if vals[0] <= 0 or vals[-1] > condition_limit * vals[0]:
            return None
        conditions.append(vals[-1] / vals[0])
        projection = transform.T @ a
        gram = adjoint(projection) @ solve(trial, projection, assume_a="pos")
        weights.append(eigvalsh(gram[edge, edge])[0])
        occupied.append(gram.diagonal().real[:nocc].mean())
    return {
        "worst_edge": float(min(weights)),
        "mean_edge": float(np.mean(weights)),
        "mean_occupied": float(np.mean(occupied)),
        "max_condition": float(max(conditions)),
    }


def optimize(run, registry, config, output):
    run = Path(run)
    if Path(output).resolve() in {
        Path(p).resolve()
        for p in (
            config,
            registry,
            run / "WAVECAR",
            run / "POTCAR",
            run / "vasprun.xml",
        )
    }:
        raise ValueError("Choose a separate registry output")
    spec = json.loads(Path(config).read_text())
    meta = metadata(run / "vasprun.xml")
    wave = Wavecar(run / "WAVECAR")
    atoms = meta["atoms"]
    if (
        meta["spinor"]
        or meta["nspin"] != 1
        or wave.ns != 1
        or meta["lreal"] not in ("F", ".FALSE.")
        or not np.allclose(wave.cell, atoms.cell, atol=1e-7)
        or not np.isclose(meta["encut"], wave.encut)
        or meta["eig"].shape != (1, wave.nk, wave.nb)
    ):
        raise ValueError(
            "Expected a matching scalar WAVECAR with reciprocal PAW projectors"
        )
    symbol = spec["element"]
    pots = read_potcar(run / "POTCAR")
    basis = read_registry(registry)
    read_registry(registry, atoms.get_chemical_symbols(), pots)
    if symbol not in atoms.get_chemical_symbols() or any(
        pots[s].title.split() != meta["titles"][s].split()
        for s in set(atoms.get_chemical_symbols())
    ):
        raise ValueError("Source species or PAW titles differ")
    shells = basis[symbol]
    selected = spec["shell_indices"]
    widths, angles = (
        np.asarray(spec["widths_A"], float),
        np.asarray(spec["angles_rad"], float),
    )
    cutoff, start = spec["cutoff_A"], spec["taper_start_A"]
    nk = spec.get("k_indices", list(range(wave.nk)))
    first, last = spec["bands"]
    nocc = int(np.ceil(meta["nelect"] / 2))
    condition = spec["max_condition"]
    samples_n = spec.get("radial_points", 769)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(type(i) is not int or not 0 <= i < len(shells) for i in selected)
        or not nk
        or len(set(nk)) != len(nk)
        or any(type(i) is not int or not 0 <= i < wave.nk for i in nk)
        or not 0 <= start < cutoff
        or not np.isfinite([start, cutoff, condition]).all()
        or type(samples_n) is not int
        or samples_n < 3
        or condition <= 1
        or type(first) is not int
        or type(last) is not int
        or not 1 <= first <= last <= wave.nb
        or not 1 <= nocc <= wave.nb
        or widths.ndim != 1
        or angles.ndim != 1
        or not len(widths)
        or not len(angles)
        or not np.isfinite(widths).all()
        or not np.isfinite(angles).all()
        or np.any(widths <= 0)
        or any(shells[i].radius[-1] > cutoff for i in selected)
    ):
        raise ValueError("Invalid radial search, band window or construction k indices")
    radius = np.linspace(0, cutoff, samples_n)
    extended = diffuse_shells(shells, selected, widths, radius, start)
    trials = {**basis, symbol: extended}
    samples = []
    edge = slice(first - 1, last)
    for i in nk:
        k, e, c = wave.read(0, i)
        if not np.allclose(k, meta["kpoints"][i], atol=1e-7, rtol=0) or not np.allclose(
            e[:, 0], meta["eig"][0, i], atol=1e-4, rtol=0
        ):
            raise ValueError("WAVECAR and XML k points or eigenvalues differ")
        phi, p, d = ao_and_metric(
            plane_waves(k, wave.cell, wave.encut), atoms, trials, pots, meta["spline"]
        )
        c = c[:, : max(last, nocc)]
        pc = adjoint(p) @ c
        if (
            np.max(abs(adjoint(c) @ c + adjoint(pc) @ d @ pc - np.eye(c.shape[1])))
            > 2e-3
        ):
            raise ValueError("Reference states fail the PAW normalization check")
        sphi = phi + p @ (d @ (adjoint(p) @ phi))
        samples.append((adjoint(phi) @ sphi, adjoint(sphi) @ c, nocc, edge))
    choices = [
        [
            (len(shells) + j * len(widths) + w, a)
            for w in range(len(widths))
            for a in angles
            if a != 0 or w == 0
        ]
        for j in range(len(selected))
    ]
    best_score = -np.inf
    for choice in product(*choices):
        local = radial_map(extended, len(shells), selected, choice)
        transform = block_diag(
            *[
                local if s == symbol else np.eye(sum(2 * sh.l + 1 for sh in basis[s]))
                for s in atoms.get_chemical_symbols()
            ]
        )
        stats = coverage(samples, transform, condition)
        if stats is None:
            continue
        score = (
            0.5 * stats["worst_edge"]
            + 0.4 * stats["mean_edge"]
            + 0.1 * stats["mean_occupied"]
        )
        if score > best_score:
            best_score, best, result = score, choice, stats
    if not np.isfinite(best_score):
        raise ValueError("No radial mixture satisfies the specified condition limit")
    chosen = list(shells)
    for i, (j, angle) in zip(selected, best):
        if angle == 0:
            continue
        sh = shells[i]
        old = CubicSpline(sh.radius, sh.radial, extrapolate=False)(
            np.maximum(radius, sh.radius[0])
        )
        radial = np.cos(angle) * np.nan_to_num(old) + np.sin(angle) * extended[j].radial
        chosen[i] = Shell(
            sh.l, radius, radial, f"{sh.label}+{extended[j].label}@{angle:g}"
        )
    with h5py.File(registry) as f:
        hashes = {
            s: g.attrs["paw_reference_sha256"]
            for s, g in f["basis"].items()
            if "paw_reference_sha256" in g.attrs
        }
    hashes[symbol] = pots[symbol].sha256
    basis[symbol] = chosen
    result["mixtures"] = [
        {"shell": i, "candidate": extended[j].label, "angle_rad": float(angle)}
        for i, (j, angle) in zip(selected, best)
    ]
    write_registry(
        basis,
        output,
        "paw-spillage-v1",
        hashes,
        {
            "search": spec,
            "result": result,
            "parent_registry_sha256": file_sha256(registry),
            "wavecar_sha256": file_sha256(run / "WAVECAR"),
            "xml_sha256": file_sha256(run / "vasprun.xml"),
        },
    )
    return result
