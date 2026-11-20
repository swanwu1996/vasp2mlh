"""Ordinary AO overlaps by bond-axis two-centre quadrature (angstrom lengths)."""

import itertools
import json
from functools import lru_cache
from math import factorial
from pathlib import Path

import h5py
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.special import lpmv, roots_legendre

from .basis import harmonics, load_basis, read_potcar
from .pipeline import output_file


def axial(l, m, cosine):
    norm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))
    return (
        norm * (np.sqrt(2) * (-1) ** m if m else 1) * lpmv(m, l, np.clip(cosine, -1, 1))
    )


@lru_cache(maxsize=16)
def directions(l):
    # A deterministic overdetermined sample to rotate a complete harmonic shell.
    j = np.arange(max(16, 4 * (2 * l + 1)))
    z = 1 - 2 * (j + 0.5) / len(j)
    az = j * np.pi * (3 - np.sqrt(5))
    xyz = np.column_stack(
        [np.sqrt(1 - z * z) * np.cos(az), np.sqrt(1 - z * z) * np.sin(az), z]
    )
    return xyz, np.linalg.pinv(harmonics(l, xyz))


def rotation(l, frame):
    xyz, inv = directions(l)
    return inv @ harmonics(l, xyz @ frame.T)


def quadrature_blocks(atoms, basis, order):
    if order < 12:
        raise ValueError("Use quadrature order >= 12")
    nodes, weights = roots_legendre(order)
    cut = {s: max(sh.radius[-1] for sh in shells) for s, shells in basis.items()}
    radial = {
        s: [CubicSpline(sh.radius, sh.radial, extrapolate=False) for sh in shells]
        for s, shells in basis.items()
    }

    def integrate(symbol, idx, r):
        sh = basis[symbol][idx]
        return np.nan_to_num(radial[symbol][idx](np.maximum(r, sh.radius[0])))

    def onsite(a, b, i, j, symbol):
        if a.l != b.l:
            return np.zeros((2 * a.l + 1, 2 * b.l + 1))
        # Use the radial quadrature that defines the normalization of each shell.
        # A single Gauss panel can miss narrow features in PAW-derived radial functions.
        if np.array_equal(a.radius, b.radius):
            val = simpson((a.radius * a.radial) * (a.radius * b.radial), x=a.radius)
        else:
            grid = np.union1d(a.radius, b.radius)
            grid = grid[grid <= min(a.radius[-1], b.radius[-1])]
            val = simpson(
                grid**2 * integrate(symbol, i, grid) * integrate(symbol, j, grid),
                x=grid,
            )
        return np.eye(2 * a.l + 1) * val

    @lru_cache(maxsize=1024)
    def axial_blocks(left, right, d):
        shells_a, shells_b = basis[left], basis[right]
        if d == 0:
            return [
                [onsite(a, b, i, j, left) for j, b in enumerate(shells_b)]
                for i, a in enumerate(shells_a)
            ]
        # Piecewise integration avoids derivative kinks at r=d and r+d=cut[right].
        lo, hi = max(0.0, d - cut[right]), min(cut[left], d + cut[right])
        pieces = sorted(
            {lo, hi, *[x for x in (d, cut[right] - d, d - cut[right]) if lo < x < hi]}
        )
        rows = []
        cols = []
        wts = []
        for low, high in itertools.pairwise(pieces):
            r = (nodes + 1) * (high - low) / 2 + low
            sl = np.abs(d - r)
            su = np.minimum(r + d, cut[right])
            s = sl[:, None] + (su - sl)[:, None] * (nodes + 1) / 2
            rows.append(np.broadcast_to(r[:, None], s.shape).ravel())
            cols.append(s.ravel())
            wts.append(
                (
                    (high - low)
                    / 2
                    * weights[:, None]
                    * (su - sl)[:, None]
                    / 2
                    * weights
                    * r[:, None]
                    * s
                    / d
                ).ravel()
            )
        r, s, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
        za = (r * r + d * d - s * s) / (2 * r * d)
        zb = (r * r - d * d - s * s) / (2 * s * d)
        av = [integrate(left, i, r) * w for i in range(len(shells_a))]
        bv = [integrate(right, j, s) for j in range(len(shells_b))]
        ya = {
            (l, m): axial(l, m, za)
            for l in {sh.l for sh in shells_a}
            for m in range(l + 1)
        }
        yb = {
            (l, m): axial(l, m, zb)
            for l in {sh.l for sh in shells_b}
            for m in range(l + 1)
        }
        result = []
        for i, a in enumerate(shells_a):
            row = []
            for j, b in enumerate(shells_b):
                block = np.zeros((2 * a.l + 1, 2 * b.l + 1))
                for m in range(min(a.l, b.l) + 1):
                    val = (
                        np.pi
                        * (2 if m == 0 else 1)
                        * np.dot(av[i] * ya[a.l, m], bv[j] * yb[b.l, m])
                    )
                    block[a.l + m, b.l + m] = block[a.l - m, b.l - m] = val
                row.append(block)
            result.append(row)
        return result

    species = atoms.get_chemical_symbols()
    ixs, jxs, offsets, displacements = neighbor_list(
        "ijSD", atoms, [cut[s] for s in species], self_interaction=True
    )
    for i, j, R, vec in zip(ixs, jxs, offsets, displacements):
        key = (*R, i, j)
        peer = (*(-R), j, i)
        if key > peer:
            continue
        d = np.linalg.norm(vec)
        left, right = species[i], species[j]
        if d < 1e-10:
            if i != j:
                raise ValueError("Coincident atomic centres")
            frame = np.eye(3)
        else:
            z = vec / d
            x = np.cross(np.eye(3)[np.argmin(abs(z))], z)
            x /= np.linalg.norm(x)
            frame = np.column_stack([x, np.cross(z, x), z])
        rots = {
            l: rotation(l, frame) for l in {sh.l for sh in basis[left] + basis[right]}
        }
        blocks = axial_blocks(left, right, round(d, 10))
        mat = np.block(
            [
                [
                    rots[a.l].T @ blocks[u][v] @ rots[b.l]
                    for v, b in enumerate(basis[right])
                ]
                for u, a in enumerate(basis[left])
            ]
        )
        yield key, mat
        if key != peer:
            yield peer, mat.T


def overlap(structure_file, config, output, potcar=None, order=None):
    if Path(output).resolve() in {
        Path(path).resolve()
        for path in (structure_file, config, potcar)
        if path is not None
    }:
        raise ValueError("Choose a separate overlap output")
    if order is None:
        order = 48
        if h5py.is_hdf5(config):
            with h5py.File(config) as registry:
                if registry.attrs.get("method") in {
                    "paw-native-v1",
                    "paw-compact-v1",
                    "paw-spillage-v1",
                }:
                    order = 96
    atoms = read(structure_file)
    basis = load_basis(
        config, read_potcar(potcar) if potcar else {}, atoms.get_chemical_symbols()
    )
    if set(basis) != set(atoms.get_chemical_symbols()):
        raise ValueError("Species and basis differ")
    ls = {s: [a.l for a in shells] for s, shells in basis.items()}
    with output_file(output) as tmp, h5py.File(tmp, "w") as f:
        f.attrs.update(
            format="vasp2mlh/overlap/1",
            metric="ordinary",
            angular_convention="real tesseral m=-l..l, Wikipedia",
            length_unit="angstrom",
            shells=json.dumps(ls),
            order=order,
        )
        f["structure/cell"] = atoms.cell.array
        f["structure/fractional"] = atoms.get_scaled_positions(wrap=False)
        f["structure/Z"] = atoms.numbers
        for s, shells in basis.items():
            for j, sh in enumerate(shells):
                g = f.create_group(f"basis/{s}/{j}")
                g.attrs["l"] = sh.l
                g["r"] = sh.radius
                g["R"] = sh.radial
        keys = []
        blocks = []
        shapes = []
        for key, block in quadrature_blocks(atoms, basis, order):
            keys.append(key)
            blocks.append(block.ravel())
            shapes.append(block.shape)
        f["pairs/indices"] = keys
        f["pairs/offsets"] = np.r_[0, np.cumsum([len(b) for b in blocks])]
        f.create_dataset("pairs/S", data=np.concatenate(blocks), compression="lzf")
        # Hard links expose the same scalar blocks to DeepH-dock as overlap.h5.
        f["atom_pairs"] = f["pairs/indices"]
        f["chunk_boundaries"] = f["pairs/offsets"]
        f["chunk_shapes"] = shapes
        f["entries"] = f["pairs/S"]


def overlap_translations(f, bounds):
    pairs = f["pairs/indices"][:]
    r, inverse = np.unique(pairs[:, :3], axis=0, return_inverse=True)
    s = np.zeros((len(r), bounds[-1], bounds[-1]))
    offsets = f["pairs/offsets"][:]
    data = f["pairs/S"][:]
    for n, (_, _, _, i, j) in enumerate(pairs):
        block = s[inverse[n], bounds[i] : bounds[i + 1], bounds[j] : bounds[j + 1]]
        block[:] = data[offsets[n] : offsets[n + 1]].reshape(block.shape)
    return r, s
