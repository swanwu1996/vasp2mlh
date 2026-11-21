"""Atom-pair blocks in the cell-periodic gauge. Spin channels stay explicit."""

from functools import lru_cache
import json

import numpy as np
from ase import Atoms

from .model import adjoint, bloch


SPARSE_FORMAT = "vasp2mlh/block-sparse/1"


def structure(f):
    return Atoms(
        numbers=f["structure/Z"][:],
        cell=f["structure/cell"][:],
        scaled_positions=f["structure/fractional"][:],
        pbc=True,
    )


def orbital_bounds(f):
    shells = json.loads(f.attrs["shells"])
    return np.r_[
        0,
        np.cumsum(
            [
                sum(2 * l + 1 for l in shells[s])
                for s in structure(f).get_chemical_symbols()
            ]
        ),
    ]


def indices(bounds, atom, spinor):
    ix = np.arange(bounds[atom], bounds[atom + 1])
    return np.r_[ix, ix + bounds[-1]] if spinor else ix


def reverse(key):
    x, y, z, i, j = key
    return (-x, -y, -z, j, i)


def pair_blocks(f):
    """Yield (R_x,R_y,R_z,i,j), H[channel,row,col], spatial S[row,col]."""
    bounds = orbital_bounds(f)
    spinor = bool(f.attrs["spinor"])
    if f.attrs["format"] == SPARSE_FORMAT:
        pairs = f["pairs"]
        h, s = pairs["H"][:], pairs["S"][:]
        offsets = pairs["offsets"][:]
        for n, key in enumerate(pairs["indices"][:]):
            key = tuple(key)
            i, j = key[-2:]
            ni, nj = np.diff(bounds)[[i, j]]
            start, stop = offsets[n : n + 2]
            factor = 2 if spinor else 1
            hh = h[:, start * factor**2 : stop * factor**2].reshape(
                len(h), factor * ni, factor * nj
            )
            ss = s[start:stop].reshape(ni, nj)
            yield key, hh, ss
            peer = reverse(key)
            if key != peer:
                yield peer, adjoint(hh), adjoint(ss)
        return
    atoms = range(len(bounds) - 1)
    si = [indices(bounds, i, False) for i in atoms]
    hi = [indices(bounds, i, spinor) for i in atoms]
    for r_idx, r in enumerate(f["real/R"][:]):
        h, s = f["real/H"][:, r_idx], f["real/S"][r_idx]
        for i in atoms:
            for j in atoms:
                hh = h[:, hi[i][:, None], hi[j]]
                ss = s[np.ix_(si[i], si[j])]
                if (i == j and not np.any(r)) or np.any(hh) or np.any(ss):
                    yield (*r, i, j), hh, ss


def matrices_k(f, k):
    """Rebuild sampled dense Bloch matrices from either real-space representation."""
    k = np.atleast_2d(k)
    if f.attrs["format"] != SPARSE_FORMAT:
        r = f["real/R"][:]
        return bloch(k, r, f["real/H"][:]), bloch(k, r, f["real/S"][:])
    bounds = orbital_bounds(f)
    spinor = bool(f.attrs["spinor"])
    dim = bounds[-1] * (1 + int(spinor))
    h = np.zeros((f.attrs["spin_channels"], len(k), dim, dim), complex)
    s = np.zeros((len(k), bounds[-1], bounds[-1]), complex)

    @lru_cache(maxsize=128)
    def translation_phase(x, y, z):
        return np.exp(2j * np.pi * (k @ [x, y, z]))[:, None, None]

    for (x, y, z, i, j), hh, ss in pair_blocks(f):
        phase = translation_phase(x, y, z)
        si, sj = slice(bounds[i], bounds[i + 1]), slice(bounds[j], bounds[j + 1])
        s[:, si, sj] += phase * ss
        # Each spin component is contiguous within a given atom block.
        ni, nj = ss.shape
        for a in range(1 + int(spinor)):
            for b in range(1 + int(spinor)):
                rows = slice(si.start + a * bounds[-1], si.stop + a * bounds[-1])
                cols = slice(sj.start + b * bounds[-1], sj.stop + b * bounds[-1])
                h[:, :, rows, cols] += (
                    phase * hh[:, None, a * ni : (a + 1) * ni, b * nj : (b + 1) * nj]
                )
    return h, s
