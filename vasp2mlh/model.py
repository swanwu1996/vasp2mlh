"""Finite-plane-wave PAW metric and discrete reciprocal/real-space transforms."""

from itertools import product

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, eigvalsh

from .basis import harmonics


class RadialTable:
    """Reuse exact rounded reciprocal radii across k points of one fixed basis."""

    def __init__(self, basis):
        self.shells = [shell for shells in basis.values() for shell in shells]
        self.values = {}
        self.capacity = max(1, min(65536, 2**20 // len(self.shells)))

    def evaluate(self, lengths):
        radii, inverse = np.unique(np.round(lengths, 12), return_inverse=True)
        keys = radii.tolist()
        missing = np.array([q for q in keys if q not in self.values])
        if len(missing):
            fresh = np.column_stack(
                [sh.transform_unique(missing) for sh in self.shells]
            )
            self.values.update(zip(missing.tolist(), fresh))
        table = np.stack([self.values[q] for q in keys])[inverse]
        if len(self.values) > self.capacity:
            self.values.clear()
        return {id(sh): table[:, i] for i, sh in enumerate(self.shells)}


def ao_and_metric(q, atoms, basis, pots, spline, radial_table=None):
    """Return AO columns and low-rank PAW overlap update S_pw = I + P D P†."""
    lengths = np.linalg.norm(q, axis=1)
    radial = radial_table.evaluate(lengths) if radial_table is not None else {}
    volume = atoms.get_volume()
    atom_types = atoms.get_chemical_symbols()
    prototypes = {}
    for symbol in dict.fromkeys(atom_types):
        pot = pots[symbol]
        shells = basis[symbol]
        angular = {
            l: harmonics(l, q) / np.sqrt(volume)
            for l in set(pot.ls + [s.l for s in shells])
        }
        aos = np.column_stack(
            [
                (radial[id(s)] if radial_table is not None else s.transform(lengths))[
                    :, None
                ]
                * angular[s.l]
                for s in shells
            ]
        )
        projs = np.column_stack(
            [
                (-1j) ** l * pot.projector(i, lengths, spline)[:, None] * angular[l]
                for i, l in enumerate(pot.ls)
            ]
        )
        # The monopole augmentation couples only identical (l,m) channels.
        labels = [(i, l, m) for i, l in enumerate(pot.ls) for m in range(-l, l + 1)]
        delta = np.array(
            [
                [
                    pot.augmentation[i, j] if (l, m) == (ll, mm) else 0.0
                    for j, ll, mm in labels
                ]
                for i, l, m in labels
            ]
        )
        prototypes[symbol] = aos, projs, delta
    aos, projs, deltas = [], [], []
    for symbol, pos in zip(atom_types, atoms.positions):
        f, p, d = prototypes[symbol]
        phase = np.exp(-1j * (q @ pos))[:, None]
        aos.append(phase * f)
        projs.append(phase * p)
        deltas.append(d)
    return np.column_stack(aos), np.column_stack(projs), block_diag(*deltas)


def adjoint(a):
    return a.conj().swapaxes(-1, -2)


def project(phi, p, delta, c, energies, spinor, complement):
    """Apply the metric separately to each spin component of the pseudo states."""
    pp = adjoint(p) @ phi
    sphi = phi + p @ (delta @ pp)
    overlap = adjoint(phi) @ sphi
    overlap = (overlap + adjoint(overlap)) / 2
    spectrum = eigvalsh(overlap)
    if spectrum[0] <= max(1e-12, spectrum[-1] * 1e-10):
        raise ValueError(
            f"Dependent PAOs in plane-wave metric: S range {spectrum[[0, -1]]}"
        )
    sfull = block_diag(overlap, overlap) if spinor else overlap
    projections = []
    gram = 0.0
    for component in np.split(c, 2 if spinor else 1):
        pc = adjoint(p) @ component
        gram = gram + adjoint(component) @ component + adjoint(pc) @ delta @ pc
        projections.append(adjoint(sphi) @ component)
    a = np.vstack(projections)
    norm_error = np.max(np.abs(gram - np.eye(len(energies))))
    if not np.isfinite(norm_error) or norm_error > 2e-3:
        raise ValueError(
            f"PAW wavefunction Gram error {norm_error:.3g}; check POTCAR, geometry and LREAL"
        )
    weights = np.sum(a.conj() * cho_solve(cho_factor(sfull), a), axis=0).real
    h = (a * (energies - complement)) @ adjoint(a) + complement * sfull
    return (h + adjoint(h)) / 2, overlap, a, weights, norm_error


def mesh_indices(k, mesh):
    mesh = np.asarray(mesh)
    if mesh.shape != (3,) or np.any(mesh <= 0) or np.prod(mesh) != len(k):
        raise ValueError(
            "Expected every point of the specified full rectangular k mesh (ISYM=-1)"
        )
    offsets = (k - k[0]) * mesh
    indices = np.rint(offsets).astype(int)
    if np.max(np.abs(offsets - indices)) > 1e-6 or len(
        np.unique(indices % mesh, axis=0)
    ) != len(k):
        raise ValueError("Duplicate, missing or off-grid k points")
    return indices % mesh


def mesh_fourier(k, mesh, residues, matrices):
    """Forward FFT on integer mesh offsets; restore the origin phase for shifted grids."""
    indices = mesh_indices(k, mesh)
    flat = np.ravel_multi_index(indices.T, mesh)
    ordered = np.take(matrices, np.argsort(flat), axis=-3)
    grid = ordered.reshape((*matrices.shape[:-3], *mesh, *matrices.shape[-2:]))
    base = np.fft.fftn(grid, axes=(-5, -4, -3), norm="forward").reshape(matrices.shape)
    return base * np.exp(-2j * np.pi * residues @ k[0])[:, None, None]


def bloch(k, translations, mr):
    phase = np.exp(2j * np.pi * np.atleast_2d(k) @ translations.T)
    return np.einsum("kr,...rij->...kij", phase, mr, optimize=True)


def shortest_images(residue, mesh, displacement, cell):
    """All tied closest periodic images for a site pair in the mesh supercell."""
    supercell = mesh[:, None] * cell
    offset = (residue + displacement) / mesh
    trial = np.rint(-offset)
    radius = np.linalg.norm((offset + trial) @ supercell) + 1e-9
    bounds = radius * np.linalg.norm(np.linalg.inv(supercell), axis=0)
    lower, upper = (
        np.ceil(-offset - bounds).astype(int),
        np.floor(-offset + bounds).astype(int),
    )
    shifts = np.array(list(product(*(range(a, b + 1) for a, b in zip(lower, upper)))))
    lengths = np.linalg.norm((shifts + offset) @ supercell, axis=1)
    return residue + mesh * shifts[lengths <= lengths.min() + 1e-9]


def localized_transform(k, mesh, atoms, counts, hs, ss):
    """Divide Fourier coefficients over the nearest pair images (Wigner-Seitz interpolation)."""
    mesh = np.array(mesh)
    residues = np.indices(mesh).reshape(3, -1).T
    hbase = mesh_fourier(k, mesh, residues, hs)
    sbase = mesh_fourier(k, mesh, residues, ss)
    bounds = np.r_[0, np.cumsum(counts)]
    frac = atoms.get_scaled_positions(wrap=False)
    spinor = hs.shape[-1] == 2 * ss.shape[-1]
    # Reuse equivalent pair displacements, including all onsite blocks.
    maps = {}
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            dt = tuple(frac[j] - frac[i])
            if dt not in maps:
                maps[dt] = [
                    shortest_images(t, mesh, dt, atoms.cell.array) for t in residues
                ]
    translations = sorted(
        {tuple(r) for sets in maps.values() for images in sets for r in images}
    )
    index = {r: i for i, r in enumerate(translations)}
    hr = np.zeros((hs.shape[0], len(index), *hs.shape[-2:]), complex)
    sr = np.zeros((len(index), *ss.shape[-2:]), complex)
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            left, right = (
                np.arange(bounds[i], bounds[i + 1]),
                np.arange(bounds[j], bounds[j + 1]),
            )
            sx = np.ix_(left, right)
            hx = (
                np.ix_(np.r_[left, left + bounds[-1]], np.r_[right, right + bounds[-1]])
                if spinor
                else sx
            )
            for t, images in enumerate(maps[tuple(frac[j] - frac[i])]):
                weights = np.exp(-2j * np.pi * (images - residues[t]) @ k[0]) / len(
                    images
                )
                for r, weight in zip(images, weights):
                    ir = index[tuple(r)]
                    hr[:, ir][:, hx[0], hx[1]] = weight * hbase[:, t][:, hx[0], hx[1]]
                    sr[ir][sx] = weight * sbase[t][sx]
    return np.array(translations), hr, sr
