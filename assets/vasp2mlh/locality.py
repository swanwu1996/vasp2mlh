"""Complete a projected band subspace by minimizing variations across the k mesh."""

import json
from itertools import product

import h5py
import numpy as np
from ase.geometry import minkowski_reduce
from scipy.spatial import ConvexHull, Voronoi

from .calibrate import metric_factors, ordinary_geometry, write_ordinary
from .model import adjoint, mesh_indices
from .pipeline import output_file, paths_collide


PRIOR_WEIGHT = 0.01
COMPLEMENT_GAP_EV = 0.1
PRIOR_OFFSET_EV = 20.0
DEGENERACY_EV = 1e-4


def retained_bands(energies, dimension, band_range, upper):
    """Keep all lower levels, a buffer band, and complete boundary multiplets."""
    if band_range is None:
        top = int(np.max((energies <= upper).sum(axis=1)))
        if top == 0:
            raise ValueError("No VASP bands below the target energy window")
    else:
        top = band_range[1]
    count = top + 1
    while count < energies.shape[1] and np.any(
        energies[:, count] - energies[:, count - 1] <= DEGENERACY_EV
    ):
        count += 1
    if count >= energies.shape[1] or count > dimension:
        raise ValueError(
            "AO dimension or NBANDS cannot resolve the target and its buffer"
        )
    return count


def neighbor_faces(cell, mesh):
    """Finite-volume Laplacian weights (shared Voronoi face / bond length)."""
    mesh = np.asarray(mesh)
    active = mesh > 1
    dims = int(sum(active))
    if dims == 0:
        return []
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T / mesh[:, None]
    reduced, transform = minkowski_reduce(reciprocal, pbc=active)
    basis = np.asarray(reduced)[active]
    axes = transform[active]
    if dims == 1:
        return [(axes[0], 0.5), (-axes[0], 0.5)]
    # Voronoi-relevant vectors of a reduced 2D/3D lattice lie in this shell.
    shifts = np.array(list(product(range(-2, 3), repeat=dims)))
    points = (shifts @ basis) @ np.linalg.qr(basis.T)[0]
    centre = np.flatnonzero(np.all(shifts == 0, axis=1))[0]
    cells = Voronoi(points)
    faces = []
    for (i, j), ridge in zip(cells.ridge_points, cells.ridge_vertices):
        if centre not in (i, j):
            continue
        if -1 in ridge:
            raise ValueError("Unbounded reciprocal-space stencil")
        peer = j if i == centre else i
        vertices = cells.vertices[ridge]
        if dims == 2:
            area = np.linalg.norm(vertices[-1] - vertices[0])
        else:
            offset = vertices - vertices.mean(axis=0)
            _, _, tangent = np.linalg.svd(offset, full_matrices=False)
            area = ConvexHull(offset @ tangent[:2].T).volume
        faces.append((shifts[peer] @ axes, area / np.linalg.norm(points[peer])))
    total = sum(weight for _, weight in faces)
    return [(shift, weight / total) for shift, weight in faces]


def complete_subspace(
    points, mesh, cell, orbital_positions, projections, energies, prior
):
    """Solve for Hermitian h(k) with locked polar-projected eigenpairs Q,E.

    In the complement, enforce eigenvalues >= E_last+gap. The convex objective
    is a gauge-transported nearest-neighbor Dirichlet energy plus a quadratic
    prior. Only source k points and their projections enter the solve.
    """
    left, singular, right = np.linalg.svd(projections, full_matrices=False)
    if np.min(singular) <= 1e-6:
        raise ValueError("Rank-deficient PAW projection of retained bands")
    q = left @ right
    locked = (q * energies[:, None, :]) @ adjoint(q)
    dim, count = q.shape[1:]
    if not np.isfinite(prior) or prior < np.max(energies) + COMPLEMENT_GAP_EV:
        raise ValueError("Localized complement prior lies inside retained bands")
    if count == dim:
        return locked, singular[:, -1] ** 2, 0
    z = np.linalg.qr(q, mode="complete")[0][:, :, count:]
    zt = adjoint(z)
    eye = np.eye(dim)
    h = locked + prior * (z @ zt)
    ix = mesh_indices(points, mesh)
    lookup = {tuple(index): i for i, index in enumerate(ix)}
    neighbors = []
    for step, weight in neighbor_faces(cell, mesh):
        peer = [lookup[tuple(index)] for index in (ix + step) % mesh]
        phase = np.exp(2j * np.pi * (orbital_positions @ (step / mesh)))
        neighbors.append((peer, weight * phase.conj()[:, None] * phase[None, :]))
    if not neighbors:
        return h, singular[:, -1] ** 2, 0
    # Hessian eigenvalues lie in [a,2-a]. Use accelerated projected gradient.
    a = PRIOR_WEIGHT
    lipschitz = 2 - a
    momentum = (np.sqrt(lipschitz) - np.sqrt(a)) / (np.sqrt(lipschitz) + np.sqrt(a))
    floor = energies[:, -1, None] + COMPLEMENT_GAP_EV

    def update(y):
        avg = sum(y[peer] * transport for peer, transport in neighbors)
        candidate = y + ((1 - a) * avg + a * prior * eye - y) / lipschitz
        ev, vec = np.linalg.eigh(zt @ candidate @ z)
        frame = z @ vec
        return locked + (frame * np.maximum(ev, floor)[:, None, :]) @ adjoint(frame)

    y = h
    for step in range(1, 501):
        new = update(y)
        if np.max(abs(new - h)) <= 1e-7 and np.max(abs(update(new) - new)) <= 1e-7:
            return new, singular[:, -1] ** 2, step
        y = new + momentum * (new - h)
        h = new
    raise ValueError("Localized PAW complement did not converge in 500 iterations")


def locality_model(model, overlap_file, output, band_range, energy_window):
    """Write the constrained Hamiltonian in the same fixed ordinary AO basis."""
    if paths_collide(output, (model, overlap_file)):
        raise ValueError("Choose a separate localized-model destination")
    with h5py.File(model) as src, h5py.File(overlap_file) as ov:
        if src.attrs["spinor"] or src["k/H"].shape[0] != 1:
            raise ValueError(
                "Localized completion currently requires scalar wavefunctions"
            )
        atoms, bounds, translations, s_real, k, sk = ordinary_geometry(src, ov)
        atoms.set_cell(src["structure/projection_cell"][:], scale_atoms=True)
        positions = np.repeat(
            atoms.get_scaled_positions(wrap=False), np.diff(bounds), axis=0
        )
        ef = float(src.attrs["fermi_eV"])
        energies = src["k/eigenvalues"][0]
        n = retained_bands(
            energies,
            bounds[-1],
            band_range,
            None if energy_window is None else ef + energy_window[1],
        )
        roots = np.empty_like(sk)
        v = np.empty((len(k), bounds[-1], n), complex)
        with output_file(output) as tmp, h5py.File(tmp, "w") as out:
            out.attrs.update(src.attrs)
            for label in src:
                if label != "real":
                    src.copy(label, out)
            for i, (metric, ordinary) in enumerate(zip(src["k/S"], sk)):
                roots[i] = metric_factors(ordinary)[0]
                projected = metric_factors(metric)[1] @ src["k/projection"][0, i]
                v[i] = projected[:, :n]
                out["k/projection"][0, i] = roots[i] @ projected
            h, weights, steps = complete_subspace(
                k,
                src["k/mesh"][:],
                atoms.cell,
                positions,
                v,
                energies[:, :n],
                ef + PRIOR_OFFSET_EV,
            )
            write_ordinary(
                out,
                src,
                atoms,
                bounds,
                translations,
                s_real,
                k,
                sk,
                (roots @ h @ roots)[None],
            )
            recipe = dict(
                version=1,
                prior_weight=PRIOR_WEIGHT,
                prior_offset_eV=PRIOR_OFFSET_EV,
                gap_eV=COMPLEMENT_GAP_EV,
                degeneracy_eV=DEGENERACY_EV,
                buffer_bands=1,
                band_range=band_range,
                energy_window_from_fermi_eV=energy_window,
            )
            out.attrs.update(
                hamiltonian_method="locality_constrained_paw",
                metric="ordinary",
                complement_policy="source_fermi_plus_20_eV",
                complement_eV=ef + PRIOR_OFFSET_EV,
                locality_recipe=json.dumps(recipe),
            )
            out["locality/min_retained_weight"] = weights
            out["locality"].attrs.update(retained_bands=n, iterations=steps)
