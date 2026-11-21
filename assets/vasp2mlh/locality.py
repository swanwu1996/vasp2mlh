"""Complete a projected band subspace by minimizing variations across the k mesh."""

import json
from itertools import product

import h5py
import numpy as np
from ase.geometry import minkowski_reduce
from scipy.linalg import block_diag
from scipy.spatial import ConvexHull, Voronoi

from .calibrate import aligned_pbe_reference, metric_factors, ordinary_geometry, write_ordinary
from .filtering import bound_hubbard_spectrum
from .model import adjoint, mesh_indices, spin_rotation
from .pipeline import output_file, paths_collide

PBE_ANCHORED = "pbe-anchored-v1"
WINDOW_FIRST_BUFFER = "window-first-buffer-2"


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


def retained_counts(energies, dimension, band_range, upper, policy):
    if policy is None:
        count = retained_bands(energies, dimension, band_range, upper)
        return np.full(len(energies), count, dtype=int)
    if policy != WINDOW_FIRST_BUFFER:
        raise ValueError(f"Unknown retained-band policy {policy}")
    top = (
        band_range[1]
        if band_range is not None
        else int(np.max((energies <= upper).sum(axis=1)))
    )
    if top < 1:
        raise ValueError("No VASP bands below the target energy window")
    # The requested manifold is common to all k. Close just the boundary
    # multiplet of the second buffer band at its own k point.
    return np.array(
        [retained_bands(row[None], dimension, (1, top + 1), None) for row in energies]
    )


def protected_intervals(energies, band_range, window):
    """Close the target interval under same-k boundary degeneracies."""
    intervals = []
    for row in energies:
        if band_range is not None:
            lo, hi = band_range[0] - 1, band_range[1]
        else:
            lo = int(np.searchsorted(row, window[0], side="left"))
            hi = int(np.searchsorted(row, window[1], side="right"))
        if lo == hi:
            intervals.append((0, 0))
            continue
        while lo > 0 and row[lo] - row[lo - 1] <= DEGENERACY_EV:
            lo -= 1
        while hi < len(row) and row[hi] - row[hi - 1] <= DEGENERACY_EV:
            hi += 1
        intervals.append((lo, hi))
    return np.asarray(intervals)


def window_first_frame(projections, joint, intervals):
    """Polar target frame followed by the polar frame of its orthogonal remainder."""
    count = projections.shape[-1]
    for lo, hi in np.unique(intervals, axis=0):
        if lo < 0 or hi > count or lo > hi:
            raise ValueError("Protected multiplet extends outside the retained space")
        if lo == hi or hi - lo == count:
            continue
        rows = np.flatnonzero((intervals == (lo, hi)).all(axis=1))
        v = projections[rows]
        left, _, right = np.linalg.svd(v[:, :, lo:hi], full_matrices=False)
        target = left @ right
        rest = np.r_[:lo, hi:count]
        others = v[:, :, rest]
        residual = others - target @ (adjoint(target) @ others)
        left, _, right = np.linalg.svd(residual, full_matrices=False)
        frame = np.empty_like(v)
        frame[:, :, lo:hi] = target
        frame[:, :, rest] = left @ right
        joint[rows] = frame
    return joint


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
    points, mesh, cell, orbital_positions, projections, energies, prior, counts=None, protected=None
):
    """Solve for Hermitian h(k) with locked polar-projected eigenpairs Q,E.

    In the complement, enforce eigenvalues >= E_last+gap. The convex objective
    is a gauge-transported nearest-neighbor Dirichlet energy plus a quadratic
    prior. Only source k points and their projections enter the solve.
    """
    nk, dim, width = projections.shape
    counts = np.full(nk, width) if counts is None else np.asarray(counts)
    if counts.shape != (nk,) or np.any(counts < 1) or np.any(counts > min(width, dim)):
        raise ValueError("Invalid retained-band counts for AO projections")
    scalar_prior = np.ndim(prior) == 0
    if not np.isfinite(prior).all():
        raise ValueError("Non-finite localized complement prior")
    weights = np.empty(nk)
    groups = []
    for count in np.unique(counts):
        ix = np.flatnonzero(counts == count)
        e = energies[ix, :count]
        v = projections[ix, :, :count]
        left, singular, right = np.linalg.svd(v, full_matrices=False)
        worst = np.argmin(singular[:, -1])
        if singular[worst, -1] <= 1e-6:
            k = np.asarray(points)[ix[worst]].tolist()
            raise ValueError(
                f"Rank-deficient PAW projection at k={k}, bands 1-{count}, "
                f"minimum singular value {singular[worst, -1]:.3g} (threshold 1e-6)"
            )
        q = left @ right
        if protected is not None:
            q = window_first_frame(v, q, protected[ix])
        locked = (q * e[:, None, :]) @ adjoint(q)
        z = np.linalg.qr(q, mode="complete")[0][:, :, count:]
        floor = e[:, -1, None] + COMPLEMENT_GAP_EV
        if z.shape[-1] and scalar_prior and prior < np.max(floor):
            raise ValueError("Localized complement prior lies inside retained bands")
        weights[ix] = singular[:, -1] ** 2
        groups.append((ix, z, locked, floor))
    target = prior * np.eye(dim) if scalar_prior else prior

    def feasible(candidate):
        projected = np.empty((nk, dim, dim), dtype=complex)
        for ix, z, locked, floor in groups:
            projected[ix] = locked
            if z.shape[-1]:
                block = (
                    candidate if candidate.ndim == 2 or len(ix) == nk else candidate[ix]
                )
                ev, vec = np.linalg.eigh(adjoint(z) @ block @ z)
                frame = z @ vec
                projected[ix] += (frame * np.maximum(ev, floor)[:, None, :]) @ adjoint(
                    frame
                )
        return projected

    h = feasible(target)
    if np.all(counts == dim):
        return h, weights, 0
    ix = mesh_indices(points, mesh)
    lookup = {tuple(index): i for i, index in enumerate(ix)}
    neighbors = []
    for step, weight in neighbor_faces(cell, mesh):
        peer = [lookup[tuple(index)] for index in (ix + step) % mesh]
        phase = np.exp(2j * np.pi * (orbital_positions @ (step / mesh)))
        neighbors.append((peer, weight * phase.conj()[:, None] * phase[None, :]))
    if not neighbors:
        return h, weights, 0
    # Hessian eigenvalues lie in [a,2-a]. Use accelerated projected gradient.
    a = PRIOR_WEIGHT
    lipschitz = 2 - a
    momentum = (np.sqrt(lipschitz) - np.sqrt(a)) / (np.sqrt(lipschitz) + np.sqrt(a))
    def update(y):
        residual = y - target
        avg = sum(residual[peer] * transport for peer, transport in neighbors)
        return feasible(y + ((1 - a) * avg - residual) / lipschitz)

    y = h
    for step in range(1, 501):
        new = update(y)
        if np.max(abs(new - h)) <= 1e-7 and np.max(abs(update(new) - new)) <= 1e-7:
            return new, weights, step
        y = new + momentum * (new - h)
        h = new
    raise ValueError("Localized PAW complement did not converge in 500 iterations")


def locality_model(model, overlap_file, output, band_range, energy_window, *, pbe_model=None, retained_policy=None, hubbard_recipe=None):
    """Write the constrained Hamiltonian in the same fixed ordinary AO basis."""
    if paths_collide(output, (model, overlap_file)):
        raise ValueError("Choose a separate localized-model destination")
    with h5py.File(model) as src, h5py.File(overlap_file) as ov:
        if src["k/H"].shape[0] != 1:
            raise ValueError(
                "Localized completion requires a scalar or spinor eigenspectrum"
            )
        spinor = bool(src.attrs["spinor"])
        atoms, bounds, translations, s_real, k, sk = ordinary_geometry(src, ov)
        atoms.set_cell(src["structure/projection_cell"][:], scale_atoms=True)
        positions = np.repeat(
            atoms.get_scaled_positions(wrap=False), np.diff(bounds), axis=0
        )
        rotation = None
        if spinor:
            positions = np.tile(positions, (2, 1))
            rotation = spin_rotation(src.attrs["saxis"])
        dim = len(positions)
        ef = float(src.attrs["fermi_eV"])
        energies = src["k/eigenvalues"][0]
        counts = retained_counts(
            energies,
            dim,
            band_range,
            None if energy_window is None else ef + energy_window[1],
            retained_policy,
        )
        n = int(max(counts))
        protected = None
        if retained_policy == WINDOW_FIRST_BUFFER:
            window = None if energy_window is None else ef + np.asarray(energy_window)
            protected = protected_intervals(energies, band_range, window)
        roots = np.empty((len(k), dim, dim), complex)
        v = np.empty((len(k), dim, n), complex)
        with output_file(output) as tmp, h5py.File(tmp, "w") as out:
            out.attrs.update(src.attrs)
            for label in src:
                if label != "real":
                    src.copy(label, out)
            for i, (metric, ordinary) in enumerate(zip(src["k/S"], sk)):
                half = metric_factors(ordinary)[0]
                inv = metric_factors(metric)[1]
                if spinor:
                    half, inv = block_diag(half, half), block_diag(inv, inv)
                roots[i] = half
                projected = inv @ src["k/projection"][0, i]
                if spinor:
                    projected = (rotation @ projected.reshape(2, -1)).reshape(
                        projected.shape
                    )

                v[i] = projected[:, :n]
                out["k/projection"][0, i] = roots[i] @ projected
            prior = ef + PRIOR_OFFSET_EV
            reference = None
            if pbe_model is not None:
                with h5py.File(pbe_model) as pbe:
                    offset, reference = aligned_pbe_reference(pbe, ov, k, ef)
                    inv_roots = np.linalg.inv(roots)
                    prior = inv_roots @ reference[0] @ inv_roots
            h, weights, steps = complete_subspace(
                k,
                src["k/mesh"][:],
                atoms.cell,
                positions,
                v,
                energies[:, :n],
                prior,
                counts,
                protected,
            )
            if hubbard_recipe is not None:
                h = bound_hubbard_spectrum(h, hubbard_recipe)
                out.attrs["hubbard_recipe"] = json.dumps(hubbard_recipe)
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
                reference=reference,
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
            if retained_policy is not None:
                recipe.update(
                    retained_policy=retained_policy,
                    buffer_bands=2,
                    polar_order="target_then_remainder",
                    boundary_multiplets="same_k",
                )
                out.attrs["locality_recipe"] = json.dumps(recipe)
                out["locality/retained_by_k"] = counts
            if pbe_model is not None:
                del out.attrs["complement_eV"]
                recipe["pbe_reference_completion"] = "locality-constrained"
                recipe["pbe_reference_prior_offset_eV"] = recipe.pop("prior_offset_eV")
                recipe["method"] = PBE_ANCHORED
                out.attrs.update(
                    hamiltonian_method=PBE_ANCHORED,
                    complement_policy="fermi_aligned_pbe_matrix",
                    pbe_alignment_eV=offset,
                    locality_recipe=json.dumps(recipe),
                )
            if spinor:
                out.attrs["source_saxis"] = src.attrs["saxis"]
                out.attrs["saxis"] = [0.0, 0.0, 1.0]
            out["locality/min_retained_weight"] = weights
            out["locality"].attrs.update(retained_bands=n, iterations=steps)
