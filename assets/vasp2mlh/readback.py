"""Verify serialized scalar ML labels against every native atom-pair block."""

import json
import pickle
import zipfile
from collections import OrderedDict, defaultdict
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from ase.io import read

from .export import OPENMX_ORDER, blocks
from .pipeline import file_sha256
from .realspace import structure
from .registry import inspect_basis


def require(condition, message):
    if not condition:
        raise ValueError(message)


def deeph_matrix(path):
    with h5py.File(path) as f:
        keys, shapes, bounds, data = (
            f[name][:]
            for name in ("atom_pairs", "chunk_shapes", "chunk_boundaries", "entries")
        )
    n = len(keys)
    require(
        keys.shape == (n, 5)
        and keys.dtype.kind in "iu"
        and shapes.shape == (n, 2)
        and shapes.dtype.kind in "iu"
        and np.all(shapes > 0)
        and bounds.shape == (n + 1,)
        and bounds.dtype.kind in "iu"
        and bounds[0] == 0
        and data.ndim == 1
        and bounds[-1] == len(data)
        and np.array_equal(np.diff(bounds), shapes.prod(axis=1)),
        f"Invalid DeepH block layout in {path.name}",
    )
    result = {
        tuple(key): data[bounds[i] : bounds[i + 1]].reshape(shapes[i])
        for i, key in enumerate(keys)
    }
    require(len(result) == n, f"Duplicate DeepH pairs in {path.name}")
    return result


def deeph_blocks(folder, model, info):
    atoms = structure(model)
    geom = read(folder / "POSCAR", format="vasp")
    require(
        np.array_equal(geom.numbers, atoms.numbers)
        and np.allclose(geom.cell, atoms.cell, atol=1e-8, rtol=0)
        and np.allclose(geom.positions, atoms.positions, atol=1e-8, rtol=0),
        "DeepH geometry or atom order differs",
    )
    meta = json.loads((folder / "info.json").read_text())
    require(
        meta["ao_sha256_by_element"] == info["sha256_by_element"]
        and meta["elements_orbital_map"] == json.loads(model.attrs["shells"])
        and meta["atoms_quantity"] == len(atoms)
        and meta["orbits_quantity"] == info["spatial_dimension"]
        and np.isclose(
            meta["fermi_energy_eV"], model.attrs["fermi_eV"], atol=1e-10, rtol=0
        )
        and np.isclose(meta["occupation"], model.attrs["nelect"], atol=1e-10, rtol=0)
        and meta["spinful"]
        == bool(model.attrs["spinor"] or model.attrs["spin_channels"] == 2)
        and not meta["orthogonal_basis"],
        "DeepH AO metadata differs",
    )
    h = deeph_matrix(folder / "hamiltonian.h5")
    s = deeph_matrix(folder / "overlap.h5")
    require(h.keys() == s.keys(), "DeepH H and S pair sets differ")
    return {key: (v, s[key]) for key, v in h.items()}


def load_graphs(path):
    """Read the PyG object array using only the classes emitted by this exporter."""
    import torch
    from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    def load_storage(data):
        return torch.load(BytesIO(data), map_location="cpu", weights_only=True)

    allowed = {
        (cls.__module__, cls.__name__): cls
        for cls in (
            Data,
            DataEdgeAttr,
            DataTensorAttr,
            GlobalStorage,
            OrderedDict,
            np.ndarray,
            np.dtype,
        )
    }
    from numpy.lib import NumpyVersion

    if NumpyVersion(np.__version__) >= "2.0.0":
        from numpy._core.multiarray import _reconstruct
    else:
        from numpy.core.multiarray import _reconstruct
    for mod in ("numpy.core.multiarray", "numpy._core.multiarray"):
        allowed[mod, "_reconstruct"] = _reconstruct
    allowed["torch.storage", "_load_from_bytes"] = load_storage
    allowed["torch._utils", "_rebuild_tensor_v2"] = torch._utils._rebuild_tensor_v2

    class GraphReader(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in allowed:
                raise ValueError(f"Unexpected class in graph archive: {module}.{name}")
            return allowed[module, name]

    with zipfile.ZipFile(path) as archive, archive.open("graph.npy") as stream:
        version = np.lib.format.read_magic(stream)
        require(version == (1, 0), "Unsupported graph array header")
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        require(
            shape == () and dtype == np.dtype(object),
            "Expected a scalar graph dictionary",
        )
        graphs = GraphReader(stream).load().item()
    require(
        isinstance(graphs, dict)
        and set(graphs) == set(range(len(graphs)))
        and all(isinstance(g, Data) for g in graphs.values()),
        "Invalid indexed HamGNN graph collection",
    )
    return graphs


def load_graph(path):
    graphs = load_graphs(path)
    require(len(graphs) == 1, "Expected one HamGNN graph for this structure")
    return graphs[0]


def hamgnn_blocks(path, model, info, nao, *, graph=None):
    from DFT_interfaces.openmx.utils import (
        au2ang,
        au2ev,
        basis_def_14,
        basis_def_19,
        basis_def_26,
    )

    g = load_graph(path) if graph is None else graph
    atoms = structure(model)
    n = len(atoms)
    require(
        json.loads(g.ao_sha256_by_element) == info["sha256_by_element"]
        and np.isclose(g.total_energy * au2ev, model.attrs["total_energy_eV"])
        and np.allclose(g.doping_charge, [model.attrs["doping_charge"]])
        and np.array_equal(g.z, atoms.numbers)
        and np.array_equal(g.node_counts, [n])
        and np.allclose(g.cell, atoms.cell.array[None] / au2ang, atol=2e-6)
        and np.allclose(g.pos, atoms.positions / au2ang, atol=2e-6),
        "HamGNN geometry, atom order or AO fingerprint differs",
    )
    masks = {14: basis_def_14, 19: basis_def_19, 26: basis_def_26}[nao]
    shells = json.loads(model.attrs["shells"])
    slots = []
    for atom in atoms:
        order = []
        for l in shells[atom.symbol]:
            order.extend(len(order) + np.array(OPENMX_ORDER[l]))
        mask = np.asarray(masks[atom.number])
        require(len(mask) == len(order), "HamGNN template disagrees with AO layout")
        slots.append(mask[np.argsort(order)])
    edges, shifts = np.asarray(g.edge_index), np.asarray(g.cell_shift)
    ne = len(shifts)
    require(
        edges.shape == (2, ne)
        and edges.dtype.kind in "iu"
        and shifts.shape == (ne, 3)
        and shifts.dtype.kind in "iu"
        and np.all((edges >= 0) & (edges < n))
        and np.allclose(g.nbr_shift, shifts @ (atoms.cell.array / au2ang), atol=3e-6),
        "Invalid HamGNN edges or cell shifts",
    )
    indices = [(*r, i, j) for r, (i, j) in zip(shifts, edges.T)]
    inverse = np.asarray(g.inv_edge_idx)
    require(
        inverse.shape == (ne,)
        and inverse.dtype.kind in "iu"
        and np.all((inverse >= 0) & (inverse < ne))
        and all(
            indices[idx] == (-a, -b, -c, j, i)
            for idx, (a, b, c, i, j) in zip(inverse, indices)
        ),
        "Invalid HamGNN reverse edges",
    )
    keys = [(0, 0, 0, i, i) for i in range(n)] + indices
    require(len(set(keys)) == len(keys), "Duplicate HamGNN edges")
    spinor = bool(model.attrs["spinor"])
    hd = nao * (2 if spinor else 1)
    s = np.asarray(g.overlap)
    if spinor:
        on = np.asarray(g.Hon) + 1j * np.asarray(g.iHon)
        off = np.asarray(g.Hoff) + 1j * np.asarray(g.iHoff)
        require(
            on.shape == (n, hd**2)
            and off.shape == (ne, hd**2)
            and g.iHon0.shape == g.iHon.shape
            and g.iHoff0.shape == g.iHoff.shape
            and not np.any(np.asarray(g.iHon0))
            and not np.any(np.asarray(g.iHoff0)),
            "Invalid HamGNN spinor components",
        )
        h = np.concatenate((on, off))
    else:
        h = np.asarray(g.hamiltonian)
        require(
            np.array_equal(h[:n], g.Hon) and np.array_equal(h[n:], g.Hoff),
            "Inconsistent HamGNN Hamiltonian tensors",
        )
    require(
        h.shape == (n + ne, hd**2)
        and s.shape == (n + ne, nao**2)
        and np.array_equal(s[:n], g.Son)
        and np.array_equal(s[n:], g.Soff),
        "Inconsistent HamGNN matrix tensors",
    )
    require(
        g.Hon0.shape == g.Hon.shape
        and g.Hoff0.shape == g.Hoff.shape
        and not np.any(np.asarray(g.Hon0))
        and not np.any(np.asarray(g.Hoff0)),
        "Invalid HamGNN reference Hamiltonian tensors",
    )
    result = {}
    spin_slots = [np.r_[a, a + nao] for a in slots] if spinor else slots
    for key, hh, ss in zip(keys, h, s):
        i, j = key[-2:]
        active = np.ix_(slots[i], slots[j])
        hblock = hh.reshape(hd, hd)[np.ix_(spin_slots[i], spin_slots[j])]
        sblock = ss.reshape(nao, nao)[active]
        require(
            np.count_nonzero(hh) == np.count_nonzero(hblock)
            and np.count_nonzero(ss) == np.count_nonzero(sblock),
            "Nonzero padded HamGNN orbitals",
        )
        result[key] = (
            hblock.astype(np.complex128 if spinor else float) * au2ev,
            sblock,
        )
    return result


def verify_export(output, fmt, model, points, nao=None):
    info = inspect_basis(model, None)
    if fmt == "deeph":
        folder = Path(output) / "deeph" / "0"
        files = [
            folder / name
            for name in ("POSCAR", "info.json", "hamiltonian.h5", "overlap.h5")
        ]
        exported = deeph_blocks(folder, model, info)
    else:
        files = [Path(output) / "graph_data.npz"]
        exported = hamgnn_blocks(files[0], model, info, nao or 14)

    result = verify_blocks(exported, fmt, model, points)
    result["file_sha256"] = {str(p.relative_to(output)): file_sha256(p) for p in files}
    return result


def verify_blocks(exported, fmt, model, points):
    # Group by atom pair so the independent-k Fourier check never allocates Nk x D x D.
    by_pair = defaultdict(list)
    for key, h, s in blocks(model):
        require(key in exported, f"Missing {fmt} block {key}")
        hh, ss = exported.pop(key)
        for actual, expected, atol in ((hh, h, 1e-5), (ss, s, 1e-7)):
            require(
                actual.shape == expected.shape
                and np.allclose(actual, expected, rtol=2e-7, atol=atol),
                f"{fmt} matrix mismatch for pair {key}",
            )
        by_pair[key[-2:]].append((key[:3], hh - h, ss - s))
    require(not exported, f"Unexpected {fmt} blocks")
    max_h = max_s = 0.0
    for entries in by_pair.values():
        shifts, dh, ds = zip(*entries)
        phase = np.exp(2j * np.pi * (points @ np.asarray(shifts).T))
        max_h = max(
            max_h, float(np.max(abs(phase @ np.asarray(dh).reshape(len(dh), -1))))
        )
        max_s = max(
            max_s, float(np.max(abs(phase @ np.asarray(ds).reshape(len(ds), -1))))
        )
    require(
        max_h <= 5e-5 and max_s <= 2e-6,
        f"{fmt} Bloch matrix mismatch: H={max_h} eV, S={max_s}",
    )
    return dict(
        status="passed",
        format=fmt,
        kpoints=len(points),
        max_delta_Hk_eV=max_h,
        max_delta_Sk=max_s,
    )
