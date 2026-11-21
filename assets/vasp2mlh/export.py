"""Framework-specific orbital order, units and atom-pair serialization."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json

import h5py
import numpy as np
from ase.io import write
from scipy.linalg import block_diag

from .pipeline import file_sha256, output_file, paths_collide
from .registry import audit
from .realspace import structure, orbital_bounds, pair_blocks


def blocks(f):
    for key, h, s in pair_blocks(f):
        yield key, h[0] if len(h) == 1 else block_diag(*h), s


def real_matrix(x, tol, name):
    error = np.max(np.abs(x.imag))
    if error > tol:
        raise ValueError(
            f"{name} requires real blocks; imaginary magnitude {error:.3g} exceeds {tol}"
        )
    return x.real


def deeph(model, destination, registry=None):
    basis_info = audit([model], registry, labels=True)[0]
    _write_deeph(model, destination, basis_info)


def _write_deeph(model, destination, basis_info):
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError(f"Choose a new or empty DeepH sample directory: {dest}")
    with h5py.File(model) as f, TemporaryDirectory(dir=dest.parent) as staging:
        tmp = Path(staging) / "sample"
        tmp.mkdir()
        atoms = structure(f)
        spinful = bool(f.attrs["spinor"] or f.attrs["spin_channels"] == 2)
        if spinful and not np.allclose(f.attrs["saxis"], [0, 0, 1]):
            raise ValueError("DeepH spin export requires VASP's default SAXIS")
        pairs, hs, ss = [], [], []
        for key, h, s in blocks(f):
            pairs.append(key)
            hs.append(h if spinful else real_matrix(h, 1e-5, "DeepH scalar H (eV)"))
            ss.append(real_matrix(s, 1e-7, "DeepH overlap"))
        for name, data in (("hamiltonian", hs), ("overlap", ss)):
            with h5py.File(tmp / f"{name}.h5", "w") as out:
                out["atom_pairs"] = pairs
                out["chunk_shapes"] = [x.shape for x in data]
                out["chunk_boundaries"] = np.r_[0, np.cumsum([x.size for x in data])]
                out["entries"] = np.concatenate([x.ravel() for x in data])
        info = dict(
            ao_sha256_by_element=basis_info["sha256_by_element"],
            atoms_quantity=len(atoms),
            orbits_quantity=int(orbital_bounds(f)[-1]),
            spinful=spinful,
            orthogonal_basis=False,
            fermi_energy_eV=f.attrs["fermi_eV"],
            elements_orbital_map=json.loads(f.attrs["shells"]),
            occupation=f.attrs["nelect"],
        )
        (tmp / "info.json").write_text(json.dumps(info, indent=2) + "\n")
        write(tmp / "POSCAR", atoms, format="vasp", direct=False, vasp5=True)
        tmp.rename(dest)


def batch_inputs(models, destination):
    if len(models) == 1 and Path(models[0]).suffix.lower() == ".json":
        from .batch import models_from_summary

        if paths_collide(destination, models):
            raise ValueError("Dataset output collides with batch summary")
        path = Path(models[0])
        fingerprint = file_sha256(path)
        return models_from_summary(path), (path, fingerprint)
    return models, None


def verify_summary_snapshot(summary, models):
    from .batch import models_from_summary

    path, fingerprint = summary
    if models_from_summary(path) != models or file_sha256(path) != fingerprint:
        raise ValueError("Batch changed during dataset export")


def deeph_batch(models, destination, registry=None):
    """Export a checked cohort as numbered DeepH samples, atomically."""
    dest = Path(destination)
    models, summary = batch_inputs(models, dest)
    reports = audit(models, registry, labels=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError("Choose a new or empty dataset directory")
    with TemporaryDirectory(dir=dest.parent) as staging:
        dataset = Path(staging) / "dataset"
        dataset.mkdir()
        for i, (model, info) in enumerate(zip(models, reports)):
            folder = dataset / str(i)
            _write_deeph(model, folder, info)
            if summary is not None:
                from .readback import deeph_blocks, verify_blocks

                with h5py.File(model) as native:
                    verify_blocks(
                        deeph_blocks(folder, native, info),
                        "deeph",
                        native,
                        native["band_validation/kpoints"][:]
                        if "band_validation/kpoints" in native
                        else native["k/points"][:8],
                    )
        if summary is not None:
            verify_summary_snapshot(summary, models)
        dataset.rename(dest)


# Source index for each OpenMX real harmonic when reading Wikipedia m=-l..l.
OPENMX_ORDER = {0: [0], 1: [2, 0, 1], 2: [2, 4, 0, 3, 1], 3: [3, 4, 2, 5, 1, 6, 0]}


def upstream_masks():
    try:
        from DFT_interfaces.openmx.utils import basis_def_14, basis_def_19, basis_def_26
    except ImportError as exc:
        raise ValueError(
            "HamGNN export needs the upstream orbital masks in the hamgnn-gpu Conda environment"
        ) from exc
    return {14: basis_def_14, 19: basis_def_19, 26: basis_def_26}


def mask_shells(nao, mask):
    dims = (
        [1] * 3 + [3] * 2 + [5] * (1 if nao == 14 else 2) + ([7] if nao == 26 else [])
    )
    start = 0
    shell_list = []
    active = set(mask)
    for dim in dims:
        block = set(range(start, start + dim))
        if block & active:
            if not block <= active:
                raise ValueError("Incomplete shell in upstream HamGNN mask")
            shell_list.append((dim - 1) // 2)
        start += dim
    return shell_list


def hamgnn_layout(shells, nao=None, *, extend=False):
    """Choose one upstream template across every species of a sample or cohort."""
    from ase.data import atomic_numbers

    templates = upstream_masks()
    matches = []
    for size in templates if nao is None else (nao,):
        layout = {}
        cost = 0
        for sym, source in shells.items():
            mask = templates[size].get(atomic_numbers[sym])
            if mask is None:
                break
            expected = mask_shells(size, mask)
            if expected != source:
                if not extend or any(
                    expected.count(l) < source.count(l) for l in set(source)
                ):
                    break
                cost += sum(2 * l + 1 for l in expected) - sum(
                    2 * l + 1 for l in source
                )
            layout[sym] = expected
        else:
            matches.append((cost, size, layout))
    if not matches:
        raise ValueError(
            f"No compatible HamGNN orbital mask for {shells} with nao_max={nao or 'auto'}. "
            "Use a compatible frozen registry, or generate one with v2h_genbasis and format='hamgnn'."
        )
    _, size, layout = min(matches, key=lambda item: item[:2])
    return size, layout


def ham_graph(f, nao):
    import torch
    from torch_geometric.data import Data
    from DFT_interfaces.openmx.utils import (
        basis_def_14,
        basis_def_19,
        basis_def_26,
        au2ang,
        au2ev,
    )

    spinor = bool(f.attrs["spinor"])
    if f.attrs["spin_channels"] != 1:
        raise ValueError(
            "Collinear magnetic labels require a spin-conditioned HamGNN configuration; use DeepH for ISPIN=2"
        )
    if spinor and not np.allclose(f.attrs["saxis"], [0, 0, 1]):
        raise ValueError("HamGNN spinor labels require default SAXIS")
    masks = {14: basis_def_14, 19: basis_def_19, 26: basis_def_26}[nao]
    atoms = structure(f)
    ls = json.loads(f.attrs["shells"])
    orders = []
    for atom in atoms:
        mask = masks.get(atom.number)
        if mask is None:
            raise ValueError(f"No upstream OpenMX {nao}-orbital mask for {atom.symbol}")
        expected = mask_shells(nao, mask)
        order = []
        for l in expected:
            order.extend(len(order) + np.array(OPENMX_ORDER[l]))
        if ls[atom.symbol] != expected:
            raise ValueError(
                f"HamGNN nao_max={nao} expects shells {expected} for {atom.symbol}"
            )
        orders.append(order)

    def padded(x, i, j, nspin=1):
        dtype = np.complex64 if np.iscomplexobj(x) else np.float32
        out = np.zeros((nao * nspin, nao * nspin), dtype=dtype)

        def indices(atom):
            mask = masks[atoms.numbers[atom]]
            order = np.array(orders[atom])
            return (
                np.concatenate([mask + s * nao for s in range(nspin)]),
                np.concatenate([order + s * len(order) for s in range(nspin)]),
            )

        mi, oi = indices(i)
        mj, oj = indices(j)
        out[np.ix_(mi, mj)] = x[np.ix_(oi, oj)]
        return out.ravel()

    onsite = {}
    edges, h_off, s_off = [], [], []
    for (a, b, c, i, j), h, s in blocks(f):
        if not spinor:
            h = real_matrix(h, 1e-5, "HamGNN H (eV)")
        hh = padded(h / au2ev, i, j, 2 if spinor else 1)
        ss = padded(real_matrix(s, 1e-7, "HamGNN overlap"), i, j)
        if i == j and (a, b, c) == (0, 0, 0):
            onsite[i] = hh, ss
        else:
            edges.append((i, j, a, b, c))
            h_off.append(hh)
            s_off.append(ss)
    lookup = {key: n for n, key in enumerate(edges)}
    inverse = [lookup[j, i, -a, -b, -c] for i, j, a, b, c in edges]
    edges = np.array(edges, dtype=int).reshape(-1, 5)
    cell = atoms.cell.array / au2ang
    shifts = edges[:, 2:]
    h = torch.from_numpy(np.stack([onsite[i][0] for i in range(len(atoms))] + h_off))
    s = torch.from_numpy(np.stack([onsite[i][1] for i in range(len(atoms))] + s_off))
    n = len(atoms)
    graph = Data(
        z=torch.LongTensor(atoms.numbers),
        pos=torch.FloatTensor(atoms.positions / au2ang),
        cell=torch.FloatTensor(cell[None]),
        node_counts=torch.LongTensor([n]),
        total_energy=float(f.attrs["total_energy_eV"]) / au2ev,
        doping_charge=torch.FloatTensor([f.attrs["doping_charge"]]),
        edge_index=torch.LongTensor(edges[:, :2].T.copy()),
        cell_shift=torch.LongTensor(shifts),
        nbr_shift=torch.FloatTensor(shifts @ cell),
        inv_edge_idx=torch.LongTensor(inverse),
        Hon=h.real[:n],
        Hoff=h.real[n:],
        overlap=s,
        Son=s[:n],
        Soff=s[n:],
        Hon0=torch.zeros_like(h.real[:n]),
        Hoff0=torch.zeros_like(h.real[n:]),
    )
    if spinor:
        graph.iHon, graph.iHoff = h.imag[:n], h.imag[n:]
        graph.iHon0, graph.iHoff0 = (
            torch.zeros_like(graph.iHon),
            torch.zeros_like(graph.iHoff),
        )
    else:
        graph.hamiltonian = h
    return graph


def hamgnn(models, output, nao=None, registry=None):
    models, summary = batch_inputs(models, output)
    if Path(output).is_symlink() or paths_collide(output, [*models, registry]):
        raise ValueError("HamGNN output collides with an input or symlink")
    reports = audit(models, registry, labels=True)
    shells = {}
    for path in models:
        with h5py.File(path) as f:
            shells.update(json.loads(f.attrs["shells"]))
    nao, _ = hamgnn_layout(shells, nao)
    graphs = {}
    for i, (path, info) in enumerate(zip(models, reports)):
        with h5py.File(path) as f:
            graphs[i] = ham_graph(f, nao)
            graphs[i].ao_sha256_by_element = json.dumps(
                info["sha256_by_element"], sort_keys=True
            )
    with output_file(output) as tmp:
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, graph=graphs)
        if summary is not None:
            from .readback import load_graphs, hamgnn_blocks, verify_blocks

            del graphs
            stored = load_graphs(tmp)
            if len(stored) != len(models):
                raise ValueError("Exported graph count differs from batch")
            for i, (path, info) in enumerate(zip(models, reports)):
                with h5py.File(path) as native:
                    exported = hamgnn_blocks(
                        tmp, native, info, nao, graph=stored.pop(i)
                    )
                    verify_blocks(
                        exported,
                        "hamgnn",
                        native,
                        native["band_validation/kpoints"][:]
                        if "band_validation/kpoints" in native
                        else native["k/points"][:8],
                    )
            verify_summary_snapshot(summary, models)
    return nao
