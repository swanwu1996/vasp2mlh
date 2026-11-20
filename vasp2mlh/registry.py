"""Portable, element-resolved AO definitions and dataset dimension checks."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .basis import Shell, load_basis, read_potcar
from .pipeline import output_file
from .realspace import SPARSE_FORMAT, structure

FORMAT = "vasp2mlh/basis-registry/1"
ANGULAR = "real tesseral m=-l..l, Wikipedia"


def fingerprint(shells):
    """Hash normalized radial functions in their actual shell order and canonical units."""
    digest = hashlib.sha256((FORMAT + "/angstrom/" + ANGULAR).encode())
    for shell in shells:
        digest.update(np.asarray([shell.l, len(shell.radius)], dtype="<i8").tobytes())
        for values in (shell.radius, shell.radial):
            digest.update(np.asarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def freeze(config, output, potcar=None):
    if Path(output).resolve() in {Path(x).resolve() for x in (config, potcar) if x}:
        raise ValueError("Choose a separate registry output")
    if h5py.is_hdf5(config):
        raise ValueError("This basis is already frozen")
    potentials = read_potcar(potcar) if potcar else {}
    specs = json.loads(Path(config).read_text())
    hashes = {
        sym: potentials[sym].sha256
        for sym, items in specs.items()
        if sym in potentials and any(item.get("source") == "potcar" for item in items)
    }
    basis = load_basis(config, potentials)
    write_registry(basis, output, "imported", hashes)


def write_registry(basis, output, method, paw_hashes=None, layout=None):
    with output_file(output) as temp, h5py.File(temp, "w") as out:
        out.attrs["method"] = method
        if layout is not None:
            out.attrs["layout"] = json.dumps(layout, sort_keys=True)
        out.attrs.update(
            format=FORMAT, angular_convention=ANGULAR, length_unit="angstrom"
        )
        for element, shells in basis.items():
            elem = out.create_group("basis/" + element)
            elem.attrs["sha256"] = fingerprint(shells)
            if paw_hashes and element in paw_hashes:
                elem.attrs["paw_reference_sha256"] = paw_hashes[element]
            for i, shell in enumerate(shells):
                group = elem.create_group(str(i))
                group.attrs.update(l=shell.l, label=shell.label)
                group["r"], group["R"] = shell.radius, shell.radial


def read_shells(group):
    shells = []
    for i in range(len(group)):
        entry = group[str(i)]
        shells.append(
            Shell(
                entry.attrs["l"],
                entry["r"][:],
                entry["R"][:],
                entry.attrs.get("label", entry.name),
                normalized=True,
            )
        )
    if not shells:
        raise ValueError(f"Empty basis at {group.name}")
    return shells


def read_registry(path, elements=None, potentials=None):
    with h5py.File(path) as f:
        if (
            f.attrs.get("format") != FORMAT
            or f.attrs.get("angular_convention") != ANGULAR
            or f.attrs.get("length_unit") != "angstrom"
        ):
            raise ValueError("Unsupported registry conventions")
        basis = {}
        for sym in sorted(set(f["basis"] if elements is None else elements)):
            if sym not in f["basis"]:
                raise ValueError(f"Registry has no definition for {sym}")
            group = f["basis"][sym]
            if (
                potentials
                and "paw_reference_sha256" in group.attrs
                and (
                    sym not in potentials
                    or potentials[sym].sha256 != group.attrs["paw_reference_sha256"]
                )
            ):
                raise ValueError(f"PAW reference differs from the registry for {sym}")
            shells = read_shells(group)
            if fingerprint(shells) != group.attrs["sha256"]:
                raise ValueError(f"Registry checksum mismatch for {sym}")
            basis[sym] = shells
        return basis


def inspect_basis(f, expected):
    atoms = structure(f)
    shells = json.loads(f.attrs["shells"])
    if set(shells) != set(atoms.get_chemical_symbols()) or set(shells) != set(
        f["basis"]
    ):
        raise ValueError("Basis and structure species disagree")
    if (
        f.attrs.get("angular_convention", ANGULAR) != ANGULAR
        or f.attrs.get("length_unit", "angstrom") != "angstrom"
        or f.attrs.get("spin_order", "spin-major") != "spin-major"
    ):
        raise ValueError("Unsupported basis conventions")
    dims = {}
    identities = {}
    for sym, angular in shells.items():
        actual = read_shells(f["basis"][sym])
        if [sh.l for sh in actual] != angular:
            raise ValueError(
                f"Orbital ordering disagrees with shell metadata for {sym}"
            )
        sig = fingerprint(actual)
        if expected is not None and expected.get(sym) != sig:
            raise ValueError(
                f"AO definition differs from the frozen basis for {sym} in {f.filename}"
            )
        dims[sym] = sum(2 * l + 1 for l in angular)
        identities[sym] = sig
    counts = np.array([dims[s] for s in atoms.get_chemical_symbols()])
    dim = int(counts.sum())
    spin = int(f.attrs.get("spin_channels", 1))
    factor = 2 if f.attrs.get("spinor", False) else 1
    kind = f.attrs["format"]
    if kind == "vasp2mlh/1":
        for group, axis in (("k", "points"), ("real", "R")):
            size = len(f[f"{group}/{axis}"])
            if f[f"{group}/H"].shape != (spin, size, dim * factor, dim * factor) or f[
                f"{group}/S"
            ].shape != (size, dim, dim):
                raise ValueError(
                    f"Matrix dimensions disagree with AO shells in {group}"
                )
    elif kind in (SPARSE_FORMAT, "vasp2mlh/overlap/1"):
        pairs = f["pairs/indices"][:]
        offset = f["pairs/offsets"][:]
        if (
            pairs.ndim != 2
            or pairs.shape[1] != 5
            or pairs.dtype.kind not in "iu"
            or offset.dtype.kind not in "iu"
            or np.any(pairs[:, 3:] < 0)
            or np.any(pairs[:, 3:] >= len(atoms))
        ):
            raise ValueError("Invalid atom-pair indices or offsets")
        sizes = counts[pairs[:, 3]] * counts[pairs[:, 4]]
        if (
            offset.shape != (len(pairs) + 1,)
            or offset[0] != 0
            or not np.array_equal(np.diff(offset), sizes)
            or f["pairs/S"].shape != (offset[-1],)
        ):
            raise ValueError("Pair dimensions disagree with AO shells")
        if kind == SPARSE_FORMAT and f["pairs/H"].shape != (
            spin,
            offset[-1] * factor**2,
        ):
            raise ValueError("Hamiltonian dimensions disagree with AO shells")
    else:
        raise ValueError(f"Unknown AO archive format {kind}")
    result = {
        "atoms": len(atoms),
        "spatial_dimension": dim,
        "orbitals_by_element": dims,
        "sha256_by_element": identities,
    }
    if kind != "vasp2mlh/overlap/1":
        result.update(hamiltonian_dimension=factor * dim, spin_channels=spin)
    return result


def audit(models, registry=None, labels=False):
    expected = (
        None
        if registry is None
        else {s: fingerprint(v) for s, v in read_registry(registry).items()}
    )
    paw_required = {}
    if registry is not None:
        with h5py.File(registry) as lock:
            paw_required = {
                s: g.attrs["paw_reference_sha256"]
                for s, g in lock["basis"].items()
                if "paw_reference_sha256" in g.attrs
            }
    known = {}
    known_paw = {}
    known_xc = set()
    reports = []
    modes = set()
    for path in models:
        with h5py.File(path) as f:
            info = inspect_basis(f, expected)
            for sym, sig in info["sha256_by_element"].items():
                if known.setdefault(sym, sig) != sig:
                    raise ValueError(f"Inconsistent radial basis for {sym} in {path}")
            paw = json.loads(f.attrs.get("paw_sha256_by_element", "{}"))
            if paw and set(paw) != set(info["orbitals_by_element"]):
                raise ValueError("Incomplete element-wise PAW provenance")
            for sym, sig in paw.items():
                if sym in paw_required and paw_required[sym] != sig:
                    raise ValueError(f"Source PAW differs from the registry for {sym}")
                if labels and known_paw.setdefault(sym, sig) != sig:
                    raise ValueError(f"Inconsistent source PAW for {sym} in {path}")
            info["source_paw_recorded"] = bool(paw)
            if labels:
                if "xc_settings" in f.attrs:
                    known_xc.add(
                        json.dumps(json.loads(f.attrs["xc_settings"]), sort_keys=True)
                    )
                    if len(known_xc) > 1:
                        raise ValueError(
                            "Dataset has different exchange-correlation settings"
                        )
                if "hamiltonian_dimension" not in info:
                    raise ValueError("Expected Hamiltonian labels")
                modes.add(
                    (
                        bool(f.attrs["spinor"]),
                        bool(f.attrs["soc"]),
                        int(f.attrs["spin_channels"]),
                        tuple(f.attrs["saxis"]) if f.attrs["spinor"] else (),
                        f.attrs["metric"],
                        f.attrs.get("hamiltonian_method", "spectral"),
                        "subspace_bands" in f.attrs,
                        f.attrs.get("alignment_width_eV"),
                        f.attrs.get("degeneracy_eV"),
                        f.attrs.get("complement_eV"),
                    )
                )
                if len(modes) > 1:
                    raise ValueError(
                        "Dataset has incompatible spin, metric or Hamiltonian definitions"
                    )
            reports.append(dict(path=str(path), **info))
    return reports
