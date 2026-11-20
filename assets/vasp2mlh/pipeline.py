"""Construct and persist the independent VASP2MLH HDF5 representation."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
import hashlib
import json
import os

import h5py
import numpy as np
from scipy.linalg import block_diag, eigvalsh

from .configuration import load_config
from .realspace import matrices_k, SPARSE_FORMAT
from .basis import load_basis, read_potcar
from .model import (
    RadialTable,
    ao_and_metric,
    project,
    mesh_indices,
    localized_transform,
    bloch,
)
from .vasp import Wavecar, plane_waves, metadata, xc_parameters


def paths_collide(output, inputs):
    target = Path(output)
    for entry in inputs:
        if entry is None:
            continue
        source = Path(entry)
        if target.resolve() == source.resolve() or (
            target.exists() and source.exists() and target.samefile(source)
        ):
            return True
    return False


@contextmanager
def output_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projection_header(run, meta, mesh):
    """Validate the WAVECAR header and XML mesh without reading coefficients."""
    atoms = meta["atoms"]
    w = Wavecar(run / "WAVECAR")
    xml_e = meta["eig"]
    if (
        xml_e.shape != (w.ns, len(meta["kpoints"]), w.nb)
        or w.nk != meta.get("wave_nk", len(meta["kpoints"]))
        or not np.allclose(w.cell, atoms.cell, atol=1e-7)
        or not np.isclose(w.encut, meta["encut"])
        or meta["lreal"] not in ("F", ".FALSE.")
    ):
        raise ValueError(
            "WAVECAR/xml mismatch or LREAL enabled; use reciprocal PAW projectors"
        )
    if meta["spinor"] and w.ns != 1:
        raise ValueError("Unexpected spinor record layout")
    k = meta["kpoints"]
    if not np.allclose(meta["weights"], 1 / len(k), atol=1e-7, rtol=0):
        raise ValueError("Fourier transform requires a uniformly weighted full mesh. Run VASP with ISYM=-1.")
    mesh = mesh if mesh is not None else meta["mesh"]
    if mesh is None:
        raise ValueError("Pass --mesh Nx Ny Nz for an explicit full k list")
    mesh_indices(k, mesh)
    return w, mesh


def build(
    run,
    config,
    output,
    mesh=None,
    complement=0.0,
    potential_run=None,
    *,
    on_kpoint=None,
):
    if not np.isfinite(complement):
        raise ValueError("Complement energy must be finite")
    run = Path(run)
    meta = metadata(run / "vasprun.xml")
    w, mesh = projection_header(run, meta, mesh)
    atoms = meta["atoms"]
    xml_e = meta["eig"]
    k = meta["kpoints"]
    points = np.empty_like(k)
    wave_indices = meta.get("wave_indices", range(len(k)))
    potentials = read_potcar(run / "POTCAR")
    symbols = atoms.get_chemical_symbols()
    basis = load_basis(config, potentials, symbols)
    if set(symbols) != set(basis) or set(symbols) != set(potentials):
        raise ValueError("POTCAR, basis and XML species must agree")
    if any(
        meta["titles"][s].split() != potentials[s].title.split() for s in set(symbols)
    ):
        raise ValueError("POTCAR title differs from vasprun.xml")
    operator = None
    if potential_run is not None:
        from .operator import ScalarOperator

        if complement != 0:
            raise ValueError(
                "A complement energy applies only to finite-band projection"
            )
        if not (run / "CHGCAR").stat().st_size:
            raise ValueError("Direct projection requires the reference charge density")
        for name in ("POTCAR", "CHGCAR"):
            if file_sha256(run / name) != file_sha256(Path(potential_run) / name):
                raise ValueError(
                    f"Direct operator input {name} differs from the source run"
                )
        operator = ScalarOperator(potential_run, meta, potentials)
        rayleigh_error = []
        residual_norm = []
    ls = {s: [sh.l for sh in shells] for s, shells in basis.items()}
    norb = sum(2 * l + 1 for s in symbols for l in ls[s])
    dim = norb * (2 if meta["spinor"] else 1)
    hs = np.empty((w.ns, len(k), dim, dim), complex)
    radial_table = RadialTable(basis)
    ss = np.empty((len(k), norb, norb), complex)
    spectral_shape = (w.ns, len(k), w.nb)
    energies, occupations, weights = (np.empty(spectral_shape) for _ in range(3))
    gram = np.empty((w.ns, len(k)))
    with output_file(output) as tmp, h5py.File(tmp, "w") as f:
        proj = f.create_dataset(
            "k/projection",
            (w.ns, len(k), dim, w.nb),
            dtype="c16",
            chunks=(1, 1, dim, w.nb),
        )
        for ik, expected in enumerate(k):
            for spin in range(w.ns):
                point, bands, coeff = w.read(spin, wave_indices[ik])
                if not np.allclose(point, expected, atol=1e-7) or not np.allclose(
                    bands[:, 0], xml_e[spin, ik], atol=1e-4, rtol=0
                ):
                    raise ValueError(
                        f"Inconsistent WAVECAR/XML k point or eigenvalues at {ik}"
                    )
                if spin == 0:
                    points[ik] = point
                    q = plane_waves(point, w.cell, w.encut)
                    ao, p, d = ao_and_metric(
                        q, atoms, basis, potentials, meta["spline"], radial_table
                    )
                if len(coeff) != len(q) * (2 if meta["spinor"] else 1):
                    raise ValueError(
                        f"Plane-wave count mismatch at k={ik}; use vasp_std or vasp_ncl, full complex storage"
                    )
                energies[spin, ik], occupations[spin, ik] = bands.T
                (
                    hs[spin, ik],
                    ss[ik],
                    proj[spin, ik],
                    weights[spin, ik],
                    gram[spin, ik],
                ) = project(ao, p, d, coeff, bands[:, 0], meta["spinor"], complement)
                if operator is not None:
                    if not np.allclose(d, operator.blocks["qij"], atol=2e-5, rtol=0):
                        raise ValueError(
                            "POTCAR and exported PAW augmentation matrices disagree"
                        )
                    mat = ao.conj().T @ operator.apply(q, point, w.cell, p, ao)
                    hs[spin, ik] = (mat + mat.conj().T) / 2
                    nocc = int(np.ceil(meta["nelect"] / 2))
                    edge = slice(max(0, nocc - 2), min(w.nb, nocc + 2))
                    ce = coeff[:, edge]
                    hce = operator.apply(q, point, w.cell, p, ce)
                    sce = ce + p @ (d @ (p.conj().T @ ce))
                    residual = hce - sce * bands[edge, 0]
                    residual_norm.append(float(np.linalg.norm(residual, axis=0).max()))
                    rayleigh_error.append(
                        float(np.abs((ce.conj() * residual).sum(axis=0)).max())
                    )
            if on_kpoint is not None:
                on_kpoint(ik + 1, len(k))
        if operator is not None:
            f["k/operator_edge_rayleigh_error_eV"] = rayleigh_error
            f["k/operator_edge_residual_norm_eV"] = residual_norm
            f.attrs["hamiltonian_method"] = "direct_scalar_paw"
            f.attrs["potential_sha256"] = file_sha256(
                Path(potential_run) / "vaspout.h5"
            )
        counts = [sum(2 * l + 1 for l in ls[s]) for s in symbols]
        r, hr, sr = localized_transform(points, mesh, atoms, counts, hs, ss)
        for name, value in {
            "structure/cell": w.cell,
            "structure/projection_cell": atoms.cell.array,
            "structure/fractional": atoms.get_scaled_positions(wrap=False),
            "structure/Z": atoms.numbers,
            "k/points": points,
            "k/mesh": mesh,
            "k/weights": meta["weights"],
            "k/H": hs,
            "k/S": ss,
            "k/eigenvalues": energies,
            "k/occupation": occupations,
            "k/projectability": weights,
            "k/gram_error": gram,
            "real/R": r,
            "real/H": hr,
            "real/S": sr,
        }.items():
            f[name] = value
        f.attrs.update(
            format="vasp2mlh/1",
            length_unit="angstrom",
            energy_unit="eV",
            angular_convention="real tesseral m=-l..l, Wikipedia",
            spin_order="spin-major",
            fourier="M(k)=sum_R exp(+2pi i k.R) M(R)",
            metric="PAW",
            spinor=meta["spinor"],
            soc=meta["soc"],
            saxis=meta["saxis"],
            spin_channels=w.ns,
            nelect=meta["nelect"],
            doping_charge=meta["doping_charge"],
            total_energy_eV=meta["energy"],
            fermi_eV=meta["efermi"],
            encut_eV=w.encut,
            complement_eV=complement,
            shells=json.dumps(ls),
            basis_config=(
                json.dumps({"registry_sha256": file_sha256(config)})
                if h5py.is_hdf5(config)
                else json.dumps(load_config(config))
            ),
            potcar_sha256=file_sha256(run / "POTCAR"),
            paw_sha256_by_element=json.dumps(
                {s: potentials[s].sha256 for s in set(symbols)}, sort_keys=True
            ),
            xc_settings=json.dumps(
                xc_parameters(meta["operator_parameters"]),
                sort_keys=True,
            ),
        )
        if (run / "CHGCAR").is_file() and (run / "CHGCAR").stat().st_size:
            f.attrs["charge_density_sha256"] = file_sha256(run / "CHGCAR")
        # A compact radial definition independent of external installation paths.
        for symbol, shells in basis.items():
            for i, sh in enumerate(shells):
                group = f.create_group(f"basis/{symbol}/{i}")
                group.attrs.update(l=sh.l, label=sh.label)
                group["r"], group["R"] = sh.radius, sh.radial


def bands(f, k):
    k = np.atleast_2d(k)
    hk, sk = matrices_k(f, k)
    results = []
    for point, h, s in zip(k, hk.swapaxes(0, 1), sk):
        if f.attrs["spinor"]:
            s = block_diag(s, s)
        if np.min(eigvalsh(s)) <= 0:
            raise ValueError(f"Interpolated overlap not positive definite at {point}")
        results.append([eigvalsh(mat, s) for mat in h])
    return np.swapaxes(results, 0, 1)


def validate(path):
    with h5py.File(path) as f:
        k = f["k/points"][:]
        result = {}
        if f.attrs["format"] == SPARSE_FORMAT:
            result["truncation"] = json.loads(f.attrs["truncation_report"])
            sk = matrices_k(f, k)[1]
        else:
            # A complete PAW spectrum resolves the identity in the AO metric.
            # This checks the reverse projection (coverage of the AO space),
            # complementary to the per-band projectability of VASP states.
            closure = []
            for ik, spatial_s in enumerate(f["k/S"]):
                metric = (
                    block_diag(spatial_s, spatial_s) if f.attrs["spinor"] else spatial_s
                )
                for spin in range(f.attrs["spin_channels"]):
                    a = f["k/projection"][spin, ik]
                    closure.append(
                        np.linalg.norm(metric - a @ a.conj().T) / np.linalg.norm(metric)
                    )
            result["ao_closure_relative_frobenius_max"] = float(max(closure))
            r = f["real/R"][:]
            sk = f["k/S"][:]
            for name in ("H", "S"):
                mr = f[f"real/{name}"][:]
                result[f"{name}_roundtrip_max"] = float(
                    np.max(np.abs(bloch(k, r, mr) - f[f"k/{name}"][:]))
                )
                inv = {tuple(x): i for i, x in enumerate(r)}
                reverse = np.take(mr, [inv[tuple(-x)] for x in r], axis=-3)
                result[f"{name}_hermiticity_max"] = float(
                    np.max(np.abs(mr - reverse.conj().swapaxes(-1, -2)))
                )
        pred = bands(f, k)
        n = min(pred.shape[-1], f["k/eigenvalues"].shape[-1])
        err = pred[..., :n] - f["k/eigenvalues"][:, :, :n]
        # Methfessel-Paxton occupations can slightly exceed the probability
        # interval. Positive weights are required for an RMS diagnostic.
        raw_occupation = f["k/occupation"][:]
        occ = np.clip(raw_occupation, 0, 1)
        result["clipped_diagnostic_occupations"] = int(
            np.count_nonzero(occ != raw_occupation)
        )
        weight = f["k/projectability"][:]
        included = occ[..., :n].sum()
        if included <= 0:
            raise ValueError("No occupied reference states in the compared spectrum")
        result.update(
            compared_lowest_bands=n,
            ao_dimension=int(pred.shape[-1]),
            vasp_bands=int(occ.shape[-1]),
            occupation_fraction_compared=float(included / occ.sum()),
            band_rmse_eV=float(np.sqrt(np.mean(err**2))),
            min_overlap_eigenvalue=float(np.linalg.eigvalsh(sk).min()),
            paw_gram_max=float(np.max(f["k/gram_error"][:])),
            occupied_spilling=float(np.sum(occ * (1 - weight)) / occ.sum()),
            occupied_band_rmse_eV=float(
                np.sqrt(np.average(err**2, weights=occ[..., :n]))
            ),
        )
    return result
