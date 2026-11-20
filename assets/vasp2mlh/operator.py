"""AO matrix elements of the scalar semilocal VASP PAW Hamiltonian."""

from pathlib import Path

import h5py
import numpy as np
from scipy import fft
from scipy.linalg import block_diag

from .model import adjoint
from .vasp import HSQDTM, metadata


def check_potential_records(f):
    for field in ("potential/total", "paw/dij", "paw/qij", "paw/lps", "paw/lmax",
                  "positions/lattice_vectors", "positions/scale", "positions/ion_types",
                  "positions/number_ion_types", "positions/position_ions", "positions/direct_coordinates"):
        if not isinstance(f.get("results/" + field), h5py.Dataset):
            raise ValueError(f"{f.filename} lacks results/{field}. Run VASP with WRT_POTENTIAL=total paw.")


def check_potential_source(source, other):
    atoms, ions = source["atoms"], other["atoms"]
    params = other["operator_parameters"]
    if params != source["operator_parameters"]:
        raise ValueError(
            "Potential and source calculations have different XC, PAW or FFT settings"
        )

    def value(name):
        return params.get(name, "")

    if (
        source["spinor"]
        or source["nspin"] != 1
        or other["spinor"]
        or other["nspin"] != 1
        or source["soc"]
        or other["soc"]
        or other["lreal"] not in ("F", ".FALSE.")
        or value("LHFCALC") == "T"
        or value("METAGGA") not in ("", "--", "F", "NONE")
        or value("LDAU") == "T"
    ):
        raise ValueError(
            "Direct operator currently supports scalar nonmagnetic semilocal LDA/GGA"
        )
    if (
        atoms.get_chemical_symbols() != ions.get_chemical_symbols()
        or not np.allclose(atoms.cell, ions.cell, atol=1e-7, rtol=0)
        or not np.allclose(
            atoms.get_scaled_positions(),
            ions.get_scaled_positions(),
            atol=1e-7,
            rtol=0,
        )
        or not np.isclose(other["encut"], source["encut"])
        or source["titles"] != other["titles"]
    ):
        raise ValueError(
            "Potential and source calculations have different structures, PAWs or cutoff"
        )


class ScalarOperator:
    """Read the potential and on-site strengths written by WRT_POTENTIAL=total paw."""

    def __init__(self, directory, source, potentials):
        directory = Path(directory)
        xml_path = directory / "vasprun.xml"
        other = metadata(xml_path)
        check_potential_source(source, other)
        atoms, ions = source["atoms"], other["atoms"]
        self.shape = tuple(
            int(other["operator_parameters"][n]) for n in ("NGX", "NGY", "NGZ")
        )
        with h5py.File(directory / "vaspout.h5") as f:
            check_potential_records(f)
            geo = f["results/positions"]
            cell = geo["lattice_vectors"][:] * geo["scale"][()]
            symbols = np.repeat(geo["ion_types"].asstr()[:], geo["number_ion_types"][:])
            positions = geo["position_ions"][:]
            if not geo["direct_coordinates"][()]:
                positions = (positions * geo["scale"][()]) @ np.linalg.inv(cell)
            delta = positions - ions.get_scaled_positions(wrap=False)
            if (
                list(symbols) != ions.get_chemical_symbols()
                or not np.allclose(cell, ions.cell, atol=1e-7, rtol=0)
                or not np.allclose(delta, np.rint(delta), atol=1e-7, rtol=0)
            ):
                raise ValueError("Potential HDF5 geometry disagrees with its XML")
            fine = f["results/potential/total"]
            if (
                fine.ndim != 4
                or fine.shape[0] != 1
                or any(n < m for n, m in zip(fine.shape[1:][::-1], self.shape))
            ):
                raise ValueError("Invalid scalar total potential grid")
            fine_k = fft.fftn(fine[0].T, norm="forward")
            idx = [np.rint(fft.fftfreq(n) * n).astype(int) for n in self.shape]
            self.potential = fft.ifftn(fine_k[np.ix_(*idx)], norm="forward").real
            paw = f["results/paw"]
            species = list(dict.fromkeys(ions.get_chemical_symbols()))
            for i, s in enumerate(species):
                ls = paw["lps"][i, : int(paw["lmax"][i])]
                if list(ls) != potentials[s].ls:
                    raise ValueError("POTCAR and PAW strength channels disagree")
            self.blocks = {}
            for key in ("dij", "qij"):
                raw = paw[key][:]
                if (
                    raw.ndim != 5
                    or raw.shape[:2] != (1, len(atoms))
                    or raw.shape[-1] != 2
                ):
                    raise ValueError("Unsupported VASP PAW strength record")
                entries = []
                for i, s in enumerate(atoms.get_chemical_symbols()):
                    size = sum(2 * l + 1 for l in potentials[s].ls)
                    v = raw[0, i, :size, :size]
                    if (
                        not np.isfinite(v).all()
                        or np.abs(v[..., 1]).max() > 1e-8
                        or not np.allclose(v[..., 0], v[..., 0].T, atol=1e-8)
                    ):
                        raise ValueError(
                            "Expected real symmetric scalar on-site strengths"
                        )
                    entries.append(v[..., 0])
                self.blocks[key] = block_diag(*entries)

    def apply(self, q, k, cell, p, vectors):
        vectors = np.asarray(vectors, dtype=complex)
        g = np.rint(q @ cell.T / (2 * np.pi) - k).astype(int)
        index = tuple((g % self.shape).T)
        if len(np.unique(np.ravel_multi_index(index, self.shape))) != len(q):
            raise ValueError("Wave basis aliases on the supplied VASP soft grid")
        kinetic = HSQDTM * np.sum(q * q, axis=1)
        hp = kinetic[:, None] * vectors + p @ (
            self.blocks["dij"] @ (adjoint(p) @ vectors)
        )
        # Transform only a batch of orbitals at a time. FFT conventions cancel in V psi.
        batch = max(1, min(32, 2**21 // int(np.prod(self.shape))))
        for start in range(0, vectors.shape[1], batch):
            sl = slice(start, start + batch)
            work = np.zeros((*self.shape, vectors[:, sl].shape[1]), complex)
            work[index] = vectors[:, sl]
            work = fft.ifftn(work, axes=(0, 1, 2), overwrite_x=True)
            work *= self.potential[..., None]
            hp[:, sl] += fft.fftn(work, axes=(0, 1, 2), overwrite_x=True)[index]
        return hp
