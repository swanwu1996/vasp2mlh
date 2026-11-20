"""Single-rank PAOFLOW 3.0.0 benchmark with the exact radial tables of a scalar model."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from PAOFLOW.PAOFLOW import PAOFLOW
from PAOFLOW.hamiltonian.do_build_pao_hamiltonian import build_Hks
from PAOFLOW.projection.do_atwfc_proj import calc_atwfc_k, read_VASP_wfc
from mpi4py import MPI
from vasp2mlh.export import structure

from vasp2mlh.pipeline import bands
from vasp2mlh.vasp import BOHR


def compare(run, model, references=()):
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("Run the comparison in one MPI rank")
    with TemporaryDirectory() as work, h5py.File(model) as f:
        if f.attrs["spinor"] or f.attrs["spin_channels"] != 1:
            raise ValueError("This PAOFLOW comparison uses scalar wavefunctions")
        configuration = {}
        for symbol, shells in json.loads(f.attrs["shells"]).items():
            folder = Path(work) / symbol
            folder.mkdir()
            configuration[symbol] = []
            for i, l in enumerate(shells):
                label = f"{shells[:i].count(l) + l + 1}{'SPDF'[l]}"
                shell = f[f"basis/{symbol}/{i}"]
                radius = shell["r"][:] / BOHR
                radial_u = radius * shell["R"][:] * BOHR**1.5
                np.savetxt(folder / f"{label}.dat", np.column_stack((radius, radial_u)))
                configuration[symbol].append(label)
        pf = PAOFLOW(workpath=work, savedir=str(Path(run).resolve()), dft="VASP")
        pf.projections(configuration=configuration, basispath=work)
        pf.projectability(pthr=0.95)
        dc = pf.data_controller
        arr, attr = dc.data_dicts()
        # build_Hks preserves the input k ordering before the optional wedge expansion.
        attr["shift_type"] = 1
        paw_weights = f["k/projectability"][0]
        reference_weights = np.sum(
            np.abs(dc.local_projections()[:, :, :, 0]) ** 2, axis=-1
        )
        orthogonal_h = build_Hks(dc)[:, :, :, 0].transpose(2, 0, 1)
        pao_bands = np.linalg.eigvalsh(orthogonal_h) + attr["Efermi"]
        # PAOFLOW uses 0,+1,-1,..., signed cubic harmonics and i**l.
        # Rotate its AO rows into our -l..l ordering and Fourier convention.
        order, signs = [], []
        shell_map = json.loads(f.attrs["shells"])
        for atom in structure(f):
            for l in shell_map[atom.symbol]:
                offset = len(order)
                for m in range(-l, l + 1):
                    index = 2 * abs(m) - (m > 0)
                    order.append(offset + index)
                    signs.append((-1) ** (l + abs(m)))
        signs = np.array(signs)
        overlap_diff, h_diff = [], []
        for ik, orth in enumerate(orthogonal_h):
            gk, _ = read_VASP_wfc(dc, ik, 0)
            orbitals = calc_atwfc_k(arr["basis"], gk, False)[order] * signs[:, None]
            soft_s = orbitals.conj() @ orbitals.T
            val, vec = np.linalg.eigh(soft_s)
            sqrt_s = (vec * np.sqrt(val)) @ vec.conj().T
            orth = orth[np.ix_(order, order)] * np.outer(signs, signs)
            h = sqrt_s @ orth @ sqrt_s + attr["Efermi"] * soft_s
            overlap_diff.append(soft_s - f["k/S"][ik])
            h_diff.append(h - f["k/H"][0, ik])
        own = bands(f, f["k/points"][:])[0]
        occ = f["k/occupation"][0]
        ref = f["k/eigenvalues"][0]
        if not np.allclose(
            ref, arr["my_eigsmat"][:, :, 0].T + attr["Efermi"], atol=1e-4
        ):
            raise ValueError("PAOFLOW reference energies/k ordering differ from model")
        n = min(pao_bands.shape[1], ref.shape[1])

        def rms(diff):
            return float(np.sqrt(np.average(diff**2, weights=occ[:, :n])))

        holdout = {}
        if references:
            from fidelity.external import score
            from vasp2mlh.realspace import orbital_bounds

            counts = np.diff(orbital_bounds(f))
            identity = np.broadcast_to(np.eye(sum(counts)), orthogonal_h.shape)
            absolute = orthogonal_h + attr["Efermi"] * identity
            for path in references:
                holdout[str(path)] = score(
                    f, absolute, identity, counts, path, int(attr["bnd"])
                )
        return dict(
            holdout=holdout,
            nonorthogonal_H_element_rmse_eV=float(
                np.sqrt(np.mean(np.abs(h_diff) ** 2))
            ),
            overlap_element_rmse=float(np.sqrt(np.mean(np.abs(overlap_diff) ** 2))),
            paoflow_pthr=0.95,
            paoflow_retained_bands=int(attr["bnd"]),
            paoflow_occupied_spilling=float(
                np.average(1 - reference_weights, weights=occ)
            ),
            occupied_projectability_mean_abs_difference=float(
                np.average(abs(reference_weights - paw_weights), weights=occ)
            ),
            paoflow_occupied_band_rmse_vs_vasp_eV=rms(pao_bands[:, :n] - ref[:, :n]),
            occupied_band_rmse_between_methods_eV=rms(own[:, :n] - pao_bands[:, :n]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("model")
    parser.add_argument("--reference", nargs="+", default=[])
    args = parser.parse_args()
    print(json.dumps(compare(args.run, args.model, args.reference), indent=2))
