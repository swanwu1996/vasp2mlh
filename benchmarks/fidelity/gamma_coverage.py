"""Measure spin-resolved edge coverage at a chosen k point in full-complex WAVECARs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigvalsh
from scipy.linalg.blas import dsyrk, zherk

from vasp2mlh.basis import read_potcar
from vasp2mlh.model import adjoint, ao_and_metric
from vasp2mlh.registry import read_registry
from vasp2mlh.vasp import Wavecar, metadata, plane_waves


def real_rows(columns, mask, multiplicity):
    weighted = columns[mask] * multiplicity
    return np.asfortranarray(np.concatenate((weighted.real, weighted.imag)))


def audit(run, registry, batch=1536, k_index=0):
    run = Path(run)
    meta = metadata(run / "vasprun.xml")
    wave = Wavecar(run / "WAVECAR")
    if (
        meta["spinor"]
        or meta["lreal"] not in ("F", ".FALSE.")
        or not 0 <= k_index < wave.nk
        or meta["eig"].shape != (wave.ns, wave.nk, wave.nb)
        or not np.allclose(wave.cell, meta["atoms"].cell, atol=1e-7, rtol=0)
        or not np.isclose(wave.encut, meta["encut"])
        or batch < 1
    ):
        raise ValueError(
            "Expected a reciprocal-projector collinear run with a valid k index"
        )
    atoms = meta["atoms"]
    pots = read_potcar(run / "POTCAR")
    species = atoms.get_chemical_symbols()
    if any(meta["titles"][s].split() != pots[s].title.split() for s in set(species)):
        raise ValueError("PAW title mismatch")
    basis = read_registry(registry, species, pots)
    point = meta["kpoints"][k_index]
    states = []
    headers = []
    for spin in range(wave.ns):
        k, b, c = wave.read(spin, k_index)
        if spin == 0:
            q = plane_waves(k, wave.cell, wave.encut)
            gamma = np.allclose(k, 0, atol=1e-12, rtol=0)
        if (
            not np.allclose(k, point, atol=1e-8, rtol=0)
            or not np.allclose(b[:, 0], meta["eig"][spin, k_index], atol=1e-4, rtol=0)
            or len(c) != len(q)
        ):
            raise ValueError("Mismatched WAVECAR/XML or half-grid Gamma storage")
        n = int(np.count_nonzero(b[:, 1] >= 0.5 - 1e-6))
        if not 2 <= n <= wave.nb - 2 or not np.all(b[:n, 1] >= 0.5 - 1e-6):
            raise ValueError("Cannot define a contiguous spin-channel edge")
        sl = slice(n - 2, n + 2)
        states.append(c[:, sl].copy())
        del c
        headers.append(
            {
                "spin": spin + 1,
                "half_occupied": bool(np.any(abs(b[:, 1] - 0.5) <= 1e-6)),
                "bands": [n - 1, n + 2],
                "occupations": b[sl, 1].tolist(),
                "energies_eV": b[sl, 0].tolist(),
            }
        )
    c = np.column_stack(states)
    reduced = []
    width = 0
    for symbol in species:
        seen = set()
        for shell in basis[symbol]:
            count = 2 * shell.l + 1
            if shell.l not in seen:
                reduced.extend(range(width, width + count))
                seen.add(shell.l)
            width += count
    # At Gamma real AO/projector columns obey X(-G)=X(G)*. The PW
    # coefficient columns can be complex and are contracted on the full grid.
    g = np.rint(q @ wave.cell.T / (2 * np.pi)).astype(int)
    canonical = (g[:, 0] > 0) | (
        (g[:, 0] == 0) & ((g[:, 1] > 0) | ((g[:, 1] == 0) & (g[:, 2] >= 0)))
    )
    if gamma and len(q) != 2 * np.count_nonzero(canonical) - 1:
        raise ValueError("Gamma stars are incomplete")
    gram = pa = ac = pc = None
    for start in range(0, len(q), batch):
        section = slice(start, start + batch)
        phi, p, d = ao_and_metric(q[section], atoms, basis, pots, meta["spline"])
        if gram is None:
            dtype = float if gamma else complex
            gram = np.zeros((width, width), dtype=dtype, order="F")
            pa = np.zeros((p.shape[1], width), dtype=dtype)
            ac = np.zeros((width, c.shape[1]), complex)
            pc = np.zeros((p.shape[1], c.shape[1]), complex)
        if gamma:
            mask = canonical[section]
            mult = np.where(np.any(g[section][mask], axis=1), np.sqrt(2.0), 1.0)[
                :, None
            ]
            x, y = real_rows(phi, mask, mult), real_rows(p, mask, mult)
            if len(x):
                gram = dsyrk(1, x, trans=1, beta=1, c=gram, overwrite_c=1)
                pa += y.T @ x
        else:
            gram = zherk(
                1, np.asfortranarray(phi), trans=2, beta=1, c=gram, overwrite_c=1
            )
            pa += adjoint(p) @ phi
        ac += adjoint(phi) @ c[section]
        pc += adjoint(p) @ c[section]
    s = np.triu(gram) + adjoint(np.triu(gram, 1)) + adjoint(pa) @ (d @ pa)
    v = ac + adjoint(pa) @ (d @ pc)
    for j, row in enumerate(headers):
        sl = slice(j * 4, j * 4 + 4)
        native = adjoint(c[:, sl]) @ c[:, sl] + adjoint(pc[:, sl]) @ d @ pc[:, sl]
        row["native_gram_max_error"] = float(abs(native - np.eye(4)).max())
        if row["native_gram_max_error"] > 2e-3:
            raise ValueError("Native PAW metric does not reproduce normalization")
        for name, indices in [("single_zeta", reduced), ("full", np.arange(width))]:
            m = s[np.ix_(indices, indices)]
            a = v[indices, sl]
            projected = adjoint(a) @ cho_solve(cho_factor(m), a)
            row[name] = {
                "dimension": len(indices),
                "weights": np.diag(projected).real.tolist(),
                "min_weight": float(eigvalsh(projected)[0]),
            }
    return {
        "formula": atoms.get_chemical_formula(),
        "atoms": len(atoms),
        "k": point.tolist(),
        "channels": headers,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("registry")
    parser.add_argument(
        "--k-index", type=int, default=0, help="zero-based WAVECAR k-point index"
    )
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))
