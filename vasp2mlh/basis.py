"""Numerical radial shells and reciprocal-space PAW data, in angstrom units."""

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.special import sph_harm_y, spherical_jn

from .vasp import BOHR


def harmonics(l, q):
    """Wikipedia real harmonics, m = -l,...,l (p_y,p_z,p_x)."""
    norm = np.linalg.norm(q, axis=1)
    theta = np.arccos(
        np.clip(np.divide(q[:, 2], norm, out=np.ones_like(norm), where=norm > 0), -1, 1)
    )
    phi = np.arctan2(q[:, 1], q[:, 0])
    y = sph_harm_y(l, np.arange(l + 1), theta[:, None], phi[:, None])
    pos = y[:, 1:] * (np.sqrt(2) * (-1.0) ** np.arange(1, l + 1))
    return np.column_stack((pos.imag[:, ::-1], y[:, 0].real, pos.real))


@dataclass
class Shell:
    l: int
    radius: np.ndarray
    radial: np.ndarray
    label: str

    def __post_init__(self):
        if (
            self.l < 0
            or self.radius.size < 3
            or self.radius[0] < 0
            or not np.all(np.diff(self.radius) > 0)
            or not np.isfinite(self.radial).all()
        ):
            raise ValueError(f"Invalid radial mesh or shell: {self.label}")
        norm = simpson((self.radius * self.radial) ** 2, x=self.radius)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"Invalid radial norm: {self.label}")
        self.radial = self.radial / np.sqrt(norm)

    def transform(self, q):
        # Integrate once per distinct radius. Bound the temporary Bessel table.
        radii, inverse = np.unique(np.round(q, 12), return_inverse=True)
        values = np.empty_like(radii)
        for start in range(0, len(radii), 256):
            sl = slice(start, start + 256)
            bessel = spherical_jn(self.l, radii[sl, None] * self.radius)
            values[sl] = simpson(bessel * (self.radius**2 * self.radial), x=self.radius)
        return 4 * np.pi * (-1j) ** self.l * values[inverse]


def numeric_rows(lines, start):
    values = []
    for line in lines[start:]:
        try:
            row = [float(v.replace("D", "E")) for v in line.split()]
        except ValueError:
            break
        values.extend(row)
    return np.asarray(values)


@dataclass
class Potential:
    element: str
    ls: list
    projectors: list
    qmax: float
    augmentation: np.ndarray
    grid: np.ndarray
    waves: list
    title: str

    def projector(self, i, q, spline):
        a = self.projectors[i]
        step = self.qmax / len(a)
        extended = np.r_[(-1) ** self.ls[i] * a[1], a]
        x = q / step
        if spline:
            interp = CubicSpline(np.arange(-1, len(a)), extended, bc_type="natural")
            return np.where(x < len(a) - 1, interp(x), 0.0)
        # Four-point Lagrange interpolation of VASP's reciprocal projector table.
        n = np.minimum(np.floor(x).astype(int), len(a) - 4)
        t = x - n
        v0, v1, v2, v3 = (extended[n + j] for j in range(4))
        value = (
            -v0 * t * (t - 1) * (t - 2)
            + 3 * v1 * (t + 1) * (t - 1) * (t - 2)
            - 3 * v2 * (t + 1) * t * (t - 2)
            + v3 * (t + 1) * t * (t - 1)
        ) / 6
        return np.where(x < len(a) - 3, value, 0.0)


def read_potcar(path):
    potentials = {}
    for block in Path(path).read_text().split("End of Dataset"):
        if not block.strip():
            continue
        lines = [line.strip() for line in block.strip().splitlines()]
        elem = re.search(r"VRHFIN\s*=\s*([A-Z][a-z]?)", block)
        if not elem or "PAW radial sets" not in lines:
            raise ValueError("Expected a VASP PAW POTCAR with radial sets")
        element = elem[1]
        if element in potentials:
            raise ValueError(f"Multiple POTCAR datasets for {element} are ambiguous")
        first = lines.index("Non local Part")
        qmax = float(lines[first - 1].split()[0])
        ls, tables, waves = [], [], []
        for i, tag in enumerate(lines):
            if tag == "Non local Part":
                l, count = map(int, lines[i + 1].split()[:2])
                ls.extend([l] * count)
            elif tag == "Reciprocal Space Part":
                tables.append(numeric_rows(lines, i + 1))
            elif tag == "pseudo wavefunction":
                waves.append(numeric_rows(lines, i + 1))
        grid = numeric_rows(lines, lines.index("grid") + 1)
        aug = next(
            i for i, line in enumerate(lines) if line.startswith("augmentation charges")
        )
        delta = numeric_rows(lines, aug + 1).reshape(len(ls), len(ls))
        if (
            len(tables) != len(ls)
            or len(waves) != len(ls)
            or any(w.shape != grid.shape for w in waves)
        ):
            raise ValueError(f"Inconsistent PAW radial records for {element}")
        potentials[element] = Potential(
            element, ls, tables, qmax, delta, grid, waves, lines[0]
        )
    return potentials


def load_basis(config, potentials):
    path = Path(config)
    specs = json.loads(path.read_text())
    basis = {}
    for symbol, shells in specs.items():
        basis[symbol] = []
        for spec in shells:
            l, z = spec["l"], spec.get("zeta", 0)
            if not isinstance(l, int) or not isinstance(z, int) or min(l, z) < 0:
                raise ValueError("l and zeta must be nonnegative integers")
            source = spec["source"]
            if source == "potcar":
                pot = potentials[symbol]
                channels = [i for i, angular in enumerate(pot.ls) if angular == l]
                if z >= len(channels):
                    raise ValueError(f"POTCAR lacks {symbol} l={l}, zeta={z}")
                r = pot.grid
                radial = pot.waves[channels[z]] / r
            else:
                file = path.parent / os.path.expandvars(spec["file"])
                if source == "openmx":
                    tag = f"pseudo.atomic.orbitals.L={l}"
                    data = file.read_text().split("<" + tag + "\n", 1)
                    if len(data) != 2:
                        raise ValueError(f"Missing {tag} in {file}")
                    rows = np.loadtxt(
                        data[1].split(tag + ">", 1)[0].splitlines(), ndmin=2
                    )
                    r, radial = rows[:, 1] * BOHR, rows[:, 2 + z]
                elif source == "table":
                    rows = np.loadtxt(file, ndmin=2)
                    r, radial = rows[:, 0], rows[:, 1 + z]
                    r = r * {"bohr": BOHR, "angstrom": 1.0}[spec["unit"]]
                else:
                    raise ValueError(f"Unknown shell source {source}")
            basis[symbol].append(Shell(l, r, radial, f"{symbol}/{source}/l{l}/z{z}"))
        if not basis[symbol]:
            raise ValueError(f"Empty basis for {symbol}")
    return basis
