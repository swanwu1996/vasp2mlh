"""Generate compact, atom-centred AO families using a specified PAW reference."""

from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline

from .configuration import load_config
from .basis import Shell, read_potcar
from .registry import write_registry


def radial_candidates(pot, r, cutoff, l, count):
    for i, angular in enumerate(pot.ls):
        if angular == l:
            native = pot.waves[i] / pot.grid
            radial = CubicSpline(pot.grid, native)(np.maximum(r, pot.grid[0]))
            radial[0] = native[0] if l == 0 else 0
            yield radial, f"partial-{i}"
    # Additional radial directions also provide polarization when no PAW channel exists.
    for j in range(count + 8):
        decay = (l + 2) * 1.6**j
        yield (r / cutoff) ** l * np.exp(-decay * r / cutoff), f"slater-{j}"


def compact_shells(pot, layout):
    cutoff = float(layout["cutoff_A"])
    counts = layout["multiplicities"]
    if (
        not np.isfinite(cutoff)
        or not 0 < cutoff <= pot.grid[-1]
        or not isinstance(counts, list)
        or not counts
        or not all(type(n) is int and n >= 0 for n in counts)
        or not any(counts)
    ):
        raise ValueError(
            f"Invalid multiplicities or cutoff outside the PAW radial grid for {pot.element}"
        )
    # A common uniform mesh makes the endpoint quadrature and boundary conditions explicit.
    r = np.linspace(0, cutoff, 513)
    t = np.clip((r / cutoff - 0.7) / 0.3, 0, 1)
    taper = 1 - t**3 * (10 - 15 * t + 6 * t**2)
    result = []
    for l, count in enumerate(counts):
        if not count:
            continue
        accepted = []

        for radial, name in radial_candidates(pot, r, cutoff, l, count):
            v = radial * taper
            original = simpson((r * v) ** 2, x=r)
            if original <= 0 or not np.isfinite(original):
                continue
            for _ in range(2):
                for other in accepted:
                    v -= simpson(r * r * v * other, x=r) * other
            norm = simpson((r * v) ** 2, x=r)
            if norm <= 1e-8 * original:
                continue
            if v[np.argmax(abs(r * v))] < 0:
                v = -v
            sh = Shell(l, r, v, f"{pot.element}/compact-v1/l{l}/{name}")
            accepted.append(sh.radial)
            result.append(sh)
            if len(accepted) == count:
                break
        if len(accepted) != count:
            raise ValueError(f"Unresolved radial rank for {pot.element}, l={l}")
    return result


COMPACT_METHOD = "paw-compact-v1"
NATIVE_METHOD = "paw-native-v1"
DEFAULT_LAYOUT = Path(__file__).with_name("compact-defaults-v1.toml")


def generate(
    output,
    potcars,
    layout_file=None,
    method=COMPACT_METHOD,
    *,
    hamgnn_layout=False,
    nao=None,
):
    if method == "auto":
        method = COMPACT_METHOD
    if method not in (COMPACT_METHOD, NATIVE_METHOD):
        raise ValueError(f"Unknown AO generation method: {method}")
    if layout_file and method == NATIVE_METHOD:
        raise ValueError("A custom layout requires paw-compact-v1")
    inputs = [*potcars] + ([layout_file] if layout_file else [])
    if Path(output).resolve() in {Path(f).resolve() for f in inputs}:
        raise ValueError("Choose a separate registry output")
    pots = {}
    for path in potcars:
        for sym, pot in read_potcar(path).items():
            if sym in pots and pots[sym].sha256 != pot.sha256:
                raise ValueError(
                    f"Multiple PAW references for {sym}; choose one per library"
                )
            pots[sym] = pot
    if not pots:
        raise ValueError("No PAW element datasets in the supplied POTCAR files")
    layout = load_config(layout_file) if layout_file else None
    if layout is not None and (not layout or not set(layout) <= set(pots)):
        raise ValueError("Supply a PAW reference for every element in the layout")
    preset = None
    if method == COMPACT_METHOD and layout is None:
        defaults = load_config(DEFAULT_LAYOUT)
        layout = {}
        for sym, pot in pots.items():
            if sym not in defaults:
                raise ValueError(
                    f"No default compact layout for {sym}; supply an explicit layout"
                )
            recipe = defaults[sym]
            layout[sym] = {
                **recipe,
                "cutoff_A": min(recipe["cutoff_A"], float(pot.grid[-1])),
            }
        preset = DEFAULT_LAYOUT.stem
    if hamgnn_layout and layout is not None:
        from .export import hamgnn_layout as select_layout

        shells = {
            sym: [l for l, n in enumerate(recipe["multiplicities"]) for _ in range(n)]
            for sym, recipe in layout.items()
        }
        size, compatible = select_layout(shells, nao, extend=layout_file is None)
        if layout_file is None:
            for sym, ls in compatible.items():
                layout[sym]["multiplicities"] = [
                    ls.count(l) for l in range(max(ls) + 1)
                ]
            preset = f"{DEFAULT_LAYOUT.stem}/hamgnn-{size}"
    basis = {}
    for sym in sorted(pots if layout is None else layout):
        pot = pots[sym]
        if layout is None:
            basis[sym] = [
                Shell(pot.ls[i], pot.grid, wave / pot.grid, f"{sym}/native/{i}")
                for i, wave in sorted(
                    enumerate(pot.waves), key=lambda pair: pot.ls[pair[0]]
                )
            ]
        else:
            basis[sym] = compact_shells(pot, layout[sym])
    if hamgnn_layout and layout is None:
        from .export import hamgnn_layout as select_layout

        select_layout({sym: [sh.l for sh in orbitals] for sym, orbitals in basis.items()}, nao)
    write_registry(
        basis,
        output,
        method,
        {s: pots[s].sha256 for s in basis},
        layout,
        preset=preset,
    )
