"""Portable, element-resolved AO definitions and dataset dimension checks."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .configuration import COLLINEAR_POLAR, BOUNDED_COLLINEAR, HSE_POLAR, SCALAR_POLAR, SPINOR_POLAR, HUBBARD_POLAR, load_config
from .basis import Shell, load_basis, read_potcar
from .pipeline import output_file
from .realspace import SPARSE_FORMAT, structure
from .vasp import is_hybrid, xc_parameters

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
    specs = load_config(config)
    hashes = {
        sym: potentials[sym].sha256
        for sym, items in specs.items()
        if sym in potentials and any(item.get("source") == "potcar" for item in items)
    }
    basis = load_basis(config, potentials)
    write_registry(basis, output, "imported", hashes)


def write_registry(basis, output, method, paw_hashes=None, layout=None, *, preset=None):
    with output_file(output) as temp, h5py.File(temp, "w") as out:
        out.attrs["method"] = method
        if method == "paw-native-v1":
            out.attrs["method_status"] = "legacy"
        if preset is not None:
            out.attrs["preset"] = preset
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


def label_definition(f):
    """Settings that define H/S, independent of the structure and energy zero."""
    attrs = f.attrs
    method = attrs.get("hamiltonian_method", "spectral")
    settings = dict(
        version=1,
        method=method,
        metric=attrs["metric"],
        spinor=bool(attrs["spinor"]),
        soc=bool(attrs["soc"]),
        channels=int(attrs["spin_channels"]),
        saxis=attrs["saxis"].tolist() if attrs["spinor"] else None,
        energy_unit=attrs.get("energy_unit", "eV"),
        fourier=attrs.get("fourier"),
    )
    for key in (
        "subspace_bands",
        "complement_eV",
        "alignment_width_eV",
        "degeneracy_eV",
        "calibration_plateau",
    ):
        if key in attrs:
            settings[key] = np.asarray(attrs[key]).tolist()
    if method == HSE_POLAR:
        settings["polar_recipe"] = json.loads(attrs["polar_recipe"])
    if method == "pbe-anchored-v1":
        settings["locality_recipe"] = json.loads(attrs["locality_recipe"])
    if method == "locality_constrained_paw":
        recipe = json.loads(attrs["locality_recipe"])
        if not np.isclose(
            settings.pop("complement_eV") - attrs["fermi_eV"],
            recipe["prior_offset_eV"],
            rtol=0,
            atol=1e-8,
        ):
            raise ValueError(
                "Localized complement energy disagrees with its label recipe"
            )
        settings["locality_recipe"] = recipe
    if method == SPINOR_POLAR and not np.isclose(
        settings["complement_eV"] - attrs["fermi_eV"],
        json.loads(attrs["polar_recipe"])["prior_offset_eV"], rtol=0, atol=1e-8
    ):
        raise ValueError("Spinor prior differs from its recorded Fermi offset")
    if method in {"hybrid_spectral_ritz", "spinor_spectral_ritz", SPINOR_POLAR}:
        settings.pop("complement_eV")
        settings["complement_policy"] = attrs["complement_policy"]
    if method in {SCALAR_POLAR, COLLINEAR_POLAR, SPINOR_POLAR, HUBBARD_POLAR, BOUNDED_COLLINEAR, "hubbard-paw-ritz-v1", "filtered_paw_ritz", "hybrid_spectral_ritz", "spinor_spectral_ritz"}:
        for key in ("filter_width_eV", "filter_floor", "filter_taper_eV"):
            settings[key] = float(attrs[key])
        settings["centre_offset_eV"] = round(
            float(attrs["filter_centre_eV"] - attrs["fermi_eV"]), 10
        )
    recipe_key = {BOUNDED_COLLINEAR: "collinear_recipe", SCALAR_POLAR: "polar_recipe", COLLINEAR_POLAR: "polar_recipe", SPINOR_POLAR: "polar_recipe", HUBBARD_POLAR: "polar_recipe"}.get(method, "hubbard_recipe")
    if recipe_key in attrs or method in {"hubbard-paw-ritz-v1", BOUNDED_COLLINEAR, SCALAR_POLAR, COLLINEAR_POLAR, SPINOR_POLAR, HUBBARD_POLAR}:
        recipe = json.loads(attrs[recipe_key])
        limit = recipe.pop("linear_through_eV")
        target = recipe["energy_window_from_fermi_eV"]
        upper = (
            attrs["fermi_eV"] + target[1]
            if target is not None
            else np.max(f["k/eigenvalues"][:, :, recipe["band_range_1based"][1] - 1])
        )
        if recipe.get("anchor_policy") == "ordered-source-envelope":
            upper = max(upper, np.max(f["k/eigenvalues"][:, :, recipe["band_range_1based"][1] - 1]))
        # XML levels may be rounded to four decimals. k/eigenvalues
        # contains the corresponding WAVECAR double precision values.
        tol = 1e-4 if target is None or recipe.get("anchor_policy") else 1e-7
        if not np.isclose(limit, upper + recipe["upper_buffer_eV"], atol=tol, rtol=0):
            raise ValueError("Inconsistent target-window spectral bound")
        if recipe.get("filter_centre_policy") == "max-fermi-plus-6-or-target-plus-2":
            if not np.isclose(
                settings.pop("centre_offset_eV"),
                max(6.0, limit - attrs["fermi_eV"]),
                atol=1e-7,
                rtol=0,
            ):
                raise ValueError("Filter centre disagrees with its target window")
        if recipe.get("filter_centre_policy") == "spinor-window-and-available-spectrum":
            centre = min(max(attrs["fermi_eV"] + 6.0, limit), attrs["filter_ceiling_eV"] - 2 * attrs["filter_taper_eV"])
            if not np.isclose(settings.pop("centre_offset_eV"), centre - attrs["fermi_eV"], atol=1e-7, rtol=0):
                raise ValueError("Spinor filter centre disagrees with its target and ceiling")
        settings[recipe_key] = recipe
    if attrs["format"] == SPARSE_FORMAT:
        report = json.loads(attrs["truncation_report"])
        for key in ("cutoff_A", "h_tol_eV", "s_tol", "relative_pruning"):
            settings[key] = report[key]
        settings["spinless_time_reversal"] = report.get("spinless_time_reversal", False)
    return settings


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
    known_hubbard = {}
    reports = []
    first_definition = None
    for path in models:
        with h5py.File(path) as f:
            info = inspect_basis(f, expected)
            for sym, sig in info["sha256_by_element"].items():
                if known.setdefault(sym, sig) != sig:
                    raise ValueError(f"Inconsistent radial basis for {sym} in {path}")
            paw = json.loads(f.attrs.get("paw_sha256_by_element", "{}"))
            if (labels or paw) and (
                not isinstance(paw, dict)
                or set(paw) != set(info["orbitals_by_element"])
                or not all(isinstance(v, str) and v.strip() for v in paw.values())
            ):
                raise ValueError("Incomplete element-wise PAW provenance")
            for sym, sig in paw.items():
                if sym in paw_required and paw_required[sym] != sig:
                    raise ValueError(f"Source PAW differs from the registry for {sym}")
                if labels and known_paw.setdefault(sym, sig) != sig:
                    raise ValueError(f"Inconsistent source PAW for {sym} in {path}")
            info["source_paw_recorded"] = bool(paw)
            if labels:
                xc = json.loads(f.attrs.get("xc_settings", "{}"))
                if not isinstance(xc, dict) or not xc:
                    raise ValueError(f"Missing XC provenance in {path}")
                for group in ("source_validation", "band_validation"):
                    if group in f:
                        source = json.loads(f[group].attrs["report"]).get(
                            "operator_settings", {}
                        )
                        recorded = xc_parameters(source)
                        if any(
                            k in recorded and v != recorded[k] for k, v in xc.items()
                        ):
                            raise ValueError(f"Conflicting XC provenance in {path}")
                        xc = {**recorded, **xc}
                        break
                if (
                    is_hybrid({"operator_parameters": xc})
                    and not {"HFSCREEN", "AEXX"} <= xc.keys()
                ):
                    raise ValueError(
                        f"Incomplete nonlocal exchange definition in {path}"
                    )
                if xc.get("GGA") in ("LI", "LIBXC") and not xc.get("LIBXC1"):
                    raise ValueError(f"Missing Libxc functional identity in {path}")
                if xc.get("LDAU") in ("T", "TRUE", ".TRUE."):
                    sites = xc.pop("HUBBARD_BY_ELEMENT", {})
                    if (
                        xc.get("LDAUTYPE") not in (1, 2, 4)
                        or not isinstance(sites, dict)
                        or set(sites) != set(paw)
                    ):
                        raise ValueError(f"Incomplete DFT+U provenance in {path}")
                    for symbol, values in sites.items():
                        if (
                            not isinstance(values, dict)
                            or set(values) != {"LDAUL", "LDAUU", "LDAUJ"}
                            or type(values["LDAUL"]) is not int
                            or not np.isfinite([values["LDAUU"], values["LDAUJ"]]).all()
                        ):
                            raise ValueError(f"Invalid DFT+U definition for {symbol}")
                        if known_hubbard.setdefault(symbol, values) != values:
                            raise ValueError(f"Different Hubbard shell, U or J for {symbol}")
                known_xc.add(json.dumps(xc, sort_keys=True))
                if len(known_xc) > 1:
                    raise ValueError(
                        "Dataset has different exchange-correlation settings"
                    )
                if "hamiltonian_dimension" not in info:
                    raise ValueError("Expected Hamiltonian labels")
                try:
                    definition = label_definition(f)
                    json.dumps(definition, allow_nan=False)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Unverifiable Hamiltonian definition in {path}: {exc}"
                    ) from exc
                if first_definition is None:
                    first_definition = definition
                if definition != first_definition:
                    changed = sorted(
                        k
                        for k in definition.keys() | first_definition.keys()
                        if definition.get(k) != first_definition.get(k)
                        or (k in definition) != (k in first_definition)
                    )
                    raise ValueError(
                        f"Incompatible Hamiltonian definitions in {path}: {', '.join(changed)}"
                    )
                info["label_definition"] = definition
            reports.append(dict(path=str(path), **info))
    return reports
