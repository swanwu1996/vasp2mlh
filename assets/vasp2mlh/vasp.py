"""Read standard VASP direct-access records and matching final-run metadata."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from ase import Atoms

BOHR = 0.529177249
HSQDTM = 13.605826 * BOHR**2


def table(node, path):
    el = None if node is None else node.find(path)
    if el is None:
        raise ValueError(f"missing {path} in vasprun.xml")
    return np.array([row.text.split() for row in el], dtype=float)


# Parameters which define VASP's exact exchange and its reciprocal-space sampling.
HYBRID_TAGS = {
    "HFSCREEN",
    "HFSCREENC",
    "HFRCUT",
    "HFALPHA",
    "LRHFCALC",
    "LTHOMAS",
    "LMODELHF",
    "LHFONE",
    "BEXX",
    "LMAXFOCK",
    "NMAXFOCKAE",
    "LFOCKAEDFT",
    "NKRED",
    "NKREDX",
    "NKREDY",
    "NKREDZ",
    "PRECFOCK",
    "ODDONLY",
    "EVENONLY",
    "SHIFTRED",
    "ENCUTFOCK",
    "AEXX_LR",
    "EXXOEP",
}


def is_hybrid(meta):
    return meta.get("operator_parameters", {}).get("LHFCALC") in {"T", ".TRUE.", "TRUE"}


def screened_hybrid_maturity(meta):
    """Feature maturity is independent of a particular sample's validation result."""
    if is_hybrid(meta) and float(meta["operator_parameters"].get("HFSCREEN", 0)) > 0:
        return {"screened_hybrid_maturity": "beta"}
    return {}


def effective_hybrid_flag(parameters):
    """Read effective exchange activation from the completed run's parameters."""
    flags = {
        "T": "T",
        "TRUE": "T",
        ".TRUE.": "T",
        "F": "F",
        "FALSE": "F",
        ".FALSE.": "F",
    }
    values = {
        flags.get((node.text or "").strip().upper())
        for node in parameters.findall(".//*[@name='LHFCALC']")
    }
    if values not in ({"T"}, {"F"}):
        raise ValueError(
            "Missing, ambiguous or invalid effective LHFCALC in VASP XML; cannot identify the functional"
        )
    return values.pop()


def spectral_mode(meta):
    if meta.get("spinor"):
        return "noncollinear"
    if is_hybrid(meta):
        return "hybrid"
    return None


def split_mesh(meta, optional):
    """Keep only the integration mesh in the training spectrum and WAVECAR map."""
    weights = meta["weights"]
    if (
        weights.ndim != 1
        or meta["kpoints"].shape != (len(weights), 3)
        or meta["eig"].ndim != 3
        or meta["eig"].shape[1] != len(weights)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.any(weights > 0)
    ):
        raise ValueError("Invalid integration weights in source XML")
    training = weights > 0
    meta["wave_indices"] = np.flatnonzero(training)
    meta["wave_nk"] = len(training)
    if not np.all(training):
        zero = dict(
            kpoints=meta["kpoints"][~training],
            eig=meta["eig"][:, ~training],
            source="zero-weight",
            wave_indices=np.flatnonzero(~training),
        )
        meta["zero_weight"] = meta["holdout"] = zero
        meta["kpoints"] = meta["kpoints"][training]
        meta["eig"] = meta["eig"][:, training]
        meta["weights"] = weights[training]
    if optional is not None:
        ks = table(optional, "./kpoints/varray[@name='kpointlist']")
        es = eigenvalues(optional)
        if ks.ndim != 2 or ks.shape[1] != 3 or es.ndim != 3 or es.shape[1] != len(ks):
            raise ValueError("Invalid KPOINTS_OPT spectrum in vasprun.xml")
        meta["holdout"] = dict(kpoints=ks, eig=es, source="KPOINTS_OPT")
    if meta["mesh"] is None:
        # mesh_indices subsequently checks every offset and multiplicity.
        wrapped = np.round(meta["kpoints"] % 1, decimals=7) % 1
        meta["mesh"] = [len(np.unique(axis)) for axis in wrapped.T]


def eigenvalues(parent):
    return np.array(
        [
            [table(k, ".")[:, 0] for k in channel]
            for channel in parent.findall("./eigenvalues/array/set/set")
        ]
    )


def xc_parameters(settings):
    names = {
        "GGA",
        "METAGGA",
        "LHFCALC",
        "LDAU",
        "XC",
        "ALDAX",
        "ALDAC",
        "AGGAX",
        "AGGAC",
        "AEXX",
        "PARAM1",
        "PARAM2",
        "GGA_COMPAT",
        "VOSKOWN",
        "LUSE_VDW",
    }
    if is_hybrid({"operator_parameters": settings}):
        names |= HYBRID_TAGS
    return {k: v for k, v in settings.items() if k in names or k.startswith("LIBXC")}


def source_fermi(calculation):
    """Use the integration-grid Fermi energy even when VASP writes KPOINTS_OPT first."""
    values = [
        dos.findtext("i[@name='efermi']")
        for dos in calculation.findall("dos")
        if "kpoints_opt" not in dos.get("comment", "").lower()
    ]
    if len(values) != 1 or values[0] is None or not np.isfinite(float(values[0])):
        raise ValueError("Missing or ambiguous source-grid Fermi energy in vasprun.xml")
    return float(values[0])


def metadata(path):
    root = ET.parse(path).getroot()
    atoms = [
        r[0].text.strip()
        for r in root.findall("./atominfo/array[@name='atoms']/set/rc")
    ]
    final = root.find("./structure[@name='finalpos']")
    cell = table(final, "./crystal/varray[@name='basis']")
    frac = table(final, "./varray[@name='positions']")
    params = root.find("parameters")

    def param(name):
        value = params.findtext(f".//*[@name='{name}']")
        if value is None:
            raise ValueError(f"Missing VASP XML parameter {name}")
        return value.strip()

    operator_names = {
        "GGA",
        "METAGGA",
        "LHFCALC",
        "LDAU",
        "LASPH",
        "LMAXMIX",
        "LUSE_VDW",
        "LSPIRAL",
        "EFIELD",
        "LDIPOL",
        "IDIPOL",
        "LMIXTAU",
        "XC",
        "ALDAX",
        "ALDAC",
        "AGGAX",
        "AGGAC",
        "AEXX",
        "PARAM1",
        "PARAM2",
        "GGA_COMPAT",
        "VOSKOWN",
        "ADDGRID",
        "NGX",
        "NGY",
        "NGZ",
        "NGXF",
        "NGYF",
        "NGZF",
    }
    operator_parameters = {
        node.attrib["name"].strip(): (node.text or "").strip().upper()
        for section in (root.find("incar"), params)
        if section is not None
        for node in section.iter()
        if node.attrib.get("name", "").strip() in operator_names | HYBRID_TAGS
        or node.attrib.get("name", "").strip().startswith("LIBXC")
    }
    operator_parameters["LHFCALC"] = effective_hybrid_flag(params)
    types = root.findall("./atominfo/array[@name='atomtypes']/set/rc")
    neutral = sum(int(row[0].text) * float(row[3].text) for row in types)
    titles = {row[1].text.strip(): row[4].text.strip() for row in types}
    mesh = root.findtext("./kpoints/generation/v[@name='divisions']")
    calcs = root.findall("./calculation")
    last = calcs[-1]
    steps = last.findall("scstep/energy/i[@name='e_fr_energy']")
    delta = (
        abs(float(steps[-1].text) - float(steps[-2].text)) if len(steps) >= 2 else None
    )
    meta = dict(
        final_energy_change_eV=delta,
        ediff_eV=float(param("EDIFF")),
        xml_path=Path(path).resolve(),
        operator_parameters=operator_parameters,
        atoms=Atoms(atoms, scaled_positions=frac, cell=cell, pbc=True),
        kpoints=table(root, "./kpoints/varray[@name='kpointlist']"),
        weights=table(root, "./kpoints/varray[@name='weights']").ravel(),
        mesh=None if mesh is None else list(map(int, mesh.split())),
        spinor=param("LNONCOLLINEAR") == "T",
        soc=param("LSORBIT") == "T",
        saxis=list(map(float, param("SAXIS").split())),
        nspin=int(param("ISPIN")),
        icharg=int(param("ICHARG")),
        writes_charge=params.findtext(".//*[@name='LCHARG']", "").strip() == "T",
        final_scf_steps=len(last.findall("scstep")),
        nelm=int(params.findtext(".//*[@name='NELM']", "0")),
        encut=float(param("ENMAX")),
        lreal=param("LREAL"),
        spline=param("NLSPLINE") == "T",
        nelect=float(param("NELECT")),
        doping_charge=float(param("NELECT")) - neutral,
        titles=titles,
        energy=float(last.findtext("./energy/i[@name='e_0_energy']")),
        efermi=source_fermi(last),
        eig=eigenvalues(last),
    )
    if spectral_mode(meta):
        split_mesh(meta, last.find("eigenvalues_kpoints_opt"))
    return meta


class Wavecar:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as file:
            first = np.fromfile(file, "<f8", 3)
            if len(first) != 3 or first[2] not in (45200, 45210, 53300, 53310):
                raise ValueError(
                    "Expected a little-endian WAVECAR with RTAG 45200/45210/53300/53310"
                )
            self.recl, self.ns, tag = map(int, first)
            self.dtype = np.dtype("<c8" if tag in (45200, 53300) else "<c16")
            if self.recl < 104 or self.ns not in (1, 2):
                raise ValueError("Invalid WAVECAR header")
            file.seek(self.recl)
            h = np.fromfile(file, "<f8", 13)
            self.nk, self.nb = map(int, h[:2])
            self.encut = h[2]
            self.cell = h[3:12].reshape(3, 3)
        header_bytes = (4 + 3 * self.nb) * 8
        self.header_records = (header_bytes + self.recl - 1) // self.recl
        if min(self.nk, self.nb, self.encut) <= 0 or (
            tag < 53000 and self.header_records != 1
        ):
            raise ValueError("Invalid WAVECAR dimensions")
        if (
            self.path.stat().st_size
            < (2 + self.ns * self.nk * (self.header_records + self.nb)) * self.recl
        ):
            raise ValueError("Truncated WAVECAR")

    def read(self, spin, ik):
        base = 2 + (spin * self.nk + ik) * (self.nb + self.header_records)
        with self.path.open("rb") as file:
            file.seek(base * self.recl)
            info = np.fromfile(file, "<f8", 4 + 3 * self.nb)
            count = int(info[0])
            if count < 1 or count * self.dtype.itemsize > self.recl:
                raise ValueError(f"Invalid coefficient count at k-point {ik}")
            coeff = np.empty((self.nb, count), dtype=complex)
            for n in range(self.nb):
                file.seek((base + self.header_records + n) * self.recl)
                coeff[n] = np.fromfile(file, self.dtype, count)
        return info[1:4], info[4:].reshape(-1, 3)[:, [0, 2]], coeff.T


def plane_waves(k, cell, encut):
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T
    radius = np.sqrt(encut / HSQDTM)
    limits = np.ceil(
        radius * np.linalg.norm(cell, axis=1) / (2 * np.pi) + np.abs(k)
    ).astype(int)
    axes = [np.r_[0 : n + 1, -n:0] for n in limits]
    z, y, x = np.meshgrid(*axes[::-1], indexing="ij")
    q = (np.column_stack((x.ravel(), y.ravel(), z.ravel())) + k) @ reciprocal
    return q[np.einsum("ij,ij->i", q, q) * HSQDTM < encut]
