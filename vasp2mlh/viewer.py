"""Render an offline band-comparison workspace with embedded local results."""

import json
from importlib.resources import files
from pathlib import Path

from .pipeline import output_file


def render(paths, destination):
    cases = []
    for filename in paths:
        with Path(filename).open() as stream:
            case = json.load(stream)
        if not case.get("fractional_k") or set(case.get("bands_eV", {})) != {
            "vasp",
            "model",
            "openmx",
        }:
            raise ValueError(f"Expected a vasp2mlh compare result in {filename}")
        cases.append(case)
    payload = json.dumps(cases, ensure_ascii=True, allow_nan=False).replace(
        "<", r"\u003c"
    )
    template = files("vasp2mlh").joinpath("viewer.html").read_text(encoding="utf-8")
    with output_file(destination) as tmp:
        tmp.write_text(template.replace("<!--LOCAL_DATA-->", payload), encoding="utf-8")
