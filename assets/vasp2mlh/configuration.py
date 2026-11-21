"""Read typed controls from a JSON or TOML file."""

import json
import os
from contextlib import contextmanager
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


# File-valued arguments shared by the ordinary CLI and TOML/JSON operations.
# Keep the leaf path intact so output symlinks can still be detected by writers.
_CONFIG_PATH_FIELDS = {
    "run",
    "reference",
    "potential_run",
    "pbe_run",
    "registry",
    "output",
    "output_dir",
    "model",
    "models",
    "potcar",
    "layout",
    "config",
    "basis_lock",
    "basis",
    "overlap",
    "structure",
    "check_points",
    "lobsterout",
    "reference_xml",
    "points",
    "destination",
    "vasp_xml",
    "openmx_band",
    "comparisons",
}


def anchor_config_paths(args, directory):
    for name in _CONFIG_PATH_FIELDS.intersection(vars(args)):
        value = getattr(args, name)
        if isinstance(value, list):
            setattr(args, name, [str(directory / item) for item in value])
        elif value is not None:
            setattr(args, name, str(directory / value))


def load_config(path):
    path = Path(path)
    with path.open("rb") as stream:
        if path.suffix.lower() == ".toml":
            config = tomllib.load(stream)
        elif path.suffix.lower() == ".json":
            config = json.load(stream)
        else:
            raise ValueError("Choose a .json or .toml control file")
    if not isinstance(config, dict):
        raise ValueError("Expected a configuration object")
    return config


BATCH_CONTROLS = {"workers", "threads", "backend", "check_only", "fast_resume"}
SCALAR_POLAR = "scalar-polar-window-v1"
HSE_POLAR = "hse-polar-window-v1"
HUBBARD_POLAR = "hubbard-polar-window-v1"
SPINOR_POLAR = "spinor-polar-window-v1"
BOUNDED_COLLINEAR = "collinear-window-ritz-v1"
DEFAULT_ENERGY_WINDOW = (-1.0, 2.0)


def runtime_controls(config, batch):
    legacy = config.get("runtime", {})
    allowed = {"threads", "backend"}
    if batch or config.get("command") == "convert":
        allowed.add("progress")
    if batch:
        allowed.update(BATCH_CONTROLS)
    if not isinstance(legacy, dict) or legacy.keys() - allowed:
        raise ValueError(f"Allowed runtime controls: {', '.join(sorted(allowed))}")
    settings = dict(legacy)
    if batch:
        for name in config.keys() & BATCH_CONTROLS:
            if name in settings and config[name] != settings[name]:
                raise ValueError(f"Conflicting values for {name}")
            settings[name] = config[name]
    for name in ("threads", "workers"):
        if name in settings and (type(settings[name]) is not int or settings[name] < 1):
            raise ValueError(f"{name} must be a positive integer")
    for key in ("check_only", "progress", "fast_resume"):
        if key in settings and type(settings[key]) is not bool:
            raise ValueError(f"runtime.{key} must be true or false")
    settings.pop(
        "progress", None
    )  # Old manifests still load; progress is always shown.
    if settings.get("backend", "auto") not in ("auto", "cython", "scipy"):
        raise ValueError("runtime.backend must be auto, cython or scipy")
    return settings


@contextmanager
def backend_setting(value):
    key = "VASP2MLH_RADIAL_BACKEND"
    previous = os.environ.get(key)
    if value is not None:
        os.environ[key] = value
    try:
        yield
    finally:
        if value is not None:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def operation_args(command, parser, values):
    """Build a namespace using the same argument definitions as the legacy CLI."""
    import argparse

    if not isinstance(values, dict):
        raise ValueError("parameters must be a table")
    if command == "convert" and "output" in values:
        values = dict(values)
        if "output_dir" in values and values["output_dir"] != values["output"]:
            raise ValueError("Conflicting output_dir and legacy output paths")
        values["output_dir"] = values.pop("output")
    actions = {a.dest: a for a in parser._actions if a.dest != "help"}
    unknown = values.keys() - actions.keys()
    if unknown:
        raise ValueError(
            f"Unknown parameters for {command}: {', '.join(sorted(unknown))}"
        )
    result = {}
    for name, action in actions.items():
        if name not in values:
            if action.required:
                raise ValueError(f"Missing parameter {name}")
            result[name] = action.default
            continue
        value = values[name]
        if action.nargs == 0:
            if type(value) is not bool:
                raise ValueError(f"{name} must be true or false")
            result[name] = value
            continue
        if action.nargs in (None, "?"):
            items = [value]
        else:
            if (
                not isinstance(value, list)
                or not value
                or (isinstance(action.nargs, int) and len(value) != action.nargs)
            ):
                raise ValueError(f"Invalid list length for {name}")
            items = value
        expected = action.type or str
        for item in items:
            if not (
                type(item) is expected or (expected is float and type(item) is int)
            ):
                raise ValueError(f"{name} expects {expected.__name__} values")
            if action.choices and item not in action.choices:
                raise ValueError(f"Invalid choice for {name}: {item}")
        result[name] = value
    for group in parser._mutually_exclusive_groups:
        count = sum(a.dest in values for a in group._group_actions)
        if count > 1 or (group.required and count != 1):
            raise ValueError(
                f"Choose one of {', '.join(a.dest for a in group._group_actions)}"
            )
    return argparse.Namespace(command=command, **result)
