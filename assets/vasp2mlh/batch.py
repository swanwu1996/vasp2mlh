"""Independent VASP conversions on a single POSIX node, with verified restart."""

import hashlib
import json
import multiprocessing
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import h5py
from threadpoolctl import threadpool_limits

from . import __version__
from .band_release import check_source, reconstruct, reference_points, resolve_pbe_run, check_pbe_pair
from .basis import radial_backend
from .configuration import (
    BATCH_CONTROLS,
    DEFAULT_ENERGY_WINDOW,
    backend_setting,
    load_config,
    runtime_controls,
)
from .pipeline import build, file_sha256, output_file, projection_header
from .operator import check_potential_records, check_potential_source
from .vasp import spectral_mode, metadata
from xml.etree.ElementTree import ParseError
from .realspace import SPARSE_FORMAT
from .progress import progress_bar
from .registry import FORMAT as REGISTRY_FORMAT, read_registry
from .sparse import sparsify


# Each spawned worker shares this one counter with the rendering parent.
_projected_points = None


def init_worker(projected):
    global _projected_points
    _projected_points = projected


def count_projected_point(done, total):
    with _projected_points.get_lock():
        _projected_points.value += 1


def read_jobs(manifest, config=None):
    manifest = Path(manifest).resolve()
    config = load_config(manifest) if config is None else config
    if not isinstance(config, dict) or not isinstance(config.get("jobs"), list):
        raise ValueError("Manifest must be an object with a jobs array")
    if not isinstance(config.get("defaults", {}), dict):
        raise ValueError("Manifest defaults must be an object")
    mode = config.get("mode", "validated")
    modes = {
        "validated": {
            "cutoff",
            "mesh",
            "energy_window",
            "band_range",
            "max_rmse",
            "max_error",
            "relative_pruning",
            "centre_offset",
            "width",
            "floor",
            "taper",
        },
        "projection": {"cutoff", "mesh", "complement"},
    }
    modes["hybrid"] = modes["noncollinear"] = modes["validated"] | {"completion_method"}
    if (
        not isinstance(mode, str)
        or mode not in modes
        or config.keys()
        - (
            {
                "command",
                "mode",
                "registry",
                "defaults",
                "jobs",
                "runtime",
                "output",
                "output_dir",
            }
            | BATCH_CONTROLS
        )
    ):
        raise ValueError("Invalid batch mode or unknown top-level manifest field")

    def path(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Expected a nonempty manifest path")
        return (manifest.parent / value).resolve()

    registry = path(config["registry"])
    jobs = []
    ids = set()
    for row in config["jobs"]:
        if not isinstance(row, dict) or not isinstance(row.get("options", {}), dict):
            raise ValueError("Each job and its options must be objects")
        label = row["id"]
        if (
            not isinstance(label, str)
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", label)
            or label in ids
        ):
            raise ValueError(f"Duplicate or invalid job id: {label!r}")
        ids.add(label)
        allowed = {"id", "run", "options"}
        if mode in ("validated", "hybrid", "noncollinear"):
            allowed.update(("reference", "potential_run"))
        if mode == "hybrid":
            allowed.add("pbe_run")
        options = {**config.get("defaults", {}), **row.get("options", {})}
        if row.keys() - allowed or options.keys() - modes[mode]:
            raise ValueError(f"Unknown field or option for {mode} job {label}")
        if mode in ("validated", "hybrid", "noncollinear") and not options.keys() & {
            "energy_window",
            "band_range",
        }:
            options["energy_window"] = list(DEFAULT_ENERGY_WINDOW)
        if mode in ("validated", "hybrid", "noncollinear") and (
            not {"cutoff", "max_rmse", "max_error"} <= options.keys()
            or len(options.keys() & {"energy_window", "band_range"}) != 1
        ):
            raise ValueError(
                f"Job {label} needs cutoff, both tolerances and exactly one window"
            )
        for key, value in options.items():
            if key in ("mesh", "band_range", "energy_window"):
                size = 3 if key == "mesh" else 2
                kinds = (int, float) if key == "energy_window" else (int,)
                if (
                    not isinstance(value, list)
                    or len(value) != size
                    or any(type(v) not in kinds for v in value)
                ):
                    raise ValueError(f"Invalid {key} array for job {label}")
            elif key == "completion_method":
                if value not in ("spectral-ritz", "locality-constrained", "pbe-anchored-v1"):
                    raise ValueError(f"Unknown hybrid method for job {label}")
            elif key == "relative_pruning":
                if type(value) is not bool:
                    raise ValueError(f"{key} must be true or false")
            elif type(value) not in (int, float):
                raise ValueError(f"{key} must be numeric for job {label}")
        job = {
            "id": label,
            "mode": mode,
            "run": path(row["run"]),
            "registry": registry,
            "options": options,
        }
        if mode in ("validated", "hybrid", "noncollinear"):
            job["reference"] = path(
                row.get("reference", row["run"])
                if mode in ("hybrid", "noncollinear")
                else row["reference"]
            )
            job["potential_run"] = path(row.get("potential_run", row["run"]))
        if "pbe_run" in row:
            job["pbe_run"] = path(row["pbe_run"])
        jobs.append(job)
    if not jobs:
        raise ValueError("Manifest has no jobs")
    return jobs


def input_paths(job):
    source = job["run"]
    paths = [
        job["registry"],
        *(source / n for n in ("WAVECAR", "vasprun.xml", "POTCAR")),
    ]
    if job["mode"] == "validated":
        potential, reference = job["potential_run"], job["reference"]
        paths += [source / "CHGCAR", potential / "vaspout.h5"]
        for folder in (potential, reference):
            paths.extend(folder / n for n in ("vasprun.xml", "POTCAR", "CHGCAR"))
        paths.extend(
            d / "OUTCAR" for d in (source, reference) if (d / "OUTCAR").exists()
        )
    elif job["mode"] in ("hybrid", "noncollinear"):
        paths.append(job["reference"] / "vasprun.xml")
        src = metadata(source / "vasprun.xml")
        directory = resolve_pbe_run(
            src, source, job.get("pbe_run"), job["options"].get("completion_method")
        )
        if directory is not None:
            paths.extend(
                directory / name for name in ("vasprun.xml", "WAVECAR", "POTCAR")
            )
            if (directory / "OUTCAR").is_file():
                paths.append(directory / "OUTCAR")
        if (source / "OUTCAR").is_file():
            paths.append(source / "OUTCAR")
    elif (source / "CHGCAR").is_file() and (source / "CHGCAR").stat().st_size:
        paths.append(source / "CHGCAR")
    return list(dict.fromkeys(paths))


def stamp(path):
    s = path.stat()
    if not path.is_file() or not s.st_size:
        raise ValueError(f"Missing or empty input {path}")
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


@lru_cache(maxsize=256)
def hash_at_stamp(path, identity):
    digest = file_sha256(path)
    if stamp(path) != identity:
        raise ValueError(f"Input changed while hashing: {path}")
    return digest


def write_json(path, data):
    with output_file(path) as tmp:
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def convert(job, output, threads, backend=None):
    started = monotonic()
    label, mode = job["id"], job["mode"]
    target = output / (label + ".h5")
    receipt = output / ".batch" / (label + ".json")
    try:
        with backend_setting(backend), threadpool_limits(limits=threads):
            files = input_paths(job)
            stamps = [stamp(p) for p in files]
            signature = {
                "version": __version__,
                "mode": mode,
                "options": job["options"],
                "backend": radial_backend(),
                "inputs": [
                    (str(p), hash_at_stamp(p, st)) for p, st in zip(files, stamps)
                ],
            }
            # Keep role assignments even if several directories share the same inputs.
            signature["directories"] = {
                k: str(job[k])
                for k in ("run", "reference", "potential_run", "pbe_run")
                if k in job
            }
            key = hashlib.sha256(
                json.dumps(signature, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            if target.exists():
                try:
                    saved = json.loads(receipt.read_text())
                except (FileNotFoundError, ValueError):
                    saved = {}
                if not isinstance(saved, dict):
                    saved = {}
                try:
                    with h5py.File(target) as f:
                        if f.attrs.get("batch_id") != label:
                            raise ValueError(
                                f"Output already exists and is not owned by this batch: {target}"
                            )
                        same = (
                            f.attrs.get("batch_signature") == key
                            and f.attrs.get("format") == SPARSE_FORMAT
                        )
                except OSError:
                    if saved.get("id") != label or saved.get("model") != target.name:
                        raise
                    same = False
                if (
                    same
                    and saved.get("signature") == key
                    and file_sha256(target) == saved.get("sha256")
                ):
                    if files != input_paths(job) or stamps != [stamp(p) for p in files]:
                        raise ValueError("Input changed during resume check")
                    return {
                        **saved,
                        "status": "skipped",
                        "seconds": monotonic() - started,
                    }
            point_callback = (
                count_projected_point if _projected_points is not None else None
            )
            with output_file(target) as tmp:
                if mode in ("validated", "hybrid", "noncollinear"):
                    if mode != (
                        spectral_mode(metadata(job["run"] / "vasprun.xml"))
                        or "validated"
                    ):
                        raise ValueError("Batch mode disagrees with source functional")
                    report = reconstruct(
                        job["run"],
                        job["registry"],
                        job["reference"],
                        tmp,
                        potential_run=job["potential_run"],
                        pbe_run=job.get("pbe_run"),
                        on_kpoint=point_callback,
                        **job["options"],
                    )
                    metrics = {
                        k: report[k]
                        for k in ("rmse_eV", "max_error_eV", "observations")
                    }
                else:
                    options = job["options"].copy()
                    cutoff = options.pop("cutoff", None)
                    with TemporaryDirectory(dir=output) as d:
                        dense = Path(d) / "projection.h5"
                        build(
                            job["run"],
                            job["registry"],
                            dense,
                            on_kpoint=point_callback,
                            **options,
                        )
                        sparsify(dense, tmp, cutoff=cutoff)
                    metrics = {}
                with h5py.File(tmp, "r+") as f:
                    f.attrs.update(
                        batch_id=label,
                        batch_mode=mode,
                        batch_signature=key,
                        radial_backend=signature["backend"],
                    )
                if files != input_paths(job) or stamps != [stamp(p) for p in files]:
                    raise ValueError("Input changed during conversion")
                result = {
                    "id": label,
                    "mode": mode,
                    "status": "converted",
                    "model": target.name,
                    "signature": key,
                    "sha256": file_sha256(tmp),
                    "seconds": monotonic() - started,
                    **metrics,
                }
            write_json(receipt, result)
            return result
    except Exception as exc:
        return {
            "id": label,
            "mode": mode,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": monotonic() - started,
        }


def preflight(job):
    for path in input_paths(job):
        stamp(path)
    run = job["run"]
    src = metadata(run / "vasprun.xml")
    if job["mode"] != "projection":
        if job["mode"] != (spectral_mode(src) or "validated"):
            raise ValueError(
                f"Batch mode {job['mode']} disagrees with source mode {spectral_mode(src) or 'validated'}"
            )
        check_source(src, run)
    wave, _ = projection_header(run, src, job["options"].get("mesh"))
    symbols = src["atoms"].get_chemical_symbols()
    basis = read_registry(job["registry"], symbols)
    counts = {s: sum(2 * sh.l + 1 for sh in shells) for s, shells in basis.items()}
    dim = sum(counts[s] for s in symbols)
    hdim = dim * (2 if src["spinor"] else 1)
    projected = len(src["kpoints"]) if spectral_mode(src) else wave.nk
    largest_mesh = projected
    if job["mode"] in ("hybrid", "noncollinear"):
        directory = resolve_pbe_run(
            src, run, job.get("pbe_run"), job["options"].get("completion_method")
        )
        if directory is not None:
            ref_pbe = check_pbe_pair(src, run, directory, job["options"].get("mesh"))
            largest_mesh = max(projected, len(ref_pbe["kpoints"]))
            projected += len(ref_pbe["kpoints"])
    if job["mode"] == "validated":
        potential_meta = (
            src
            if run == job["potential_run"]
            else metadata(job["potential_run"] / "vasprun.xml")
        )
        check_potential_source(src, potential_meta)
        with h5py.File(job["potential_run"] / "vaspout.h5") as potential:
            check_potential_records(potential)
    if job["mode"] != "projection":
        reference_points(src, job["reference"])
    return {
        "kpoints": projected,
        "orbitals": dim,
        "dense_hs_bytes": 16
        * largest_mesh
        * (wave.ns * hdim**2 + dim**2),
    }


def run_batch(manifest, output=None, workers=None, threads=None, check_only=None):
    import fcntl

    manifest = Path(manifest).resolve()
    config = load_config(manifest)
    if config.get("command", "convert-batch") != "convert-batch":
        raise ValueError("Expected a convert-batch configuration")
    settings = runtime_controls(config, batch=True)
    workers = settings.get("workers", 1) if workers is None else workers
    threads = settings.get("threads", 1) if threads is None else threads
    check_only = settings.get("check_only", False) if check_only is None else check_only
    if output is None:
        if (
            "output_dir" in config
            and "output" in config
            and config["output"] != config["output_dir"]
        ):
            raise ValueError("Conflicting output_dir and legacy output paths")
        destination = config.get("output_dir", config.get("output"))
        if not isinstance(destination, str) or not destination.strip():
            raise ValueError("Specify the output directory in the configuration")
        output = manifest.parent / destination
    if workers < 1 or threads < 1:
        raise ValueError("Worker and thread counts must be positive")
    jobs = read_jobs(manifest, config)
    with h5py.File(jobs[0]["registry"]) as registry:
        if registry.attrs.get("format") != REGISTRY_FORMAT:
            raise ValueError("Batch conversion requires a frozen AO registry")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = output / ".batch"
    state.mkdir(exist_ok=True)
    summary_file = output / ("preflight.json" if check_only else "summary.json")
    # Reserve all destinations before writing, including inputs referenced by other jobs.
    destinations = [summary_file, state / "lock"]
    for job in jobs:
        destinations.extend(
            [output / (job["id"] + ".h5"), state / (job["id"] + ".json")]
        )
    protected = {
        Path(manifest).resolve(),
        *(p.resolve() for job in jobs for p in input_paths(job)),
    }
    if any(p.is_symlink() or p.resolve() in protected for p in destinations):
        raise ValueError("A batch destination is a symlink or collides with an input")
    context = multiprocessing.get_context("spawn")
    projected = context.Value("Q", 0) if not check_only else None
    label = (
        "Preflight" if check_only else f"Convert ({min(workers, len(jobs))} workers)"
    )
    with (
        (state / "lock").open("a") as lock,
        progress_bar(label, len(jobs), "job", projected=projected) as bar,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = {}
        write_json(summary_file, {"status": "running", "total": len(jobs)})

        counts = Counter()

        def record(result):
            results[result["id"]] = result
            status = result["status"]
            counts[status] += 1
            if status == "failed":
                message = f"{result['id']} failed: {result['error']}"
                if sys.stderr.isatty():
                    bar.write(message, file=sys.stderr)
                else:
                    print(message, file=sys.stderr, flush=True)
            detail = ", ".join(f"{name}={n}" for name, n in counts.items())
            bar.set_postfix_str(f"{detail}, last={result['id']}", refresh=False)
            bar.update(1)

        if check_only:
            for job in jobs:
                try:
                    details = preflight(job)
                    record({"id": job["id"], "status": "ready", **details})
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    IndexError,
                    ParseError,
                    AttributeError,
                ) as exc:
                    record({"id": job["id"], "status": "failed", "error": str(exc)})
        else:
            pool = ProcessPoolExecutor(
                max_workers=min(workers, len(jobs)),
                mp_context=context,
                initializer=init_worker,
                initargs=(projected,),
            )
            try:
                futures = {
                    pool.submit(
                        convert, job, output, threads, settings.get("backend")
                    ): job["id"]
                    for job in jobs
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "id": futures[future],
                            "status": "failed",
                            "error": str(exc),
                        }
                    record(result)
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
        rows = [results[job["id"]] for job in jobs]
        summary = {
            "status": "failed"
            if any(r["status"] == "failed" for r in rows)
            else "complete",
            "mode": jobs[0]["mode"],
            "check_only": check_only,
            "jobs": rows,
        }
        write_json(summary_file, summary)
        return summary
