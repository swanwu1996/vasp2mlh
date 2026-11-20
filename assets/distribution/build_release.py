"""Build the Linux CPU binary distribution in a clean environment with the project installed."""

import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

from vasp2mlh import __version__
from vasp2mlh.basis import radial_backend


def main():
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise SystemExit("This release target requires Linux x86_64")
    if radial_backend() != "cython":
        raise SystemExit("Install the compiled radial kernel before building")
    root = Path(__file__).resolve().parents[2]
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    name = f"vasp2mlh-{__version__}-linux-x86_64"
    with TemporaryDirectory(dir=dist) as temp:
        work = Path(temp).resolve()
        product = work / name
        product.mkdir()
        # Build from the installed wheel. The application code and dependencies are
        # explicitly located by the interpreter; the research tree is never collected.
        import vasp2mlh

        package = Path(vasp2mlh.__file__).resolve().parent
        args = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--onefile",
            "--name",
            "vasp2mlh",
            "--distpath",
            str(product),
            "--specpath",
            str(work),
            "--workpath",
            str(work / "build"),
            "--hidden-import",
            "ase.io.vasp",
            "--hidden-import",
            "ase.io.extxyz",
            "--hidden-import",
            "vasp2mlh._radial",
            "--add-data",
            f"{package / 'viewer.html'}:vasp2mlh",
            "--add-data",
            f"{package / 'compact-defaults-v1.toml'}:vasp2mlh",
        ]
        for dep in ("numpy", "scipy", "h5py", "ase", "threadpoolctl"):
            args.extend(("--copy-metadata", dep))
        for optional in (
            "torch",
            "torch_geometric",
            "DFT_interfaces",
            "matplotlib",
            "IPython",
            "tkinter",
            "pytest",
        ):
            args.extend(("--exclude-module", optional))
        # Outside the source directory, even the launch script's import path is clean.
        entry = work / "launch.py"
        shutil.copyfile(root / "assets/distribution/launch.py", entry)
        subprocess.run([*args, str(entry)], cwd=work, check=True)
        for alias in ("vasp2mlh_valid", "vasp2mlh_genbasis"):
            os.link(product / "vasp2mlh", product / alias)
        templates = product / "configs"
        templates.mkdir()
        shutil.copy(package / "compact-defaults-v1.toml", templates)
        for filename in ("mlh.toml", "validated.toml", "projection.json", "basis.toml", "layout.toml", "bands.toml", "report.json", "deeph.toml"):
            shutil.copyfile(root / "configs" / filename, templates / filename)
        target = dist / (name + ".tar.gz")
        packed = work / target.name
        with tarfile.open(packed, "w:gz", compresslevel=2) as archive:
            archive.add(product, arcname=name)
        digest = hashlib.sha256(packed.read_bytes()).hexdigest()
        os.replace(packed, target)
        (dist / (target.name + ".sha256")).write_text(f"{digest}  {target.name}\n")
        print(target)


if __name__ == "__main__":
    main()
