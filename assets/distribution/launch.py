"""Entrypoint for the standalone executable, including spawned conversion workers."""

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    import sys
    from pathlib import Path

    from vasp2mlh.cli import genbasis_main, main, valid_main

    entrypoints = {"v2h_genbasis": genbasis_main, "v2h_valid": valid_main}
    entrypoints.get(Path(sys.argv[0]).name, main)()
