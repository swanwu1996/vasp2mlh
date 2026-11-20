"""Entrypoint for the standalone executable, including spawned conversion workers."""

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    from pathlib import Path
    import sys
    from vasp2mlh.cli import genbasis_main, main, valid_main

    if Path(sys.argv[0]).name == "vasp2mlh_valid":
        valid_main()
    elif Path(sys.argv[0]).name == "vasp2mlh_genbasis":
        genbasis_main()
    else:
        main()
