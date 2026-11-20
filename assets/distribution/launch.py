"""Entrypoint for the standalone executable, including spawned conversion workers."""

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    from vasp2mlh.cli import main

    main()
