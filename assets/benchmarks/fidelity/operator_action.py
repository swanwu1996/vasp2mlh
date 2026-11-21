"""Score the action on independent PAW-projected wavefunctions."""

import argparse
import json
from pathlib import Path

from vasp2mlh.operator_action import audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("models", nargs="+")
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--bands",
        type=int,
        nargs=2,
        help="inclusive ordered band indices, starting at 1",
    )
    target.add_argument(
        "--energy-window",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="relative to the model source Fermi energy, eV",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.run, args.models, args.energy_window, args.bands), indent=2
        )
    )
