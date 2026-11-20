"""Build the optional CPU radial kernel."""

import os

from setuptools import Extension, setup

extensions = []
if os.environ.get("VASP2MLH_NO_EXT") != "1":
    from Cython.Build import cythonize

    extensions = cythonize(
        [Extension("vasp2mlh._radial", ["assets/vasp2mlh/_radial.pyx"], optional=True)],
        build_dir="build/cython",
        compiler_directives={"language_level": 3},
    )
setup(ext_modules=extensions)
