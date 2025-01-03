import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Extension configuration for Cython
ext = Extension(
    name="c_func",
    sources=[
        "noisy_bnsp/c/c_func.pyx",
        "noisy_bnsp/c/ns_cross_exp.c",
        "noisy_bnsp/c/ns_cross_gamma.c",
        "noisy_bnsp/c/emp.c",
    ],
    include_dirs=[np.get_include()],
    libraries=["gsl", "gslcblas"],
    extra_compile_args=["-O3"],
    extra_link_args=["-O3"],
)

setup(
    ext_modules=cythonize([ext]),
    include_dirs=[np.get_include()],
)
