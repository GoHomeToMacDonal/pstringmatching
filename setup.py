from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "pstringmatching",
        ["csrc/main.cpp"],
        extra_compile_args=["-Wno-sign-compare", "-fopenmp", "-O2"],
        extra_link_args=["-fopenmp"]
    ),
]

setup(ext_modules=ext_modules)
