from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="pyehtdata",
    version="0.0.0pre-alpha",
    description="A test bed for the common data structure in python for the EHT",
    long_description=long_description,
    url="https://github.com/astrosmili/pyehtdata/",
    author="Kazunori Akiyama",
    author_email="kakiyama@mit.edu",
    license="MIT",
    keywords="imaging astronomy EHT",
    packages=find_packages(exclude=["doc*", "test*"]),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "xarray",
        "zarr",
        "netcdf4",
        "h5py",
        "astropy",
        "scikit-image",
        "tqdm",
    ]
)
