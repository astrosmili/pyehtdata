[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

# pyehtdata

This is a development version of a potential data library designed for the Event Horizon Telescope (EHT) and next generation EHT (ngEHT).

The aim of this library is
- Natively handling five dimensional images (two dimensional images over frequency, polaization and time grids)
- Providing more readable and intuitive data access to uv-domain datasets (etc, visibilities, closure quantities, calibration tables)
- Offering the data I/O not only with the conventional data formats for the EHT (FITS for images, UVFITS for visibilities), but also with other public data format for the N-D labeled multi-dimensional data used in the science community (e.g. [NETCDF4 (based on HDF5)](https://www.unidata.ucar.edu/software/netcdf/), [Zarr](https://zarr.readthedocs.io/en/stable)) more easily accessible from many other launguages (etc., Julia, C/C++, Fortran, etc).
- Offer lazy loading of data to handle a potentially large data set on the disk that cannot be fully loaded into the memory. 
- Sounce independent data editing / flagging of images and visibilities.

To achive the above aims, this library provides data structure based on [Xarray](https://docs.xarray.dev/en/stable/)'s Dataset for both uv- and image-domain data. This allows to build up a readable data structure easily accessible even without this particular library.

# Installation
You can install this library by
```
python setup.py install
```
or
```
pip install .
```

# Documentation
Documentation is in preparation.


