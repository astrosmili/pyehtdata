#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''

import warnings
from tkinter import E


class XarrayDataset(object):
    """
    This is a base Class for the data classes in this library that has
    a single xarray data set.
    """
    # Data type name
    datatype = "ehtdata_base"

    # Default Group Name for the I/O
    group = "base"

    # Data Set
    #   This supposed to include the data set
    ds = None

    def __init__(self, ds):
        """
        Initialize the instance.

        Args:
            ds (xarray.Dataset): Input Dataset
        """
        self.ds = ds.copy()

        if "datatype" not in ds.attrs.keys():
            warnings.warn("Input dataset doesn't have an attribute for datatype. Adding the datatype of '{0}'".format(
                self.datatype), UserWarning)
            self.ds.attrs["datatype"] = self.datatype
        elif ds.attrs["datatype"] != self.datatype:
            ValueError("Input dataset has the datatype of '{0}', which is not compatible with this class." % (
                ds.attrs["datatype"]))

    def _repr_html_(self):
        return self.ds._repr_html_()

    @classmethod
    def __cls__init__(cls, ds):
        return cls(ds=ds)

    @classmethod
    def load_dataset(cls, infile, engine="netcdf4", group=None, **args_open_dataset):
        """
        Open a dataset from a specified input file. This library
        will use xarray.open_dataset() function, and assume that the input file
        is in the netCDF4 (using 'netcdf4' or 'h5netcdf' engines) or zarr ('zarr' engine)
        format, allowing to store multiple datasets in a single data container
        using HDF5 or zarr. The dataset will be loaded from the group specified
        by the argument 'group'.

        Args:
            infile (string): input data container (netcdf4 file or zarr directory)
            engine (string): Engine to use when reading files. 
              This library only accepts "netcdf4", "h5netcdf" and "zarr" engines.
              See the documentation for xarray.open_dataset for details. 
            group (string): the group name of data set. If not specified, 
              it will use the default group name of the class in cls.group.
            **args_open_dataset: other arguments for xarray.open_dataset

        Returns:
            Loaded object
        """
        from xarray import open_dataset

        if engine not in ["netcdf4", "h5netcdf", "zarr"]:
            raise ValueError(
                "available engines: 'netcdf4', 'h5netcdf', 'zarr'")

        if group is None:
            groupname = cls.group
        else:
            groupname = group

        ds = open_dataset(infile, group=groupname,
                          engine=engine, **args_open_dataset)

        return cls(ds=ds)

    def copy(self):
        """
        Replicate this object.

        Returns:
            Replicated data
        """
        return self.__cls__init__(ds=self.ds.copy())

    def chunk(self, **args):
        self.ds = self.ds.chunk(**args)

    def to_netcdf(self, outfile, mode="w", group=None, engine="netcdf4", **args_to_netcdf):
        """
        Save to zarr. Dataset of this object will be saved to
        the specified zarr using the xarray.Dataset.to_zarr() method.

        Args:
            outzarr (string): output filename
            mode (string): Persistence mode. See the documentation for
                xarray.Dataset.to_netcdf().
            engine (string): Engine to use when reading files. 
              This library only accepts "netcdf4" and "h5netcdf" engines.
              See the documentation for xarray.Dataset.to_netcdf() for details. 
            group (string): the group name of data set. If not specified, 
              it will use the default group name of the class in cls.group.
            **args_to_zarr: other arguments for xarray.Dataset.to_zarr()
        """
        from xarray import open_zarr
        import os

        if engine not in ["netcdf4", "h5netcdf"]:
            raise ValueError("available engines: 'netcdf4', 'h5netcdf'")

        if group is None:
            groupname = self.group
        else:
            groupname = group

        if mode == "w" and os.path.isfile(outfile):
            os.system("rm -f {0}".format(outfile))

        self.ds.to_netcdf(outfile, mode=mode, engine=engine,
                          group=groupname, format="NETCDF4", **args_to_netcdf)

    def to_zarr(self, outfile, mode="w", group=None, **args_to_zarr):
        """
        Save to zarr. Dataset of this object will be saved to
        the specified zarr using the xarray.Dataset.to_zarr() method.

        Args:
            outzarr (string): output filename
            mode (string): Persistence mode. See the documentation for
                xarray.Dataset.to_zarr().
            group (string): the group name of data set. If not specified, 
              it will use the default group name of the class in cls.group.
            **args_to_zarr: other arguments for xarray.Dataset.to_zarr()
        """
        from xarray import open_zarr

        if group is None:
            groupname = self.group
        else:
            groupname = group

        self.ds.to_zarr(outfile, mode=mode, group=groupname, **args_to_zarr)
