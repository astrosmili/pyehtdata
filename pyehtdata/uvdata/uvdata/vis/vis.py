#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
from ....util.xarrds import XarrayDataset


class VisData(XarrayDataset):
    """
    Visibility Data:
    This class is storing visibilities with dimension compatible with uvfits.
    """
    # Data type name
    datatype = "ehtdata_visibility"

    # Group Name of zarr file
    group = "visibility"

    def set_scan(self, scangap=None, nseg=2.):
        """
        Detect scans based on the specified scan gap.

        Args:
            scangap (float or astropy.units.Quantity, optional):
                Minimal time seperation between scans.
                If not specfied, this will be guessed from data segmentation (see nseg).
                If a float value is specified, its unit is assumuted to be in seconds.
                Defaults to None.
            nseg (float, optional):
                If scangap is None, the minimal time seperation between scans
                will be set to nseg * minimal_data_segementation_time.
                Defaults to 2.
        """
        from astropy.units import Quantity
        from ....util.units import DAY
        from numpy import unique, diff, median, int64, where, hstack, zeros

        # Get a non-redundant set of MJD
        mjdset, mjdinv = unique(
            self.ds.mjd, return_index=False, return_inverse=True, return_counts=False)

        # Take difference
        mjddiff = diff(mjdset)
        scanid = zeros(mjddiff.size)

        # Get scangap in day
        if not scangap:
            scangap_day = median(mjddiff)*nseg
        elif isinstance(scangap, Quantity):
            scangap_day = scangap.to(DAY).value
        else:
            scangap_day = scangap / 86400.

        # Detect Scan
        scanid[where(mjddiff <= scangap_day)] = 0
        scanid[where(mjddiff > scangap_day)] = 1
        scanid = int64(hstack((0, scanid)).cumsum())
        self.ds.coords["scanid"] = ("data", scanid[mjdinv])

    def gen_scandata(self):
        """
        Generate a scan dataset based on the scan ID in Table
        """
        from numpy import zeros, int64, float64
        from xarray import Dataset
        from ..scan import ScanData

        if "scanid" not in self.ds.coords.keys():
            raise ValueError(
                "scan ID is not set yet. Please run set_scan method first.")

        scangroup = self.ds.groupby("scanid")
        Nscan = len(scangroup)
        scanid = zeros(Nscan, dtype=int64)
        mjdst = zeros(Nscan, dtype=float64)
        dmjdst = zeros(Nscan, dtype=float64)
        mjded = zeros(Nscan, dtype=float64)
        dmjded = zeros(Nscan, dtype=float64)
        for scan in scangroup:
            idx = scan[0]
            ds = scan[1]
            scanid[idx] = idx
            mjdst[idx] = ds.mjd.data[0]-ds.dmjd.data[0]/2.
            mjded[idx] = ds.mjd.data[-1]+ds.dmjd.data[-1]/2.

        ds = Dataset(
            coords=dict(
                mjdst=("scan", mjdst),
                mjded=("scan", mjded)
            )
        )
        return ScanData(ds=ds)

    @classmethod
    def load_dataset(cls, infile, engine="netcdf4", group=None, chunks="auto", **args_open_dataset):
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
                          engine=engine, chunks=chunks, **args_open_dataset)

        if engine in ["netcdf4", "h5netcdf"]:
            return ds.isel(ReIm=0) + 1j * ds.isel(ReIm=1)

        return cls(ds=ds)

    def to_netcdf(self, outfile, mode="w", group=None, engine="netcdf4", **args_to_netcdf):
        """
        Save to netcdf. Dataset of this object will be saved to
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
        from xarray import concat
        import os

        if engine not in ["netcdf4", "h5netcdf"]:
            raise ValueError("available engines: 'netcdf4', 'h5netcdf'")

        if group is None:
            groupname = self.group
        else:
            groupname = group

        if mode == "w" and os.path.isfile(outfile):
            os.system("rm -f {0}".format(outfile))

        ds = self.ds.expand_dims('ReIm', axis=-1)  # Add ReIm axis at the end
        ds = concat([ds.real, ds.imag], dim='ReIm')
        return ds.to_netcdf(outfile, mode=mode, engine=engine,
                            group=groupname, format="NETCDF4", **args_to_netcdf)
