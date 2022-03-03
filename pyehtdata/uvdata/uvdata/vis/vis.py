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
    name = "Visibility Dataset"

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
