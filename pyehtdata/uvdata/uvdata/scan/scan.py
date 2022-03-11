#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
from ....util.xarrds import XarrayDataset
from ....util.units import DAY, MIN, SEC

# Logger
from logging import getLogger
logger = getLogger(__name__)


class ScanData(XarrayDataset):
    """
    Scan Dataset:
    This class is storing Scan Metadata for UV data sets
    mainly using xarray.Dataset
    """
    # Data type name
    datatype = "ehtdata_scan"

    # Group Name of zarr file
    group = "scan"

    def segment_scans(self, tap=10*SEC):
        """
        Segment scans by a given accumulation period. Generate a pandas
        DataFrame storeing mjd, dmjd (accumulation period) and scanid.

        Args:
            tap (float or astropy.Quantity, optional):
                Accumulation Period. Negative gives timestamps for scan-averaged data.
                Defaults to 10 seconds. If float is specifed, the unit is
                assumuted to be seconds.

        Returns:
            timetable (pandas.DataFrame): timetable storing mjd, dmjd and scanid
        """
        from pandas import concat
        from astropy.units import Quantity

        Nscan = self.ds.dims["scan"]
        mjdst = self.ds.mjdst.values
        mjded = self.ds.mjded.values

        if tap > 0:
            if isinstance(tap, Quantity):
                tap_d = tap.to_value(DAY)
            else:
                tap_d = tap/86400.
        else:
            tap_d = tap

        def gen_timetable(mjdst, mjded, tap, scanid):
            from numpy import arange
            from pandas import DataFrame

            # scan length
            tscan = mjded - mjdst

            # set ap
            if tap <= 0:
                tap_tmp = tscan
            else:
                if tap > tscan:
                    logger.warning("Scan {0:d}: tap is larger than the scan length. Use scan length for tap.".format(
                        scanid))
                    tap_tmp = tscan
                else:
                    tap_tmp = tap

            tab = DataFrame()
            tab["mjd"] = mjdst + arange(tap_tmp/2, tscan, tap_tmp)
            Nseg = len(tab)
            tab["dmjd"] = [tap_tmp for i in range(Nseg)]
            tab["scanid"] = [scanid for i in range(Nseg)]
            return tab

        timetab = concat([gen_timetable(mjdst=mjdst[isc], mjded=mjded[isc],
                                        tap=tap_d, scanid=isc) for isc in range(Nscan)])

        return timetab

    @classmethod
    def from_pattern(cls, tstart, tend, tscan=6*MIN, tint=20*MIN):
        '''
        generate a scandata based on input scan patterns

        Args:
            tstart (astropy.time.Time): The start time of the observations
            tend (astropy.time.Time): The end time of the observations
            tscan (astropy.units.Quantity): Scan length (default=6 minutes)
            tint (astropy.units.Quantity): Scan interval (default=20 minutes)
        Returns:
            uvdata.ScanData instance
        '''
        from numpy import arange
        from xarray import Dataset

        # total window
        ttot = tend - tstart

        # convert units
        ttot_jd = ttot.jd
        tint_jd = tint.to_value(unit=DAY)
        tscn_jd = tscan.to_value(unit=DAY)

        # scan time
        tscan_st_jd = arange(0, ttot_jd, tscn_jd+tint_jd)
        mjdst = tstart.mjd + tscan_st_jd
        mjded = mjdst + tscn_jd

        ds = Dataset(
            coords=dict(
                mjdst=("scan", mjdst),
                mjded=("scan", mjded)
            )
        )

        return cls(ds=ds)

    def plot_scan(self, timeunit="utchour", **plotargs):
        """
        Plot scan id as a function of time

        Args:
            timeunit (str, optional):
                time unit for x-axis.
                available units: utchour, utcday, mjd
            **plotargs: arguments for pyplot.plot used in this function
        """
        from numpy import floor
        import matplotlib.pyplot as plt

        Nscan = self.ds.dims["scan"]
        mjdst = self.ds.mjdst.values
        mjded = self.ds.mjded.values

        if "utch" in timeunit.lower():
            mjdref = floor(mjdst.min())
            xst = (mjdst - mjdref) * 24
            xed = (mjded - mjdref) * 24
            xlabel = "UTC (hour) from MJD=%d" % (int(mjdref))
        elif "utcd" in timeunit.lower():
            mjdref = floor(mjdst.min())
            xst = mjdst - mjdref
            xed = mjded - mjdref
            xlabel = "UTC (day) from MJD=%d" % (int(mjdref))
        elif "mjd" in timeunit.lower():
            xst = mjdst.copy()
            xed = mjded.copy()
            xlabel = "MJD"

        for i in range(Nscan):
            plt.plot([xst[i], xed[i]], [i, i], **plotargs)
        plt.xlabel(xlabel)
        plt.ylabel("Scan ID")
