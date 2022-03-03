#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
from ....util.xarrds import XarrayDataset


class FreqData(XarrayDataset):
    """
    Frequency Dataset:
    This class is storing Frequency Metadata for UV data sets
    mainly using xarray.Dataset
    """
    # Data type name
    name = "Frequency Dataset"

    # Group Name of zarr file
    group = "frequency"

    def get_shape(self):
        return (self.ds.spw_freq.size, self.ds.Nch)

    def recalc_freq(self):
        from numpy import zeros, arange

        # reset index
        ds = self.ds

        # get the number of spw and ch
        Nspw, Nch = self.get_shape()

        # create an array
        freqarr = zeros([Nspw, Nch], dtype="float64")
        chidarr = arange(Nch)

        # compute frequency
        for ispw in range(Nspw):
            spw_freq = ds.spw_freq.data[ispw]
            sideband = ds.sideband.data[ispw]
            ch_bw = ds.ch_bw.data[ispw]
            freqarr[ispw] = spw_freq + sideband * ch_bw * chidarr

        ds["freq"] = (["spw", "ch"], freqarr)
