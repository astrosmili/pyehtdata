#!/usr/bin/env python
# -*- coding: utf-8 -*-


class UVData(object):
    # filename if data are loading lazily
    filename = "in memomry"

    # Meta data
    ant = None  # antenna meta data
    freq = None  # frequency meta data
    scan = None  # scan information
    src = None  # source information

    # visibility products
    vis = None  # visibilities
    bs = None  # bi-spectra
    ca = None  # closure amplitudes

    def __init__(self, filename=None, ant=None, freq=None, scan=None, src=None,
                 vis=None, bs=None, ca=None):

        if filename is not None:
            self.filename = filename

        if ant is not None:
            self.ant = ant.copy()

        if freq is not None:
            self.freq = freq.copy()

        if scan is not None:
            self.scan = scan.copy()

        if src is not None:
            self.src = src.copy()

        if vis is not None:
            self.vis = vis.copy()

        if bs is not None:
            self.vis = vis.copy()

        if ca is not None:
            self.vis = vis.copy()

    def __repr__(self):
        outlines = []
        outlines.append("filename: %s" % (self.filename))
        outlines.append("Attributes:")

        if self.ant is not None:
            outlines.append("  ant: antenna-based static metadata")

        if self.freq is not None:
            outlines.append("  freq: frequency setup")

        if self.scan is not None:
            outlines.append("  scan: scan table")

        if self.src is not None:
            outlines.append("  src: source information")

        if self.vis is not None:
            outlines.append("  vis: scan-segmented visibility data set")

        return "\n".join(outlines)

    @classmethod
    def load_zarr(cls, infile, group=""):
        from .io.zarr import zarr2UVData
        return zarr2UVData(infile, group=group)

    @classmethod
    def load_netcdf(cls, infile, group=""):
        from .io.netcdf import netcdf2UVData
        return netcdf2UVData(infile, group=group)

    def to_zarr(self, outfile, group="", mode="w"):
        from .io.zarr import UVData2zarr
        UVData2zarr(self, outfile, group=group, mode=mode)

    def to_netcdf(self, outfile, group="", mode="w"):
        from .io.netcdf import UVData2netcdf
        UVData2netcdf(self, outfile, group=group, mode=mode)
