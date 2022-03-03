#!/usr/bin/env python
# -*- coding: utf-8 -*-


class UVData(object):
    # zarr file
    zarrfile = None

    # Meta data
    ant = None  # antenna meta data
    freq = None  # frequency meta data
    scan = None  # scan information
    src = None  # source information

    # visibility products
    vis = None  # visibilities
    bs = None  # bi-spectra
    ca = None  # closure amplitudes

    def __init__(self, zarrfile, ant=None, freq=None, scan=None, src=None,
                 vis=None, bs=None, ca=None):

        self.zarrfile = zarrfile

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
        outlines.append("zarr file: %s" % (self.zarrfile))
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
    def load_zarr(cls, inzarr):
        from .io.zarr import zarr2UVData
        return zarr2UVData(inzarr)

    def to_zarr(self, outzarr):
        from .io.zarr import UVData2zarr
        UVData2zarr(self, outzarr)
