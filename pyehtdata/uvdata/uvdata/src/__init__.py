#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
from .src import SrcData


def zarr2UVData():
    import zarr
    from ... import AntData, FreqData, ScanData, VisTable, SrcData

    inzarr = 'test.zarr'

    z = zarr.open(inzarr, mode='r')
    keys = list(z.group_keys())

    inputs = dict()
    if "antenna" in keys:
        inputs["ant"] = AntData.load_zarr(inzarr)
    if "frequency" in keys:
        inputs["freq"] = FreqData.load_zarr(inzarr)
    if "scan" in keys:
        inputs["scan"] = ScanData.load_zarr(inzarr)
    if "source" in keys:
        inputs["src"] = SrcData.load_zarr(inzarr)
    if "visibility_table" in keys:
        inputs["vistab"] = VisTable.load_zarr(inzarr)
