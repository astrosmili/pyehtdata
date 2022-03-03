#!/usr/bin/env python
# -*- coding: utf-8 -*-


def zarr2UVData(inzarr):
    import zarr
    from .. import AntData, FreqData, ScanData, VisData, SrcData, UVData

    # Open zarr file
    z = zarr.open(inzarr, mode='r')  # note: no need to close
    keys = list(z.group_keys())  # get group keys

    inputs = dict()
    inputs["zarrfile"] = inzarr
    if "antenna" in keys:
        inputs["ant"] = AntData.load_zarr(inzarr)
    if "frequency" in keys:
        inputs["freq"] = FreqData.load_zarr(inzarr)
    if "scan" in keys:
        inputs["scan"] = ScanData.load_zarr(inzarr)
    if "source" in keys:
        inputs["src"] = SrcData.load_zarr(inzarr)
    if "visibility" in keys:
        inputs["vis"] = VisData.load_zarr(inzarr)

    return UVData(**inputs)


def UVData2zarr(uvd, outzarr):
    import zarr

    # Open zarr file
    z = zarr.open(outzarr, mode='w')  # note: no need to close
    keys = list(z.group_keys())  # get group keys

    if uvd.ant is not None:
        uvd.ant.to_zarr(outzarr)
    if uvd.scan is not None:
        uvd.scan.to_zarr(outzarr)
    if uvd.freq is not None:
        uvd.freq.to_zarr(outzarr)
    if uvd.src is not None:
        uvd.src.to_zarr(outzarr)
    if uvd.vis is not None:
        uvd.vis.to_zarr(outzarr)
