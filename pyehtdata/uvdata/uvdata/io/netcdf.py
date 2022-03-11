#!/usr/bin/env python
# -*- coding: utf-8 -*-


def netcdf2UVData(infile, group=""):
    import os
    import netCDF4
    from .. import AntData, FreqData, ScanData, VisData, SrcData, UVData

    # Open zarr file and check groups
    ds = netCDF4.Dataset(infile, mode="r")

    groups = group.split("/")
    while "" in groups:
        groups.remove("")
    if len(groups) > 0:
        tmpds = ds
        for tmpgrp in groups:
            tmpds = tmpds.groups[tmpgrp]
        keys = tmpds.groups.keys()
    else:
        keys = ds.groups.keys()

    inputs = dict()
    inputs["filename"] = infile
    if "antenna" in keys:
        inputs["ant"] = AntData.load_dataset(
            infile, group=os.path.join(group, AntData.group))
    if "frequency" in keys:
        inputs["freq"] = FreqData.load_dataset(
            infile, group=os.path.join(group, FreqData.group))
    if "scan" in keys:
        inputs["scan"] = ScanData.load_dataset(
            infile, group=os.path.join(group, ScanData.group))
    if "source" in keys:
        inputs["src"] = SrcData.load_dataset(
            infile, group=os.path.join(group, SrcData.group))
    if "visibility" in keys:
        inputs["vis"] = VisData.load_dataset(
            infile, group=os.path.join(group, VisData.group))

    return UVData(**inputs)


def UVData2netcdf(uvd, outfile, group="", mode="w"):
    import os
    curmode = mode
    if uvd.ant is not None:
        uvd.ant.to_netcdf(outfile, group=os.path.join(
            group, uvd.ant.group), mode=curmode)
        curmode = "a"
    if uvd.scan is not None:
        uvd.scan.to_netcdf(outfile, group=os.path.join(
            group, uvd.scan.group), mode=curmode)
        curmode = "a"
    if uvd.freq is not None:
        uvd.freq.to_netcdf(outfile, group=os.path.join(
            group, uvd.freq.group), mode=curmode)
        curmode = "a"
    if uvd.src is not None:
        uvd.src.to_netcdf(outfile, group=os.path.join(
            group, uvd.src.group), mode=curmode)
        curmode = "a"
    if uvd.vis is not None:
        uvd.vis.to_netcdf(outfile, group=os.path.join(
            group, uvd.vis.group), mode=curmode)
        curmode = "a"
