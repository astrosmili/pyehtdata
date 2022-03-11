#!/usr/bin/env python
# -*- coding: utf-8 -*-


def zarr2UVData(infile, group=""):
    import zarr
    import os
    from .. import AntData, FreqData, ScanData, VisData, SrcData, UVData

    # Open zarr file
    z = zarr.open(infile, mode='r')  # note: no need to close
    groups = group.split("/")
    while "" in groups:
        groups.remove("")
    if len(groups) > 0:
        tmpds = z
        for tmpgrp in groups:
            tmpds = tmpds[tmpgrp]
        keys = list(tmpds.group_keys())  # get group keys
    else:
        keys = list(z.group_keys())

    inputs = dict()
    inputs["filename"] = infile
    if "antenna" in keys:
        inputs["ant"] = AntData.load_dataset(
            infile, engine="zarr", group=os.path.join(group, AntData.group))
    if "frequency" in keys:
        inputs["freq"] = FreqData.load_dataset(
            infile, engine="zarr", group=os.path.join(group, FreqData.group))
    if "scan" in keys:
        inputs["scan"] = ScanData.load_dataset(
            infile, engine="zarr", group=os.path.join(group, ScanData.group))
    if "source" in keys:
        inputs["src"] = SrcData.load_dataset(
            infile, engine="zarr", group=os.path.join(group, SrcData.group))
    if "visibility" in keys:
        inputs["vis"] = VisData.load_dataset(
            infile, engine="zarr", group=os.path.join(group, VisData.group))

    return UVData(**inputs)


def UVData2zarr(uvd, outfile, group="", mode="w"):
    import zarr
    import os

    if mode == "w":
        z = zarr.open(outfile, mode='w')

    if uvd.ant is not None:
        uvd.ant.to_zarr(outfile, group=os.path.join(
            group, uvd.ant.group), mode=mode)
    if uvd.scan is not None:
        uvd.scan.to_zarr(outfile, group=os.path.join(
            group, uvd.scan.group), mode=mode)
    if uvd.freq is not None:
        uvd.freq.to_zarr(outfile, group=os.path.join(
            group, uvd.freq.group), mode=mode)
    if uvd.src is not None:
        uvd.src.to_zarr(outfile, group=os.path.join(
            group, uvd.src.group), mode=mode)
    if uvd.vis is not None:
        uvd.vis.to_zarr(outfile, group=os.path.join(
            group, uvd.vis.group), mode=mode)
