#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
from ....util.xarrds import XarrayDataset


class SrcData(XarrayDataset):
    """
    Source Dataset:
    This class is storing Source Metadata for UV data sets
    mainly using xarray.Dataset
    """
    # Data type name
    name = "Source Dataset"

    # Group Name of zarr file
    group = "source"
