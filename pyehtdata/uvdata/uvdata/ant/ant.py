#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
from ....util.xarrds import XarrayDataset


class AntData(XarrayDataset):
    """
    Antenna-based Dataset:
    This class is storing Antenna Metadata for UV data sets
    mainly using xarray.Dataset
    """
    # Data type name
    datatype = "ehtdata_antenna"

    # Group Name of zarr file
    group = "antenna"

    # Mandatory Coordinates, dimensions, default values
    coords_mandatory = dict(
        antname=(["ant"], "ant", "Antenna Name"),
        anttype=(["ant"], "g", "Antenna type ('g'round or 's'pace)"),
        x=(["ant"], 0., "Geocentric coordinates in m"),
        y=(["ant"], 0., "Geocentric coordinates in m"),
        z=(["ant"], 0., "Geocentric coordinates in m"),
        el_max=(["ant"], 90., "Maximum elevation in degree"),
        el_min=(["ant"], 0., "Maximum elevation in degree"),
        fr_pa_coeff=(
            ["ant"], 1., "Coeffcient for the parallactic angle to compute the field rotation angle."),
        fr_el_coeff=(
            ["ant"], 0., "Coeffcient for the elevation angle to compute the field rotation angle."),
        fr_offset=(
            ["ant"], 0., "Offset angle in degree to compute the field rotation angle."),
        sefd0=(["ant", "pol"], 1000., "Zenith SEFD in Jy"),
        tau0=(["ant"], 0., "Zenith opacity"),
        pol=(["pol"], ["R", "L"], "Polarization"),
    )

    def init_coords(self):
        from xarray import Dataset
        from numpy import full

        hasds = False
        if hasattr(self, "ds"):
            if self.ds is not None:
                hasds = True

        if not hasds:
            self.ds = Dataset()

        coordkeys = self.ds.coords.keys()
        for key in self.coords_mandatory.keys():
            if key not in coordkeys:
                coordshape = [
                    self.ds.dims[dimkey] for dimkey in self.coords_mandatory[key][0]
                ]

                coordvals = full(
                    shape=coordshape,
                    fill_value=self.coords_mandatory[key][1]
                )

                self.ds.coords[key] = (
                    self.coords_mandatory[key][0],
                    coordvals
                )
