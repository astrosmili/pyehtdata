#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a sub-module of smili2 handling image data.
'''
__author__ = "Smili Developer Team"


def load_fits_ehtim(infits):
    """
    Load a FITS Image in ehtim's format into an imdata.Image instance.

    Args:
        infits (str or astropy.io.fits.HDUList):
            input FITS filename or HDUList instance

    Returns:
        imdata.Image: loaded image]
    """
    import astropy.io.fits as pf
    from numpy import abs
    from astropy.coordinates import SkyCoord
    from ....util.units import DEG
    from ..image import gen_image

    isfile = False
    if isinstance(infits, str):
        hdulist = pf.open(infits)
        isfile = True
    elif isinstance(infits, pf.HDUList):
        hdulist = infits.copy()

    # number of the Stokes Parameter
    ns = len(hdulist)

    # ra axis
    xdeg = hdulist[0].header["OBSRA"]
    nx = hdulist[0].header["NAXIS1"]
    dx = abs(hdulist[0].header["CDELT1"])
    ixref = hdulist[0].header["CRPIX1"]-1

    # dec axis
    ydeg = hdulist[0].header["OBSDEC"]
    ny = hdulist[0].header["NAXIS2"]
    dy = abs(hdulist[0].header["CDELT2"])
    iyref = hdulist[0].header["CRPIX2"]-1

    # stokes axis
    ns = len(hdulist)

    # time axis
    mjd = [hdulist[0].header["MJD"]]

    # frequency
    freq = [hdulist[0].header["FREQ"]]

    # source
    source = hdulist[0].header["OBJECT"]
    srccoord = SkyCoord(ra=xdeg*DEG, dec=ydeg*DEG)

    # telescope
    instrument = hdulist[0].header["TELESCOP"]

    outimage = gen_image(
        nx=nx, ny=ny,
        dx=dx, dy=dy,
        angunit="deg",
        ixref=ixref, iyref=iyref,
        mjd=mjd,
        freq=freq,
        ns=ns,
        source=source,
        srccoord=srccoord,
        instrument=instrument
    )

    #
    # Copy data from the fits hdu to the Image class instance
    #
    # outimage dims=["mjd", "freq", "stokes", "y", "x"]
    # fits hdu dims=["stokes", "freq", "y", "x"]
    #
    for istk in range(ns):
        outimage.ds.image[0, 0, istk] = hdulist[istk].data.copy()

    if isfile:
        hdulist.close()

    # update angunit
    outimage.set_angunit()

    return outimage


def save_image_to_fits_ehtim(image, outfits=None, overwrite=True, idx=(0, 0)):
    '''
    save the image(s) to the image FITS file or HDUList in the eht-imaging
    library's format

    Args:
        image (imdata.Image object):
            Input Image
        outfits (string; default is None):
            FITS file name. If not specified, then HDUList object will be
            returned.
        overwrite (boolean):
            It True, an existing file will be overwritten.
        idx (list):
            Index for (MJD, FREQ)
    Returns:
        HDUList object if outfits is None
    '''
    from astropy.io.fits import PrimaryHDU, ImageHDU, HDUList
    from ....util.units import conv

    if len(idx) != 2:
        raise ValueError(
            "idx must have the length of 2; should be index for (MJD, FREQ)")

    # Get the number of stokes parameters
    ns = image.ds.dims["stokes"]

    # Get the Image Array
    if len(idx) != 2:
        raise ValueError(
            "idx must be a two dimensional index for (mjd, freq)")
    else:
        imarr = image.ds.image.data[idx]
        imjd, ifreq = idx

    # Create HDUs
    #   Some conversion factor
    rad2deg = conv("rad", "deg")

    hdulist = []
    # current EHTIM format assumes each HDU / stokes parameter
    for ipol in range(ns):
        stokes = image.ds.coords["stokes"].data[ipol]

        if ipol == 0:
            hdu = PrimaryHDU(imarr[ipol])
        else:
            hdu = ImageHDU(imarr[ipol], name=stokes)

        # set header
        hdu.header.set("FITSGEN", "SMILI2")
        hdu.header.set("OBJECT", image.ds.attrs["source"])
        hdu.header.set("CTYPE1", "RA---SIN")
        hdu.header.set("CTYPE2", "DEC--SIN")
        hdu.header.set("CDELT1", -image.ds.attrs["dx"]*rad2deg)
        hdu.header.set("CDELT2", image.ds.attrs["dy"]*rad2deg)
        hdu.header.set("OBSRA", image.ds.attrs["x"]*rad2deg)
        hdu.header.set("OBSDEC", image.ds.attrs["y"]*rad2deg)
        hdu.header.set("FREQ", image.ds.coords["freq"].data[ifreq])
        hdu.header.set("CRPIX1", image.ds.attrs["ixref"]+1)
        hdu.header.set("CRPIX2", image.ds.attrs["iyref"]+1)
        hdu.header.set("MJD", image.ds.coords["mjd"].data[imjd])
        hdu.header.set("TELESCOP", image.ds.attrs["instrument"])
        hdu.header.set("BUNIT", "JY/PIXEL")
        hdu.header.set("STOKES", stokes)

        # appended to HDUList
        hdulist.append(hdu)

    # Convert the list of HDUs to HDUList
    hdulist = HDUList(hdulist)

    # return or write HDUList
    if outfits is None:
        return hdulist
    else:
        hdulist.writeto(outfits, overwrite=True)
