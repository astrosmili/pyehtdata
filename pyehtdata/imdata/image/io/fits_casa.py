#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a sub-module of smili2 handling image data.
'''
__author__ = "Smili Developer Team"

# Logger
from logging import getLogger
logger = getLogger(__name__)


def load_fits_casa(infits):
    """
    Load a FITS Image in CASA's format into an imdata.Image instance.
    Args:
        infits (str or astropy.io.fits.HDUList):
            input FITS filename or HDUList instance
    Returns:
        imdata.Image: loaded image
    """
    import astropy.io.fits as pf
    from numpy import abs, deg2rad, arange
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from ....util.units import DEG
    from ..image import gen_image

    # FITS file: name or hdulidt
    isfile = False
    if isinstance(infits, str):
        hdulist = pf.open(infits)
        isfile = True
    elif isinstance(infits, pf.HDUList):
        hdulist = infits.copy()

    hdu = hdulist[0]

    # for k, v in hdu.header.items():
    #     pass

    eqx = hdu.header["EQUINOX"]

    # ra axis
    if 'OBSRA' in hdu.header:
        xdeg = hdu.header["OBSRA"]
    else:
        xdeg = hdu.header["CRVAL1"]
    nx = hdu.header["NAXIS1"]
    dx = abs(deg2rad(hdu.header["CDELT1"]))
    ixref = hdu.header["CRPIX1"] - 1
    ra_prj = hdu.header["CTYPE1"]
    if ra_prj != 'RA---SIN':
        logger.warning("Projection CTYPE1='%s' is not RA---SIN." % ra_prj)

    # dec axis
    if 'OBSDEC' in hdu.header:
        ydeg = hdu.header["OBSDEC"]
    else:
        ydeg = hdu.header["CRVAL2"]
    ny = hdu.header["NAXIS2"]
    dy = abs(deg2rad(hdu.header["CDELT2"]))
    iyref = hdu.header["CRPIX2"] - 1
    dec_prj = hdu.header["CTYPE2"]
    if dec_prj != 'DEC--SIN':
        logger.warning("Projection CTYPE1='%s' is not DEC--SIN." % dec_prj)

    # frequency
    nfreq = hdu.header["NAXIS3"]
    fref = hdu.header["CRVAL3"]
    fdel = hdu.header["CDELT3"]
    ifref = hdu.header["CRPIX3"] - 1
    # funit = hdu.header["CUNIT3"]
    # freq = CRVAL3 + CDELT3*(np.arange(NAXIS3) - CRPIX3 + 1)
    freq = fref + fdel*(arange(nfreq) - ifref)

    # stokes axis
    nstk = hdu.header["NAXIS4"]

    # time axis
    isot = hdu.header["DATE-OBS"]  # In the ISO time format
    tim = Time(isot, format='isot', scale='utc')  # An astropy.time object
    mjd = [tim.mjd]  # Modified Julian Date

    # telescope
    instrument = hdu.header["TELESCOP"]

    # source
    source = hdu.header["OBJECT"]

    # get the source locaiton
    #   (1) RADESYS is the coordinate system
    if 'RADESYS' in hdu.header:
        coordsys = hdu.header['RADESYS'].lower()
    else:
        coordsys = "icrs"

    #   (2) Equinox for the coordinate system
    if 'EQUINOX' in hdu.header:
        equinox = Time(hdu.header['EQUINOX'], format="jyear")
    else:
        equinox = tim
        if coordsys != "icrs":
            logger.warning(
                "Input fits does not have a header information for equinox. Use the Date-OBS.")
    #   argument for skycoord
    scargs = dict(
        ra=xdeg*DEG,
        dec=ydeg*DEG,
        frame=coordsys
    )
    if "fk" in scargs["frame"]:
        scargs["equinox"] = equinox
    elif scargs["frame"] != "gcrs":
        scargs["obstime"] = equinox
    srccoord = SkyCoord(**scargs)

    img = gen_image(
        nx=nx, ny=ny,
        dx=dx, dy=dy, angunit="rad",
        ixref=ixref, iyref=iyref,
        mjd=mjd,
        freq=freq,
        ns=nstk,
        source=source,
        srccoord=srccoord,
        instrument=instrument
    )

    #
    # Copy data from the fits hdu to the Image class instance img
    #
    # img dims=["mjd",    "freq", "stokes", "y", "x"]
    # hdu dims=["stokes", "freq", "y", "x"]
    #
    img.ds.image.data[0, :] = hdu.data.swapaxes(0, 1)

    if isfile:
        hdulist.close()

    # update angunit
    img.set_angunit()

    return img

#
# File Exporters
#


def to_fits_casa(image, outfits=None, overwrite=True, imjd=0):
    '''
    Save the image(s) to the image FITS file or HDUList in the CASA format
    Args:
        outfits (string; default is None):
            FITS file name. If not specified, then HDUList object will be
            returned.
        overwrite (boolean):
            It True, an existing file will be overwritten.
        imjd (int):
            Index for MJD
            Only a image.data.data slice for specific time 'pixel'
            passed in imjd is saved in a FITS file
    Returns:
        HDUList object if outfits is None
    '''
    from astropy.io.fits import PrimaryHDU, HDUList
    from astropy.time import Time
    from ....util.units import conv

    #
    # img dims=["mjd", "freq", "stokes", "y", "x"]
    # hdu dims=["stokes", "freq", "y", "x"]
    #
    # Get a slice of the Image Array for the given imjd,
    # swapping the axes from [freq, stokes, :, :] to [stokes, freq, :, :]
    # to conform with the CASA HDU standard.
    #
    imarr = image.ds.image.data[imjd].swapaxes(0, 1)  # Change

    #
    # Get reference frequency fref and frequency increment
    # from the frequency array freq according to the formula
    # freq = fref + fdel*(arange(nfreq) - ifref)
    #
    freq = image.ds.freq.data
    nfreq = len(freq)
    fref = freq[0]
    fdel = (freq[1] - freq[0]) if nfreq > 1 else 1e9
    ifref = 1

    #
    # Create HDUs
    #
    #   Some conversion factor(s)
    rad2deg = conv("rad", "deg")

    hdulist = []

    hdu = PrimaryHDU(imarr)

    # set header
    hdu.header.set("OBJECT", image.ds.attrs["source"])

    hdu.header.set("CTYPE1", "RA---SIN")
    hdu.header.set("CRVAL1", image.ds.attrs["obsra"]*rad2deg)
    hdu.header.set("CDELT1", -image.ds.attrs["dx"]*rad2deg)
    hdu.header.set("CRPIX1", image.ds.attrs["ixref"]+1)
    hdu.header.set("CUNIT1", "deg")

    hdu.header.set("CTYPE2", "DEC--SIN")
    hdu.header.set("CRVAL2", image.ds.attrs["y"]*rad2deg)
    hdu.header.set("CDELT2", image.ds.attrs["dy"]*rad2deg)
    hdu.header.set("CRPIX2", image.ds.attrs["iyref"]+1)
    hdu.header.set("CUNIT2", "deg")

    hdu.header.set("CTYPE3", "FREQ")
    hdu.header.set("CRVAL3", fref)
    hdu.header.set("CDELT3", fdel)
    hdu.header.set("CRPIX3", float(ifref))
    hdu.header.set("CUNIT3", 'Hz')

    hdu.header.set("CTYPE4", "STOKES")
    hdu.header.set("CRVAL4", 1.0)
    hdu.header.set("CDELT4", 1.0)
    hdu.header.set("CRPIX4", 1.0)
    hdu.header.set("CUNIT4", '')

    hdu.header.set("OBSRA", image.ds.attrs["x"]*rad2deg)
    hdu.header.set("OBSDEC", image.ds.attrs["y"]*rad2deg)
    hdu.header.set("FREQ", freq[ifref-1])

    mjd = image.data["mjd"].data[imjd]
    tim = Time(mjd, format='mjd', scale='utc')  # Time object
    dt = tim.datetime64  # Same as tim.isot, but higher precision
    isot = str(dt)
    hdu.header.set("DATE-OBS", isot)
    hdu.header.set("MJD", mjd)

    hdu.header.set("TELESCOP", image.ds.attrs["instrument"])
    hdu.header.set("BUNIT", "JY/PIXEL")
#        hdu.header.set("STOKES", stokes)

    if 'coordsys' in image.ds.attrs:
        hdu.header.set('RADESYS', image.ds.attrs['coordsys'])

    if 'equinox' in image.ds.attrs:
        hdu.header.set('EQUINOX', image.ds.attrs['equinox'])

    # appended to HDUList
    hdulist.append(hdu)

    # Convert the list of HDUs to HDUList
    hdulist = HDUList(hdulist)

    # return or write HDUList
    if outfits is None:
        return hdulist
    else:
        hdulist.writeto(outfits, overwrite=True)
