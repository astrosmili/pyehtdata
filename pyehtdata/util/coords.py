#!/usr/bin/env python
# -*- coding: utf-8 -*-


def xyz2lstazelfra(x, y, z, utc, skycoord, fr_pa_coeff=1., fr_el_coeff=0., fr_offset=0.):
    """
    Compute LST, azimuth, elevation, parallactic and field rotation angles
    (in radian) from utc, source coordinates, and geocentric coodinates.

    Args:
        x, y, z (float like): geocentorial coordinates in meters
        utc (astropy.time.Time): utc time stamps. should be array like.
        skycoord (astropy.coordinates.SkyCoord): source cooridnate
        fr_pa_coeff, fr_el_coeff, fr_offset (float):
            Coefficients (dimensionless) and offset angle (in radian)
            used to compute the field rotation angle from the elevation
            and parallactic angles by the followng equation:
                fra = fr_pa_coeff * par + fr_el_coeff * el + fr_offset

    Returns:
        numpy array with the dimension of [5, Ntime]. The first indices are
        for lst in hour, az, el, par, and fra in radian
    """
    from numpy import asarray, cos, sin, tan, arctan2
    from astropy.coordinates import AltAz, EarthLocation, GCRS
    from .units import M, HOUR2RAD

    # Set antenna location
    location = EarthLocation(x=x, y=y, z=z, unit=M)
    lat = location.lat.radian

    # get lst
    lst = utc.sidereal_time(
        kind="apparent", longitude=location.lon, model="IAU2006A").hour

    # azimuth & elevation
    altaz = skycoord.transform_to(AltAz(location=location, obstime=utc))
    el = altaz.alt.radian
    az = altaz.az.radian

    # GCRS's RA and Dec
    gcrs = skycoord.transform_to(GCRS(obstime=utc))
    ra = gcrs.ra.radian
    dec = gcrs.dec.radian

    # compute pallactic angle
    H = lst*HOUR2RAD - ra
    cosH = cos(H)
    sinH = sin(H)
    tanlat = tan(lat)
    cosdec = cos(dec)
    sindec = sin(dec)
    par = arctan2(sinH, cosdec*tanlat - sindec*cosH)  # par in radian

    # compute field rotation angle
    fra = fr_pa_coeff * par + fr_el_coeff * el + fr_offset  # fra in radian

    return asarray([lst, az, el, par, fra])


def xyz2uvwsec(gst, ra, dec, x1, y1, z1, x2, y2, z2):
    """
    Convert from xyz to uvw

    Args:
        gst (float): gst in radians
        ra, dec (float; scaler or array): Right Ascention and Declination
        x1, y1, z1 (float array): geocentric coordinates of the antenna 1 in meters
        x2, y2, z2 (float array): geocentric coordinates of the antenna 2 in meters

    Returns:
        [type]: [description]
    """
    from astropy.constants import c
    from numpy import cos, sin

    # constants
    c = c.si.value

    # baseline vector
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2

    # cos, sin
    cosdec = cos(dec)
    sindec = sin(dec)
    cosGH = cos(gst - ra)
    sinGH = sin(gst - ra)

    # Earth-rotate baseline vector
    bl_x = cosGH * dx - sinGH * dy
    bl_y = sinGH * dx + cosGH * dy
    bl_z = dz

    # compute projections
    u = bl_y
    v = -bl_x * sindec + bl_z * cosdec
    w = +bl_x * cosdec + bl_z * sindec

    return u/c, v/c, w/c
