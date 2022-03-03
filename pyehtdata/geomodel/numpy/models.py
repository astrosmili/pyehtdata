#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule of smili.geomodel, containing functions to
calculate visibilities and images of some geometric models.

Here, we note that the whole design of this module is imspired by
Lindy Blackburn's python module 'modmod', and
we would like to thank Lindy to share his idea.
'''
from ..util.units import conv
from .geomodel import GeoModel


def Gaussian(x0=0, y0=0, totalflux=1, majsize=1, minsize=None, pa=0, dx=1., dy=None, angunit="mas"):
    '''
    Create geomodel.geomodel.GeoModel Object for the specified Gaussian
    Args:
        totalflux (float, optional): Total flux density of the model. Defaults to 1.
        x0, y0 (float, optional): The position of the centorid. Defaults to 0.
        majsize, minsize (float, optional): Major/Minor-axis FWHM size of the Gaussian.
        pa (float, optional): Position Angle of the Gaussian in degree.
        dx, dy (float, optional): The pixel size of the image. Defaults to 1.

        angunit (str, optional): The angular unit. Defaults to "mas".
        x0=0.0, y0=0.0: position of the centorid
        totalflux=1.0: total flux of the Gaussian
        majsize, minsize: Major/Minor-axis FWHM size of the Gaussian
        pa: Position Angle of the Gaussian in degree
        angunit="mas": angular unit of x0, y0, majsize, minsize (uas, mas, asec, amin, deg, rad)
    Returns:
    Returns:
        geomodel.geomodel.GeoModel Object for the specified Gaussian
    '''
    from numpy import sqrt, exp, log, pi

    if minsize is None:
        minsize = majsize

    if dy is None:
        dy = dx

    # convert angular units
    factor = conv(angunit, "rad")
    dxrad = dx * factor
    dyrad = dy * factor

    # define a Gaussian with F=1 jy, size = 1 (angunit)
    sigma = factor/sqrt(8*log(2))

    def V(u, v):
        return exp(-2*pi*pi*(u*u+v*v)*sigma*sigma)

    def I(x, y):
        return dxrad*dyrad/2/pi/sigma/sigma*exp(-(x*x+y*y)/2/sigma/sigma)

    output = GeoModel(V=V, I=I)

    # transform Gaussian, so that it will be elliptical Gaussian
    if totalflux != 1:
        output = output * totalflux
    if majsize != 1 or minsize != 1:
        output = output.scale(hx=minsize, hy=majsize)
    if pa != 0:
        output = output.rotate(deltaPA=pa, deg=True)
    if x0 != 0 or y0 != 0:
        output = output.shift(deltax=x0, deltay=y0, angunit=angunit)
    return output


def Rectangular(totalflux=1., x0=0., y0=0., Lx=1., Ly=None, dx=1., dy=None, angunit="mas"):
    """
    Create a geomodel.GeoModel instance for the specified RectAngular function.

    Args:
        totalflux (float, optional): Total flux density of the model. Defaults to 1.
        x0, y0 (float, optional): The position of the centorid. Defaults to 0.
        Lx, Ly (float, optional): The length of the function. Defaults to 1.
        dx, dy (float, optional): The pixel size of the image. Defaults to 1.
        angunit (str, optional): The angular unit. Defaults to "mas".

    Returns:
        [type]: [description]
    """
    '''
    Create geomodel.geomodel.GeoModel Object for the specified RectAngular function
    Args:
        x0=0.0, y0=0.0: position of the centorid
        totalflux=1.0: total flux of the model in Jy
        Lx, Ly: the size of the rectangular function.
        dx, dy: the pixel size of the image
        angunit="mas": angular unit of x0, y0, dx, dy (uas, mas, asec, amin, deg, rad)
    Returns:
        geomodel.GeoModel: The instance of the specified RectAngular model
    '''
    from numpy import where, abs, sinc, pi, exp, isnan
    from ..util.units import conv

    if dy is None:
        dy = dx
    if Ly is None:
        Ly = Lx

    # convert angular units
    factor = conv(angunit, "rad")

    x0rad = x0 * factor
    y0rad = y0 * factor
    Lxrad = abs(factor*Lx)
    Lyrad = abs(factor*Ly)
    dxrad = abs(factor*dx)
    dyrad = abs(factor*dy)

    # get the mean intensity = Total flux / (the number of pixels in the rect angular)
    Imean = totalflux / Lxrad / Lyrad * dxrad * dyrad

    def I(x, y):
        xnorm = abs((x-x0rad)/Lxrad)
        ynorm = abs((y-y0rad)/Lyrad)
        return where(xnorm <= 0.5, 1, 0)*where(ynorm <= 0.5, 1, 0)*Imean

    def V(u, v):
        unorm = Lxrad*u
        vnorm = Lyrad*v
        amp = totalflux * sinc(unorm)*sinc(vnorm)
        phase = 2*pi*(u*x0rad+v*y0rad)
        return amp*exp(1j*phase)

    return GeoModel(V=V, I=I)


def Triangular(totalflux=1, x0=0, y0=0, dx=1, dy=None, angunit="mas"):
    '''
    Create geomodel.geomodel.GeoModel Object for the specified Triangular function
    Args:
        x0=0.0, y0=0.0: position of the centroid
        totalflux=1.0: total flux of the Gaussian
        dx, dy: the size of the Triangular function
        angunit="mas": angular unit of x0, y0, dx, dy (uas, mas, asec, amin, deg, rad)
    Returns:
        geomodel.geomodel.GeoModel Object for the specified Triangular function
    '''
    from numpy import where, abs, sinc, pi, square
    from ..util.units import conv

    # define a Gaussian with F=1 jy, size = 1 (angunit)
    factor = conv(angunit, "rad")
    x0rad = x0 * factor
    y0rad = y0 * factor
    dxrad = abs(factor*dx)
    dyrad = abs(factor*dy)
    dxyinv = 1./dxrad/dyrad

    def I(x, y):
        xnorm = abs(x-x0rad/dxrad)
        ynorm = abs(y-y0rad/dyrad)
        return totalflux*dxyinv*where(xnorm <= 1, (1-xnorm)/dxrad, 0)*where(ynorm <= 1, (1-ynorm)/dyrad, 0)

    def V(u, v):
        unorm = dxrad*u
        vnorm = dyrad*v
        return totalflux*square(sinc(unorm)*sinc(vnorm))

    output = GeoModel(V=V, I=I)
    if x0 != 0 or y0 != 0:
        output = output.shift(deltax=x0, deltay=y0, angunit=angunit)
    return output
