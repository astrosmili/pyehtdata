#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a sub-module of smili2 handling image data.
'''
__author__ = "Smili Developer Team"
from ...util.xarrds import XarrayDataset

# Logger
from logging import getLogger
logger = getLogger(__name__)


def plot_imshow(
        image,
        scale="linear",
        dyrange=100,
        gamma=0.5,
        vmax=None,
        vmin=None,
        relative=False,
        angunit=None,
        fluxunit="Jy",
        saunit="pixel",
        axisoff=False,
        axislabel=True,
        cmap="afmhot",
        idx=(0, 0, 0),
        interpolation="bilinear",
        **imshow_args):
    '''
    Plot the image.
    To change the angular unit, please change IMFITS.angunit.

    Args:
        scale(str; default="linear"):
            Transfar function. Availables are "linear", "log", "gamma"
        dyrange(float; default=100):
            Dynamic range of the log color contour.
        gamma(float; default=1/2.):
            Gamma parameter for scale = "gamma".
        vmax(float):
            The maximum value of the color contour.
        vmin(float):
            The minimum value of the color contour.
            If logscale = True, dyrange will be used to set vmin.
        relative(boolean, default=True):
            If True, vmin will be the relative value to the peak or vmax.
        fluxunit(string):
            Unit for the flux desity(Jy, mJy, uJy, K, si, cgs)
        saunit(string):
            Angular Unit for the solid angle(pixel, uas, mas, asec or arcsec,
            amin or arcmin, degree, beam).
        axisoff(boolean, default=False):
            If True, plotting without any axis label, ticks, and lines.
            This option is superior to the axislabel option.
        axislabel(boolean, default=True):
            If True, plotting the axislabel.
        colorbar(boolean, default=False):
            If True, the colorbar will be shown.
        colorbarprm(dic, default={}):
            parameters for pyplot.colorbar
        idx(integer):
            which image will be plotted.
        **imshow_args:
            Args will be input in matplotlib.pyplot.imshow
    '''
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, PowerNorm
    from numpy import where, abs, isnan, isinf
    import dask

    # Get angular unit
    if angunit is None:
        angunit = image.ds.attrs["angunit"]
    imextent = image.get_imextent(angunit)

    # Get images to be plotted
    if len(idx) != 3:
        raise ValueError("len(idx) should be 3 [i_time, i_freq, i_pol]")
    imarr = image.get_imarray(fluxunit=fluxunit, saunit=saunit)[idx]
    if isinstance(imarr, dask.array.core.Array):
        imarr = imarr.compute()
    imarr[isnan(imarr)] = 0
    imarr[isinf(imarr)] = 0

    if vmax is None:
        peak = imarr.max()
    else:
        peak = vmax

    if scale.lower() == "log":
        vmin = None
        norm = LogNorm(vmin=peak/dyrange, vmax=peak)
        imarr[where(imarr < peak/dyrange)] = peak/dyrange
    elif scale.lower() == "gamma":
        if vmin is not None and relative:
            vmin *= peak
        elif vmin is None:
            vmin = 0.
        norm = PowerNorm(vmin=vmin, vmax=peak, gamma=gamma)
        imarr[where(abs(imarr) < 0)] = 0
    elif scale.lower() == "linear":
        if vmin is not None and relative:
            vmin *= peak
        norm = None
    else:
        raise ValueError(
            "Invalid scale parameters. Available: 'linear', 'log', 'gamma'")

    im = plt.imshow(imarr, origin="lower", extent=imextent,
                    vmin=vmin, vmax=vmax,
                    cmap=cmap, interpolation=interpolation, norm=norm,
                    **imshow_args)

    # Axis Label
    if axislabel:
        plot_xylabel(angunit)

    # Axis off
    if axisoff:
        plt.axis("off")

    return im


def plot_xylabel(image_or_angunit, **labelargs):
    """
    Add xy label in the existing image plot.

    Args:
        image_or_angunit (imdata.Image or str):
            Angular Units.
        **labelargs:
            Arguments for pyplot.xlabel / ylable functions.
    """
    from ...util.plot import get_angunitlabel
    from matplotlib.pyplot import xlabel, ylabel
    from ..image import Image

    if isinstance(image_or_angunit, Image):
        angunit = image_or_angunit.angunit
    else:
        angunit = image_or_angunit
    angunitlabel = get_angunitlabel(angunit)

    xlabel("Relative RA (%s)" % (angunitlabel), **labelargs)
    ylabel("Relative Dec (%s)" % (angunitlabel), **labelargs)


def plot_beam(image, boxfc=None, boxec=None, beamfc=None, beamec="white",
              lw=1., alpha=0.5, x0=0.05, y0=0.05, boxsize=1.5, zorder=None):
    '''
    Plot beam in the header.
    To change the angular unit, please change IMFITS.angunit.

    Args:
        x0, y0(float, default=0.05):
            leftmost, lowermost location of the box
            if relative = True, the value is on transAxes coordinates
        relative(boolean, default=True):
            If True, the relative coordinate to the current axis will be
            used to plot data
        boxsize(float, default=1.5):
            Relative size of the box to the major axis size.
        boxfc, boxec(color formatter):
            Face and edge colors of the box
        beamfc, beamec(color formatter):
            Face and edge colors of the beam
        lw(float, default=1): linewidth
        alpha(float, default=0.5): transparency parameter (0 < 1)
                                    (0 < 1) for the face color
    '''
    from ...util.units import conv
    from ...util.plot import arrays_ellipse, arrays_box
    from matplotlib.pyplot import plot, fill, gca
    from numpy import max

    angunit = image.ds.attrs["angunit"]
    angconv = conv("rad", angunit)

    majsize = image.meta["bmaj"].val * angconv
    minsize = image.meta["bmin"].val * angconv
    pa = image.meta["bpa"].val

    offset = max([majsize, minsize])/2*boxsize

    # get the current axes
    ax = gca()

    # center
    xedge, yedge = ax.transData.inverted().transform(
        ax.transAxes.transform((x0, y0)))
    xcen = xedge - offset
    ycen = yedge + offset

    # get ellipce shapes
    xe, ye = arrays_ellipse(xcen, ycen, majsize, minsize, pa)

    xb, yb = arrays_box(xcen, ycen, offset*2, offset*2)

    # output dictionary
    outdic = dict(
        boxfill=None,
        beamfill=None,
        box=None,
        beam=None
    )

    # Box fill
    if boxfc is not None:
        outdic["boxfill"] = fill(xb, yb, fc=boxfc, alpha=alpha, zorder=zorder)
    # Beam fill
    if beamfc is not None:
        outdic["beamfill"] = fill(
            xe, ye, fc=beamfc, alpha=alpha, zorder=zorder)
    # Beam
    if beamec is not None:
        outdic["beam"] = plot(xe, ye, lw, color=beamec, zorder=zorder)
    # Box
    if boxec is not None:
        outdic["box"] = plot(xb, yb, lw, color=boxec, zorder=zorder)

    return outdic


def plot_scalebar(x, y, length, ha="center", color="white", lw=1, **plotargs):
    '''
    Plot a scale bar

    Args:
        x, y ( in the unit of the current plot):
            x, y coordinates of the scalebar
        length (in the unit of the current plot):
            length of the scale bar
        ha(str, default="center"):
            The horizontal alignment of the bar.
            Available options is ["center", "left", "right"]
        plotars:
            Arbital arguments for pyplot.plot.
    Returns:
        output from pyplot.plot
    '''
    from numpy import abs
    from matplotlib.pyplot import plot

    if ha.lower() == "center":
        xmin = x-abs(length)/2
        xmax = x+abs(length)/2
    elif ha.lower() == "left":
        xmin = x - abs(length)
        xmax = x
    elif ha.lower() == "right":
        xmin = x
        xmax = x + abs(length)
    else:
        raise ValueError("ha must be center, left or right")

    # plot
    return plot([xmax, xmin], [y, y], color=color, lw=lw, **plotargs)
