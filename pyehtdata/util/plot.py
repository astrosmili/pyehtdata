#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This module describes data classes and related functions to handle UVFITS data.
'''
__author__ = "Smili Developer Team"

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt


def matplotlibrc(nrows=1, ncols=1, width=250, height=250):
    # Get this from LaTeX using \showthe\columnwidth
    fig_width_pt = width*ncols
    fig_height_pt = height*nrows
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_height_pt*inches_per_pt  # height in inches
    fig_size = [fig_width, fig_height]
    params = {'axes.labelsize': 13,
              'axes.titlesize': 13,
              'legend.fontsize': 14,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'figure.figsize': fig_size,
              'figure.dpi': 300
              }
    matplotlib.rcParams.update(params)


def reset_matplotlibrc():
    matplotlib.rcdefaults()


def get_angunitlabel(angunit=None):
    '''
    Get the angular unit of the specifed angunit. If not given,
    it will be taken by self.angunit
    '''
    # Axis Label
    if angunit.lower().find("pixel") == 0:
        unit = "pixel"
    elif angunit.lower().find("uas") == 0:
        unit = r"$\rm \mu$as"
    elif angunit.lower().find("mas") == 0:
        unit = "mas"
    elif angunit.lower().find("arcsec") * angunit.lower().find("asec") == 0:
        unit = "arcsec"
    elif angunit.lower().find("arcmin") * angunit.lower().find("amin") == 0:
        unit = "arcmin"
    elif angunit.lower().find("deg") == 0:
        unit = "deg"
    else:
        raise ValueError("Angular Unit '{}' is not available" % (angunit))
    return unit


def arrays_ellipse(xcen, ycen, Dmaj, Dmin, PA):
    """
    [summary]

    Args:
        Dmaj (float-like): Major-axis Diameter
        Dmin (float-lile): Minor-axis Diameter
        PA (float-like): Position Angle in radian

    Returns:
        [type]: [description]
    """
    from numpy import array, cos, sin, deg2rad, arange, dot
    theta = deg2rad(arange(0.0, 360.0, 1.0))
    x = 0.5 * Dmin * sin(theta)
    y = 0.5 * Dmaj * cos(theta)

    cosrt = cos(-PA)
    sinrt = sin(-PA)
    R = array([
        [cosrt, -sinrt],
        [sinrt,  cosrt],
    ])
    x, y = dot(R, array([x, y]))
    return x+xcen, y+ycen


def arrays_box(xcen, ycen, Lx, Ly):
    """
    [summary]

    Args:
        Lx ([type]): [description]
        Ly ([type]): [description]

    Returns:
        [type]: [description]
    """
    from numpy import array
    x = array([0, Lx, Lx, 0, 0]) - Lx/2.
    y = array([0, 0, Ly, Ly, 0]) - Ly/2.
    return x+xcen, y+ycen
