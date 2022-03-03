#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule of smili.geomodel, containing functions to
calculate visibilities and images of some geometric models.

Here, we note that the whole design of this module is imspired by
Lindy Blackburn's python module 'modmod', and
we would like to thank Lindy to share his idea.
'''


class GeoModel(object):
    @staticmethod
    def V(u, v): return (u-u)*(v-v)

    @staticmethod
    def I(x, y): return (x-x)*(y-y)

    def __init__(self, V=None, I=None):
        if V is None:
            self.V = lambda u, v: (u-u)*(v-v)
        else:
            self.V = V

        if I is None:
            self.I = lambda x, y: (x-x)*(y-y)
        else:
            self.I = I

    def __add__(self, other):
        if type(self) == type(other):
            def V(u, v): return self.V(u, v) + other.V(u, v)
            def I(x, y): return self.I(x, y) + other.I(x, y)
            return GeoModel(V=V, I=I)
        else:
            raise ValueError(
                "Addition can be calculated only between the same type of objects")

    def __iadd__(self, other):
        if type(self) == type(other):
            def V(u, v): return self.V(u, v) + other.V(u, v)
            def I(x, y): return self.I(x, y) + other.I(x, y)
            return GeoModel(V=V, I=I)
        else:
            raise ValueError(
                "Addition can be calculated only between the same type of objects")

    def __sub__(self, other):
        if type(self) == type(other):
            def V(u, v): return self.V(u, v) - other.V(u, v)
            def I(x, y): return self.I(x, y) - other.I(x, y)
            return GeoModel(V=V, I=I)
        else:
            raise ValueError(
                "Subtraction can be calculated only between the same type of objects")

    def __isub__(self, other):
        if type(self) == type(other):
            def V(u, v): return self.V(u, v) - other.V(u, v)
            def I(x, y): return self.I(x, y) - other.I(x, y)
            return GeoModel(V=V, I=I)
        else:
            raise ValueError(
                "Subtraction can be calculated only between the same type of objects")

    def __mul__(self, other):
        def V(u, v): return self.V(u, v) * other
        def I(x, y): return self.I(x, y) * other
        return GeoModel(V=V, I=I)

    def __imul__(self, other):
        def V(u, v): return self.V(u, v) * other
        def I(x, y): return self.I(x, y) * other
        return GeoModel(V=V, I=I)

    def __truediv__(self, other):
        def V(u, v): return self.V(u, v) / other
        def I(x, y): return self.I(x, y) / other
        return GeoModel(V=V, I=I)

    def __itruediv__(self, other):
        def V(u, v): return self.V(u, v) / other
        def I(x, y): return self.I(x, y) / other
        return GeoModel(V=V, I=I)

    def shift(self, deltax=0., deltay=0., angunit="mas"):
        from numpy import pi, exp
        from ..util.units import conv
        angunit2rad = conv(angunit, "rad")
        dx = deltax * angunit2rad
        dy = deltay * angunit2rad
        def V(u, v): return self.V(u, v) * exp(1j*2*pi*(u*dx+v*dy))
        def I(x, y): return self.I(x-dx, y-dy)
        return GeoModel(V=V, I=I)

    def rotate(self, deltaPA=0., deg=True):
        from numpy import pi, cos, sin
        if deg:
            dPA = deltaPA * pi / 180
        else:
            dPA = deltaPA
        cosdpa = cos(dPA)
        sindpa = sin(dPA)
        def x1(x, y): return x * cosdpa - y * sindpa
        def y1(x, y): return x * sindpa + y * cosdpa
        def V(u, v): return self.V(x1(u, v), y1(u, v))
        def I(x, y): return self.I(x1(x, y), y1(x, y))
        return GeoModel(V=V, I=I)

    def scale(self, hx=1., hy=None):
        if hy is None:
            hy = hx

        def V(u, v): return self.V(u*hx, v*hy)
        def I(x, y): return self.I(x/hx, y/hy)/hx/hy
        return GeoModel(V=V, I=I)

    def Vre(self, u, v):
        from numpy import real
        return real(self.V(u, v))

    def Vim(self, u, v):
        from numpy import imag
        return imag(self.V(u, v))

    def Vamp(self, u, v):
        '''
        Return theano symbolic represenation of the visibility amplitude

        Args:
            u, v: uv-coordinates
        Return:
            amplitude
        '''
        from numpy import abs
        return abs(self.V(u, v))

    def logVamp(self, u, v):
        '''
        Return theano symbolic represenation of the visibility amplitude

        Args:
            u, v: uv-coordinates
        Return:
            amplitude
        '''
        from numpy import log
        return log(self.Vamp(u, v))

    def Vphase(self, u, v):
        '''
        Return theano symbolic represenation of the visibility phase

        Args:
            u, v: uv-coordinates
        Return:
            phase in radian
        '''
        from numpy import angle
        return angle(self.V(u, v))

    # Bi-spectrum
    def B(self, u1, v1, u2, v2, u3, v3):
        '''
        Return theano symbolic represenation of the real part of the Bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            real part of the bi-spectrum
        '''
        V1 = self.V(u1, v1)
        V2 = self.V(u2, v2)
        V3 = self.V(u3, v3)
        return V1*V2*V3

    def Bre(self, u1, v1, u2, v2, u3, v3):
        '''
        Return theano symbolic represenation of the imaginary part of the Bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            imaginary part of the bi-spectrum
        '''
        from numpy import real
        return real(self.B(u1, v1, u2, v2, u3, v3))

    def Bim(self, u1, v1, u2, v2, u3, v3):
        '''
        Return theano symbolic represenation of the imaginary part of the Bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            imaginary part of the bi-spectrum
        '''
        from numpy import imag
        return imag(self.B(u1, v1, u2, v2, u3, v3))

    def Bamp(self, u1, v1, u2, v2, u3, v3):
        '''
        Return theano symbolic represenation of the amplitude of the Bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            amplitude of the bi-spectrum
        '''
        from numpy import abs
        return abs(self.B(u1, v1, u2, v2, u3, v3))

    def Bphase(self, u1, v1, u2, v2, u3, v3):
        '''
        Return theano symbolic represenation of the phase of the Bi-spectrum.
        if given uv-coodinates are closed, this will be the closure phase.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            phase of the bi-spectrum
        '''
        from numpy import angle
        return angle(self.B(u1, v1, u2, v2, u3, v3))

    # Closure Amplitudes
    def Camp(self, u1, v1, u2, v2, u3, v3, u4, v4):
        '''
        Return theano symbolic represenation of the phase of the Bi-spectrum.
        if given uv-coodinates are closed, this will be the closure phase.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            phase of the bi-spectrum
        '''
        Vamp1 = self.Vamp(u1, v1)
        Vamp2 = self.Vamp(u2, v2)
        Vamp3 = self.Vamp(u3, v3)
        Vamp4 = self.Vamp(u4, v4)
        return Vamp1*Vamp2/Vamp3/Vamp4

    def logCamp(self, u1, v1, u2, v2, u3, v3, u4, v4):
        '''
        Return theano symbolic represenation of the phase of the Bi-spectrum.
        if given uv-coodinates are closed, this will be the closure phase.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            phase of the bi-spectrum
        '''
        logVamp1 = self.logVamp(u1, v1)
        logVamp2 = self.logVamp(u2, v2)
        logVamp3 = self.logVamp(u3, v3)
        logVamp4 = self.logVamp(u4, v4)
        return logVamp1+logVamp2-logVamp3-logVamp4


# -------------------------------------------------------------------------------
# Some calculations
# -------------------------------------------------------------------------------
def dphase(phase1, phase2):
    from numpy import arctan2, cos, sin
    dphase = phase2 - phase1
    return arctan2(sin(dphase), cos(dphase))

# -------------------------------------------------------------------------------
# Phase for symmetric sources
# -------------------------------------------------------------------------------


def phaseshift(u, v, x0=0, y0=0, angunit="mas"):
    '''
    Phase of a symmetric object (Gaussians, Point Sources, etc).
    This function also can be used to compute a phase shift due to positional shifnp.
    Args:
        u, v (mandatory): uv coordinates in lambda
        x0=0, y0=0: position of centorid or positional shift in anguninp.
        angunit="mas": angular unit of x0, y0 (uas, mas, asec, amin, deg, rad)
    return:
        phase in rad
    '''
    from numpy import pi
    from ..util.units import conv
    return 2*pi*(u*x0+v*y0)*conv(angunit, "rad")
