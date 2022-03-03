#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule of smili.GeoModel, containing functions to
calculate visibilities and images of some geometric models.

Here, we note that the whole design of this module is imspired by
Lindy Blackburn's python module 'modmod', and
we would like to thank Lindy to share his idea.
'''


class GeoModel(object):
    @staticmethod
    def Vis(u, v):
        return (u-u)*(v-v)

    @staticmethod
    def Img(x, y):
        return (x-x)*(y-y)

    def __init__(self, Vis=None, Img=None):
        """
        A Class for Geometric Model

        Args:
            Vis (Jax-compatible function, optional):
                Visibility function in Jy.
                It should be in a form of Vis(u, v),
                where (u, v) is uv-coodinates in lambda.
            Img (Jax-compatible function, optional):
                Image function in Jy/pixel.
                It should be in a form of Img(x, y),
                where (x, y) is xy-coodinates in radians.
        """
        if Vis is None:
            self.Vis = lambda u, v: (u-u)*(v-v)
        else:
            self.Vis = Vis

        if Img is None:
            self.Img = lambda x, y: (x-x)*(y-y)
        else:
            self.Img = Img

    def __add__(self, other):
        if isinstance(other, GeoModel):
            def Vis(u, v): return self.Vis(u, v) + other.Vis(u, v)
            def Img(x, y): return self.Img(x, y) + other.Img(x, y)
            return GeoModel(Vis=Vis, Img=Img)
        else:
            raise ValueError(
                "Addition can be calculated only between the same type of objects")

    def __iadd__(self, other):
        if isinstance(other, GeoModel):
            def Vis(u, v): return self.Vis(u, v) + other.Vis(u, v)
            def Img(x, y): return self.Img(x, y) + other.Img(x, y)
            return GeoModel(Vis=Vis, Img=Img)
        else:
            raise ValueError(
                "Addition can be calculated only between the same type of objects")

    def __sub__(self, other):
        if isinstance(other, GeoModel):
            def Vis(u, v): return self.Vis(u, v) - other.Vis(u, v)
            def Img(x, y): return self.Img(x, y) - other.Img(x, y)
            return GeoModel(Vis=Vis, Img=Img)
        else:
            raise ValueError(
                "Subtraction can be calculated only between the same type of objects")

    def __isub__(self, other):
        if isinstance(other, GeoModel):
            def Vis(u, v): return self.Vis(u, v) - other.Vis(u, v)
            def Img(x, y): return self.Img(x, y) - other.Img(x, y)
            return GeoModel(Vis=Vis, Img=Img)
        else:
            raise ValueError(
                "Subtraction can be calculated only between the same type of objects")

    def __mul__(self, other):
        def Vis(u, v): return self.Vis(u, v) * other
        def Img(x, y): return self.Img(x, y) * other
        return GeoModel(Vis=Vis, Img=Img)

    def __imul__(self, other):
        def Vis(u, v): return self.Vis(u, v) * other
        def Img(x, y): return self.Img(x, y) * other
        return GeoModel(Vis=Vis, Img=Img)

    def __truediv__(self, other):
        def Vis(u, v): return self.Vis(u, v) / other
        def Img(x, y): return self.Img(x, y) / other
        return GeoModel(Vis=Vis, Img=Img)

    def __itruediv__(self, other):
        def Vis(u, v): return self.Vis(u, v) / other
        def Img(x, y): return self.Img(x, y) / other
        return GeoModel(Vis=Vis, Img=Img)

    def shift(self, dx=0., dy=0.):
        """
        Shift Image

        Args:
            dx, dy (float, optional): offset in radians
        Returns:
            GeoModel object
        """
        from jax.numpy import pi, exp
        def Vis(u, v): return self.Vis(u, v) * exp(1j*2*pi*(u*dx+v*dy))
        def Img(x, y): return self.Img(x-dx, y-dy)
        return GeoModel(Vis=Vis, Img=Img)

    def rotate(self, dPA=0.):
        """
        Rotate Image

        Args:
            dPA (float, optional): rotation angle in radians

        Returns:
            GeoModel object
        """
        from jax.numpy import cos, sin
        cosdpa = cos(dPA)
        sindpa = sin(dPA)
        def x1(x, y): return x * cosdpa - y * sindpa
        def y1(x, y): return x * sindpa + y * cosdpa
        def Vis(u, v): return self.Vis(x1(u, v), y1(u, v))
        def Img(x, y): return self.Img(x1(x, y), y1(x, y))
        return GeoModel(Vis=Vis, Img=Img)

    def scale(self, hx=1., hy=None):
        """
        Scale x and y axis

        Args:
            hx, hy (float, optional): Scaling factor. Defaults to 1..

        Returns:
            GeoModel
        """
        if hy is None:
            hy = hx

        def Vis(u, v): return self.Vis(u*hx, v*hy)
        def Img(x, y): return self.Img(x/hx, y/hy)/hx/hy
        return GeoModel(Vis=Vis, Img=Img)

    def Vre(self, u, v):
        """
        Real part of the complex visbility

        Args:
            u, v (float): uv-coordinates
        """
        from jax.numpy import real
        return real(self.Vis(u, v))

    def Vim(self, u, v):
        """
        Imag part of the complex visbility

        Args:
            u, v (float): uv-coordinates
        """
        from jax.numpy import imag
        return imag(self.Vis(u, v))

    def Vamp(self, u, v):
        """
        Amplitudes of the complex visbility

        Args:
            u, v (float): uv-coordinates
        """
        from jax.numpy import abs
        return abs(self.Vis(u, v))

    def logVamp(self, u, v):
        """
        Log amplitudes of the complex visbility

        Args:
            u, v (float): uv-coordinates in lambda
        """
        from jax.numpy import log
        return log(self.Vamp(u, v))

    def Vphase(self, u, v):
        """
        Phase of the complex visbility

        Args:
            u, v (float): uv-coordinates in lambda
        """
        from jax.numpy import angle
        return angle(self.Vis(u, v))

    # Bi-spectrum
    def B(self, u1, v1, u2, v2, u3, v3):
        '''
        Bi-spectrum

        Args:
            un, vn (n=1, 2, 3): uv-coordinates in lambda
        Return:
            real part of the bi-spectrum
        '''
        V1 = self.Vis(u1, v1)
        V2 = self.Vis(u2, v2)
        V3 = self.Vis(u3, v3)
        return V1*V2*V3

    def Bre(self, u1, v1, u2, v2, u3, v3):
        '''
        Real part of the bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates in lambda
        '''
        from jax.numpy import real
        return real(self.B(u1, v1, u2, v2, u3, v3))

    def Bim(self, u1, v1, u2, v2, u3, v3):
        '''
        Imag part of the bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates in lambda
        '''
        from jax.numpy import imag
        return imag(self.B(u1, v1, u2, v2, u3, v3))

    def Bamp(self, u1, v1, u2, v2, u3, v3):
        '''
        Amplitude of the bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates in lambda
        '''
        from jax.numpy import abs
        return abs(self.B(u1, v1, u2, v2, u3, v3))

    def Bphase(self, u1, v1, u2, v2, u3, v3):
        '''
        Phase of the bi-spectrum. (i.e. closure phases)

        Args:
            un, vn (n=1, 2, 3): uv-coordinates in lambda
        '''
        from jax.numpy import angle
        return angle(self.B(u1, v1, u2, v2, u3, v3))

    # Closure Amplitudes
    def Camp(self, u1, v1, u2, v2, u3, v3, u4, v4):
        '''
        Closure amplitudes

        Args:
            un, vn (n=1, 2, 3, 4): uv-coordinates in lambda
        '''
        Vamp1 = self.Vamp(u1, v1)
        Vamp2 = self.Vamp(u2, v2)
        Vamp3 = self.Vamp(u3, v3)
        Vamp4 = self.Vamp(u4, v4)
        return Vamp1*Vamp2/Vamp3/Vamp4

    def logCamp(self, u1, v1, u2, v2, u3, v3, u4, v4):
        '''
        Log Closure amplitudes

        Args:
            un, vn (n=1, 2, 3, 4): uv-coordinates in lambda
        '''
        logVamp1 = self.logVamp(u1, v1)
        logVamp2 = self.logVamp(u2, v2)
        logVamp3 = self.logVamp(u3, v3)
        logVamp4 = self.logVamp(u4, v4)
        return logVamp1+logVamp2-logVamp3-logVamp4


def dphase(phase1, phase2):
    from jax.numpy import arctan2, cos, sin
    dphase = phase2 - phase1
    return arctan2(sin(dphase), cos(dphase))


"""
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
"""
