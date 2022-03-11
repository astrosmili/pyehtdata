#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a sub-module of smili2 handling image data.
'''
__author__ = "Smili Developer Team"
from ...util.xarrds import XarrayDataset
from ... import util
import copy
import numpy as np
import itertools
import scipy.ndimage as sn
import h5py
import tqdm

# Logger
from logging import getLogger
logger = getLogger(__name__)


class Image(XarrayDataset):
    '''
    The Class to handle five dimensional images

    Attributes:
        ds (xarray.Dataset): Image data
    '''
    # Data type name
    datatype = "ehtdata_image"

    # Default Group Name for the I/O
    group = "image"

    # Data Set
    #   This supposed to include the data set
    ds = None

    # data set dimension
    ds_dims = ("time", "freq", "pol", "y", "x")

    def init_attrs(self):
        '''
        Initialize the metadata.
        '''
        self.ds.attrs["source"] = "No Name"
        self.ds.attrs["instrument"] = "No Name"
        self.ds.attrs["ra"] = 0.
        self.ds.attrs["dec"] = 0.
        self.ds.attrs["coordsys"] = "ICRS"
        self.ds.attrs["equinox"] = -1.
        self.ds.attrs["dx"] = 1.
        self.ds.attrs["dy"] = 1.
        self.ds.attrs["ixref"] = 1.
        self.ds.attrs["iyref"] = 1.
        self.ds.attrs["beam_maj"] = 0.
        self.ds.attrs["beam_min"] = 0.
        self.ds.attrs["beam_pa"] = 0.
        self.ds.attrs["pulse_type"] = "rect"

    def init_pol(self):
        '''
        Add the polarization information to the image data array.
        '''
        ns = self.ds.dims["pol"]

        if ns == 1:
            self.ds.coords["pol"] = ("pol", ["I"])
        elif ns == 4:
            self.ds.coords["pol"] = ("pol", ["I", "Q", "U", "V"])
        else:
            raise ValueError("Current version of this library accepts only"
                             "single or full polarization images.")

    def init_xygrid(self):
        '''
        Add the xy coordinates (in radians) to the image data array.
        '''
        from numpy import arange

        axis_list = ["x", "y"]
        sign_list = [-1, 1]

        for iaxis in range(2):
            axis = axis_list[iaxis]
            sign = sign_list[iaxis]

            # get some meta info
            nx = self.ds.dims[axis]
            dx = self.ds.attrs["d"+axis]
            ixref = self.ds.attrs["i{}ref".format(axis)]

            # compute coordinates
            self.ds.coords[axis] = (axis, sign*dx*(arange(nx)-ixref))

    def chunk(self, chunks=None, **args):
        if chunks is None:
            chunks = dict(
                time=1,
                freq=1,
                pol=1,
                y=self.ds.dims["y"],
                x=self.ds.dims["x"]
            )
        self.ds = self.ds.chunk(chunks=chunks, **args)

    def set_source(self, source="M87", srccoord=None):
        '''
        Set the source name and the source coordinate to the metadata.
        If source coordinate is not given, it will be taken from the CDS.

        Args:
            source (str; default="M87"):
                Source Name
            srccoord (astropy.coordinates.Skycoord object; default=None):
                Source position. If not specified, it is automatically pulled
                from the CDS.
        '''
        from astropy.coordinates import SkyCoord

        if source is None and srccoord is None:
            raise ValueError("source or srccoord must be specified.")

        if source is not None:
            self.ds.attrs["source"] = source
            if srccoord is None:
                srccoord = SkyCoord.from_name(source)

        if srccoord is not None:
            # parse coordinates
            self.ds.attrs["ra"] = srccoord.ra.rad
            self.ds.attrs["dec"] = srccoord.dec.rad
            # coordinate system
            self.ds.attrs["coordsys"] = srccoord.name
            # equinox
            if "icrs" in srccoord.name:
                self.ds.attrs["equinox"] = -1
            elif "fk" in srccoord.name:
                self.ds.attrs["equinox"] = srccoord.equinox.jyear
            elif "gcrs" in srccoord.name:
                self.ds.attrs["equinox"] = srccoord.obstime.jyear
            else:
                raise ValueError(
                    "Coordinate System %s is not supported currently." % (srccoord.name))

    def set_instrument(self, instrument="EHT"):
        """
        Set the metadata for the instrument with a
        specified name of the instrument.

        Args:
            instrument (str): The instrument name. Defaults to "EHT".
        """
        if instrument is not None:
            self.ds.attrs["instrument"] = instrument

    def set_mjd(self, mjd):
        """
        Set the MJD infromation for the image data array.

        Args:
            mjd (float or array): MJD
        """
        from numpy import isscalar, asarray

        if isscalar(mjd):
            mjd_arr = asarray([mjd], dtype="float64")
        else:
            mjd_arr = mjd.copy()

        self.ds.coords["mjd"] = ("time", mjd_arr)

    def set_freq(self, freq):
        """
        Set the frequency infromation for the image data array.

        Args:
            freq (float or array): Frequency in Hz
        """
        from numpy import isscalar, asarray

        if isscalar(freq):
            freq_arr = asarray([freq], dtype="float64")
        else:
            freq_arr = freq.copy()

        self.ds.coords["freq"] = ("freq", freq_arr)

    def set_angunit(self, angunit="auto"):
        from numpy import amax
        from ...util.units import Unit, DEG

        if angunit != "auto":
            self.ds.attrs["angunit"] = angunit
            return

        angunits = ["deg", "arcmin", "arcsec", "mas", "uas"]
        xmax = amax(self.get_imextent(angunit="deg"))*DEG

        for angunit in angunits:
            self.ds.attrs["angunit"] = angunit
            if xmax < 0.1*Unit(angunit):
                continue
            else:
                break

    def set_beam(self, majsize=0., minsize=0., pa=0., scale=1., angunit=None):
        '''
        Set beam parameters into headers.

        Args:
            majsize, minsize(float, default=0):
                major/minor-axis FWHM size
            scale(float, default=1):
                scaling factor that will be multiplied to maj/min size.
            pa(float, default=0):
                position angle in deg
        '''
        from ...util.units import conv
        from numpy import deg2rad

        # get the conversion factor for the angular unit
        if angunit is None:
            angunit = self.angunit
        factor = conv(angunit, "rad")

        # beam paramters
        self.ds.attrs["beam_maj"] = majsize * factor * scale
        self.ds.attrs["beam_min"] = minsize * factor * scale
        self.ds.attrs["beam_pa"] = deg2rad(pa)

    def get_bconv(self, fluxunit="Jy", saunit="pixel"):
        from ...util.units import conv, Unit
        from numpy import log, pi

        # pixel size (in radian)
        dx = self.ds.attrs["dx"]
        dy = self.ds.attrs["dy"]
        dxdy = dx*dy  # solid angle of the pixel

        if "k" in fluxunit.lower():
            from astropy.constants import c, k_B
            nu = self.ds.coords["freq"]
            Jy2K = c.si.value ** 2 / \
                (2 * k_B.si.value * nu ** 2) / dxdy * 1e-26
            return Jy2K * conv("K", fluxunit)
        else:
            fluxconv = conv("Jy", fluxunit)

            if "pix" in saunit.lower() or "px" in saunit.lower():
                saconv = 1
            elif "beam" in saunit.lower():
                # get beamsize (in radian)
                bmaj = self.ds.attrs["beam_maj"]
                bmin = self.ds.attrs["beam_min"]

                # beamsolid angle
                beamsa = bmaj*bmin*pi/(4*log(2))
                saconv = conv(dxdy*Unit("rad**2"), beamsa*Unit("rad**2"))
            else:
                saconv = conv(dxdy*Unit("rad**2"), saunit)

            return fluxconv/saconv

    def get_xygrid(self, twodim=False, angunit=None):
        '''
        Get the xy coordinates of the image

        Args:
            angunit(string): Angular unit(uas, mas, asec or arcsec,
                             amin or arcmin, degree)
            twodim(boolean): It True, the 2D grids will be returned.
                             Otherwise, the 1D arrays will be returned
        Returns:
            x, y: x, y coordinates in the specified unit.
        '''
        from ...util.units import conv
        from numpy import meshgrid

        # get the conversion factor for the angular unit
        if angunit is None:
            angunit = self.ds.attrs["angunit"]
        factor = conv("rad", angunit)

        x = self.ds.coords["x"].data * factor
        y = self.ds.coords["y"].data * factor
        if twodim:
            x, y = meshgrid(x, y)
        return x, y

    def get_imextent(self, angunit=None):
        '''
        Get the field of view of the image for the pyplot.imshow function.

        Args:
          angunit(string): Angular unit

        Returns:
          [xmax, xmin, ymin, ymax]: extent of the image
        '''
        from ...util.units import conv
        from numpy import asarray

        # get the conversion factor for the angular unit
        if angunit is None:
            angunit = self.ds.attrs["angunit"]
        factor = conv("rad", angunit)

        dx = self.ds.attrs["dx"]
        dy = self.ds.attrs["dy"]
        nx = self.ds.dims["x"]
        ny = self.ds.dims["y"]
        ixref = self.ds.attrs["ixref"]
        iyref = self.ds.attrs["iyref"]

        xmax = -dx * (0 - ixref - 0.5)
        xmin = -dx * (nx - 1 - ixref + 0.5)
        ymax = dy * (ny - 1 - iyref + 0.5)
        ymin = dy * (0 - iyref - 0.5)

        return asarray([xmax, xmin, ymin, ymax]) * factor

    def get_imarray(self, fluxunit="Jy", saunit="pixel"):
        bconv = self.get_bconv(fluxunit=fluxunit, saunit=saunit)
        converted = self.ds.image * bconv
        return converted.data.copy()

    def get_uvgrid(self, twodim=False):
        """
        Get the uv coordinates of the image on the Fourier domain

        Args:
            twodim(boolean): It True, the 2D grids will be returned.
                             Otherwise, the 1D arrays will be returned
        Returns:
            u, v: u, v coordinates.
        """
        from numpy import meshgrid
        from numpy.fft import fftfreq, ifftshift

        # get the shape of array
        nx = self.ds.dims["x"]
        ny = self.ds.dims["y"]

        # pixel size
        dxrad = self.ds.attrs["dx"]
        dyrad = self.ds.attrs["dy"]

        # create uv grids
        ug = ifftshift(fftfreq(nx, d=-dxrad))
        vg = ifftshift(fftfreq(ny, d=dyrad))

        if twodim:
            return meshgrid(ug, vg)
        else:
            return ug, vg

    def get_uvextent(self):
        """
        Get the field of view of the image on the Fourier transform
        for the pyplot.imshow function. Here we assume that the Fourier image
        is created by the get_visarr method.

        Returns:
          [umax, umin, vmin, vmax]: extent of the Fourier transformed image.
        """
        from numpy import asarray
        ug, vg = self.get_uvgrid()
        du_half = (ug[1] - ug[0])/2
        dv_half = (vg[1] - vg[0])/2
        return asarray([ug[0]-du_half, ug[-1]+du_half, vg[0]-dv_half,
                        vg[-1]+dv_half])

    def get_vis(self, idx=(0, 0, 0), apply_pulsefunc=True):
        """
        Get an array of visibilities computed from the image.

        Args:
            idx (tuple, optional):
                An index, or a list of indice of the image.
                The index should be in the form of (time, freq, pol).
                Defaults to (0, 0, 0). If you specify None, then
                visibilities will be computed for every index of images.
            apply_pulsefunc (bool, optional):
                If True, the pulse function specified in the meta data
                will be applied. Defaults to True.

        Returns:
            numpy.ndarray:
                full complex visibilities. The array shape will
                depend on idx.
        """
        from numpy import pi, exp, asarray, unravel_index
        from numpy.fft import fftshift, ifftshift, fft2

        # get array
        imarr = self.get_imarray()
        nt, nf, ns, ny, nx = imarr.shape
        ixref = self.ds.attrs["ixref"]
        iyref = self.ds.attrs["iyref"]
        dxrad = self.ds.attrs["dx"]
        dyrad = self.ds.attrs["dy"]

        # get uv grid
        ug, vg = self.get_uvgrid(twodim=True)

        # adjust phase
        ix_cen = nx//2 + nx % 2  # image center index used in np.fft.fft2
        iy_cen = ny//2 + ny % 2
        dix = ix_cen - ixref  # shift in the pixel unit
        diy = iy_cen - iyref
        viskernel = exp(1j*2*pi*(-dxrad*dix*ug + dyrad*diy*vg))

        # mutiply the pulse function
        # if apply_pulsefunc:
        #    viskernel *= self.get_pulsefunc()(ug, vg)

        # define FFT function
        def dofft(imarr2d):
            return ifftshift(fft2(fftshift(imarr2d)))

        # compute full complex visibilities
        if idx is None:
            shape3d = (nt, nf, ns)
            vis = asarray([dofft(imarr[unravel_index(i, shape=shape3d)])
                           for i in range(nt*nf*ns)]).reshape([nt, nf, ns,
                                                               ny, nx])
            vis[:, :, :] *= viskernel
        else:
            ndim = asarray(idx).ndim
            if ndim == 1:
                vis = dofft(imarr[idx])*viskernel
            elif ndim == 2:
                nidx = len(idx)
                vis = asarray([dofft(imarr[tuple(idx[i])])
                               for i in range(nidx)]).reshape([nidx, ny, nx])
                vis[:] *= viskernel
            else:
                raise ValueError("Invalid dimension of the input index.")
        return vis

    def get_skycoord(self):
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        from ...util.units import RAD

        args = dict(
            ra=self.ds.attrs["x"] * RAD,
            dec=self.ds.attrs["y"] * RAD,
            frame=self.ds.attrs["coordsys"]
        )
        if "fk" in args["frame"]:
            args["equinox"] = Time(self.ds.attrs["equinox"], format="jyear")
        elif args["frame"] != "gcrs":
            args["obstime"] = Time(self.ds.attrs["equinox"], format="jyear")

        return SkyCoord(**args)

    def get_beam(self, angunit=None):
        '''
        Get beam parameters.

        Args:
            angunit(string): Angular Unit
        Return:
            dic: the beam parameter information
        '''
        from ...util.units import conv
        from numpy import rad2deg

        if angunit is None:
            angunit = self.ds.attrs["angunit"]

        factor = conv("rad", angunit)

        outdic = {}
        outdic["scale"] = 1.0
        outdic["majsize"] = self.ds.attrs["beam_maj"] * factor
        outdic["minsize"] = self.ds.attrs["beam_min"] * factor
        outdic["pa"] = rad2deg(self.ds.attrs["beam_pa"])
        outdic["angunit"] = angunit
        return outdic

    def get_pulsefunc(self):
        ptype = self.ds.attrs["ptype"].lower()

        if "delta" in ptype:
            def pulsefunc(u, v): return 1
        elif "rect" in ptype:
            from ...geomodel import Rectangular
            dxrad = self.ds.attrs["dx"]
            dyrad = self.ds.attrs["dy"]
            pulsefunc = Rectangular(
                Lx=dxrad, Ly=dyrad, dx=dxrad, dy=dyrad, angunit="rad").V
        else:
            raise ValueError("unknown pulse type: %s" % (ptype))
        return pulsefunc

    def add_geomodel(self, geomodel, idx=(0, 0, 0), inplace=False):
        """
        Add a specified geometric model to the image

        Args:
            geomodel (geomodel.GeoModel):
                The input geometric model.
            idx (tuble):
                An index of the image where the Gaussians to be added.
                Should be in the format of (time, freq, pol).
                Default to (0,0,0).
            inplace (bool, optional):
                If False, return the convolved image. Defaults to False.

        Returns:
            imdata.Image: if inplace==False.
        """
        if inplace:
            outimage = self
        else:
            outimage = self.copy()

        # get x,y coordinates
        xg, yg = self.get_xygrid(angunit="rad", twodim=True)

        # compute the intensity
        imarr = geomodel.Img(xg, yg)

        # add data
        outimage.data[idx] += imarr

        if not inplace:
            return outimage

    def add_gauss(self, totalflux=1., x0=0., y0=0., majsize=1., minsize=None,
                  pa=0., scale=1., angunit="uas", inplace=False):
        """
        Add a Gaussian to the image.

        Args:
            totalflux (float, optional):
                total flux density in Jy.
            x0, y0 (float, optional):
                The centoroid position of the kernel.
            majsize, minsize (float, optional):
                The major- and minor- axis FWHM size of the kernel.
            pa (int, optional):
                The position angle of the kernel.
            scale (int, optional):
                The scaling factor to be applied to the kernel size.
            angunit (str, optional):
                The angular unit for the centroid location and kernel size.
                Defaults to "mas".
            inplace (bool, optional):
                If False, return the convolved image. Defaults to False.

        Returns:
            imdata.Image: the convolved image if inplace==False.
        """
        from ...geomodel import Gaussian
        from ..util.units import conv

        # scale the axis size
        majsize_scaled = majsize * scale
        if minsize is None:
            minsize_scaled = majsize_scaled
        else:
            minsize_scaled = minsize * scale

        factor = conv("rad", angunit)
        dx = self.ds.attrs["dx"] * factor
        dy = self.ds.attrs["dy"] * factor

        # initialize Gaussian model
        geomodel = Gaussian(x0=x0, y0=y0, dx=dx, dy=dy, majsize=majsize_scaled,
                            minsize=minsize_scaled, pa=pa, angunit=angunit)

        # run convolution
        if inplace:
            self.add_geomodel(geomodel=geomodel, inplace=inplace)
        else:
            return self.add_geomodel(geomodel=geomodel, inplace=inplace)

    #  Convolution with Geometric Models

    def convolve_geomodel(self, geomodel, inplace=False):
        """
        Convolve the image with an input geometrical model

        Args:
            geomodel (geomodel.GeoModel):
                The input geometric model.
            inplace (bool, optional):
                If False, return the convolved image. Defaults to False.

        Returns:
            imdata.Image: the convolved image if inplace==False.
        """
        from jax.numpy import real, unravel_index, conj
        from jax.numpy.fft import fftshift, ifftshift, fft2, ifft2
        import jax
        import dask

        if inplace:
            outimage = self
        else:
            outimage = self.copy()

        # get the array shape
        nt, nf, ns, ny, nx = self.ds.image.shape
        self.chunk()
        imarr = self.ds.image.data
        shape3d = (nt, nf, ns)

        # get uv coordinates and compute kernel
        # conj is applied because radio astronomy uses
        # "+" for the fourier exponent
        ug, vg = self.get_uvgrid(twodim=True)
        convkernel = conj(geomodel.Vis(ug, vg))

        # define convolve functions
        def dofft(imarr2d):
            return ifftshift(fft2(fftshift(imarr2d)))

        def doifft(vis2d):
            return real(ifftshift(ifft2(fftshift(vis2d))))

        def convolve2d(imarr2d):
            imarr2d = jax.lax.stop_gradient(imarr2d)
            return doifft(dofft(imarr2d)*convkernel)

        convolve2d = jax.jit(convolve2d)

        # run fft convolve
        output = [dask.delayed(convolve2d)(imarr[unravel_index(i, shape=shape3d)])
                  for i in range(nt*nf*ns)]
        output = dask.array.array(dask.compute(output))
        outimage.ds.image.data = output.reshape((nt, nf, ns, ny, nx))

        # return the output image
        if inplace is False:
            return outimage

    def convolve_gauss(self, x0=0., y0=0., majsize=1., minsize=None, pa=0.,
                       scale=1., angunit="uas", inplace=False):
        """
        Gaussian Convolution.

        Args:
            x0, y0 (float, optional):
                The centoroid position of the kernel.
            majsize, minsize (float, optional):
                The major- and minor- axis FWHM size of the kernel.
            pa (int, optional):
                The position angle of the kernel.
            scale (int, optional):
                The scaling factor to be applied to the kernel size.
            angunit (str, optional):
                The angular unit for the centroid location and kernel size.
                Defaults to "mas".
            inplace (bool, optional):
                If False, return the convolved image. Defaults to False.

        Returns:
            imdata.Image: the convolved image if inplace==False.
        """
        from ...geomodel import Gaussian

        # scale the axis size
        majsize_scaled = majsize * scale
        if minsize is None:
            minsize_scaled = majsize_scaled
        else:
            minsize_scaled = minsize * scale

        # initialize Gaussian model
        geomodel = Gaussian(x0=x0, y0=y0, majsize=majsize_scaled,
                            minsize=minsize_scaled, pa=pa, angunit=angunit)
        # run convolution
        if inplace:
            self.convolve_geomodel(geomodel=geomodel, inplace=inplace)
        else:
            return self.convolve_geomodel(geomodel=geomodel, inplace=inplace)

    def convolve_rectangular(self, x0=0., y0=0., Lx=1., Ly=None, angunit="mas",
                             inplace=False):
        """
        Convolution with a rectangular kernel

        Args:
            x0, y0 (float, optional):
                The centoroid position of the kernel.
            Lx, Ly (float, optional):
                The width of the kernel.
            angunit (str, optional):
                The angular unit for the centroid location and kernel size.
                Defaults to "mas".
            inplace (bool, optional):
                If False, return the convolved image. Defaults to False.

        Returns:
            imdata.Image: the convolved image if inplace==False.
        """
        from ...geomodel import Rectangular
        from ..util.units import conv

        # get the pixel size of the image
        rad2aunit = conv("rad", angunit)
        dx = self.ds.attrs["dx"] * rad2aunit
        dy = self.ds.attrs["dy"] * rad2aunit

        # initialize Gaussian model
        geomodel = Rectangular(x0=x0, y0=y0, Lx=Lx, Ly=Ly, dx=dx, dy=dy,
                               angunit=angunit)

        # run convolution
        if inplace:
            self.convolve_geomodel(geomodel=geomodel, inplace=inplace)
        else:
            return self.convolve_geomodel(geomodel=geomodel, inplace=inplace)

    def imshow(self,
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
            index(integer):
            **imshow_args: Args will be input in matplotlib.pyplot.imshow

        Returns:
            output from plt.imshow
        '''
        from .plot import plot_imshow
        return plot_imshow(
            image=self,
            scale=scale, dyrange=dyrange, gamma=gamma,
            vmax=vmax, vmin=vmin,
            relative=relative,
            fluxunit=fluxunit, saunit=saunit, angunit=angunit,
            axisoff=axisoff, axislabel=axislabel,
            cmap=cmap,
            idx=idx,
            interpolation=interpolation,
            **imshow_args
        )

    def plot_xylabel(self, **labelargs):
        """
        Add xy label in the existing image plot.

        Args:
            **labelargs:
                Arguments for pyplot.xlabel / ylable functions.
        """
        from .plot import plot_xylabel
        plot_xylabel(self, **labelargs)

    def plot_beam(self, boxfc=None, boxec=None, beamfc=None, beamec="white",
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

        Returns:
            dictionary of outputs from plt.plot and plt.fill
        '''
        from .plot import plot_beam
        return plot_beam(
            image=self,
            boxfc=boxfc, boxec=boxec, beamfc=beamfc, beamec=beamec, lw=lw,
            alpha=alpha, x0=x0, y0=y0, boxsize=boxsize, zorder=zorder
        )

    @staticmethod
    def plot_scalebar(x, y, length, ha="center", color="white", lw=1,
                      **plotargs):
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
        '''
        from .plot import plot_scalebar
        return plot_scalebar(
            x=x, y=y, length=length, ha=ha, color=color, lw=lw, **plotargs
        )

    def to_fits(self, outfits=None, idx=None, fitsfmt="casa", overwrite=True):
        '''
        save the image(s) to the image FITS file or HDUList

        Args:
            outfits (string; default is None):
                FITS file name. If not specified, then HDUList object will be
                returned.
            idx (list):
                Index for images. For ehtim & CASA formats,
                it should be (MJD, FREQ).
            fitsfmr (str):
                The format for FITS file. Default is "casa".
                Available formats are ["casa", "ehtim"]
            overwrite (boolean):
                It True, an existing file will be overwritten.
        Returns:
            HDUList object if outfits is None
        '''
        from .io.fits_ehtim import save_image_to_fits_ehtim
        from .io.fits_casa import save_image_to_fits_casa

        if fitsfmt.lower() == "casa":
            output = save_image_to_fits_casa(
                self, outfits=outfits, overwrite=overwrite, idx=idx
            )
        elif fitsfmt.lower() == "ehtim":
            output = save_image_to_fits_ehtim(
                self, outfits=outfits, overwrite=overwrite, idx=idx
            )
        else:
            raise ValueError("fitsfmt=%s is not supported." % (fitsfmt))

        return output

    def to_hdf5_ehtim(self, file_name):
        '''
        Save image object in a hdf5 format

        Args:
            file_name (string): output hdf5 file name
        '''
        Ntime, Nfreq, Npol, Nx, Ny = self.ds.image.shape

        if Nx != Ny:
            raise ValueError(
                "Grid numbers of x, y coordinates are different Nx(=%d) != Ny(=%d)" % (Nx, Ny))
        if len(list(set(self.ds.freq.data))) > 1:
            raise ValueError(
                "This function cannot be used for multi epoch hdf5")
        if len(list(set(self.ds.pol.data))) > 1 or self.ds.pol.data[0] != "I":
            raise ValueError("This function is only used for Stokes I")

        mjd = int(self.ds.mjd.data[0])
        times = (self.ds.mjd.data - mjd)*24

        with h5py.File(file_name, "w") as file:
            head = file.create_dataset('header', (0,), dtype="S10")
            head.attrs['mjd'] = np.string_(str(mjd))
            head.attrs['psize'] = np.string_(str(Nx))
            head.attrs['source'] = np.string_(str(self.ds.source))
            head.attrs['ra'] = np.string_(str(self.ds.ra))
            head.attrs['dec'] = np.string_(str(self.ds.dec))
            head.attrs['rf'] = np.string_(str(self.ds.freq.data[0]))
            # If including polarization the following term should be updated
            head.attrs['polrep'] = np.string_('stokes')
            head.attrs['pol_prim'] = np.string_(self.ds.pol.data[0])
            dset = file.create_dataset("times", data=times, dtype='f8')

            # image table (Nt * Ny * Nx)
            name = "I"
            frames = np.stack(
                [np.flipud(self.ds.image.data[it, 0, 0, :, :]) for it in range(Ntime)])
            dset = file.create_dataset(name, data=frames, dtype='f8')

    def cpimage(self, im, save_totalflux=False, order=1):
        outim = copy.deepcopy(self)

        Ntime, Nfreq, Npol, Nx0, Ny0 = im.ds["image"].data.shape
        dx0 = im.ds.dx
        dy0 = im.ds.dy
        Nxr0 = im.ds.ixref
        Nyr0 = im.ds.iyref

        dummy, dummy, dummy, Nx1, Ny1 = outim.ds["image"].data.shape
        dx1 = outim.ds.dx
        dy1 = outim.ds.dy
        Nxr1 = outim.ds.ixref
        Nyr1 = outim.ds.iyref

        # output grid information
        coord = np.zeros([2, Nx1 * Ny1])
        xgrid = (np.arange(Nx1) + 1 - Nxr1) * dx1 / dx0 + Nxr0 - 1
        ygrid = (np.arange(Ny1) + 1 - Nyr1) * dy1 / dy0 + Nyr0 - 1
        x, y = np.meshgrid(xgrid, ygrid)
        coord[0, :] = y.reshape(Nx1 * Ny1)
        coord[1, :] = x.reshape(Nx1 * Ny1)

        for itime, ifreq, ipol in itertools.product(range(Ntime), range(Nfreq), range(Npol)):
            data = im.ds["image"].data[itime, ifreq, ipol]

            outim.ds["image"].data[itime, ifreq, ipol] = sn.map_coordinates(
                data, coord, order=order,
                mode='constant', cval=0.0, prefilter=True).reshape([Ny1, Nx1]
                                                                   ) * dx1 * dy1 / dx0 / dy0

            # Flux Scaling
            if save_totalflux:
                totalflux = im.totalflux(
                    itime=itime, ifreq=ifreq, ipol=ipol)
                outim.ds["image"].data[0, 0, 0] *= totalflux / \
                    outim.totalflux(itime=itime, ifreq=ifreq, ipol=ipol)

        # outim.update_fits()
        return outim

    def totalflux(self, fluxunit="Jy", itime=0, ifreq=0, ipol=0):
        '''
        Calculate the total flux density of the image

        Args:
          itime (integer): index for the image frame
          ipol (integer): index for polarization parameter at which the total flux will be calculated
          ifreq (integer): index for Frequency at which the total flux will be calculated
        '''
        return self.ds["image"].data[itime, ifreq, ipol].sum() * util.fluxconv("Jy", fluxunit)

    def lightcurve(self, fluxunit="Jy", ifreq=0, ipol=0):
        Ntime, dummy, dummy, dummy, dummy = self.ds["image"].data.shape
        return np.array([self.totalflux(fluxunit, itime, ifreq, ipol) for itime in range(Ntime)])

    ###########


def gen_image(
        nx=128, ny=None,
        dx=2., dy=None,
        ixref=None, iyref=None,
        angunit="uas",
        mjd=[0.],
        freq=[230e9],
        ns=1,
        source="M87",
        srccoord=None,
        instrument="EHT",
        **args):
    """
    Generate a blank image.

    Args:
        nx (int, optional): [description]. Defaults to 128.
        ny ([type], optional): [description]. Defaults to None.
        dx ([type], optional): [description]. Defaults to 2..
        dy ([type], optional): [description]. Defaults to None.
        ixref ([type], optional): [description]. Defaults to None.
        iyref ([type], optional): [description]. Defaults to None.
        angunit (str, optional): [description]. Defaults to "uas".
        mjd (list, optional): [description]. Defaults to [0.].
        freq (list, optional): [description]. Defaults to [230e9].
        ns (int, optional): [description]. Defaults to 1.
        source (str, optional): [description]. Defaults to "M87".
        srccoord ([type], optional): [description]. Defaults to None.
        instrument (str, optional): [description]. Defaults to "VLBI".
    """
    from numpy import float64, int32, abs, isscalar, zeros
    from xarray import Dataset
    from ...util.units import conv
    #  dx & dy
    factor = conv(angunit, "rad")
    dxrad = abs(float64(dx*factor))
    if dy is None:
        dyrad = dxrad
    else:
        dyrad = abs(float64(dy*factor))

    #  nx & ny
    nx = int32(nx)
    if ny is None:
        ny = nx
    else:
        ny = int32(ny)

    # ixref & iyref
    if ixref is None:
        ixref = nx/2 - 0.5
    else:
        ixref = float64(ixref)
    if iyref is None:
        iyref = ny/2 - 0.5
    else:
        iyref = float64(iyref)

    # ns
    ns = int32(ns)

    # nf
    if isscalar(freq):
        nf = int32(1)
    else:
        nf = len(freq)

    # nt
    if isscalar(mjd):
        nt = int32(1)
    else:
        nt = len(mjd)

    # create an empty image
    outim = Image(
        ds=Dataset(
            data_vars=dict(
                image=(Image.ds_dims, zeros(
                    [nt, nf, ns, ny, nx], dtype=float64))
            )
        )
    )

    # initialize meta data
    outim.init_attrs()
    outim.ds.attrs["dx"] = dxrad
    outim.ds.attrs["dy"] = dyrad
    outim.ds.attrs["ixref"] = ixref
    outim.ds.attrs["iyref"] = iyref
    outim.ds.attrs["angunit"] = angunit
    outim.set_source(source, srccoord)
    outim.set_instrument(instrument)

    # add coordinates
    outim.init_pol()
    outim.init_xygrid()
    outim.set_mjd(mjd)
    outim.set_freq(freq)

    return outim


def load_hdf5(hdf5file, angunit="uas"):
    '''
    Read data from the movie hdf5 file
    Args:
        hdf5file (string): hdf5 file name. currently the format is based on 2021 ehtim library.
    Returns:
        imdata.Image object
    '''

    # Read hdf5
    with h5py.File(hdf5file, "r") as file:
        head = file['header']
        mjd = int(head.attrs['mjd'].astype(str))
        times = np.float64(file['times'][:]/24)+mjd
        dx = np.float64(head.attrs['psize'].astype(
            str)) * util.angconv("rad", angunit)
        ra = np.float64(head.attrs['ra'].astype(str))
        dec = np.float64(head.attrs['dec'].astype(str))
        freq = np.float64(head.attrs['rf'].astype(str))
        polrep = head.attrs['polrep'].astype(str)
        pol_prim = head.attrs['pol_prim'].astype(str)
        source = head.attrs["source"].astype(str)
        data = file[pol_prim][:]  # time, y, x
        Ntime, Ny, Nx = data.shape

    # intialization of image object
    mov = gen_image(nx=Nx, ny=Ny, dx=dx, dy=dx, ixref=None, iyref=None,
                    angunit=angunit, mjd=times, freq=[freq], ns=1,
                    source=source, srccoord=None, instrument="EHT")

    logger.info("Construct movie frame")
    logger.info("Note: so far hdf5 includes single pol (=I) and single band")
    #logger.info("Note: need to invert the y direction")
    for itime in tqdm.tqdm(range(Ntime)):
        mov.ds["image"].data[itime, 0, 0] = data[itime, ::-1]

    return mov


def load_fits(infits, fitsfmt="casa"):
    """
    [summary]

    Args:
        infits (string or HDUList):
            Input FITS file name or HDUList object.
        fitsfmr (str):
            The format for FITS file. Default is "casa".
            Available formats are ["casa", "ehtim"]

    Returns:
        imdata.Image object
    """
    from .io.fits_ehtim import load_fits_ehtim
    from .io.fits_casa import load_fits_casa

    if fitsfmt.lower() == "casa":
        return load_fits_casa(infits)
    elif fitsfmt.lower() == "ehtim":
        return load_fits_ehtim(infits)
    else:
        raise ValueError("fitsfmt=%s is not supported." % (fitsfmt))
