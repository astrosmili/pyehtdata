#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import int64

# Logger
from logging import getLogger
logger = getLogger(__name__)


# Dictionary for pol labels and their IDs in UVFITS
polid2name = {
    "+1": "I",
    "+2": "Q",
    "+3": "U",
    "+4": "V",
    "-1": "RR",
    "-2": "LL",
    "-3": "RL",
    "-4": "LR",
    "-5": "XX",
    "-6": "YY",
    "-7": "XY",
    "-8": "YX",
}
polname2id = {}
for key in polid2name.keys():
    polname2id[polid2name[key]] = int64(key)


def uvfits2UVData(inuvfits, scangap=None, nseg=2, outfile=None, group="", format="netcdf", mode="w", printlevel=0):
    """
    Load an uvfits file. Currently, this function can read only single-source,
    single-frequency-setup, single-array data correctly.

    Args:
        uvfits (string or pyfits.HDUList object):
            Input uvfits data
        scangap (float or astropy.units.Quantity, optional):
            Minimal time seperation between scans.
            If not specfied, this will be guessed from data segmentation (see nseg).
            If a float value is specified, its unit is assumuted to be in seconds.
            Defaults to None.
        nseg (float, optional):
            If scangap is None, the minimal time seperation between scans
            will be set to nseg * minimal_data_segementation_time.
            Defaults to 2.
        printlevel (integer, optional):
            print some notes. 0: silient 3: maximum level
    Returns:
        uvdata.UVData object
    """
    import astropy.io.fits as pf
    import zarr
    from ..uvdata import UVData

    # check input files
    if isinstance(inuvfits, type("")):
        hdulist = pf.open(inuvfits)
        closehdu = True
    else:
        hdulist = inuvfits
        closehdu = False

    # print HDU info if requested.
    if printlevel > 0:
        hdulist.info()
        print("")

    # load data
    ghdu, antab, fqtab = uvfits2HDUs(hdulist)

    # create zarr file
    #z = zarr.open(outzarr, mode="w")

    def save_ds(ds):
        import os
        groupname = os.path.join(group, ds.group)
        if outfile is not None:
            if format == "zarr":
                freqds.to_zarr(outfile, group=groupname, mode=mode)
            elif format == "netcdf":
                freqds.to_netcdf(outfile, group=groupname, mode=mode)
            del ds

    # Load info from HDU
    #   Frequency
    freqds = uvfits2freq(ghdu=ghdu, antab=antab, fqtab=fqtab)
    del fqtab
    save_ds(freqds)
    #   Antenna
    antds = uvfits2ant(antab=antab)
    del antab
    save_ds(antds)
    #   Source
    srcds = uvfits2src(ghdu=ghdu)
    save_ds(srcds)
    #   Visibilities
    visds = uvfits2vis(ghdu=ghdu)
    del ghdu

    # close HDU if this is loaded from a file
    if closehdu:
        hdulist.close()

    # Detect scans and save visibilities and scaninfo to zarr file
    visds.set_scan(scangap=scangap, nseg=2)
    save_ds(visds)

    scands = visds.gen_scandata()
    save_ds(scands)

    if outfile is None:
        uvd = UVData(
            freq=freqds,
            src=srcds,
            scan=scands,
            vis=visds,
            ant=antds
        )
        return uvd
    else:
        if format == "zarr":
            from .zarr import zarr2UVData
            return zarr2UVData(outfile, group=group)
        elif format == "netcdf":
            from .netcdf import netcdf2UVData
            return netcdf2UVData(outfile, group=group)


def uvfits2HDUs(hdulist):
    """
    Read HDUList, and get PrimaryHDU & HDUS for AIPS AN/FQ Tables

    Args:
        hdulist (astropy.io.fits.HDUList): hdulist

    Returns:
        Group HDU
        HDU for AIPS AN Table
        HDU for AIPS FQ Table
    """
    hduinfo = hdulist.info(output=False)
    Nhdu = len(hduinfo)

    fqtab = None
    antab = None
    ghdu = None
    for ihdu in range(Nhdu):
        hduname = hduinfo[ihdu][1]
        if "PRIMARY" in hduname.upper():
            if ghdu is not None:
                logger.warning("This UVFITS has more than two Primary HDUs.")
                logger.warning("The later one will be taken.")
            ghdu = hdulist[ihdu]
        elif "FQ" in hduname.upper():
            if fqtab is not None:
                logger.warning("This UVFITS has more than two AIPS FQ tables.")
                logger.warning("The later one will be taken.")
            fqtab = hdulist[ihdu]
        elif "AN" in hduname.upper():
            if antab is not None:
                logger.warning("This UVFITS has more than two AIPS AN tables.")
                logger.warning("The later one will be taken.")
            antab = hdulist[ihdu]

    return ghdu, antab, fqtab


def uvfits2vis(ghdu):
    """
    Load the array information from uvfits's AIPS AN table into the SMILI format.

    Args:
        ghdu (astropy.io.fits.HDU): Group (Primary) HDU

    Returns:
        VisData: complex visibility in SMILI format
    """
    from ..vis.vis import VisData
    from astropy.time import Time
    from xarray import Dataset
    from numpy import float64, int32, int64, zeros, where, power
    from numpy import abs, sign, isinf, isnan, finfo, unique, modf, arange, min, diff

    # read visibilities
    #    uvfits's original dimension is [data,dec,ra,if,ch,pol,complex]
    Ndata, Ndec, Nra, dammy, dammy, Npol, dammy = ghdu.data.data.shape
    del dammy
    if Nra > 1 or Ndec > 1:
        logger.warning(
            "GroupHDU has more than single coordinates (Nra, Ndec)=(%d, %d)." % (Nra, Ndec))
        logger.warning("We will pick up only the first one.")
    vis_ghdu = ghdu.data.data[:, 0, 0, :]  # to [data,if,ch,pol,complex]

    # get visibilities, errors, and flag (flagged, removed,)
    vcmp = float64(vis_ghdu[:, :, :, :, 0]) + 1j * \
        float64(vis_ghdu[:, :, :, :, 1])
    sigma = float64(power(abs(vis_ghdu[:, :, :, :, 2]), -0.5))
    flag = int32(sign(vis_ghdu[:, :, :, :, 2]))

    # check sigma
    idx = where(isinf(sigma))
    sigma[idx] = 0
    flag[idx] = 0

    idx = where(isnan(sigma))
    sigma[idx] = 0
    flag[idx] = 0

    idx = where(sigma < finfo(float64).eps)
    sigma[idx] = 0
    flag[idx] = 0

    # Read Random Parameters
    paridxes = [None for i in range(9)]
    parnames = ghdu.data.parnames
    Npar = len(parnames)
    jd1 = zeros(Ndata)
    jd2 = zeros(Ndata)
    for i in range(Npar):
        parname = parnames[i]
        if "UU" in parname:
            paridxes[0] = i+1
            usec = float64(ghdu.data.par(i))
        if "VV" in parname:
            paridxes[1] = i+1
            vsec = float64(ghdu.data.par(i))
        if "WW" in parname:
            paridxes[2] = i+1
            wsec = float64(ghdu.data.par(i))
        if "DATE" in parname:
            if paridxes[3] is None:
                paridxes[3] = i+1
                jd1 = float64(ghdu.data.par(i))
            elif paridxes[4] is None:
                paridxes[4] = i+1
                jd2 = float64(ghdu.data.par(i))
            else:
                errmsg = "Random Parameters have too many 'DATE' columns."
                raise ValueError(errmsg)
        if "BASELINE" in parname:
            paridxes[5] = i+1
            bl = float64(ghdu.data.par(i))
        if "SOURCE" in parname:
            paridxes[6] = i+1
            srcid = int32(ghdu.data.par(i))
        if "INTTIM" in parname:
            paridxes[7] = i+1
            inttim = float64(ghdu.data.par(i))
        if "FREQSEL" in parname:
            paridxes[8] = i+1
            freqsel = int32(ghdu.data.par(i))

    # convert JD to MJD
    mjd = Time(jd1, jd2, format="jd").mjd

    # warn if it is an apparently multi source file
    if paridxes[6] is not None:
        if len(unique(srcid)) > 1:
            logger.warning(
                "Group HDU contains data on more than a single source.")
            logger.warning(
                "It will likely cause a problem since SMILI assumes a singlesource UVFITS.")

    # Integration time in the unit of day
    if paridxes[7] is None:
        logger.warning(
            "Group HDU do not have a random parameter for the integration time.")
        logger.warning(
            "It will be estimated with a minimal time interval of data.")
        dmjd = min(abs(diff(unique(mjd))))
    else:
        dmjd = inttim/86400

    # warn if data are apparently with multi IF setups
    if paridxes[8] is not None:
        if len(unique(freqsel)) > 1:
            logger.warning(
                "Group HDU contains data on more than a frequency setup.")
            logger.warning(
                "It will likely cause a problem since SMILI assumes a UVFITS with a single setup.")

    # antenna ID
    subarray, bl = modf(bl)
    subarray = int64(100*(subarray)+1)
    antid1 = int64(bl//256)-1
    antid2 = int64(bl % 256)-1
    if len(unique(subarray)) > 1:
        logger.warning("Group HDU contains data with 2 or more subarrays.")
        logger.warning(
            "It will likely cause a problem, since SMILI assumes UVFITS for a single subarray.")

    # read polarizations
    polids = ghdu.header["CDELT3"] * \
        (arange(Npol)+1-ghdu.header["CRPIX3"])+ghdu.header["CRVAL3"]
    pol = [polid2name["%+d" % (polid)] for polid in polids]

    # form a data array
    ds = Dataset(
        data_vars=dict(
            vis=(["data", "spw", "ch", "pol"], vcmp)
        ),
        coords=dict(
            mjd=("data", mjd),
            dmjd=("data", dmjd),
            usec=("data", usec),
            vsec=("data", vsec),
            wsec=("data", wsec),
            antid1=("data", antid1),
            antid2=("data", antid2),
            flag=(["data", "spw", "ch", "pol"], flag),
            sigma=(["data", "spw", "ch", "pol"], sigma),
            pol=(["pol"], pol),
        )
    )
    return VisData(ds=ds.sortby(["mjd", "antid1", "antid2"]))


def uvfits2ant(antab):
    """
    Load the rray information from uvfits's AIPS AN table into the SMILI format.

    Args:
        antab (astropy.io.fits.HDU): HDU for AIPS AN table

    Returns:
        AntData: array information in SMILI format
    """
    from numpy import asarray, zeros, ones, unique
    from ..ant.ant import AntData
    from xarray import Dataset

    # The array name
    name = antab.header["ARRNAM"]

    # Number of Antenna
    Nant = len(antab.data)

    # Anteanna Name
    antname = antab.data["ANNAME"].tolist()

    # XYZ Coordinates
    xyz = antab.data["STABXYZ"]

    # Parse Field Rotation Information
    #   See AIPS MEMO 117
    #      0: ALT-AZ, 1: Eq, 2: Orbit, 3: X-Y, 4: Naismith-R, 5: Naismith-L
    #      6: Manual
    mntsta = antab.data["MNTSTA"]
    fr_pa_coeff = ones(Nant)
    fr_el_coeff = zeros(Nant)
    fr_offset = zeros(Nant)
    for i in range(Nant):
        if mntsta[i] == 0:  # azel
            fr_pa_coeff[i] = 1
            fr_el_coeff[i] = 0
        elif mntsta[i] == 1:  # Equatorial
            fr_pa_coeff[i] = 0
            fr_el_coeff[i] = 0
        elif mntsta[i] == 4:  # Nasmyth-R
            fr_pa_coeff[i] = 1
            fr_el_coeff[i] = 1
        elif mntsta[i] == 5:  # Nasmyth-L
            fr_pa_coeff[i] = 1
            fr_el_coeff[i] = -1
        else:
            logger.warning("MNTSTA %d at Station %s is not supported currently." % (
                mntsta[i], antname[i]))

    # check polarization
    pola = unique(antab.data["POLTYA"])
    polb = unique(antab.data["POLTYB"])
    if len(pola) > 1 or len(polb) > 1:
        msg = "POLTYA or POLTYB have more than a single polarization"
        logger.error(msg)
        raise ValueError(msg)
    pol = [pola[0], polb[0]]

    # assume all of them are ground array
    anttype = asarray(["g" for i in range(Nant)], dtype="U8")

    antdata = AntData(
        ds=Dataset(
            coords=dict(
                antname=("ant", antname),
                x=("ant", xyz[:, 0]),
                y=("ant", xyz[:, 1]),
                z=("ant", xyz[:, 2]),
                fr_pa_coeff=("ant", fr_pa_coeff),
                fr_el_coeff=("ant", fr_el_coeff),
                fr_offset=("ant", fr_offset),
                anttype=("ant", anttype),
                pol=("pol", pol)
            ),
            attrs=dict(
                name=name,
            ),
        )
    )
    antdata.init_coords()

    return antdata


def uvfits2freq(ghdu, antab, fqtab):
    """
    Load the frequency information from uvfits HDUs into the SMILI format.

    Args:
        ghdu (astropy.io.fits.HDU): Group (Primary) HDU
        antab (astropy.io.fits.HDU): HDU for AIPS AN table
        fqtab (astropy.io.fits.HDU): HDU for AIPS FQ table

    Returns:
        FreqData: Loaded frequency table
    """
    from ..freq import FreqData
    from xarray import Dataset
    from numpy import float64

    # read meta data from antenna table
    reffreq = antab.header["FREQ"]  # reference frequency in GHz
    # use array name because uvfits doesn't have such meta data
    name = antab.header["ARRNAM"]

    # get number of channels
    dammy, dammy, dammy, Nspw, Nch, dammy, dammy = ghdu.data.data.shape
    del dammy

    # read data from frequency table
    nfrqsel = len(fqtab.data["FRQSEL"])
    if nfrqsel > 1:
        logger.warning(
            "Input FQ Tables have more than single FRQSEL. We only handle a uvfits with single FRQSEL.")

    # read meta information
    def arraylize(input):
        from numpy import isscalar, array
        if isscalar(input):
            return array([input])
        else:
            return input
    spwfreq = arraylize(float64(fqtab.data["IF FREQ"][0]))
    chbw = arraylize(float64(fqtab.data["CH WIDTH"][0]))
    sideband = arraylize(float64(fqtab.data["SIDEBAND"][0]))

    # check the consistency between the number of if in FQ Table and GroupHDU
    if len(spwfreq) != Nspw:
        raise ValueError(
            "Group HDU has %d IFs, which is inconsistent with FQ table with %d IFs" % (
                Nspw, len(spwfreq))
        )

    # Make FreqTable
    dataset = Dataset(
        coords=dict(
            spw_freq=("spw", reffreq+spwfreq),
            ch_bw=("spw", chbw),
            sideband=("spw", sideband)
        ),
        attrs=dict(
            name=name,
            Nch=Nch,
        )
    )
    freq = FreqData(dataset)
    freq.recalc_freq()

    return freq


def uvfits2src(ghdu):
    """
    Load the source information from uvfits HDUs into the SMILI format.

    Args:
        ghdu (astropy.io.fits.HDU): Group (Primary) HDU

    Returns:
        SrcData: Loaded frequency table
    """
    from ..src.src import SrcData
    from xarray import Dataset

    # source info
    srcname = ghdu.header["OBJECT"]
    ra = ghdu.header["CRVAL6"]
    dec = ghdu.header["CRVAL7"]
    if "EQUINOX" in ghdu.header.keys():
        equinox = ghdu.header["EQUINOX"]
        coordsys = "fk5"
    elif "EPOCH" in ghdu.header.keys():
        equinox = ghdu.header["EPOCH"]
        coordsys = "fk5"
    else:
        equinox = -1

    if equinox < 0:
        equinox = -1
        coordsys = "icrs"

    src = Dataset(
        attrs=dict(
            name=srcname,
            ra=ra,
            dec=dec,
            equinox=equinox,
            coordsys=coordsys
        ),
    )

    return SrcData(src)
