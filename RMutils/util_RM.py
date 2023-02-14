#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common procedures used with RM-synthesis scripts."""

# =============================================================================#
#                                                                             #
# NAME:     util_RM.py                                                        #
#                                                                             #
# PURPOSE:  Common procedures used with RM-synthesis scripts.                 #
#                                                                             #
# REQUIRED: Requires the numpy and scipy modules.                             #
#                                                                             #
# MODIFIED: 16-Nov-2018 by J. West                                            #
#                                                                             #
# CONTENTS:                                                                   #
#                                                                             #
#  do_rmsynth_planes   ... perform RM-synthesis on Q & U data cubes           #
#  get_rmsf_planes     ... calculate the RMSF for a cube of data              #
#  do_rmclean_hogbom   ... perform Hogbom RM-clean on a dirty FDF             #
#  fits_make_lin_axis  ... create an array of absica values for a lin axis    #
#  extrap              ... interpolate and extrapolate an array               #
#  fit_rmsf            ... fit a Gaussian to the main lobe of the RMSF        #
#  gauss1D             ... return a function to evaluate a 1D Gaussian        #
#  detect_peak         ... detect the extent of a peak in a 1D array          #
#  measure_FDF_parms   ... measure parameters of a Faraday dispersion func    #
#  norm_cdf            ... calculate the CDF of a Normal distribution         #
#  cdf_percentile      ... return the value at the given percentile of a CDF  #
#  calc_sigma_add      ... calculate most likely additional scatter           #
#  calc_normal_tests   ... calculate metrics measuring deviation from normal  #
#  measure_qu_complexity  ... measure the complexity of a q & u spectrum      #
#  measure_fdf_complexity  ... measure the complexity of a clean FDF spectrum #
#                                                                             #
# DEPRECATED CODE ------------------------------------------------------------#
#                                                                             #
#  do_rmsynth          ... perform RM-synthesis on Q & U data by spectrum     #
#  get_RMSF            ... calculate the RMSF for a 1D wavelength^2 array     #
#  do_rmclean          ... perform Hogbom RM-clean on a dirty FDF             #
#  plot_complexity     ... plot the residual, PDF and CDF (deprecated)        #
#                                                                             #
# =============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2015 - 2018 Cormac R. Purcell                                 #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the "Software"),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
# =============================================================================#

import math as m
import sys
from typing import Tuple, Union

import numpy as np
from scipy import stats

from RMutils.mpfit import mpfit
from RMutils.util_misc import (
    MAD,
    calc_mom2_FDF,
    calc_parabola_vertex,
    create_pqu_spectra_burn,
    progress,
    toscalar,
)
from RMutils.logger import logger

# Constants
C = 2.99792458e8


# -----------------------------------------------------------------------------#
def do_rmsynth_planes(
    data_Q: np.ndarray,
    data_U: np.ndarray,
    lambda_sq_arr_m2: np.ndarray,
    phi_arr_radm2: np.ndarray,
    weight_arr: Union[None, np.ndarray] = None,
    lam0_sq_m2: Union[None, np.ndarray] = None,
    n_bits: int = 32,
):
    """Perform RM-synthesis on Stokes Q and U cubes (1,2 or 3D). This version
    of the routine loops through spectral planes and is faster than the pixel-
    by-pixel code. This version also correctly deals with isolated clumps of
    NaN-flagged voxels within the data-cube (unlikely in interferometric cubes,
    but possible in single-dish cubes). Input data must be in standard python
    [z,y,x] order, where z is the frequency axis in ascending order.

    data_Q           ... 1, 2 or 3D Stokes Q data array
    data_U           ... 1, 2 or 3D Stokes U data array
    lambda_sq_arr_m2  ... vector of wavelength^2 values (assending freq order)
    phi_arr_radm2    ... vector of trial Faraday depth values
    weight_arr       ... vector of weights, default [None] is Uniform (all 1s)
    n_bits           ... precision of data arrays [32]
    verbose         ... print feedback during calculation [False]
    log             ... function to be used to output messages [print]

    """

    # Default data types
    dtFloat = f"float{n_bits}"
    dtComplex = f"complex{2 * n_bits}"

    # Set the weight array
    if weight_arr is None:
        weight_arr = np.ones(lambda_sq_arr_m2.shape, dtype=dtFloat)
    weight_arr = np.where(np.isnan(weight_arr), 0.0, weight_arr)

    # Sanity check on array sizes
    if not weight_arr.shape == lambda_sq_arr_m2.shape:
        logger.error("Lambda^2 and weight arrays must be the same shape.")
        return None, None
    if not data_Q.shape == data_U.shape:
        logger.error("Stokes Q and U data arrays must be the same shape.")
        return None, None
    nDims = len(data_Q.shape)
    if not nDims <= 3:
        logger.error("data dimensions must be <= 3.")
        return None, None
    if not data_Q.shape[0] == lambda_sq_arr_m2.shape[0]:
        logger.error(f"Data depth does not match lambda^2 vector ({data_Q.shape[0]} vs {lambda_sq_arr_m2.shape[0]}).")
        logger.error("Check that data is in [z, y, x] order.")
        return None, None

    # Reshape the data arrays to 3 dimensions
    if nDims == 1:
        data_Q = np.reshape(data_Q, (data_Q.shape[0], 1, 1))
        data_U = np.reshape(data_U, (data_U.shape[0], 1, 1))
    elif nDims == 2:
        data_Q = np.reshape(data_Q, (data_Q.shape[0], data_Q.shape[1], 1))
        data_U = np.reshape(data_U, (data_U.shape[0], data_U.shape[1], 1))

    # Create a complex polarised cube, B&dB Eqns. (8) and (14)
    # Array has dimensions [nFreq, nY, nX]
    pCube = (data_Q + 1j * data_U) * weight_arr[:, np.newaxis, np.newaxis]

    # Check for NaNs (flagged data) in the cube & set to zero
    mskCube = np.isnan(pCube)
    pCube = np.nan_to_num(pCube)

    # If full planes are flagged then set corresponding weights to zero
    mskPlanes = np.sum(np.sum(~mskCube, axis=1), axis=1)
    mskPlanes = np.where(mskPlanes == 0, 0, 1)
    weight_arr *= mskPlanes

    # Initialise the complex Faraday Dispersion Function cube
    nX = data_Q.shape[-1]
    nY = data_Q.shape[-2]
    nPhi = phi_arr_radm2.shape[0]
    FDFcube = np.zeros((nPhi, nY, nX), dtype=dtComplex)

    # lam0_sq_m2 is the weighted mean of lambda^2 distribution (B&dB Eqn. 32)
    # Calculate a global lam0_sq_m2 value, ignoring isolated flagged voxels
    K = 1.0 / np.sum(weight_arr)
    if lam0_sq_m2 is None:
        lam0_sq_m2 = K * np.sum(weight_arr * lambda_sq_arr_m2)
    if not np.isfinite(lam0_sq_m2):  # Can happen if all channels are NaNs/zeros
        lam0_sq_m2 = 0.0

    # The K value used to scale each FDF spectrum must take into account
    # flagged voxels data in the datacube and can be position dependent
    weightCube = np.invert(mskCube) * weight_arr[:, np.newaxis, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        KArr = np.true_divide(1.0, np.sum(weightCube, axis=0))
        KArr[KArr == np.inf] = 0
        KArr = np.nan_to_num(KArr)

    # Do the RM-synthesis on each plane
    a = lambda_sq_arr_m2 - lam0_sq_m2
    for i, phi_plane in tqdm(enumerate(phi_arr_radm2), total=nPhi, desc="RM-synthesis by channel"):
        arg = np.exp(-2.0j * phi_plane * a)[:, np.newaxis, np.newaxis]
        FDFcube[i, :, :] = KArr * np.sum(pCube * arg, axis=0)

    # Remove redundant dimensions in the FDF array
    FDFcube = np.squeeze(FDFcube)

    return FDFcube, lam0_sq_m2


# -----------------------------------------------------------------------------#
def get_rmsf_planes(
    lambda_sq_arr_m2,
    phi_arr_radm2,
    weight_arr=None,
    mskArr=None,
    lam0_sq_m2=None,
    double=True,
    fitRMSF=False,
    fitRMSFreal=False,
    n_bits=32,
    verbose=False,
    log=print,
):
    """Calculate the Rotation Measure Spread Function from inputs. This version
    returns a cube (1, 2 or 3D) of RMSF spectra based on the shape of a
    boolean mask array, where flagged data are True and unflagged data False.
    If only whole planes (wavelength channels) are flagged then the RMSF is the
    same for all pixels and the calculation is done once and replicated to the
    dimensions of the mask. If some isolated voxels are flagged then the RMSF
    is calculated by looping through each wavelength plane, which can take some
    time. By default the routine returns the analytical width of the RMSF main
    lobe but can also use MPFIT to fit a Gaussian.

    lambda_sq_arr_m2  ... vector of wavelength^2 values (assending freq order)
    phi_arr_radm2    ... vector of trial Faraday depth values
    weight_arr       ... vector of weights, default [None] is no weighting
    maskArr         ... cube of mask values used to shape return cube [None]
    lam0_sq_m2       ... force a reference lambda^2 value (def=calculate) [None]
    double          ... pad the Faraday depth to double-size [True]
    fitRMSF         ... fit the main lobe of the RMSF with a Gaussian [False]
    fitRMSFreal     ... fit RMSF.real, rather than abs(RMSF) [False]
    n_bits           ... precision of data arrays [32]
    verbose         ... print feedback during calculation [False]
    log             ... function to be used to output messages [print]

    """

    # Default data types
    dtFloat = "float" + str(n_bits)
    dtComplex = "complex" + str(2 * n_bits)

    # For cleaning the RMSF should extend by 1/2 on each side in phi-space
    if double:
        nPhi = phi_arr_radm2.shape[0]
        nExt = np.ceil(nPhi / 2.0)
        resampIndxArr = np.arange(2.0 * nExt + nPhi) - nExt
        phi2Arr = extrap(resampIndxArr, np.arange(nPhi, dtype="int"), phi_arr_radm2)
    else:
        phi2Arr = phi_arr_radm2

    # Set the weight array
    if weight_arr is None:
        weight_arr = np.ones(lambda_sq_arr_m2.shape, dtype=dtFloat)
    weight_arr = np.where(np.isnan(weight_arr), 0.0, weight_arr)

    # Set the mask array (default to 1D, no masked channels)
    if mskArr is None:
        mskArr = np.zeros_like(lambda_sq_arr_m2, dtype="bool")
        nDims = 1
    else:
        mskArr = mskArr.astype("bool")
        nDims = len(mskArr.shape)

    # Sanity checks on array sizes
    if not weight_arr.shape == lambda_sq_arr_m2.shape:
        logger.error("wavelength^2 and weight arrays must be the same shape.")
        return None, None, None, None
    if not nDims <= 3:
        logger.error("mask dimensions must be <= 3.")
        return None, None, None, None
    if not mskArr.shape[0] == lambda_sq_arr_m2.shape[0]:
        logger.error("mask depth does not match lambda^2 vector (%d vs %d).", end=" ")
        (mskArr.shape[0], lambda_sq_arr_m2.shape[-1])
        log("     Check that the mask is in [z, y, x] order.")
        return None, None, None, None

    # Reshape the mask array to 3 dimensions
    if nDims == 1:
        mskArr = np.reshape(mskArr, (mskArr.shape[0], 1, 1))
    elif nDims == 2:
        mskArr = np.reshape(mskArr, (mskArr.shape[0], mskArr.shape[1], 1))

    # Create a unit cube for use in RMSF calculation (negative of mask)
    # CVE: unit cube removed: it wasn't accurate for non-uniform weights, and was no longer used

    # Initialise the complex RM Spread Function cube
    nX = mskArr.shape[-1]
    nY = mskArr.shape[-2]
    nPix = nX * nY
    nPhi = phi2Arr.shape[0]
    RMSFcube = np.ones((nPhi, nY, nX), dtype=dtComplex)

    # If full planes are flagged then set corresponding weights to zero
    xySum = np.sum(np.sum(mskArr, axis=1), axis=1)
    mskPlanes = np.where(xySum == nPix, 0, 1)
    weight_arr *= mskPlanes

    # Check for isolated clumps of flags (# flags in a plane not 0 or nPix)
    flagTotals = np.unique(xySum).tolist()
    try:
        flagTotals.remove(0)
    except Exception:
        pass
    try:
        flagTotals.remove(nPix)
    except Exception:
        pass
    do1Dcalc = True
    if len(flagTotals) > 0:
        do1Dcalc = False

    # lam0Sq is the weighted mean of LambdaSq distribution (B&dB Eqn. 32)
    # Calculate a single lam0_sq_m2 value, ignoring isolated flagged voxels
    K = 1.0 / np.nansum(weight_arr)
    lam0_sq_m2 = K * np.nansum(weight_arr * lambda_sq_arr_m2)

    # Calculate the analytical FWHM width of the main lobe
    fwhmRMSF = 3.8 / (np.nanmax(lambda_sq_arr_m2) - np.nanmin(lambda_sq_arr_m2))

    # Do a simple 1D calculation and replicate along X & Y axes
    if do1Dcalc:
        if verbose:
            log("Calculating 1D RMSF and replicating along X & Y axes.")

        # Calculate the RMSF
        a = (-2.0 * 1j * phi2Arr).astype(dtComplex)
        b = lambda_sq_arr_m2 - lam0_sq_m2
        RMSFArr = K * np.sum(weight_arr * np.exp(np.outer(a, b)), 1)

        # Fit the RMSF main lobe
        fitStatus = -1
        if fitRMSF:
            if verbose:
                log("Fitting Gaussian to the main lobe.")
            if fitRMSFreal:
                mp = fit_rmsf(phi2Arr, RMSFArr.real)
            else:
                mp = fit_rmsf(phi2Arr, np.abs(RMSFArr))
            if mp is None or mp.status < 1:
                pass
                logger.error("failed to fit the RMSF.")
                log("     Defaulting to analytical value.")
            else:
                fwhmRMSF = mp.params[2]
                fitStatus = mp.status

        # Replicate along X and Y axes
        RMSFcube = np.tile(RMSFArr[:, np.newaxis, np.newaxis], (1, nY, nX))
        fwhmRMSFArr = np.ones((nY, nX), dtype=dtFloat) * fwhmRMSF
        statArr = np.ones((nY, nX), dtype="int") * fitStatus

    # Calculate the RMSF at each pixel
    else:
        if verbose:
            log("Calculating RMSF by channel.")

        # The K value used to scale each RMSF must take into account
        # isolated flagged voxels data in the datacube
        weightCube = np.invert(mskArr) * weight_arr[:, np.newaxis, np.newaxis]
        with np.errstate(divide="ignore", invalid="ignore"):
            KArr = np.true_divide(1.0, np.sum(weightCube, axis=0))
            KArr[KArr == np.inf] = 0
            KArr = np.nan_to_num(KArr)

        # Calculate the RMSF for each plane
        if verbose:
            progress(40, 0)
        a = lambda_sq_arr_m2 - lam0_sq_m2
        for i in range(nPhi):
            if verbose:
                progress(40, ((i + 1) * 100.0 / nPhi))
            arg = np.exp(-2.0j * phi2Arr[i] * a)[:, np.newaxis, np.newaxis]
            RMSFcube[i, :, :] = KArr * np.sum(weightCube * arg, axis=0)

        # Default to the analytical RMSF
        fwhmRMSFArr = np.ones((nY, nX), dtype=dtFloat) * fwhmRMSF
        statArr = np.ones((nY, nX), dtype="int") * (-1)

        # Fit the RMSF main lobe
        if fitRMSF:
            if verbose:
                log("Fitting main lobe in each RMSF spectrum.")
                log("> This may take some time!")
                progress(40, 0)
            k = 0
            for i in range(nX):
                for j in range(nY):
                    k += 1
                    if verbose:
                        progress(40, ((i + 1) * 100.0 / nPhi))
                    if fitRMSFreal:
                        mp = fit_rmsf(phi2Arr, RMSFcube[:, j, i].real)
                    else:
                        mp = fit_rmsf(phi2Arr, np.abs(RMSFcube[:, j, i]))
                    if not (mp is None or mp.status < 1):
                        fwhmRMSFArr[j, i] = mp.params[2]
                        statArr[j, i] = mp.status

    # Remove redundant dimensions
    RMSFcube = np.squeeze(RMSFcube)
    fwhmRMSFArr = np.squeeze(fwhmRMSFArr)
    statArr = np.squeeze(statArr)

    return RMSFcube, phi2Arr, fwhmRMSFArr, statArr


# -----------------------------------------------------------------------------#
def do_rmclean_hogbom(
    dirtyFDF,
    phi_arr_radm2,
    RMSFArr,
    phi2Arr_radm2,
    fwhmRMSFArr,
    cutoff,
    maxIter=1000,
    gain=0.1,
    mskArr=None,
    n_bits=32,
    verbose=False,
    doPlots=False,
    pool=None,
    chunksize=None,
    log=print,
    window=0,
):
    """Perform Hogbom CLEAN on a cube of complex Faraday dispersion functions
    given a cube of rotation measure spread functions.

    dirtyFDF       ... 1, 2 or 3D complex FDF array
    phi_arr_radm2   ... 1D Faraday depth array corresponding to the FDF
    RMSFArr        ... 1, 2 or 3D complex RMSF array
    phi2Arr_radm2  ... double size 1D Faraday depth array of the RMSF
    fwhmRMSFArr    ... scalar, 1D or 2D array of RMSF main lobe widths
    cutoff         ... clean cutoff (+ve = absolute values, -ve = sigma) [-1]
    maxIter        ... maximun number of CLEAN loop interations [1000]
    gain           ... CLEAN loop gain [0.1]
    mskArr         ... scalar, 1D or 2D pixel mask array [None]
    n_bits          ... precision of data arrays [32]
    verbose        ... print feedback during calculation [False]
    doPlots        ... plot the final CLEAN FDF [False]
    pool           ... thread pool for multithreading (from schwimmbad) [None]
    chunksize      ... number of pixels to be given per thread (for 3D) [None]
    log            ... function to be used to output messages [print]
    window         ... Only clean in ±RMSF_FWHM window around first peak [False]

    """

    # Default data types
    dtFloat = "float" + str(n_bits)
    dtComplex = "complex" + str(2 * n_bits)

    # Sanity checks on array sizes
    nPhi = phi_arr_radm2.shape[0]
    if nPhi != dirtyFDF.shape[0]:
        logger.error("'phi2Arr_radm2' and 'dirtyFDF' are not the same length.")
        return None, None, None
    nPhi2 = phi2Arr_radm2.shape[0]
    if not nPhi2 == RMSFArr.shape[0]:
        logger.error("missmatch in 'phi2Arr_radm2' and 'RMSFArr' length.")
        return None, None, None
    if not (nPhi2 >= 2 * nPhi):
        logger.error("the Faraday depth of the RMSF must be twice the FDF.")
        return None, None, None
    nDims = len(dirtyFDF.shape)
    if not nDims <= 3:
        logger.error("FDF array dimensions must be <= 3.")
        return None, None, None
    if not nDims == len(RMSFArr.shape):
        logger.error("the input RMSF and FDF must have the same number of axes.")
        return None, None, None
    if not RMSFArr.shape[1:] == dirtyFDF.shape[1:]:
        logger.error("the xy dimesions of the RMSF and FDF must match.")
        return None, None, None
    if mskArr is not None:
        if not mskArr.shape == dirtyFDF.shape[1:]:
            logger.error("pixel mask must match xy dimesnisons of FDF cube.")
            log(
                "     FDF[z,y,z] = {:}, Mask[y,x] = {:}.".format(
                    dirtyFDF.shape, mskArr.shape
                ),
                end=" ",
            )

            return None, None, None
    else:
        mskArr = np.ones(dirtyFDF.shape[1:], dtype="bool")

    # Reshape the FDF & RMSF array to 3 dimensions and mask array to 2
    if nDims == 1:
        dirtyFDF = np.reshape(dirtyFDF, (dirtyFDF.shape[0], 1, 1))
        RMSFArr = np.reshape(RMSFArr, (RMSFArr.shape[0], 1, 1))
        mskArr = np.reshape(mskArr, (1, 1))
        fwhmRMSFArr = np.reshape(fwhmRMSFArr, (1, 1))
    elif nDims == 2:
        dirtyFDF = np.reshape(dirtyFDF, list(dirtyFDF.shape[:2]) + [1])
        RMSFArr = np.reshape(RMSFArr, list(RMSFArr.shape[:2]) + [1])
        mskArr = np.reshape(mskArr, (dirtyFDF.shape[1], 1))
        fwhmRMSFArr = np.reshape(fwhmRMSFArr, (dirtyFDF.shape[1], 1))
    iterCountArr = np.zeros_like(mskArr, dtype="int")

    # Determine which pixels have components above the cutoff
    absFDF = np.abs(np.nan_to_num(dirtyFDF))
    mskCutoff = np.where(np.max(absFDF, axis=0) >= cutoff, 1, 0)
    xyCoords = np.rot90(np.where(mskCutoff > 0))

    # Feeback to user
    if verbose:
        nPix = dirtyFDF.shape[-1] * dirtyFDF.shape[-2]
        nCleanPix = len(xyCoords)
        log("Cleaning {:}/{:} spectra.".format(nCleanPix, nPix))

    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    # Residual is initially copies of dirty FDF, so that pixels that are not
    #  processed get correct values (but will be overridden when processed)
    residFDF = dirtyFDF.copy()
    ccArr = np.zeros(dirtyFDF.shape, dtype=dtComplex)
    cleanFDF = np.zeros_like(dirtyFDF)

    # Loop through the pixels containing a polarised signal
    inputs = [[yi, xi, dirtyFDF] for yi, xi in xyCoords]
    rmc = RMcleaner(
        RMSFArr,
        phi2Arr_radm2,
        phi_arr_radm2,
        fwhmRMSFArr,
        iterCountArr,
        maxIter,
        gain,
        cutoff,
        n_bits,
        verbose,
        window,
    )

    if pool is None:
        if verbose:
            progress(40, 0)
            i = 0
        output = []
        for pix in inputs:
            output.append(rmc.cleanloop(pix))
            if verbose:
                progress(40, ((i) * 100.0 / nCleanPix))
                i += 1
    else:
        if verbose:
            log(
                "(Progress bar is not supported for parallel mode. Please wait for the code to finish."
            )
        if chunksize is not None:
            output = list(pool.map(rmc.cleanloop, inputs, chunksize=chunksize))
        else:
            output = list(pool.map(rmc.cleanloop, inputs))
        pool.close()
    # Put data back in correct shape
    #    ccArr = np.reshape(np.rot90(np.stack([model for _, _, model in output]), k=-1),dirtyFDF.shape)
    #    cleanFDF = np.reshape(np.rot90(np.stack([clean for clean, _, _ in output]), k=-1),dirtyFDF.shape)
    #    residFDF = np.reshape(np.rot90(np.stack([resid for _, resid, _ in output]), k=-1),dirtyFDF.shape)
    for i in range(len(inputs)):
        yi = inputs[i][0]
        xi = inputs[i][1]
        ccArr[:, yi, xi] = output[i][2]
        cleanFDF[:, yi, xi] = output[i][0]
        residFDF[:, yi, xi] = output[i][1]

    # Restore the residual to the CLEANed FDF (moved outside of loop:
    # will now work for pixels/spectra without clean components)
    cleanFDF += residFDF

    # Remove redundant dimensions
    cleanFDF = np.squeeze(cleanFDF)
    ccArr = np.squeeze(ccArr)
    iterCountArr = np.squeeze(iterCountArr)
    residFDF = np.squeeze(residFDF)

    return cleanFDF, ccArr, iterCountArr, residFDF


# -----------------------------------------------------------------------------#


class RMcleaner:
    """Allows do_rmclean_hogbom to be run in parallel
    Designed around use of schwimmbad parallelization tools.
    """

    def __init__(
        self,
        RMSFArr,
        phi2Arr_radm2,
        phi_arr_radm2,
        fwhmRMSFArr,
        iterCountArr,
        maxIter=1000,
        gain=0.1,
        cutoff=0,
        nbits=32,
        verbose=False,
        window=0,
    ):
        self.RMSFArr = RMSFArr
        self.phi2Arr_radm2 = phi2Arr_radm2
        self.phi_arr_radm2 = phi_arr_radm2
        self.fwhmRMSFArr = fwhmRMSFArr
        self.iterCountArr = iterCountArr
        self.maxIter = maxIter
        self.gain = gain
        self.cutoff = cutoff
        self.verbose = verbose
        self.nbits = nbits
        self.window = window

    def cleanloop(self, args):
        return self._cleanloop(*args)

    def _cleanloop(self, yi, xi, dirtyFDF):
        dirtyFDF = dirtyFDF[:, yi, xi]
        # Initialise arrays to hold the residual FDF, clean components, clean FDF
        residFDF = dirtyFDF.copy()
        ccArr = np.zeros_like(dirtyFDF)
        cleanFDF = np.zeros_like(dirtyFDF)
        RMSFArr = self.RMSFArr[:, yi, xi]
        fwhmRMSFArr = self.fwhmRMSFArr[yi, xi]

        # Find the index of the peak of the RMSF
        indxMaxRMSF = np.nanargmax(RMSFArr)

        # Calculate the padding in the sampled RMSF
        # Assumes only integer shifts and symmetric
        nPhiPad = int((len(self.phi2Arr_radm2) - len(self.phi_arr_radm2)) / 2)

        iterCount = 0
        while np.max(np.abs(residFDF)) >= self.cutoff and iterCount < self.maxIter:
            # Get the absolute peak channel, values and Faraday depth
            indxPeakFDF = np.argmax(np.abs(residFDF))
            peakFDFval = residFDF[indxPeakFDF]
            phiPeak = self.phi_arr_radm2[indxPeakFDF]

            # A clean component is "loop-gain * peakFDFval
            CC = self.gain * peakFDFval
            ccArr[indxPeakFDF] += CC

            # At which channel is the CC located at in the RMSF?
            indxPeakRMSF = indxPeakFDF + nPhiPad

            # Shift the RMSF & clip so that its peak is centred above this CC
            shiftedRMSFArr = np.roll(RMSFArr, indxPeakRMSF - indxMaxRMSF)[
                nPhiPad:-nPhiPad
            ]

            # Subtract the product of the CC shifted RMSF from the residual FDF
            residFDF -= CC * shiftedRMSFArr

            # Restore the CC * a Gaussian to the cleaned FDF
            cleanFDF += gauss1D(CC, phiPeak, fwhmRMSFArr)(self.phi_arr_radm2)
            iterCount += 1
            self.iterCountArr[yi, xi] = iterCount

        # Create a mask for the pixels that have been cleaned
        mask = np.abs(ccArr) > 0
        dPhi = self.phi_arr_radm2[1] - self.phi_arr_radm2[0]
        fwhmRMSFArr_pix = fwhmRMSFArr / dPhi
        for i in np.where(mask)[0]:
            start = int(i - fwhmRMSFArr_pix / 2)
            end = int(i + fwhmRMSFArr_pix / 2)
            mask[start:end] = True
        residFDF_mask = np.ma.array(residFDF, mask=~mask)
        # Clean again within mask
        while (
            np.ma.max(np.ma.abs(residFDF_mask)) >= self.window
            and iterCount < self.maxIter
        ):
            if residFDF_mask.mask.all():
                break
            # Get the absolute peak channel, values and Faraday depth
            indxPeakFDF = np.ma.argmax(np.abs(residFDF_mask))
            peakFDFval = residFDF_mask[indxPeakFDF]
            phiPeak = self.phi_arr_radm2[indxPeakFDF]

            # A clean component is "loop-gain * peakFDFval
            CC = self.gain * peakFDFval
            ccArr[indxPeakFDF] += CC

            # At which channel is the CC located at in the RMSF?
            indxPeakRMSF = indxPeakFDF + nPhiPad

            # Shift the RMSF & clip so that its peak is centred above this CC
            shiftedRMSFArr = np.roll(RMSFArr, indxPeakRMSF - indxMaxRMSF)[
                nPhiPad:-nPhiPad
            ]

            # Subtract the product of the CC shifted RMSF from the residual FDF
            residFDF -= CC * shiftedRMSFArr

            # Restore the CC * a Gaussian to the cleaned FDF
            cleanFDF += gauss1D(CC, phiPeak, fwhmRMSFArr)(self.phi_arr_radm2)
            iterCount += 1
            self.iterCountArr[yi, xi] = iterCount

            # Remake masked residual FDF
            residFDF_mask = np.ma.array(residFDF, mask=~mask)

        cleanFDF = np.squeeze(cleanFDF)
        residFDF = np.squeeze(residFDF)
        ccArr = np.squeeze(ccArr)

        return cleanFDF, residFDF, ccArr


# -----------------------------------------------------------------------------#
def fits_make_lin_axis(head, axis=0, dtype="f4"):
    """Create an array containing the axis values, assuming a simple linear
    projection scheme. Axis selection is zero-indexed."""

    axis = int(axis)
    if head["NAXIS"] < axis + 1:
        return []

    i = str(int(axis) + 1)
    start = head["CRVAL" + i] + (1 - head["CRPIX" + i]) * head["CDELT" + i]
    stop = (
        head["CRVAL" + i] + (head["NAXIS" + i] - head["CRPIX" + i]) * head["CDELT" + i]
    )
    nChan = int(abs(start - stop) / head["CDELT" + i] + 1)

    return np.linspace(start, stop, nChan).astype(dtype)


# -----------------------------------------------------------------------------#
def extrap(x, xp, yp):
    """
    Wrapper to allow np.interp to linearly extrapolate at function ends.

    np.interp function with linear extrapolation
    http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate
    -give-a-an-extrapolated-result-beyond-the-input-ran

    """
    y = np.interp(x, xp, yp)
    y = np.where(x < xp[0], yp[0] + (x - xp[0]) * (yp[0] - yp[1]) / (xp[0] - xp[1]), y)
    y = np.where(
        x > xp[-1], yp[-1] + (x - xp[-1]) * (yp[-1] - yp[-2]) / (xp[-1] - xp[-2]), y
    )
    return y


# -----------------------------------------------------------------------------#
def fit_rmsf(xData, yData, thresh=0.4, ampThresh=0.4):
    """
    Fit the main lobe of the RMSF with a Gaussian function.
    """

    try:

        # Detect the peak and mask off the sidelobes
        msk1 = detect_peak(yData, thresh)
        msk2 = np.where(yData < ampThresh, 0.0, msk1)
        if sum(msk2) < 4:
            msk2 = msk1
        validIndx = np.where(msk2 == 1.0)
        xData = xData[validIndx]
        yData = yData[validIndx]

        # Estimate starting parameters
        a = 1.0
        b = xData[np.argmax(yData)]
        w = np.nanmax(xData) - np.nanmin(xData)
        inParms = [
            {"value": a, "fixed": False, "parname": "amp"},
            {"value": b, "fixed": False, "parname": "offset"},
            {"value": w, "fixed": False, "parname": "width"},
        ]

        # Function which returns another function to evaluate a Gaussian
        def gauss(p):
            a, b, w = p
            gfactor = 2.0 * m.sqrt(2.0 * m.log(2.0))
            s = w / gfactor

            def rfunc(x):
                y = a * np.exp(-((x - b) ** 2.0) / (2.0 * s**2.0))
                return y

            return rfunc

        # Function to evaluate the difference between the model and data.
        # This is minimised in the least-squared sense by the fitter
        def errFn(p, fjac=None):
            status = 0
            return status, gauss(p)(xData) - yData

        # Use mpfit to perform the fitting
        mp = mpfit(errFn, parinfo=inParms, quiet=True)

        return mp

    except Exception:
        return None


# -----------------------------------------------------------------------------#
def gauss1D(amp=1.0, mean=0.0, fwhm=1.0):
    """Function which returns another function to evaluate a Gaussian"""

    gfactor = 2.0 * m.sqrt(2.0 * m.log(2.0))
    sigma = fwhm / gfactor

    def rfunc(x):
        return amp * np.exp(-((x - mean) ** 2.0) / (2.0 * sigma**2.0))

    return rfunc


# -----------------------------------------------------------------------------#


def detect_peak(a, thresh=0.4):
    """Detect the extent of the peak in the array by moving away, in both
    directions, from the peak channel amd looking for where the value drops
    below a threshold.
    Returns a mask array like the input array with 1s over the extent of the
    peak and 0s elsewhere."""

    # Find the peak
    iPk = np.argmax(a)  # If the peak is flat, this is the left index

    # find first point below threshold right of peak
    ishift = np.where(a[iPk:] < thresh)[0][0]
    iR = iPk + ishift
    iL = iPk - ishift + 1

    msk = np.zeros_like(a)
    msk[iL:iR] = 1

    # DEBUG PLOTTING
    if False:
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.step(np.arange(len(a)), a, where="mid", label="arr")
        ax.step(np.arange(len(msk)), msk * 0.5, where="mid", label="msk")
        ax.axhline(0, color="grey")
        ax.axvline(iPk, color="k", linewidth=3.0)
        ax.axhline(thresh, color="magenta", ls="--")
        ax.set_xlim([iL - 20, iR + 20])
        leg = ax.legend(
            numpoints=1,
            loc="upper right",
            shadow=False,
            borderaxespad=0.3,
            ncol=1,
            bbox_to_anchor=(1.00, 1.00),
        )
        fig.show()
    return msk


# -----------------------------------------------------------------------------#
def measure_FDF_parms(
    FDF,
    phiArr,
    fwhmRMSF,
    dFDF=None,
    lamSqArr_m2=None,
    lam0Sq=None,
    snrDoBiasCorrect=5.0,
):
    """
    Measure standard parameters from a complex Faraday Dispersion Function.
    Currently this function assumes that the noise levels in the Stokes Q
    and U spectra are the same.
    Returns a dictionary containing measured parameters.
    """

    # Determine the peak channel in the FDF, its amplitude and index
    absFDF = np.abs(FDF)
    indxPeakPIchan = (
        np.nanargmax(absFDF[1:-1]) + 1
    )  # Masks out the edge channels, since they can't be fit to.
    ampPeakPIchan = absFDF[indxPeakPIchan]

    # Measure the RMS noise in the spectrum after masking the peak
    dPhi = np.nanmin(np.diff(phiArr))
    fwhmRMSF_chan = np.ceil(fwhmRMSF / dPhi)
    iL = int(max(0, indxPeakPIchan - fwhmRMSF_chan * 2))
    iR = int(min(len(absFDF), indxPeakPIchan + fwhmRMSF_chan * 2))
    absFDFmsked = absFDF.copy()
    absFDFmsked[iL:iR] = np.nan
    absFDFmsked = absFDFmsked[np.where(absFDFmsked == absFDFmsked)]
    if float(len(absFDFmsked)) / len(absFDF) < 0.3:
        dFDFcorMAD = MAD(absFDF)
        dFDFrms = np.sqrt(np.mean(absFDF**2))
    else:
        dFDFcorMAD = MAD(absFDFmsked)
        dFDFrms = np.sqrt(np.mean(absFDFmsked**2))

    # Default to using the measured FDF if a noise value has not been provided
    if dFDF is None:
        dFDF = dFDFcorMAD

    # Measure the RM of the peak channel
    phiPeakPIchan = phiArr[indxPeakPIchan]
    dPhiPeakPIchan = fwhmRMSF * dFDF / (2.0 * ampPeakPIchan)
    snrPIchan = ampPeakPIchan / dFDF

    # Correct the peak for polarisation bias (POSSUM report 11)
    ampPeakPIchanEff = ampPeakPIchan
    if snrPIchan >= snrDoBiasCorrect:
        ampPeakPIchanEff = np.sqrt(ampPeakPIchan**2.0 - 2.3 * dFDF**2.0)

    # Calculate the polarisation angle from the channel
    peakFDFimagChan = FDF.imag[indxPeakPIchan]
    peakFDFrealChan = FDF.real[indxPeakPIchan]
    polAngleChan_deg = (
        0.5 * np.degrees(np.arctan2(peakFDFimagChan, peakFDFrealChan)) % 180
    )
    dPolAngleChan_deg = np.degrees(dFDF / (2.0 * ampPeakPIchan))

    # Calculate the derotated polarisation angle and uncertainty
    polAngle0Chan_deg = (
        np.degrees(np.radians(polAngleChan_deg) - phiPeakPIchan * lam0Sq) % 180
    )
    nChansGood = np.sum(np.where(lamSqArr_m2 == lamSqArr_m2, 1.0, 0.0))
    varLamSqArr_m2 = (
        np.sum(lamSqArr_m2**2.0) - np.sum(lamSqArr_m2) ** 2.0 / nChansGood
    ) / (nChansGood - 1)
    dPolAngle0Chan_rad = np.sqrt(
        dFDF**2.0
        * nChansGood
        / (4.0 * (nChansGood - 2.0) * ampPeakPIchan**2.0)
        * ((nChansGood - 1) / nChansGood + lam0Sq**2.0 / varLamSqArr_m2)
    )
    dPolAngle0Chan_deg = np.degrees(dPolAngle0Chan_rad)

    # Determine the peak in the FDF, its amplitude and Phi using a
    # 3-point parabolic interpolation
    phiPeakPIfit = None
    dPhiPeakPIfit = None
    ampPeakPIfit = None
    snrPIfit = None
    ampPeakPIfitEff = None
    indxPeakPIfit = None
    peakFDFimagFit = None
    peakFDFrealFit = None
    polAngleFit_deg = None
    dPolAngleFit_deg = None
    polAngle0Fit_deg = None
    dPolAngle0Fit_deg = None

    # Only do the 3-point fit if peak is 1-channel from either edge
    if indxPeakPIchan > 0 and indxPeakPIchan < len(FDF) - 1:
        phiPeakPIfit, ampPeakPIfit = calc_parabola_vertex(
            phiArr[indxPeakPIchan - 1],
            absFDF[indxPeakPIchan - 1],
            phiArr[indxPeakPIchan],
            absFDF[indxPeakPIchan],
            phiArr[indxPeakPIchan + 1],
            absFDF[indxPeakPIchan + 1],
        )

        snrPIfit = ampPeakPIfit / dFDF

        # In rare cases, a parabola can be fitted to the edge of the spectrum,
        # producing a unreasonably large RM and polarized intensity.
        # In these cases, everything should get NaN'd out.
        if np.abs(phiPeakPIfit) > np.max(np.abs(phiArr)):
            phiPeakPIfit = np.nan
            ampPeakPIfit = np.nan

        # Error on fitted Faraday depth (RM) is same as channel, but using fitted PI
        dPhiPeakPIfit = fwhmRMSF * dFDF / (2.0 * ampPeakPIfit)

        # Correct the peak for polarisation bias (POSSUM report 11)
        ampPeakPIfitEff = ampPeakPIfit
        if snrPIfit >= snrDoBiasCorrect:
            ampPeakPIfitEff = np.sqrt(ampPeakPIfit**2.0 - 2.3 * dFDF**2.0)

        # Calculate the polarisation angle from the fitted peak
        # Uncertainty from Eqn A.12 in Brentjens & De Bruyn 2005
        indxPeakPIfit = np.interp(
            phiPeakPIfit, phiArr, np.arange(phiArr.shape[-1], dtype="f4")
        )
        peakFDFimagFit = np.interp(phiPeakPIfit, phiArr, FDF.imag)
        peakFDFrealFit = np.interp(phiPeakPIfit, phiArr, FDF.real)
        polAngleFit_deg = (
            0.5 * np.degrees(np.arctan2(peakFDFimagFit, peakFDFrealFit)) % 180
        )
        dPolAngleFit_deg = np.degrees(dFDF / (2.0 * ampPeakPIfit))

        # Calculate the derotated polarisation angle and uncertainty
        # Uncertainty from Eqn A.20 in Brentjens & De Bruyn 2005
        polAngle0Fit_deg = (
            np.degrees(np.radians(polAngleFit_deg) - phiPeakPIfit * lam0Sq)
        ) % 180
        dPolAngle0Fit_rad = np.sqrt(
            dFDF**2.0
            * nChansGood
            / (4.0 * (nChansGood - 2.0) * ampPeakPIfit**2.0)
            * ((nChansGood - 1) / nChansGood + lam0Sq**2.0 / varLamSqArr_m2)
        )
        dPolAngle0Fit_deg = np.degrees(dPolAngle0Fit_rad)

    # Store the measurements in a dictionary and return
    mDict = {
        "dFDFcorMAD": toscalar(dFDFcorMAD),
        "dFDFrms": toscalar(dFDFrms),
        "phiPeakPIchan_rm2": toscalar(phiPeakPIchan),
        "dPhiPeakPIchan_rm2": toscalar(dPhiPeakPIchan),
        "ampPeakPIchan": toscalar(ampPeakPIchan),
        "ampPeakPIchanEff": toscalar(ampPeakPIchanEff),
        "dAmpPeakPIchan": toscalar(dFDF),
        "snrPIchan": toscalar(snrPIchan),
        "indxPeakPIchan": toscalar(indxPeakPIchan),
        "peakFDFimagChan": toscalar(peakFDFimagChan),
        "peakFDFrealChan": toscalar(peakFDFrealChan),
        "polAngleChan_deg": toscalar(polAngleChan_deg),
        "dPolAngleChan_deg": toscalar(dPolAngleChan_deg),
        "polAngle0Chan_deg": toscalar(polAngle0Chan_deg),
        "dPolAngle0Chan_deg": toscalar(dPolAngle0Chan_deg),
        "phiPeakPIfit_rm2": toscalar(phiPeakPIfit),
        "dPhiPeakPIfit_rm2": toscalar(dPhiPeakPIfit),
        "ampPeakPIfit": toscalar(ampPeakPIfit),
        "ampPeakPIfitEff": toscalar(ampPeakPIfitEff),
        "dAmpPeakPIfit": toscalar(dFDF),
        "snrPIfit": toscalar(snrPIfit),
        "indxPeakPIfit": toscalar(indxPeakPIfit),
        "peakFDFimagFit": toscalar(peakFDFimagFit),
        "peakFDFrealFit": toscalar(peakFDFrealFit),
        "polAngleFit_deg": toscalar(polAngleFit_deg),
        "dPolAngleFit_deg": toscalar(dPolAngleFit_deg),
        "polAngle0Fit_deg": toscalar(polAngle0Fit_deg),
        "dPolAngle0Fit_deg": toscalar(dPolAngle0Fit_deg),
    }

    return mDict


# -----------------------------------------------------------------------------#
def norm_cdf(mean=0.0, std=1.0, N=50, xArr=None):
    """Return the CDF of a normal distribution between -6 and 6 sigma, or at
    the values of an input array."""

    if xArr is None:
        x = np.linspace(-6.0 * std, 6.0 * std, N)
    else:
        x = xArr
    y = norm.cdf(x, loc=mean, scale=std)

    return x, y


# -----------------------------------------------------------------------------#
def cdf_percentile(x, p, q=50.0):
    """Return the value at a given percentile of a cumulative distribution
    function."""

    # Determine index where cumulative percentage is achieved
    try:  # Can fail if NaNs present, so return NaN in this case.
        i = np.where(p > q / 100.0)[0][0]
    except:
        return np.nan

    # If at extremes of the distribution, return the limiting value
    if i == 0 or i == len(x):
        return x[i]

    # or interpolate between the two bracketing values in the CDF
    else:
        m = (p[i] - p[i - 1]) / (x[i] - x[i - 1])
        c = p[i] - m * x[i]
        return (q / 100.0 - c) / m


# -----------------------------------------------------------------------------#
def calc_sigma_add(xArr, yArr, dyArr, yMed=None, noise=None, nSamp=1000, suffix=""):
    """Calculate the most likely value of additional scatter, assuming the
    input data is drawn from a normal distribution. The total uncertainty on
    each data point Y_i is modelled as dYtot_i**2 = dY_i**2 + dYadd**2."""

    # Measure the median and MADFM of the input data if not provided.
    # Used to overplot a normal distribution when debugging.
    if yMed is None:
        yMed = np.median(yArr)
    if noise is None:
        noise = MAD(yArr)

    # Sample the PDF of the additional noise term from a limit near zero to
    # a limit of the range of the data, including error bars
    yRng = np.nanmax(yArr + dyArr) - np.nanmin(yArr - dyArr)
    sigmaAddArr = np.linspace(yRng / nSamp, yRng, nSamp)

    # Model deviation from Gaussian as an additional noise term.
    # Loop through the range of i additional noise samples and calculate
    # chi-squared and sum(ln(sigma_total)), used later to calculate likelihood.
    nData = len(xArr)
    chiSqArr = np.zeros_like(sigmaAddArr)
    lnSigmaSumArr = np.zeros_like(sigmaAddArr)
    for i, sigmaAdd in enumerate(sigmaAddArr):
        sigmaSqTot = dyArr**2.0 + sigmaAdd**2.0
        lnSigmaSumArr[i] = np.nansum(np.log(np.sqrt(sigmaSqTot)))
        chiSqArr[i] = np.nansum((yArr - yMed) ** 2.0 / sigmaSqTot)
    dof = nData - 1
    chiSqRedArr = chiSqArr / dof

    # Calculate the PDF in log space and normalise the peak to 1
    lnProbArr = (
        -np.log(sigmaAddArr)
        - nData * np.log(2.0 * np.pi) / 2.0
        - lnSigmaSumArr
        - chiSqArr / 2.0
    )
    lnProbArr -= np.nanmax(lnProbArr)
    probArr = np.exp(lnProbArr)

    # Normalise the area under the PDF to be 1
    A = np.nansum(probArr * np.diff(sigmaAddArr)[0])
    probArr /= A

    # Calculate the cumulative PDF
    CPDF = np.cumsum(probArr) / np.nansum(probArr)

    # Calculate the mean of the distribution and the +/- 1-sigma limits
    sigmaAdd = cdf_percentile(x=sigmaAddArr, p=CPDF, q=50.0)
    sigmaAddMinus = cdf_percentile(x=sigmaAddArr, p=CPDF, q=15.72)
    sigmaAddPlus = cdf_percentile(x=sigmaAddArr, p=CPDF, q=84.27)
    mDict = {
        "sigmaAdd" + suffix: toscalar(sigmaAdd),
        "dSigmaAddMinus" + suffix: toscalar(sigmaAdd - sigmaAddMinus),
        "dSigmaAddPlus" + suffix: toscalar(sigmaAddPlus - sigmaAdd),
    }

    # Return the curves to be plotted in a separate dictionary
    pltDict = {
        "sigmaAddArr" + suffix: sigmaAddArr,
        "chiSqRedArr" + suffix: chiSqRedArr,
        "probArr" + suffix: probArr,
        "xArr" + suffix: xArr,
        "yArr" + suffix: yArr,
        "dyArr" + suffix: dyArr,
    }

    # DEBUG PLOTS
    if False:

        # Setup for the figure
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(18.0, 10.0))

        # Plot the data and the +/- 1-sigma levels
        ax1 = fig.add_subplot(231)
        ax1.errorbar(x=xArr, y=yArr, yerr=dyArr, ms=4, fmt="o")
        ax1.axhline(yMed, color="grey", zorder=10)
        ax1.axhline(yMed + noise, color="r", linestyle="--", zorder=10)
        ax1.axhline(yMed - noise, color="r", linestyle="--", zorder=10)
        ax1.set_title(r"Input Data")
        ax1.set_xlabel(r"$\lambda^2$")
        ax1.set_ylabel("Amplitude")

        # Plot the histogram of the data overlaid by the normal distribution
        H = 1.0 / np.sqrt(2.0 * np.pi * noise**2.0)
        xNorm = np.linspace(yMed - 3 * noise, yMed + 3 * noise, 1000)
        yNorm = H * np.exp(-0.5 * ((xNorm - yMed) / noise) ** 2.0)
        fwhm = noise * (2.0 * np.sqrt(2.0 * np.log(2.0)))
        ax2 = fig.add_subplot(232)
        nBins = 15
        n, b, p = ax2.hist(yArr, nBins, normed=1, histtype="step")
        ax2.plot(xNorm, yNorm, color="k", linestyle="--", linewidth=2)
        ax2.axvline(yMed, color="grey", zorder=11)
        ax2.axvline(yMed + fwhm / 2.0, color="r", linestyle="--", zorder=11)
        ax2.axvline(yMed - fwhm / 2.0, color="r", linestyle="--", zorder=11)
        ax2.set_title(r"Distribution of Data Compared to Normal")
        ax2.set_xlabel(r"Amplitude")
        ax2.set_ylabel(r"Normalised Counts")

        # Plot the ECDF versus a normal CDF
        ecdfArr = np.array(list(range(nData))) / float(nData)
        ySrtArr = np.sort(yArr)
        ax3 = fig.add_subplot(233)
        ax3.step(ySrtArr, ecdfArr, where="mid")
        x, y = norm_cdf(mean=yMed, std=noise, N=1000)
        ax3.plot(x, y, color="k", linewidth=2, linestyle="--", zorder=1)
        ax3.set_title(r"CDF of Data Compared to Normal")
        ax3.set_xlabel(r"Amplitude")
        ax3.set_ylabel(r"Normalised Counts")

        # Plot reduced chi-squared
        ax4 = fig.add_subplot(234)
        ax4.step(x=sigmaAddArr, y=chiSqRedArr, linewidth=1.5, where="mid")
        ax4.axhline(1.0, color="r", linestyle="--")
        ax4.set_title(r"$\chi^2_{\rm reduced}$ vs $\sigma_{\rm additional}$")
        ax4.set_xlabel(r"$\sigma_{\rm additional}$")
        ax4.set_ylabel(r"$\chi^2_{\rm reduced}$")

        # Plot the probability distribution function
        ax5 = fig.add_subplot(235)
        ax5.step(x=sigmaAddArr, y=probArr, linewidth=1.5, where="mid")
        ax5.axvline(sigmaAdd, color="grey", linestyle="-", linewidth=1.5)
        ax5.axvline(sigmaAddMinus, color="r", linestyle="--", linewidth=1.0)
        ax5.axvline(sigmaAddPlus, color="r", linestyle="--", linewidth=1.0)
        ax5.set_title("Relative Likelihood")
        ax5.set_xlabel(r"$\sigma_{\rm additional}$")
        ax5.set_ylabel(r"P($\sigma_{\rm additional}$|data)")

        # Plot the CPDF
        ax6 = fig.add_subplot(236)
        ax6.step(x=sigmaAddArr, y=CPDF, linewidth=1.5, where="mid")
        ax6.set_ylim(0, 1.05)
        ax6.axvline(sigmaAdd, color="grey", linestyle="-", linewidth=1.5)
        ax6.axhline(0.5, color="grey", linestyle="-", linewidth=1.5)
        ax6.axvline(sigmaAddMinus, color="r", linestyle="--", linewidth=1.0)
        ax6.axvline(sigmaAddPlus, color="r", linestyle="--", linewidth=1.0)
        ax6.set_title("Cumulative Likelihood")
        ax6.set_xlabel(r"$\sigma_{\rm additional}$")
        ax6.set_ylabel(r"Cumulative Likelihood")

        # Zoom in
        ax5.set_xlim(0, sigmaAdd + (sigmaAddPlus - sigmaAdd) * 4.0)
        ax6.set_xlim(0, sigmaAdd + (sigmaAddPlus - sigmaAdd) * 4.0)

        # Show the figure
        fig.subplots_adjust(
            left=0.07, bottom=0.07, right=0.97, top=0.94, wspace=0.25, hspace=0.25
        )
        fig.show()

        # Feedback to user
        print(
            "sigma_add(q) = %.4g (+%3g, -%3g)"
            % (mDict["sigmaAddQ"], mDict["dSigmaAddPlusQ"], mDict["dSigmaAddMinusQ"])
        )
        print(
            "sigma_add(u) = %.4g (+%3g, -%3g)"
            % (mDict["sigmaAddU"], mDict["dSigmaAddPlusU"], mDict["dSigmaAddMinusU"])
        )
        input()

    return mDict, pltDict


# -----------------------------------------------------------------------------#
def calc_normal_tests(inArr, suffix=""):
    """Calculate metrics measuring deviation of an array from Normal."""

    # Perfrorm the KS-test
    KS_z, KS_p = kstest(inArr, "norm")

    # Calculate the Anderson test
    AD_z, AD_crit, AD_sig = anderson(inArr, "norm")

    # Calculate the skewness (measure of symmetry)
    # abs(skewness) < 0.5 =  approx symmetric
    skewVal = skew(inArr)
    SK_z, SK_p = skewtest(inArr)

    # Calculate the kurtosis (tails compared to a normal distribution)
    kurtosisVal = kurtosis(inArr)
    KUR_z, KUR_p = kurtosistest(inArr)

    # Return dictionary
    mDict = {
        "KSz" + suffix: toscalar(KS_z),
        "KSp" + suffix: toscalar(KS_p),
        # "ADz" + suffix: toscalar(AD_z),
        # "ADcrit" + suffix: toscalar(AD_crit),
        # "ADsig" + suffix: toscalar(AD_sig),
        "skewVal" + suffix: toscalar(skewVal),
        "SKz" + suffix: toscalar(SK_z),
        "SKp" + suffix: toscalar(SK_p),
        "kurtosisVal" + suffix: toscalar(kurtosisVal),
        "KURz" + suffix: toscalar(KUR_z),
        "KURp" + suffix: toscalar(KUR_p),
    }

    return mDict


# -----------------------------------------------------------------------------#
def measure_qu_complexity(
    freqArr_Hz, qArr, uArr, dqArr, duArr, fracPol, psi0_deg, RM_radm2, specF=1
):

    # Create a RM-thin model to subtract
    pModArr, qModArr, uModArr = create_pqu_spectra_burn(
        freqArr_Hz=freqArr_Hz,
        fracPolArr=[fracPol],
        psi0Arr_deg=[psi0_deg],
        RMArr_radm2=[RM_radm2],
    )
    lamSqArr_m2 = np.power(C / freqArr_Hz, 2.0)
    ndata = len(lamSqArr_m2)

    # Subtract the RM-thin model to create a residual q & u
    qResidArr = qArr - qModArr
    uResidArr = uArr - uModArr

    # Calculate value of additional scatter term for q & u (max likelihood)
    mDict = {}
    pDict = {}
    m1D, p1D = calc_sigma_add(
        xArr=lamSqArr_m2[: int(ndata / specF)],
        yArr=(qResidArr / dqArr)[: int(ndata / specF)],
        dyArr=(dqArr / dqArr)[: int(ndata / specF)],
        yMed=0.0,
        noise=1.0,
        suffix="Q",
    )
    mDict.update(m1D)
    pDict.update(p1D)
    m2D, p2D = calc_sigma_add(
        xArr=lamSqArr_m2[: int(ndata / specF)],
        yArr=(uResidArr / duArr)[: int(ndata / specF)],
        dyArr=(duArr / duArr)[: int(ndata / specF)],
        yMed=0.0,
        noise=1.0,
        suffix="U",
    )
    mDict.update(m2D)
    pDict.update(p2D)

    # Calculate the deviations statistics
    # Done as a test for the paper, not usually offered to user.
    # mDict.update( calc_normal_tests(qResidArr/dqArr, suffix="Q") )
    # mDict.update( calc_normal_tests(uResidArr/duArr, suffix="U") )

    return mDict, pDict


# -----------------------------------------------------------------------------#
def measure_fdf_complexity(phiArr, FDF):

    # Second moment of clean component spectrum
    mom2FDF = calc_mom2_FDF(FDF, phiArr)

    return toscalar(mom2FDF)