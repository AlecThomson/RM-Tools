#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     do_RM-clean.py                                                    #
#                                                                             #
# PURPOSE:  Run RM-clean on a  cube of dirty Faraday dispersion functions.    #
#                                                                             #
# MODIFIED: 15-May-2016 by C. Purcell                                         #
# MODIFIED: 23-October-2019 by A. Thomson                                     #
#                                                                             #
#=============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2015 Cormac R. Purcell                                        #
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
#=============================================================================#

import sys
import os
import time
from dask.array.core import to_zarr
import numpy as np
import astropy.io.fits as pf
import dafits
import dask.array as da
from dask.distributed import Client, performance_report
import zarr

from RMutils.util_RM import do_rmclean_hogbom
from RMutils.util_RM import fits_make_lin_axis

C = 2.997924538e8 # Speed of light [m/s]


#-----------------------------------------------------------------------------#
def run_rmclean(fitsFDF, fitsRMSF, cutoff, maxIter=1000, gain=0.1, nBits=32,
                verbose = True, log = print):
    """Run RM-CLEAN on a 2/3D FDF cube given an RMSF cube stored as FITS.

    If you want to run RM-CLEAN on arrays, just use util_RM.do_rmclean_hogbom.

    Args:
        fitsFDF (str): Name of FDF FITS file.
        fitsRMSF (str): Name of RMSF FITS file
        cutoff (float): CLEAN cutoff in flux units

    Kwargs:
        maxIter (int): Maximum number of CLEAN iterations per pixel.
        gain (float): CLEAN loop gain.
        nBits (int): Precision of floating point numbers.
        pool (multiprocessing Pool): Pool function from multiprocessing
            or schwimmbad
        chunksize (int): Number of chunks which it submits to the process pool
            as separate tasks. The (approximate) size of these chunks can be
            specified by setting chunksize to a positive integer.
        verbose (bool): Verbosity.
        log (function): Which logging function to use.

    Returns:
        cleanFDF (ndarray): Cube of RMCLEANed FDFs.
        ccArr (ndarray): Cube of RMCLEAN components (i.e. the model).
        iterCountArr (ndarray): Cube of number of RMCLEAN iterations.
        residFDF (ndarray): Cube of residual RMCLEANed FDFs.
        head (fits.header): Header of FDF FITS file for template.

    """


    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)

    # Read the FDF
    dirtyFDF, head, FD_axis = read_FDF_cubes(fitsFDF)



    phiArr_radm2 = fits_make_lin_axis(head, axis=FD_axis-1, dtype=dtFloat)

    # Read the RMSF

    RMSFArr, headRMSF,FD_axis = read_FDF_cubes(fitsRMSF)
    fwhmRMSFArr = dafits.read(fitsRMSF.replace('_real','_FWHM').replace('_im','_FWHM').replace('_tot','_FWHM'))
    fwhmRMSFArr = da.squeeze(fwhmRMSFArr)
    phi2Arr_radm2 = fits_make_lin_axis(headRMSF, axis=FD_axis-1, dtype=dtFloat)

    startTime = time.time()

    # Do the clean
    cleanFDF, ccArr, iterCountArr, residFDF  = \
        do_rmclean_hogbom(dirtyFDF         = dirtyFDF,
                          phiArr_radm2     = phiArr_radm2,
                          RMSFArr          = RMSFArr,
                          phi2Arr_radm2    = phi2Arr_radm2,
                          fwhmRMSFArr      = fwhmRMSFArr,
                          cutoff           = cutoff,
                          maxIter          = maxIter,
                          gain             = gain,
                          verbose          = verbose,
                          doPlots          = False)

    endTime = time.time()
    cputime = (endTime - startTime)
    if (verbose): log("> RM-clean completed in %.2f seconds." % cputime)
    if (verbose): log("Saving the clean FDF and ancillary FITS files")




    #Move FD axis back to original position, and restore dimensionality:
    old_Ndim,FD_axis=find_axes(head)
    new_Ndim=cleanFDF.ndim
    #The difference is the number of dimensions that need to be added:
    if old_Ndim-new_Ndim != 0: #add missing (degenerate) dimensions back in
        cleanFDF=np.expand_dims(cleanFDF,axis=tuple(range(old_Ndim-new_Ndim)))
        ccArr=np.expand_dims(ccArr,axis=tuple(range(old_Ndim-new_Ndim)))
        residFDF=np.expand_dims(residFDF,axis=tuple(range(old_Ndim-new_Ndim)))
    #New dimensions are added to the beginning of the axis ordering 
    #(revserse of FITS ordering)
    
    #Move the FDF axis to it's original spot. Hopefully this means that all
    # axes are in their original place after all of that.
    cleanFDF=da.moveaxis(cleanFDF,old_Ndim-new_Ndim,old_Ndim-FD_axis)
    ccArr=da.moveaxis(ccArr,old_Ndim-new_Ndim,old_Ndim-FD_axis)
    residFDF=da.moveaxis(residFDF,old_Ndim-new_Ndim,old_Ndim-FD_axis)

    return cleanFDF, ccArr, iterCountArr, residFDF, head


def writefits(cleanFDF, ccArr, iterCountArr, residFDF, headtemp, nBits=32,
            prefixOut="", outDir="", write_separate_FDF=False, verbose=True, log=print):
    """Write data to disk in FITS


    Output files:
        Default:
            FDF_clean.fits: RMCLEANed FDF, in 3 extensions: Q,U, and PI.
            FDF_CC.fits: RMCLEAN components, in 3 extensions: Q,U, and PI.
            CLEAN_nIter.fits: RMCLEAN iterations.

        write_seperate_FDF=True:
            FDF_clean_real.fits and FDF_CC.fits are split into
            three constituent components:
                FDF_clean_real.fits: Stokes Q
                FDF_clean_im.fits: Stokes U
                FDF_clean_tot.fits: Polarized Intensity (sqrt(Q^2+U^2))
                FDF_CC_real.fits: Stokes Q
                FDF_CC_im.fits: Stokes U
                FDF_CC_tot.fits: Polarized Intensity (sqrt(Q^2+U^2))
                CLEAN_nIter.fits: RMCLEAN iterations.
    Args:
        cleanFDF (ndarray): Cube of RMCLEANed FDFs.
        ccArr (ndarray): Cube of RMCLEAN components (i.e. the model).
        iterCountArr (ndarray): Cube of number of RMCLEAN iterations.
        residFDF (ndarray): Cube of residual RMCLEANed FDFs.

    Kwargs:
        prefixOut (str): Prefix for filenames.
        outDir (str): Directory to save files.
        write_seperate_FDF (bool): Write Q, U, and PI separately?
        verbose (bool): Verbosity.
        log (function): Which logging function to use.
    """
    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2*nBits)


    if outDir=='':  #To prevent code breaking if file is in current directory
        outDir='.'
    # Write cubes to Zarr to begin computation
    FDF_zarr_complex = outDir + "/" + prefixOut + "FDF_clean_complex.zarr"
    FDF_zarr_real = outDir + "/" + prefixOut + "FDF_clean_real.zarr"
    FDF_zarr_imag = outDir + "/" + prefixOut + "FDF_clean_imag.zarr"
    FDF_zarr_tot = outDir + "/" + prefixOut + "FDF_clean_tot.zarr"

    cleanFDF.to_zarr(FDF_zarr_complex, overwrite=True)
    cleanFDF = da.from_zarr(FDF_zarr_complex)

    FDFcube_real = da.real(cleanFDF).astype(dtFloat)
    FDFcube_imag = da.imag(cleanFDF).astype(dtFloat)
    FDFcube_tot = da.absolute(cleanFDF).astype(dtFloat)

    FDFcube_real.to_zarr(FDF_zarr_real, overwrite=True)
    FDFcube_imag.to_zarr(FDF_zarr_imag, overwrite=True)
    FDFcube_tot.to_zarr(FDF_zarr_tot, overwrite=True)

    FDFcubeArr_real = zarr.open(FDF_zarr_real, mode="r")
    FDFcubeArr_imag = zarr.open(FDF_zarr_imag, mode="r")
    FDFcubeArr_tot = zarr.open(FDF_zarr_tot, mode="r")
    
    # Save the clean FDF
    if not write_separate_FDF:
        fitsFileOut = outDir + "/" + prefixOut + "FDF_clean.fits"

        if(verbose): log("> %s" % fitsFileOut)
        hdu0 = pf.PrimaryHDU(FDFcubeArr_real, headtemp)
        hdu1 = pf.ImageHDU(FDFcubeArr_imag, headtemp)
        hdu2 = pf.ImageHDU(FDFcubeArr_tot, headtemp)
        hduLst = pf.HDUList([hdu0, hdu1, hdu2])
        hduLst.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        hduLst.close()
    else:
        hdu0 = pf.PrimaryHDU(FDFcubeArr_real, headtemp)
        fitsFileOut = outDir + "/" + prefixOut + "FDF_clean_real.fits"
        hdu0.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        if (verbose): log("> %s" % fitsFileOut)
        hdu1 = pf.PrimaryHDU(FDFcubeArr_imag, headtemp)
        fitsFileOut = outDir + "/" + prefixOut + "FDF_clean_im.fits"
        hdu1.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        if (verbose): log("> %s" % fitsFileOut)
        hdu2 = pf.PrimaryHDU(FDFcubeArr_tot, headtemp)
        fitsFileOut = outDir + "/" + prefixOut + "FDF_clean_tot.fits"
        hdu2.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        if (verbose): log("> %s" % fitsFileOut)

    # Write cubes to Zarr to begin computation
    FDF_CC_zarr_complex = outDir + "/" + prefixOut + "FDF_CC_clean_complex.zarr"
    FDF_CC_zarr_real = outDir + "/" + prefixOut + "FDF_CC_clean_real.zarr"
    FDF_CC_zarr_imag = outDir + "/" + prefixOut + "FDF_CC_clean_imag.zarr"
    FDF_CC_zarr_tot = outDir + "/" + prefixOut + "FDF_CC_clean_tot.zarr"

    ccArr.to_zarr(FDF_CC_zarr_complex, overwrite=True)
    ccArr = da.from_zarr(FDF_CC_zarr_complex)

    FDF_CCcube_real = da.real(ccArr).astype(dtFloat)
    FDF_CCcube_imag = da.imag(ccArr).astype(dtFloat)
    FDF_CCcube_tot = da.absolute(ccArr).astype(dtFloat)

    FDF_CCcube_real.to_zarr(FDF_CC_zarr_real, overwrite=True)
    FDF_CCcube_imag.to_zarr(FDF_CC_zarr_imag, overwrite=True)
    FDF_CCcube_tot.to_zarr(FDF_CC_zarr_tot, overwrite=True)

    FDF_CCcube_real = zarr.open(FDF_CC_zarr_real, mode="r")
    FDF_CCcube_imag = zarr.open(FDF_CC_zarr_imag, mode="r")
    FDF_CCcube_tot = zarr.open(FDF_CC_zarr_tot, mode="r")

    if not write_separate_FDF:
    #Save the complex clean components as another file.
        fitsFileOut = outDir + "/" + prefixOut + "FDF_CC.fits"
        if (verbose): log("> %s" % fitsFileOut)
        hdu0 = pf.PrimaryHDU(FDF_CCcube_real, headtemp)
        hdu1 = pf.ImageHDU(FDF_CCcube_imag, headtemp)
        hdu2 = pf.ImageHDU(FDF_CCcube_tot, headtemp)
        hduLst = pf.HDUList([hdu0, hdu1, hdu2])
        hduLst.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        hduLst.close()
    else:
        hdu0 = pf.PrimaryHDU(FDF_CCcube_real, headtemp)
        fitsFileOut = outDir + "/" + prefixOut + "FDF_CC_real.fits"
        hdu0.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        if (verbose): log("> %s" % fitsFileOut)
        hdu1 = pf.PrimaryHDU(FDF_CCcube_imag, headtemp)
        fitsFileOut = outDir + "/" + prefixOut + "FDF_CC_im.fits"
        hdu1.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        if (verbose): log("> %s" % fitsFileOut)
        hdu2 = pf.PrimaryHDU(FDF_CCcube_tot, headtemp)
        fitsFileOut = outDir + "/" + prefixOut + "FDF_CC_tot.fits"
        hdu2.writeto(fitsFileOut, output_verify="fix", overwrite=True)
        if (verbose): log("> %s" % fitsFileOut)

    #Because there can be problems with different axes having different FITS keywords,
    #don't try to remove the FD axis, but just make it degenerate.

    if headtemp['NAXIS'] > 2:
        headtemp["NAXIS3"] = 1
    if headtemp['NAXIS'] == 4:
        headtemp["NAXIS4"] = 1

    # Save the iteration count mask
    fitsFileOut = outDir + "/" + prefixOut + "CLEAN_nIter.fits"
    zarrFileOut = fitsFileOut.replace(".fits", ".zarr")

    # Using np.newaxis, as da.expand_dims doesn't work with dask.array
    iterCountArr.astype(dtFloat)[np.newaxis].to_zarr(zarrFileOut, overwrite=True)
    iterCountArr = zarr.open(zarrFileOut, mode="r")

    if (verbose): log("> %s" % fitsFileOut)
    headtemp["BUNIT"] = "Iterations"
    hdu0 = pf.PrimaryHDU(iterCountArr,
                         headtemp)
    hduLst = pf.HDUList([hdu0])
    hduLst.writeto(fitsFileOut, output_verify="fix", overwrite=True)
    hduLst.close()


#Old method (for multi-extension files)
def read_FDF_cube(filename):
    """Read in a FDF/RMSF cube. Figures out which axis is Faraday depth and
    puts it first (in numpy order) to accommodate the rest of the code.
    Returns: (complex_cube, header,FD_axis)
    """
    HDULst = pf.open(filename, "readonly", memmap=True)
    head = HDULst[0].header.copy()
    FDFreal = HDULst[0].data
    FDFimag = HDULst[1].data
    complex_cube = FDFreal + 1j * FDFimag

    #Identify Faraday depth axis (assumed to be last one if not explicitly found)
    Ndim=head['NAXIS']
    FD_axis=Ndim
    #Check for FD axes:
    for i in range(1,Ndim+1):
        try:
            if 'FDEP' in head['CTYPE'+str(i)].upper():
                FD_axis=i
        except:
            pass #The try statement is needed for if the FITS header does not
                 # have CTYPE keywords.

    #Move FD axis to first place in numpy order.
    if FD_axis != Ndim:
        complex_cube=da.moveaxis(complex_cube,Ndim-FD_axis,0)

    #Remove degenerate axes to prevent problems with later steps.
    complex_cube=complex_cube.squeeze()

    return complex_cube, head,FD_axis

def find_axes(header):
    """Idenfities how many axes are present in a FITS file, and which is the
    Faraday depth axis. Necessary for bookkeeping on cube dimensionality,
    given that RM-clean only supports 3D cubes, but data may be 4D files."""
    Ndim=header['NAXIS']
    FD_axis=Ndim
    #Check for FD axes:
    for i in range(1,Ndim+1):
        try:
            if 'FDEP' in header['CTYPE'+str(i)].upper():
                FD_axis=i
        except:
            pass #The try statement is needed for if the FITS header does not
                 # have CTYPE keywords.
    return Ndim,FD_axis

def read_FDF_cubes(filename):
    """Read in a FDF/RMSF cube. Input filename can be any of real, imag, or tot components.
    Figures out which axis is Faraday depth and
    puts it first (in numpy order) to accommodate the rest of the code.
    Returns: (complex_cube, header,FD_axis)
    """
    FDFreal, head = dafits.read(filename.replace('_tot','_real').replace('_im','_real'), return_header=True)

    FDFimag = dafits.read(filename.replace('_tot','_real').replace('_im','_real'), return_header=False)
    complex_cube = FDFreal + 1j * FDFimag

    #Identify Faraday depth axis (assumed to be last one if not explicitly found)
    Ndim,FD_axis=find_axes(head)

    #Move FD axis to first place in numpy order.
    if FD_axis != Ndim:
        complex_cube=da.moveaxis(complex_cube,Ndim-FD_axis,0)

    #Remove degenerate axes to prevent problems with later steps.
    complex_cube=complex_cube.squeeze()

    return complex_cube, head,FD_axis


#-----------------------------------------------------------------------------#
def main():
    import argparse
    """
    Start the function to perform RM-clean if called from the command line.
    """

    # Help string to be shown using the -h option
    descStr = """
    Run RM-CLEAN on a cube of Faraday dispersion functions (FDFs), applying
    a cube of rotation measure spread functions created by the script
    'do_RMsynth_3D.py'. Saves cubes of deconvolved FDFs & clean-component
    spectra, and a pixel map showing the number of iterations performed.
    Set any of the multiprocessing options to enable parallelization
    (otherwise, pixels will be processed serially).

    Expects that the input is in the form of the Stokes-separated
    (single extension) FITS cubes produced by do_RMsynth_3D.
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("fitsFDF", metavar="FDF_dirty.fits", nargs=1,
                        help="FITS cube containing the dirty FDF.\n(Can be any of the FDF output cubes from do_RMsynth_3D.py)")
    parser.add_argument("fitsRMSF", metavar="RMSF.fits", nargs=1,
                        help="FITS cube containing the RMSF and FWHM image.\n(Cans be any of the RMSF output cubes from do_RMsynth_3D.py)")
    parser.add_argument("-c", dest="cutoff", type=float, nargs=1,
                        default=1.0, help="CLEAN cutoff in flux units")
    parser.add_argument("-n", dest="maxIter", type=int, default=1000,
                        help="Maximum number of CLEAN iterations per pixel [1000].")
    parser.add_argument("-g", dest="gain", type=float, default=0.1,
                        help="CLEAN loop gain [0.1].")
    parser.add_argument("-o", dest="prefixOut", default="",
                        help="Prefix to prepend to output files [None].")
    parser.add_argument("-f", dest="write_separate_FDF", action="store_false",
                        help="Store different Stokes as FITS extensions [False, store as separate files].")

    parser.add_argument("-v", dest="verbose", action="store_true",
                        help="Verbose [False].")

    parser.add_argument("--client", default=None,
                    help="Dask client address [None - creates local Client]")
    parser.add_argument("--report", default=None,
                    help="Dask report name [None - not saved]")
    args = parser.parse_args()


    verbose = args.verbose
    # Sanity checks
    for f in args.fitsFDF + args.fitsRMSF:
        if not os.path.exists(f):
            print("File does not exist: '%s'." % f)
            sys.exit()

    dataDir, _ = os.path.split(args.fitsFDF[0])
    
    report = args.report
    if report is not None:
        report = dataDir + '/' + report + '.html'
        if verbose:
            print(f"Saving report to '{report}'")
    client = Client(args.client)
    with performance_report(report):
        if verbose:
            print(f"Dask client running at '{client.dashboard_link}'")
        # Run RM-CLEAN on the cubes
        cleanFDF, ccArr, iterCountArr, residFDF, headtemp = run_rmclean(fitsFDF     = args.fitsFDF[0],
                                                            fitsRMSF    = args.fitsRMSF[0],
                                                            cutoff      = args.cutoff,
                                                            maxIter     = args.maxIter,
                                                            gain        = args.gain,
                                                            nBits       = 32,
                                                            verbose = verbose)
        # Write results to disk
        writefits(cleanFDF,
                ccArr,
                iterCountArr,
                residFDF,
                headtemp,
                prefixOut           = args.prefixOut,
                outDir              = dataDir,
                write_separate_FDF  = args.write_separate_FDF,
                verbose             = verbose)

#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
