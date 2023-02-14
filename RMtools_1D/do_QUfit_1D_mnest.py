#!/usr/bin/env python

# =============================================================================#
#                                                                             #
# NAME:     do_QUfit_1D_nest.py                                               #
#                                                                             #
# PURPOSE:  Code to simultaneously fit Stokes Q/I and U/I spectra with a      #
#           Faraday active models.                                            #
#                                                                             #
# MODIFIED: 12-Mar-2018 by C. Purcell                                         #
#                                                                             #
# CONTENTS:                                                                   #
#                                                                             #
#   main           ... parse command line arguments and initiate procedure    #
#   run_qufit      ... main function of the QU-fitting procedure              #
#   prior_call     ... return the prior tranform fucntion given prior limits  #
#   lnlike_call    ... return a function to evaluate ln(like) given the data  #
#   chisq_model    ... calculate the chi-squared for the model                #
#   wrap_chains    ... wrap and shift chains of periodic parameters           #
#   init_mnest     ... initialise multinest using a default dictionary        #
#   merge_two_dicts .. merge two dictionaries                                 #
#                                                                             #
# =============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2018 Cormac R. Purcell                                        #
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

import argparse
import copy
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from shutil import Error

import bilby
import matplotlib.pyplot as plt
import numpy as np

from RMtools_1D.do_RMsynth_1D import readFile
from RMutils import corner
from RMutils.util_misc import calculate_StokesI_model, create_frac_spectra, toscalar
from RMutils.util_plotTk import CustomNavbar, plot_Ipqu_spectra_fig

C = 2.997924538e8  # Speed of light [m/s]


# -----------------------------------------------------------------------------#
def main():
    """
    Start the run_qufit procedure if called from the command line.

    Run QU-fitting on polarised spectra (1D) stored in an ASCII file. The
    Stokes I spectra is first fit with a polynomial and the resulting model
    used to create fractional q = Q/I and u = U/I spectra. If the 'noStokesI'
    option is given, the input data are assumed to be fractional already.

    The script uses the Nested Sampling algorithm (Skilling 2004) to find
    the best fitting parameters, given a prior function on each free parameter.
    The sampling algorithm also calculates the Bayesian evidence, which can
    be used for model comparison. Factors of >10 between models mean that one
    model is strongly favoured over the other.

    Models and priors are  specified as Python code in files called 'mX.py'
    within the 'models_ns' directory. See the existing files for examples
    drawn from the paper Sokoloff et al. 1998, MNRAS 229, pg 189.

    Main algorithm handles the command line interface, passing all arguments
    one to run_qufit().
    """

    # Help string to be shown using the -h option
    descStr = """
    Run QU-fitting on polarised spectra (1D) stored in an ASCII file. The
    Stokes I spectra is first fit with a polynomial and the resulting model
    used to create fractional q = Q/I and u = U/I spectra. If the 'noStokesI'
    option is given, the input data are assumed to be fractional already.

    The script uses the Nested Sampling algorithm (Skilling 2004) to find
    the best fitting parameters, given a prior function on each free parameter.
    The sampling algorithm also calculates the Bayesian evidence, which can
    be used for model comparison. Factors of >10 between models mean that one
    model is strongly favoured over the other.

    Models and priors are  specified as Python code in files called 'mX.py'
    within the 'models_ns' directory. See the existing files for examples
    drawn from the paper Sokoloff et al. 1998, MNRAS 229, pg 189.
    """

    # Parse the command line options
    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "dataFile",
        metavar="dataFile.dat",
        nargs=1,
        help="ASCII file containing Stokes spectra & errors.",
    )
    parser.add_argument(
        "-m", dest="modelNum", type=int, default=1, help="model number to fit [1]."
    )
    parser.add_argument(
        "-f",
        dest="fit_function",
        type=str,
        default="log",
        help="Stokes I fitting function: 'linear' or ['log'] polynomials.",
    )
    parser.add_argument(
        "-o",
        dest="polyOrd",
        type=int,
        default=2,
        help="polynomial order to fit to I spectrum: 0-5 supported, 2 is default.\nSet to negative number to enable dynamic order selection.",
    )
    parser.add_argument(
        "-i",
        dest="noStokesI",
        action="store_true",
        help="ignore the Stokes I spectrum [False].",
    )
    parser.add_argument(
        "-p", dest="showPlots", action="store_true", help="show the plots [False]."
    )
    parser.add_argument(
        "-d",
        dest="debug",
        action="store_true",
        help="turn on debugging messages/plots [False].",
    )
    parser.add_argument(
        "-v", dest="verbose", action="store_true", help="verbose mode [False]."
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="dynesty",
        help="Which sampler to use with Bilby (see https://lscsoft.docs.ligo.org/bilby/samplers.html) [dynesty].",
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=1,
        help="Number of cores to use for sampling [1].",
    )
    parser.add_argument(
        "--nlive",
        type=int,
        default=300,
        help="Number of live points to use for sampling [300].",
    )
    args = parser.parse_args()

    # Sanity checks
    if not os.path.exists(args.dataFile[0]):
        print("File does not exist: '%s'." % args.dataFile[0])
        sys.exit()

    prefixOut, ext = os.path.splitext(args.dataFile[0])
    data = readFile(args.dataFile[0], 32, verbose=args.verbose, debug=args.debug)

    # Run the QU-fitting procedure
    run_qufit(
        data=data,
        modelNum=args.modelNum,
        polyOrd=args.polyOrd,
        nBits=32,
        noStokesI=args.noStokesI,
        showPlots=args.showPlots,
        debug=args.debug,
        verbose=args.verbose,
        fit_function=args.fit_function,
        sampler=args.sampler,
        ncores=args.ncores,
        nlive=args.nlive,
        prefixOut=prefixOut,
    )


def load_model(modelNum, verbose=False):
    if verbose:
        print("\nLoading the model from 'models_ns/m%d.py' ..." % modelNum)
    # First check the working directory for a model. Failing that, try the install directory.
    try:
        spec = importlib.util.spec_from_file_location(
            "m%d" % modelNum, "models_ns/m%d.py" % modelNum
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod] = mod
        spec.loader.exec_module(mod)
    except FileNotFoundError:
        try:
            RMtools_dir = os.path.dirname(importlib.util.find_spec("RMtools_1D").origin)
            spec = importlib.util.spec_from_file_location(
                "m%d" % modelNum, RMtools_dir + "/models_ns/m%d.py" % modelNum
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod] = mod
            spec.loader.exec_module(mod)
        except:
            print(
                "Model could not be found! Please make sure model is present either in {}/models_ns/, or in {}/RMtools_1D/models_ns/".format(
                    os.getcwd(), RMtools_dir
                )
            )
            sys.exit()
    return mod


# -----------------------------------------------------------------------------#
def run_qufit(
    data,
    modelNum,
    polyOrd=2,
    nBits=32,
    noStokesI=False,
    showPlots=False,
    debug=False,
    verbose=False,
    sampler="dynesty",
    fit_function="log",
    ncores=1,
    nlive=1000,
    prefixOut="prefixOut",
):
    """Carry out QU-fitting using the supplied parameters:
    data (list): Contains frequency and polarization data as either:
        [freq_Hz, I, Q, U, dI, dQ, dU]
            freq_Hz (array_like): Frequency of each channel in Hz.
            I (array_like): Stokes I intensity in each channel.
            Q (array_like): Stokes Q intensity in each channel.
            U (array_like): Stokes U intensity in each channel.
            dI (array_like): Error in Stokes I intensity in each channel.
            dQ (array_like): Error in Stokes Q intensity in each channel.
            dU (array_like): Error in Stokes U intensity in each channel.
        or
        [freq_Hz, q, u,  dq, du]
            freq_Hz (array_like): Frequency of each channel in Hz.
            q (array_like): Fractional Stokes Q intensity (Q/I) in each channel.
            u (array_like): Fractional Stokes U intensity (U/I) in each channel.
            dq (array_like): Error in fractional Stokes Q intensity in each channel.
            du (array_like): Error in fractional Stokes U intensity in each channel.
    modelNum (int, required): number of model to be fit to data. Models and
         priors are specified as Python code in files called 'mX.py' within
        the 'models_ns' directory.
    outDir (str): relative or absolute path to save outputs to. Defaults to
        working directory.
    polyOrd (int): Order of polynomial to fit to Stokes I spectrum (used to
        normalize Q and U values). Defaults to 3 (cubic).
    nBits (int): number of bits to use in internal calculations.
    noStokesI (bool): set True if the Stokes I spectrum should be ignored.
    showPlots (bool): Set true if the spectrum and parameter space plots
        should be displayed.
    sigma_clip (float): How many standard deviations to clip around the
        mean of each mode in the parameter postierors.
    debug (bool): Display debug messages.
    verbose (bool): Print verbose messages/results to terminal.

    Returns: nothing. Results saved to files and/or printed to terminal."""

    # Output prefix is derived from the input file name
    nestOut = f"{prefixOut}_m{modelNum}_{sampler}/"

    # Parse the data array
    # freq_Hz, I, Q, U, dI, dQ, dU
    try:
        (freq_arr_Hz, I_arr, Q_arr, U_arr, dI_arr, dQ_arr, dU_arr) = data
        print("\nFormat [freq_Hz, I, Q, U, dI, dQ, dU]")
    except Exception:
        # freq_Hz, Q, U, dQ, dU
        try:
            (freq_arr_Hz, Q_arr, U_arr, dQ_arr, dU_arr) = data
            print("\nFormat [freq_Hz, Q, U,  dQ, dU]")
            noStokesI = True
        except Exception:
            print("\nError: Failed to parse data file!")
            if debug:
                print(traceback.format_exc())
            return

    # If no Stokes I present, create a dummy spectrum = unity
    if noStokesI:
        print("Note: no Stokes I data - assuming fractional polarisation.")
        I_arr = np.ones_like(Q_arr)
        dI_arr = np.zeros_like(Q_arr)

    # Convert to GHz for convenience
    lam_sq_arr_m2 = np.power(C / freq_arr_Hz, 2.0)

    # Fit the Stokes I spectrum and create the fractional spectra
    data_arr = create_frac_spectra(
        freq_arr=freq_arr_Hz,
        I_arr=I_arr,
        Q_arr=Q_arr,
        U_arr=U_arr,
        dI_arr=dI_arr,
        dQ_arr=dQ_arr,
        dU_arr=dU_arr,
        polyOrd=polyOrd,
        verbose=True,
        fit_function=fit_function,
    )
    (IMod_arr, q_arr, u_arr, dq_arr, du_arr, IfitDict) = data_arr

    # Plot the data and the Stokes I model fit
    print("Plotting the input data and spectral index fit.")
    freqHir_arr_Hz = np.linspace(freq_arr_Hz[0], freq_arr_Hz[-1], 10000)
    IModHir_arr = calculate_StokesI_model(IfitDict, freqHir_arr_Hz)
    specFig = plt.figure(facecolor="w", figsize=(10, 6))
    plot_Ipqu_spectra_fig(
        freq_arr_Hz=freq_arr_Hz,
        I_arr=I_arr,
        q_arr=q_arr,
        u_arr=u_arr,
        dI_arr=dI_arr,
        dq_arr=dq_arr,
        du_arr=du_arr,
        freqHir_arr_Hz=freqHir_arr_Hz,
        IMod_arr=IModHir_arr,
        fig=specFig,
    )

    # Use the custom navigation toolbar
    try:
        specFig.canvas.toolbar.pack_forget()
        CustomNavbar(specFig.canvas, specFig.canvas.toolbar.window)
    except Exception:
        pass

    # Display the figure
    if showPlots:
        specFig.canvas.draw()
        specFig.show()

    # -------------------------------------------------------------------------#

    # Load the model and parameters from the relevant file
    mod = load_model(modelNum, verbose=True)

    model = mod.model

    # Let's time the sampler
    startTime = time.time()

    parNames = []
    priorTypes = []
    labels = []
    bounds = []
    wraps = []
    for key, prior in mod.priors.items():
        if prior.__class__.__name__ == "Constraint":
            continue
        parNames.append(key)
        priorTypes.append(prior.__class__.__name__)
        labels.append(prior.latex_label)
        bounds.append([prior.minimum, prior.maximum])
        wraps.append(prior.boundary)
    nDim = len(parNames)
    fixedMsk = [0 if x == "DeltaFunction" else 1 for x in priorTypes]
    nFree = sum(fixedMsk)

    # Set the prior function given the bounds of each parameter
    priors = mod.priors

    # Set the likelihood function given the data
    lnlike = lnlike_call(parNames, lam_sq_arr_m2, q_arr, dq_arr, u_arr, du_arr, modelNum)
    # Let's time the sampler
    startTime = time.time()

    result = bilby.run_sampler(
        likelihood=lnlike,
        priors=priors,
        sampler=sampler,
        nlive=nlive,
        npool=ncores,
        outdir=nestOut,
        label="m%d" % modelNum,
        plot=True,
    )

    # Do the post-processing on one processor
    endTime = time.time()
    # Best guess here - taking the maximum likelihood value
    lnLike = np.max(result.log_likelihood_evaluations)
    lnEvidence = result.log_evidence
    dLnEvidence = result.log_evidence_err

    # Get the best-fitting values & uncertainties

    p = [None] * nDim
    errPlus = [None] * nDim
    errMinus = [None] * nDim
    # g = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
    for i in range(nDim):
        summary = result.get_one_dimensional_median_and_error_bar(parNames[i])
        # Get stats around modal value
        p[i], errPlus[i], errMinus[i] = (
            summary.median,
            summary.plus,
            summary.minus,
        )

    # Calculate goodness-of-fit parameters
    nData = 2.0 * len(lam_sq_arr_m2)
    dof = nData - nFree - 1
    chi_sq = chisq_model(parNames, p, lam_sq_arr_m2, q_arr, dq_arr, u_arr, du_arr, model)
    chi_sqRed = chi_sq / dof
    AIC = 2.0 * nFree - 2.0 * lnLike
    AICc = 2.0 * nFree * (nFree + 1) / (nData - nFree - 1) - 2.0 * lnLike
    BIC = nFree * np.log(nData) - 2.0 * lnLike

    # Summary of run
    print("")
    print("-" * 80)
    print("SUMMARY OF SAMPLING RUN:")
    print("#-PROCESSORS  = %d" % ncores)
    print("RUN-TIME      = %.2f" % (endTime - startTime))
    print("DOF           = %d" % dof)
    print("CHISQ:        = %.3g" % chi_sq)
    print("CHISQ RED     = %.3g" % chi_sqRed)
    print("AIC:          = %.3g" % AIC)
    print("AICc          = %.3g" % AICc)
    print("BIC           = %.3g" % BIC)
    print("ln(EVIDENCE)  = %.3g" % lnEvidence)
    print("dLn(EVIDENCE) = %.3g" % dLnEvidence)
    print("")
    print("-" * 80)
    print("RESULTS:\n")
    for i in range(len(p)):
        print("%s = %.4g (+%3g, -%3g)" % (parNames[i], p[i], errPlus[i], errMinus[i]))
    print("-" * 80)
    print("")

    # Create a save dictionary and store final p in values
    outFile = f"{prefixOut}_m{modelNum}_{sampler}.json"
    # IfitDict["p"] = toscalar(IfitDict["p"].tolist())
    saveDict = {
        "parNames": toscalar(parNames),
        "labels": toscalar(labels),
        "values": toscalar(p),
        "errPlus": toscalar(errPlus),
        "errMinus": toscalar(errMinus),
        "bounds": toscalar(bounds),
        "priorTypes": toscalar(priorTypes),
        "wraps": toscalar(wraps),
        "dof": toscalar(dof),
        "chi_sq": toscalar(chi_sq),
        "chi_sqRed": toscalar(chi_sqRed),
        "AIC": toscalar(AIC),
        "AICc": toscalar(AICc),
        "BIC": toscalar(BIC),
        "ln(EVIDENCE) ": toscalar(lnEvidence),
        "dLn(EVIDENCE)": toscalar(dLnEvidence),
        "nFree": toscalar(nFree),
        "Imodel": ",".join([str(x.astype(np.float32)) for x in IfitDict["p"]]),
        "Imodel_errs": ",".join(
            [str(x.astype(np.float32)) for x in IfitDict["perror"]]
        ),
        "IfitChi_sq": toscalar(IfitDict["chi_sq"]),
        "IfitChi_sqRed": toscalar(IfitDict["chi_sqRed"]),
        "IfitPolyOrd": toscalar(IfitDict["polyOrd"]),
        "Ifitfreq0": toscalar(IfitDict["reference_frequency_Hz"]),
    }

    for k, v in saveDict.items():
        if isinstance(v, np.float_):
            saveDict[k] = float(v)
        elif isinstance(v, np.int_):
            saveDict[k] = int(v)
        elif isinstance(v, np.ndarray):
            saveDict[k] = v.tolist()
        elif isinstance(v, np.bool_):
            saveDict[k] = bool(v)
    json.dump(saveDict, open(outFile, "w"))
    outFile = f"{prefixOut}_m{modelNum}_{sampler}.dat"
    FH = open(outFile, "w")
    for k, v in saveDict.items():
        FH.write("%s=%s\n" % (k, v))
    FH.close()
    print("Results saved in JSON and .dat format to:\n '%s'\n" % outFile)

    # Plot the posterior samples in a corner plot
    # chains =  aObj.get_equal_weighted_posterior()
    # chains = wrap_chains(chains, wraps, bounds, p)[:, :nDim]
    # iFixed = [i for i, e in enumerate(fixedMsk) if e==0]
    # chains = np.delete(chains, iFixed, 1)
    # for i in sorted(iFixed, reverse=True):
    #     del(labels[i])
    #     del(p[i])

    cornerFig = result.plot_corner()

    # Save the posterior chains to ASCII file

    # Plot the data and best-fitting model
    lam_sqHir_arr_m2 = np.linspace(lam_sq_arr_m2[0], lam_sq_arr_m2[-1], 10000)
    freqHir_arr_Hz = C / np.sqrt(lam_sqHir_arr_m2)
    IMod_arr = calculate_StokesI_model(IfitDict, freqHir_arr_Hz)
    pDict = {k: v for k, v in zip(parNames, p)}
    quMod_arr = model(pDict, lam_sqHir_arr_m2)
    model_dict = {
        "model": model,
        "parNames": parNames,
        "posterior": result.posterior,
    }
    specFig.clf()
    plot_Ipqu_spectra_fig(
        freq_arr_Hz=freq_arr_Hz,
        I_arr=I_arr,
        q_arr=q_arr,
        u_arr=u_arr,
        dI_arr=dI_arr,
        dq_arr=dq_arr,
        du_arr=du_arr,
        freqHir_arr_Hz=freqHir_arr_Hz,
        IMod_arr=IMod_arr,
        qMod_arr=quMod_arr.real,
        uMod_arr=quMod_arr.imag,
        model_dict=model_dict,
        fig=specFig,
    )
    specFig.canvas.draw()

    # Save the figures
    outFile = prefixOut + "fig_m%d_specfit.pdf" % modelNum
    specFig.set_canvas(specFig.canvas)
    specFig.savefig(outFile)
    print("Plot of best-fitting model saved to:\n '%s'\n" % outFile)
    outFile = prefixOut + "fig_m%d_corner.pdf" % modelNum
    cornerFig.set_canvas(cornerFig.canvas)
    cornerFig.savefig(outFile)
    print("Plot of posterior samples saved to \n '%s'\n" % outFile)

    # Display the figures
    if showPlots:
        specFig.show()
        cornerFig.show()


# -----------------------------------------------------------------------------#
class lnlike_call(bilby.Likelihood):
    """Returns a function to evaluate the log-likelihood"""

    def __init__(self, parNames, lam_sq_arr_m2, q_arr, dq_arr, u_arr, du_arr, modelNum):

        self.parNames = parNames
        self.lam_sq_arr_m2 = lam_sq_arr_m2
        self.q_arr = q_arr
        self.dq_arr = dq_arr
        self.u_arr = u_arr
        self.du_arr = du_arr
        self.modelNum = modelNum
        pDict = {k: None for k in parNames}
        super().__init__(parameters=pDict)

    def log_likelihood(self):
        # Evaluate the model and calculate the joint ln(like)
        # Silva 2006
        model = load_model(self.modelNum).model
        quMod = model(self.parameters, self.lam_sq_arr_m2)
        dqu_arr = np.sqrt(np.power(self.dq_arr, 2) + np.power(self.du_arr, 2))
        chi_sqQ = np.nansum(np.power((self.q_arr - quMod.real) / self.dq_arr, 2))
        chi_sqU = np.nansum(np.power((self.u_arr - quMod.imag) / self.dq_arr, 2))
        nData = len(dqu_arr)
        logLike = (
            -nData * np.log(2.0 * np.pi)
            - 2.0 * np.nansum(np.log(dqu_arr))
            - chi_sqQ / 2.0
            - chi_sqU / 2.0
        )

        return logLike


# -----------------------------------------------------------------------------#
def chisq_model(parNames, p, lam_sq_arr_m2, q_arr, dq_arr, u_arr, du_arr, model):
    """Calculate the chi^2 for the current model, given the data."""

    # Evaluate the model and calculate chisq
    pDict = {k: v for k, v in zip(parNames, p)}
    quMod = model(pDict, lam_sq_arr_m2)
    chisq = np.nansum(
        np.power((q_arr - quMod.real) / dq_arr, 2)
        + np.power((u_arr - quMod.imag) / du_arr, 2)
    )

    return chisq


# -----------------------------------------------------------------------------#
def wrap_chains(chains, wraps, bounds, p, verbose=False):

    # Get the indices of the periodic parameters
    iWrap = [i for i, e in enumerate(wraps) if e != 0]

    # Loop through the chains for periodic parameters
    for i in iWrap:
        wrapLow = bounds[i][0]
        wrapHigh = bounds[i][1]
        rng = wrapHigh - wrapLow

        # Shift the wrapping to centre on the best fit value
        wrapCent = wrapLow + (wrapHigh - wrapLow) / 2.0
        wrapLow += p[i] - wrapCent
        wrapHigh += p[i] - wrapCent
        chains[:, i] = ((chains[:, i] - wrapLow) % rng) + wrapLow
        if verbose:
            print(
                "Wrapped parameter '%d' in range [%s, %s] ..." % (i, wrapLow, wrapHigh)
            )

    return chains


# -----------------------------------------------------------------------------#
def init_mnest():
    """Initialise MultiNest arguments"""

    argsDict = {
        "LogLikelihood": "",
        "Prior": "",
        "n_dims": 0,
        "n_params": 0,
        "n_clustering_params": 0,
        "wrapped_params": None,
        "importance_nested_sampling": False,
        "multimodal": False,
        "const_efficiency_mode": False,
        "n_live_points": 100,
        "evidence_tolerance": 0.5,
        "sampling_efficiency": "model",
        "n_iter_before_update": 500,
        "null_log_evidence": -1.0e90,
        "max_modes": 100,
        "mode_tolerance": -1.0e90,
        "outputfiles_basename": "",
        "seed": -1,
        "verbose": True,
        "resume": True,
        "context": 0,
        "write_output": True,
        "log_zero": -1.0e100,
        "max_iter": 0,
        "init_MPI": False,
        "dump_callback": None,
    }
    return argsDict


# -----------------------------------------------------------------------------#
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
