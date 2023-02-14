# =============================================================================#
#                          MODEL DEFINITION FILE                              #
# =============================================================================#
import bilby
import numpy as np
from bilby.core.prior import Constraint, PriorDict


# -----------------------------------------------------------------------------#
# Function defining the model.                                                #
#                                                                             #
#  pDict       = Dictionary of parameters, created by parsing inParms, below. #
#  lam_sq_arr_m2 = _array of lambda-squared values                               #
#  qu_arr       = Complex array containing the Re and Im spectra.              #
# -----------------------------------------------------------------------------#
def model(pDict, lam_sq_arr_m2):
    """Two separate Faraday components, averaged within same telescope beam
    (i.e., unresolved), with a common Burn depolarisation term."""

    # Calculate the complex fractional q and u spectra
    p_arr1 = pDict["fracPol1"] * np.ones_like(lam_sq_arr_m2)
    p_arr2 = pDict["fracPol2"] * np.ones_like(lam_sq_arr_m2)
    qu_arr1 = p_arr1 * np.exp(
        2j * (np.radians(pDict["psi01_deg"]) + pDict["RM1_radm2"] * lam_sq_arr_m2)
    )
    qu_arr2 = p_arr2 * np.exp(
        2j * (np.radians(pDict["psi02_deg"]) + pDict["RM2_radm2"] * lam_sq_arr_m2)
    )
    qu_arr = (qu_arr1 + qu_arr2) * np.exp(
        -2.0 * pDict["sigmaRM_radm2"] ** 2.0 * lam_sq_arr_m2**2.0
    )

    return qu_arr


# -----------------------------------------------------------------------------#
# Priors for the above model.                                                 #
# See https://lscsoft.docs.ligo.org/bilby/prior.html for details.             #
#                                                                             #
# -----------------------------------------------------------------------------#
def converter(parameters):
    """
    Function to convert between sampled parameters and constraint parameter.

    Parameters
    ----------
    parameters: dict
        Dictionary containing sampled parameter values, 'RM1_radm2', 'RM1_radm2'.

    Returns
    -------
    dict: Dictionary with constraint parameter 'delta_RM1_RM2_radm2' added.
    """
    converted_parameters = parameters.copy()
    converted_parameters["delta_RM1_RM2_radm2"] = (
        parameters["RM1_radm2"] - parameters["RM2_radm2"]
    )
    converted_parameters["sum_p1_p2"] = parameters["fracPol1"] + parameters["fracPol2"]
    return converted_parameters


priors = PriorDict(conversion_function=converter)

priors["fracPol1"] = bilby.prior.Uniform(
    minimum=0.001,
    maximum=1.0,
    name="fracPol1",
    latex_label="$p_1$",
)
priors["fracPol2"] = bilby.prior.Uniform(
    minimum=0.001,
    maximum=1.0,
    name="fracPol2",
    latex_label="$p_2$",
)

priors["psi01_deg"] = bilby.prior.Uniform(
    minimum=0,
    maximum=180.0,
    name="psi01_deg",
    latex_label="$\psi_{0,1}$ (deg)",
    boundary="periodic",
)
priors["psi02_deg"] = bilby.prior.Uniform(
    minimum=0,
    maximum=180.0,
    name="psi02_deg",
    latex_label="$\psi_{0,2}$ (deg)",
    boundary="periodic",
)

priors["RM1_radm2"] = bilby.prior.Uniform(
    minimum=-1100.0,
    maximum=1100.0,
    name="RM1_radm2",
    latex_label="$\phi_1$ (rad m$^{-2}$)",
)
priors["RM2_radm2"] = bilby.prior.Uniform(
    minimum=-1100.0,
    maximum=1100.0,
    name="RM2_radm2",
    latex_label="$\phi_2$ (rad m$^{-2}$)",
)
priors["delta_RM1_RM2_radm2"] = Constraint(
    minimum=0,
    maximum=2200.0,
    name="delta_RM1_RM2_radm2",
    latex_label="$\Delta\phi_{1,2}$ (rad m$^{-2}$)",
)

priors["sigmaRM_radm2"] = bilby.prior.Uniform(
    minimum=0,
    maximum=100.0,
    name="sigmaRM_radm2",
    latex_label="$\sigma_{RM}$ (rad m$^{-2}$)",
)
priors["sum_p1_p2"] = Constraint(
    minimum=0.001,
    maximum=1,
    name="sum_p1_p2",
    latex_label="$p_1+p_2$)",
)
