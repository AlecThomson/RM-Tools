# =============================================================================#
#                          MODEL DEFINITION FILE                              #
# =============================================================================#
import bilby
import numpy as np


# -----------------------------------------------------------------------------#
# Function defining the model.                                                #
#                                                                             #
#  pDict       = Dictionary of parameters, created by parsing inParms, below. #
#  lam_sq_arr_m2 = _array of lambda-squared values                               #
#  qu_arr       = Complex array containing the Re and Im spectra.              #
# -----------------------------------------------------------------------------#
def model(pDict, lam_sq_arr_m2):
    """Simple Faraday thin source."""

    # Calculate the complex fractional q and u spectra
    p_arr = pDict["fracPol"] * np.ones_like(lam_sq_arr_m2)
    qu_arr = p_arr * np.exp(
        2j * (np.radians(pDict["psi0_deg"]) + pDict["RM_radm2"] * lam_sq_arr_m2)
    )

    return qu_arr


# -----------------------------------------------------------------------------#
# Priors for the above model.                                                 #
# See https://lscsoft.docs.ligo.org/bilby/prior.html for details.             #
#                                                                             #
# -----------------------------------------------------------------------------#
priors = {
    "fracPol": bilby.prior.Uniform(
        minimum=0.001, maximum=1.0, name="fracPol", latex_label="$p$"
    ),
    "psi0_deg": bilby.prior.Uniform(
        minimum=0,
        maximum=180.0,
        name="psi0_deg",
        latex_label="$\psi_0$ (deg)",
        boundary="periodic",
    ),
    "RM_radm2": bilby.prior.Uniform(
        minimum=-1100.0,
        maximum=1100.0,
        name="RM_radm2",
        latex_label="RM (rad m$^{-2}$)",
    ),
}
