'''
Library of lateral buckling calculations
'''

from typing import Tuple
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import minimize

def friction_distribution(low_estimate: float, best_estimate: float,
                          high_estimate: float, fit_type: str) -> Tuple[
                              float, float, float, float, float, float, float, float]:
    '''
    Compute the parameters of the lognormal friction factor distribution (axial or lateral)
    by minimising the root mean square error (RMSE) between geotechnical estimates and
    back-calculated friction factors from the lognormal distribution.

    Parameters
    ----------
    low_estimate : float
       Low estimate (LE) friction factor, representing the 5th percentile.
    best_estimate : float
        Best estimate (BE) friction factor, representing the 50th percentile.
    high_estimate : float
        High estimate (HE) friction factor, representing the 95th percentile.
    fit_type : str
        Type of fit to perform: 'LE_BE_HE' (fit to LE, BE, and HE), 'LE_BE' (fit to LE and BE),
        or 'BE_HE' (fit to BE and HE).

    Returns
    -------
    mean_friction : float
        Mean of the lognormal friction factor distribution.
    std_friction : float
        Standard deviation of the lognormal friction factor distribution.
    location_param : float
        Location parameter of the lognormal friction factor distribution.
    scale_param : float
        Scale parameter of the lognormal friction factor distribution.
    rmse_le_be_he : float
        RMSE for the 'LE_BE_HE' fit type.
    rmse_le_be : float
        RMSE for the 'LE_BE' fit type.
    rmse_be_he : float
        RMSE for the 'BE_HE' fit type.
    rmse_best_fit : float
        RMSE for the best fit type.

    Notes
    -----
    The function calculates the parameters of the lognormal friction factor distribution
    based on LE at 5th percentile, BE at 50th percentile, and HE at 95th percentile

    Examples
    --------
    >>> friction_distribution(0.5, 1.0, 1.5, 'LE_BE_HE')
    (np.float64(0.9684082990173732), np.float64(0.30043235817282576), np.float64(-0.0780466572081312), np.float64(0.3031342048564706), np.float64(0.5617726470692103), np.float64(0.9249212712925357), np.float64(1.5228213095679706), np.float64(0.05765844141916946))
    '''

    # Calculate RMSE based on the parameters location and scale
    def objective(params):
        location_param, scale_param = params
        if fit_type == 'LE_BE_HE':
            le_fit = lognorm(scale_param, 0.0, np.exp(location_param)).ppf(0.05)
            be_fit = lognorm(scale_param, 0.0, np.exp(location_param)).ppf(0.50)
            he_fit = lognorm(scale_param, 0.0, np.exp(location_param)).ppf(0.95)
            error = np.sqrt(((le_fit - low_estimate)**2 + (be_fit - best_estimate)**2
                             + (he_fit - high_estimate)**2) / 3.0)
        elif fit_type == 'LE_BE':
            le_fit = lognorm(scale_param, 0.0, np.exp(location_param)).ppf(0.05)
            be_fit = lognorm(scale_param, 0.0, np.exp(location_param)).ppf(0.50)
            error = np.sqrt(((le_fit - low_estimate)**2 + (be_fit - best_estimate)**2) / 2.0)
        elif fit_type == 'BE_HE':
            be_fit = lognorm(scale_param, 0.0, np.exp(location_param)).ppf(0.50)
            he_fit = lognorm(scale_param, 0.0, np.exp(location_param)).ppf(0.95)
            error = np.sqrt(((be_fit - best_estimate)**2 + (he_fit - high_estimate)**2) / 2.0)
        else:
            error = 1.0E+06
        return error

    # Random initial guess
    initial_location = np.mean([np.log(low_estimate), np.log(best_estimate), np.log(high_estimate)])
    initial_scale = np.std([np.log(low_estimate), np.log(best_estimate), np.log(high_estimate)],
                           ddof=1)
    initial_guess = [initial_location, initial_scale]

    # Use minimize to find the parameters that minimize RMSE
    result = minimize(objective, initial_guess, method='Nelder-Mead')
    location_param, scale_param = result.x

    # Calculate lognormal parameters based on the optimized lognormal distribution
    mean_friction = np.exp(location_param + scale_param**2 / 2)
    std_friction = np.sqrt((np.exp(scale_param**2) - 1) * \
                           np.exp(2 * location_param + scale_param**2))

    # Calculate the fitted values
    le_fit = lognorm(scale_param, 0.0, np.exp(location_param)).ppf(0.05)
    be_fit = lognorm(scale_param, 0.0, np.exp(location_param)).ppf(0.50)
    he_fit = lognorm(scale_param, 0.0, np.exp(location_param)).ppf(0.95)

    # Calculate RMSE
    if fit_type == 'LE_BE_HE':
        rmse = np.sqrt(((le_fit - low_estimate)**2 + (be_fit - best_estimate)**2 +\
                        (he_fit - high_estimate)**2) / 3.0)
    elif fit_type == 'LE_BE':
        rmse = np.sqrt(((le_fit - low_estimate)**2 + (be_fit - best_estimate)**2) / 2.0)
    elif fit_type == 'BE_HE':
        rmse = np.sqrt(((be_fit - best_estimate)**2 + (he_fit - high_estimate)**2) / 2.0)

    return mean_friction, std_friction, location_param, scale_param, le_fit, be_fit, he_fit, rmse
