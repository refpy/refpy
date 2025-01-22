'''
Library of lateral buckling calculations
'''

from typing import Tuple
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import minimize

def calculate_friction_distribution(low_estimate: float, best_estimate: float, high_estimate: float, fit_type: str) -> Tuple[float, float, float, float, float, float, float, float]:
    '''
    Compute the parameters of the lognormal friction factor distribution (axial or lateral)
    based on the minimum root mean square error (RMSE) between geotechnical and back-calculated
    from the lognormal distribution friction factors

    Parameters
    ----------
    low_estimate : float
       Low estimate (LE) friction factor
    best_estimate : float
        Best estimate (BE) friction factor
    high_estimate : float
        High estimate (HE) friction factor
    fit_type : str
        Type of fit: 'LE_BE_HE', 'LE_BE', or 'BE_HE'

    Returns
    -------
    mean_friction : float
        Mean of the lognormal friction factor distribution
    std_friction : float
        Standard deviation of the lognormal friction factor distribution
    location_param : float
        Location parameter of the lognormal friction factor distribution
    scale_param : float
        Scale parameter of the lognormal friction factor distribution
    le_fit : float
        Back-calculated LE friction factor from the lognormal friction factor distribution
    be_fit : float
        Back-calculated BE friction factor from the lognormal friction factor distribution
    he_fit : float
        Back-calculated HE friction factor from the lognormal friction factor distribution
    rmse : float
        RMSE between the LE, BE, and HE geotechnical and back-calculated friction factors

    Notes
    -----
    The function calculates the parameters of the lognormal friction factor distribution
    based on LE at 5th percentile, BE at 50th percentile, and HE at 95th percentile
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
        return error

    # Random initial guess
    initial_location = np.mean([np.log(low_estimate), np.log(best_estimate), np.log(high_estimate)])
    initial_scale = np.std([np.log(low_estimate), np.log(best_estimate), np.log(high_estimate)], ddof=1)
    initial_guess = [initial_location, initial_scale]

    # Use minimize to find the parameters that minimize RMSE
    result = minimize(objective, initial_guess, method='Nelder-Mead')
    location_param, scale_param = result.x

    # Calculate lognormal parameters based on the optimized lognormal distribution
    mean_friction = np.exp(location_param + scale_param**2 / 2)
    std_friction = np.sqrt((np.exp(scale_param**2) - 1) * np.exp(2 * location_param + scale_param**2))

    # Calculate the fitted values
    le_fit = lognorm(scale_param, 0.0, np.exp(location_param)).ppf(0.05)
    be_fit = lognorm(scale_param, 0.0, np.exp(location_param)).ppf(0.50)
    he_fit = lognorm(scale_param, 0.0, np.exp(location_param)).ppf(0.95)

    # Calculate RMSE
    if fit_type == 'LE_BE_HE':
        rmse = np.sqrt(((le_fit - low_estimate)**2 + (be_fit - best_estimate)**2 + (he_fit - high_estimate)**2) / 3.0)
    elif fit_type == 'LE_BE':
        rmse = np.sqrt(((le_fit - low_estimate)**2 + (be_fit - best_estimate)**2) / 2.0)
    elif fit_type == 'BE_HE':
        rmse = np.sqrt(((be_fit - best_estimate)**2 + (he_fit - high_estimate)**2) / 2.0)

    return mean_friction, std_friction, location_param, scale_param, le_fit, be_fit, he_fit, rmse

def calc_oos(section_type, bound, length, sleeper_mean, sleeper_std):

    '''
    Compute the parameters of the lognormal OOS distribution
    (straight, bend or sleeper).

    Parameters
    ----------
    section_type : string
        Section type: 'Straight' or 'Bend' or 'Sleeper'
    bound : string
        For nonimally straight sections, bound of the OOS distribution
    length : string
        The length of section
    sleeper_mean : float
        Mean of the OOS distribution for the sleeper
    sleeper_std : float
        STD of the OOS distribution for the sleeper
        
    Returns
    -------
    mean : float
        Mean of the lognromal OOS distribution
    std : float
        Standard deviation of the lognormal OOS distribution
    ref_length : float
        Reference length for each section
    location : float
        Location parameter of the lognormal OOS distribution
    scale : float
        scale parameter of the lognormal OOS distribution
    '''

    # Calculate mean and std for nominally straight sections based on F110 page 108 and 109
    if section_type == 'Straight':
        if bound == 'LE':
            mean, std = 0.93, 0.27
        elif bound == 'BE':
            mean, std = 1.26, 0.33
        elif bound == 'HE':
            mean, std = 1.58, 0.31
        ref_length = 1000.0

    # Calculate mean and std for bend section based on F110 Eq B.9
    elif section_type == 'Bend':
        mean = np.maximum(0.45, 0.67 - 0.22 * int(length) / 1000.0)
        std = 0.3 * mean
        ref_length = np.minimum(1000.0, int(length))

    # Calculate mean and std for sleepers
    elif section_type == 'Sleeper':
        mean = sleeper_mean
        std = sleeper_std
        ref_length = 'Discrete'

    # Calculate location and scale parameters of the lognormal OOS distribution
    loc = np.log(mean**2 / np.sqrt(mean**2 + std**2))
    scale = np.sqrt(np.log(1 + std**2 / mean**2))

    return mean, std, ref_length, loc, scale

def calc_oos_elem_length(ref_len, elem_len, loc, scale):

    '''
    Fucntion to change the reference length of the OOS distribution to
    a given elemental length

    Parameters
    ----------
    ref_len : float
        OOS reference length
    elem_len : float
        Elemental  length
    loc_cbf : float
        Location parameter of the lognormal CBF distribution
    scale_cbf : float
        scale paramater of the lognormal CBF distribution

    Returns
    -------
    oos: array
        Array with the OOS critical factors
    pdf_oos_elem_len : array
        Array with the PDF of the lognormal CBF distribution
        with a given elemental length
    cdf_oos_elem_len : array
        Array with the CDF of the lognormal CBF distribution
        with a given elemental length
    '''

    # Ratio between the OOS reference length and the elemental length
    n = ref_len / elem_len

    # OOS Range
    min_oos = lognorm(scale, 0.0, np.exp(loc)).ppf(0.001)
    max_oos = lognorm(scale, 0.0, np.exp(loc)).ppf(0.999)
    oos = np.arange(min_oos, max_oos, 0.0001)

    # OOS PDF and CDF - OOS Ref. Length
    pdf_oos = lognorm(scale, 0.0, np.exp(loc)).pdf(oos)
    cdf_oos = lognorm(scale, 0.0, np.exp(loc)).cdf(oos)

    # OOS PDF and CDF - Elemental Length
    pdf_oos_elem_len = 1/n * pdf_oos * (1 - cdf_oos)**(1 / n - 1)
    cdf_oos_elem_len = 1 - (1 - cdf_oos)**(1 / n)

    return oos, pdf_oos_elem_len, cdf_oos_elem_len

def calc_cbf(route_type, loc_fric, scale_fric, loc_oos, scale_oos, ea, ei, sw, radius, h):

    '''
    Compute the parameters of a lognormal distribution for CBF (straight or bend sections).

    Parameters
    ----------
    section_type : string
        Section type: 'Straight' or 'Bend' or 'Sleeper'
    loc_fric : float
        Location parameter of the lognormal friction distribution
    scale_fric : float
        scale paramater of the lognormal friction distribution
    loc_OOS : float
        Location parameter of the lognormal OOS distribution
    scale_OOS : float
        scale paramater of the lognormal OOS distribution
    ea : float
        Axial stiffness
    ei : float
        Bending stiffness
    sw : float
        Submerged weight
    radius : float
        Bend radius
    h : float
        Sleeper height

    Returns
    -------
    loc_cbf : float
        Location parameter of the lognormal CBF distribution
    scale_cbf : float
        scale paramater of the lognormal CBF distribution
    mean_cbf : float
        Mean of the lognormal CBF distribution
    std_cbf : float
        Standard deviation of the lognormal CBF distribution
    '''

    if route_type == 'Straight':
        s_char = 2.26 * ea**0.25 * ei**0.25 * sw**0.5
        loc_cbf = 0.5 * loc_fric + loc_oos + np.log(s_char)
        scale_cbf = np.sqrt((0.5 * scale_fric)**2 + scale_oos**2)

    elif route_type == 'Bend': #! 'Curve'
        loc_cbf = loc_fric + loc_oos + np.log(int(radius)) + np.log(sw)
        scale_cbf = np.sqrt(scale_fric**2 + scale_oos**2)

    if route_type == 'Sleeper':
        prop_uhb = 4.0 * np.sqrt(ei * sw / h)
        mean_oos= np.exp(loc_oos + scale_oos**2 / 2)
        std_oos = np.sqrt((np.exp(scale_oos**2) - 1) * np.exp(2 * loc_oos + scale_oos**2))
        mean_cbf = mean_oos * prop_uhb
        std_cbf = std_oos * prop_uhb
        loc_cbf = loc_oos + np.log(prop_uhb)
        scale_cbf = scale_oos

    else:
        mean_cbf = np.exp(loc_cbf + scale_cbf**2 / 2)
        std_cbf = np.sqrt((np.exp(scale_cbf**2) - 1) * np.exp(2 * loc_cbf + scale_cbf**2))

    return loc_cbf, scale_cbf, mean_cbf, std_cbf

def calc_hobbs_residual(ea, ei, eaf, sw, mua, mul):

    '''
    Find the buckle length based on known EAF and Hobbs equation
    and calculate the post buckle length and post buckle force.

    Parameters
    ----------
    ea : float
        Axial stiffness
    ei : float
        Bending stiffness
    eaf : float
        Effective axial force
    sw : float
        Submerged weight
    mua : float
        Axial friction factor
    mul : float
        Lateral friction factor

    Returns
    -------
    residual_buckle_force : float
        Residual buckle force
    overall_buckle_length : float
        Residual buckle length
    '''

    # Positive EAF conversion
    eaf = np.abs(eaf)

    # Hobbs equation (mode 3) for the post-buckling force and buckle length (residual)
    def equation_hobbs(l, eaf, ei, mua, ea, mul, sw):
        y = 34.06 * (ei / l**2) + 1.294 * mua * sw * l * (np.sqrt(1 + 1.668e-4 * (
            ea * (mul * sw)**2 * l**5) / (mua * sw * ei**2)) - 1)

        return abs(y - eaf)

    # Set an initial guess for the length of the central lobe of the buckle
    central_lobe_length = 999.0

    # Find the value L_BH to get the minimum value of S_0_Inlet using scipy.optimize.minimize
    result = fsolve(equation_hobbs, central_lobe_length,
                    args=(eaf, ei, mua, ea, mul, sw))
    residual_central_lobe_length = result[0]
    goalseek_control = equation_hobbs(residual_central_lobe_length, eaf, ei, mua, ea, mul, sw)

    # Calculate buckle length and post-buckle force based on L_BH
    residual_buckle_length = 2.588 * residual_central_lobe_length
    residual_buckle_force = 34.06 * ei / residual_central_lobe_length**2

    return residual_buckle_force, residual_buckle_length, goalseek_control
