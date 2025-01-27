'''
Library of limit states
'''

import numpy as np

def lcc(od: float, wt: float, ca: float, linepipe_type: str, temp: np.ndarray, pi: np.ndarray,
        pe: np.ndarray, eaf: np.ndarray, bm: np.ndarray, safety_class: str) -> np.ndarray:
    """
    Calculate the limit state function for local buckling.

    Parameters
    ----------
    od : float
        Outer diameter of the pipe.
    wt : float
        Wall thickness of the pipe.
    ca : float
        Corrosion allowance.
    linepipe_type : str
        Type of the linepipe ('smls' or 'sawl').
    temp : np.ndarray
        Temperature array.
    pi : np.ndarray
        Internal pressure array.
    pe : np.ndarray
        External pressure array.
    eaf : np.ndarray
        Effective axial force array.
    bm : np.ndarray
        Bending moment array.
    safety_class : str
        Safety class ('Low', 'Medium', or 'High').

    Returns
    -------
    lcc : np.ndarray
        Limit state function for local buckling.
    """
    # Define partial safety factors
    gamma_f = 1.1
    gamma_c = 1.0
    gamma_m = 1.15

    # Define safety class factor
    if safety_class == 'Low':
        gamma_sclb = 1.04
    elif safety_class == 'Medium':
        gamma_sclb = 1.14
    else:
        gamma_sclb = 0.0

    # Define fabrication factor based on linepipe type
    if linepipe_type == 'smls':
        alpha_fab = 1.00
    elif linepipe_type == 'sawl':
        alpha_fab = 0.93

    # Set effective axial force to zero if positive
    eaf[eaf > 0.0] = 0.0
    ssd = gamma_f * gamma_c * eaf  # Calculate design effective axial force
    msd = gamma_f * gamma_c * np.abs(bm)  # Calculate design bending moment

    # Calculate temperature derating
    temp_derating = np.where(temp < 50, 0.0,
                             np.where(temp < 100, 0.6E+06 * (temp - 50.0),
                                      30E+06 + 0.4E+06 * (temp - 100.0)))
    smys = 450.0E+06 - temp_derating  # Calculate specified minimum yield strength
    smts = 535.0E+06 - temp_derating  # Calculate specified minimum tensile strength

    # Calculate various parameters for local buckling
    t2 = wt - ca
    fcb = np.minimum(smys, smts / 1.15)
    sp = smys * np.pi * (od - t2) * t2
    mp = smys * t2 * (od - t2)**2
    pb = 2 * t2 / (od - t2) * fcb * 2 / np.sqrt(3)
    pel = 2 * 207.0E+9 * (t2 / od)**3 / (1 - 0.3**2)
    pp = smys * alpha_fab * 2 * t2 / od
    oval = 0.01

    # Initialize collapse pressure array
    pc = np.empty(0)
    beta = (60 - od / t2) / 90
    alpha_c = (1 - beta) + beta * (smts / smys)

    # Calculate collapse pressure using bisection method
    for i in range(fcb.size):
        pc_min = 0.0
        pc_max = 1000.0E+5
        while np.amax(np.abs(pc_max - pc_min)) > 0.1:
            pc_mean = 0.5 * (pc_min + pc_max)
            f1 = (pc_min - pel) * (pc_min**2 - pp[i]**2) - pc_min * pel * pp[i] * oval * od / t2
            f2 = (pc_max - pel) * (pc_max**2 - pp[i]**2) - pc_max * pel * pp[i] * oval * od / t2
            f3 = (pc_mean - pel) * (pc_mean**2 - pp[i]**2) - pc_mean * pel * pp[i] * oval * od / t2
            if f1 * f3 > 0.0:
                pc_min = pc_mean
            else:
                pc_max = pc_mean
        pc = np.append(pc, pc_mean)

    # Calculate gamma_p factor
    gamma_p = 1 - 3 * beta * (1 - (pi - pe) / pb)
    gamma_p[(pi - pe) / pb <= 2 / 3] = 1 - beta

    # Calculate the limit state function for local buckling
    lcc = ((gamma_m * gamma_sclb * msd / (alpha_c * mp) + 
            (gamma_m * gamma_sclb * ssd / (alpha_c * sp))**2)**2 + 
           (gamma_p * (pi - pe) / (alpha_c * pb))**2)

    # Adjust lcc values based on external pressure
    for i in range(lcc.size):
        if pe[i] > pi[i]:
            lcc[i] = ((gamma_m * gamma_sclb * msd[i] / (alpha_c[i] * mp[i]) + 
                       (gamma_m * gamma_sclb * ssd[i] / (alpha_c[i] * sp[i]))**2)**2 + 
                      (gamma_m * gamma_sclb * pe[i] / pc[i])**2)

    return lcc  # Return the limit state function for local buckling
