'''
Library of DNV limit states
'''

import numpy as np
from refpy.pipe_properties import wall_thickness_corroded

def temperature_derating_stress(material: np.ndarray,
                                temperature: np.ndarray) -> np.ndarray:
    '''
    Calculate the temperature derating stress of a material.

    Parameters
    ----------
    material : numpy.ndarray
        The types of materials: 1 for 'CMn' or '13CR' & 2 for '22Cr' or '25CR'.
    temperature : numpy.ndarray
        The temperatures at which to calculate the derating stress.

    Returns
    -------
    numpy.ndarray
        The derating stress values for the given materials and temperatures.

    Raises
    ------
    ValueError
        If a material is not supported.

    Examples
    --------
    >>> materials = np.array([1, 1, 2, 2])
    >>> temperatures = np.array([80.0, 110.0, 80.0, 110.0])
    >>> temperature_derating_stress(materials, temperatures)
    array([18000000., 34000000., 70000000., 95000000.])
    '''
    derating_array = np.empty(0)
    for mat, temp in zip(material, temperature):
        if mat == 1:
            derating_value = np.interp(temp,
                                       [50.0, 100.0, 200.0],
                                       [0.0, 30.0E+06, 70.0E+06])
        elif mat == 2:
            derating_value = np.interp(temp,
                                       [20.0, 50.0, 100.0, 200.0],
                                       [0.0, 40.0E+06, 90.0E+06, 140.0E+06])
        else:
            raise ValueError('Material not supported')
        derating_array = np.append(derating_array, derating_value)
    return derating_array

def yield_stress(material: np.ndarray, smys: np.ndarray, temperature: np.ndarray,
                 material_strength_factor: np.ndarray) -> np.ndarray:
    '''
    Calculate the yield stress of a material.

    Parameters
    ----------
    material : numpy.ndarray
        The types of materials: 1 for 'CMn' or '13CR' & 2 for '22Cr' or '25CR'.
    smys : numpy.ndarray
        The specified minimum yield strengths of the materials.
    temperature : numpy.ndarray
        The temperatures at which to calculate the yield stress.
    material_strength_factor : np.ndarray
        The material strength factor.

    Returns
    -------
    numpy.ndarray
        The yield stress values of the materials at the given temperatures
        and strength factor.

    See Also
    --------
    temperature_derating_stress : Calculate the temperature derating stress of a material.

    Examples
    --------
    >>> materials = np.array([1, 1, 2, 2])
    >>> smys = np.array([450.0E+06, 450.0E+06, 550.0E+06, 550.0E+06])
    >>> temperatures = np.array([80.0, 110.0, 80.0, 110.0])
    >>> material_strength_factor = np.array([0.96, 0.96, 0.96, 0.96])
    >>> yield_stress(materials, smys, temperatures, material_strength_factor)
    array([4.1472e+08, 3.9936e+08, 4.6080e+08, 4.3680e+08])
    '''
    derating_value = temperature_derating_stress(material, temperature)
    return (smys - derating_value) * material_strength_factor

def tensile_strength(material: np.ndarray, smts: np.ndarray, temperature: np.ndarray,
                     material_strength_factor: float) -> np.ndarray:
    '''
    Calculate the tensile strength of a material.

    Parameters
    ----------
    material : numpy.ndarray
        The types of materials: 1 for 'CMn' or '13CR' & 2 for '22Cr' or '25CR'.
    smts : numpy.ndarray
        The specified minimum tensile strengths of the materials.
    temperature : numpy.ndarray
        The temperatures at which to calculate the tensile strength.
    material_strength_factor : numpy.ndarray
        The material strength factor.

    Returns
    -------
    numpy.ndarray
        The tensile strength values of the materials at the given temperatures
        and strength factor.

    See Also
    --------
    temperature_derating_stress : Calculate the temperature derating stress of a material.

    Examples
    --------
    >>> materials = np.array([1, 1, 2, 2])
    >>> smts = np.array([485.0E+06, 485.0E+06, 590.0E+06, 590.0E+06])
    >>> temperatures = np.array([80.0, 110.0, 80.0, 110.0])
    >>> material_strength_factor = np.array([0.96, 0.96, 0.96, 0.96])
    >>> tensile_strength(materials, smts, temperatures, material_strength_factor)
    array([4.4832e+08, 4.3296e+08, 4.9920e+08, 4.7520e+08])
    '''
    derating_value = temperature_derating_stress(material, temperature)
    return (smts - derating_value) * material_strength_factor

def characteristic_material_burst_strength(material: np.ndarray, smys: np.ndarray,
                                           smts: np.ndarray, temperature: np.ndarray,
                                           material_strength_factor: np.ndarray) -> np.ndarray:
    '''
    Calculate the characteristic material burst strength.

    Parameters
    ----------
    material : numpy.ndarray
        The types of materials: 1 for 'CMn' or '13CR' & 2 for '22Cr' or '25CR'.
    smys : numpy.ndarray
        The specified minimum yield strengths of the materials.
    smts : numpy.ndarray
        The specified minimum tensile strengths of the materials.
    temperature : numpy.ndarray
        The temperatures at which to calculate the burst strength.
    material_strength_factor : numpy.ndarray
        The material strength factor.

    Returns
    -------
    numpy.ndarray
        The characteristic material burst strength values of the materials
        at the given temperatures and strength factor.

    See Also
    --------
    yield_stress : Calculate the yield stress of a material.
    tensile_strength : Calculate the tensile strength of a material.

    Examples
    --------
    >>> materials = np.array([1, 1, 2, 2])
    >>> smys = np.array([450.0E+06, 450.0E+06, 550.0E+06, 550.0E+06])
    >>> smts = np.array([600.0E+06, 600.0E+06, 700.0E+06, 700.0E+06])
    >>> temperatures = np.array([80.0, 110.0, 80.0, 110.0])
    >>> material_strength_factor = np.array([0.96, 0.96, 0.96, 0.96])
    >>> characteristic_material_burst_strength(materials, smys, smts, temperatures, material_strength_factor)
    array([4.1472e+08, 3.9936e+08, 4.6080e+08, 4.3680e+08])
    '''
    yield_stress_value = yield_stress(material, smys, temperature,
                                      material_strength_factor)
    tensile_strength_value = tensile_strength(material, smts, temperature,
                                              material_strength_factor)
    return np.minimum(yield_stress_value, tensile_strength_value / 1.15)

def burst_pressure(outer_diameter: np.ndarray, wall_thickness: np.ndarray,
                   corrosion_allowance: np.ndarray, material: np.ndarray,
                   smys: np.ndarray, smts: np.ndarray, temperature: np.ndarray,
                   material_strength_factor: np.ndarray):
    '''
    Calculate the burst pressure of a pipeline.

    Parameters
    ----------
    outer_diameter : float
        The outer diameter of the pipeline.
    wall_thickness : float
        The wall thickness of the pipeline.
    corrosion_allowance : float
        The corrosion allowance to be subtracted from the wall thickness.
    material : numpy.ndarray
        The types of materials: 1 for 'CMn' or '13CR' & 2 for '22Cr' or '25CR'.
    smys : numpy.ndarray
        The specified minimum yield strengths of the materials.
    smts : numpy.ndarray
        The specified minimum tensile strengths of the materials.
    temperature : numpy.ndarray
        The temperatures at which to calculate the burst pressure.
    material_strength_factor : float
        The material strength factor.

    Returns
    -------
    numpy.ndarray
        The burst pressure values of the pipeline.

    Raises
    ------
    ValueError
        If a material is not supported.

    See Also
    --------
    characteristic_material_burst_strength : Calculate the characteristic material burst strength.
    wall_thickness_corroded : Calculate the corroded wall thickness.
    
    >>> outer_diameter = np.array([0.2731, 0.3239, 0.2731, 0.3239])
    >>> wall_thickness = np.array([0.0127, 0.0159, 0.0127, 0.0159])
    >>> corrosion_allowance = np.array([0.003, 0.003, 0.003, 0.003])    
    >>> materials = np.array([1, 1, 2, 2])
    >>> smys = np.array([450.0E+06, 450.0E+06, 550.0E+06, 550.0E+06])
    >>> smts = np.array([600.0E+06, 600.0E+06, 700.0E+06, 700.0E+06])
    >>> temperatures = np.array([80.0, 110.0, 80.0, 110.0])
    >>> material_strength_factor = np.array([0.96, 0.96, 0.96, 0.96])
    >>> burst_pressure(outer_diameter, wall_thickness, corrosion_allowance, materials, smys, smts, temperatures, material_strength_factor)
    array([35270393.70222808, 38255444.18258572, 39189326.33580898,
           41841892.07470313])
    '''
    fcb = characteristic_material_burst_strength(material, smys, smts,
                                                 temperature, material_strength_factor)
    wtcorr = wall_thickness_corroded(wall_thickness, corrosion_allowance)
    return (2.0 * wtcorr) / (outer_diameter - wtcorr) * fcb * 2.0 / np.sqrt(3.0)
