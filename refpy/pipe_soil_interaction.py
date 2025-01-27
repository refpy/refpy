"""
Module for calculating the capacity of soils using various models.
"""

import numpy as np

def downward_undrained_model1(diameter, surface_roughness, density,
                              surcharge_on_seabed, burial_depth):
    """
    Calculate the vertical bearing capacity of soil using the downward undrained model.

    Parameters
    ----------
    diameter : float
        Diameter of the pipe.
    surface_roughness : str
        Surface roughness of the pipe ('Smooth' or 'Rough').
    density : float
        Density of the soil.
    surcharge_on_seabed : float
        Stress or surcharge at the elevation of the seabed.
    burial_depth : float
        Depth of burial.

    Returns
    -------
    depth : np.ndarray
        Array of depth values.
    vertical_bearing_capacity : np.ndarray
        Array of vertical bearing capacity values.
    
    Examples
    --------
    >>> downward_undrained_model1(0.2731, 'Smooth', 1000.0, 100, 1.0)
    (array([0.   , 0.005, ...0.995, 1.   ]), array([   0.        ,  920.50185636, ...7141.92857341, 7163.61163085]))
    """
    # Create an array of depth values
    depth = np.linspace(0.005, 1, 200)

    # Initialize an empty array for width
    width = np.empty(depth.size)
    for i in range(depth.size):
        if depth[i] < diameter / 2:
            width[i] = 2 * np.sqrt(diameter * depth[i] - depth[i] ** 2)
        else:
            width[i] = diameter

    # Initialize an empty array for undrained shear strength
    undrained_shear_strength = np.empty(depth.size)
    for i in range(depth.size):
        if depth[i] < diameter / 2 * (1 - np.sqrt(2) / 2.):
            undrained_shear_strength[i] = 0.
        else:
            undrained_shear_strength[i] = depth[i] + diameter / 2 * (np.sqrt(2) - 1) - width[i] / 2

    # Calculate the initial shear strength
    initial_shear_strength = surcharge_on_seabed + density * undrained_shear_strength

    # Calculate the friction factor
    friction_factor = np.divide(np.multiply(density, width), initial_shear_strength)
    friction_factor_reference = np.linspace(0, 16, 17)

    # Determine the friction factor based on surface roughness
    if surface_roughness == 'Smooth':
        friction_factor_values = np.array([1.00, 1.12, 1.19, 1.24, 1.28, 1.31, 1.34, 1.36, 1.38,
                                           1.39, 1.40, 1.41, 1.42, 1.43, 1.44, 1.445, 1.45])
    else:
        friction_factor_values = np.array([1.00, 1.23, 1.36, 1.44, 1.50, 1.55, 1.59, 1.61, 1.64,
                                           1.66, 1.67, 1.69, 1.70, 1.71, 1.72, 1.730, 1.74])

    # Interpolate the friction factor
    interpolated_friction_factor = np.interp(
        friction_factor, friction_factor_reference, friction_factor_values)

    # Calculate the vertical bearing capacity
    vertical_bearing_capacity = interpolated_friction_factor * (
        5.14 * initial_shear_strength + density * width / 4 + density * 9.807 * burial_depth) * width

    # Calculate the average shear strength
    average_shear_strength_1 = (surcharge_on_seabed + initial_shear_strength) / 2.
    average_shear_strength_2 = vertical_bearing_capacity / width / 5.14

    # Calculate the depth correction factor
    depth_correction_factor = 0.3 * average_shear_strength_1 / average_shear_strength_2 * \
        np.arctan2( undrained_shear_strength, width)

    # Initialize an empty array for the bearing capacity modifier
    bearing_capacity_modifier = np.empty(depth.size)
    for i in range(depth.size):
        if depth[i] < diameter / 2:
            bearing_capacity_modifier[i] = np.arcsin(width[i] / diameter) * diameter ** 2 / 4 + \
                width[i] * diameter / 4 * np.cos(np.arcsin(width[i] / diameter))
        else:
            bearing_capacity_modifier[i] = np.pi * diameter ** 2 / 8 +\
                diameter * (depth[i] - diameter / 2)

    vertical_bearing_capacity = vertical_bearing_capacity * (1 + depth_correction_factor) +\
        density * 9.807 * bearing_capacity_modifier
    depth = np.append(0, depth)
    vertical_bearing_capacity = np.append(0, vertical_bearing_capacity)
    return depth, vertical_bearing_capacity
