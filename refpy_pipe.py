'''
Library of pipeline main properties
'''

def calc_inner_diameter(outer_diameter, wall_thickness):
    """
    Calculate the pipe inner diameter

    Parameters
    ----------
    outer_diameter : float
        Outer diameter of the pipe
    wall_thickness : float
        Wall thickness of the pipe

    Returns
    -------
    inner_diameter : float
        Inner diameter of the pipe
    """
    inner_diameter = outer_diameter - 2.0 * wall_thickness
    return inner_diameter

def calc_inner_area(outer_diameter, wall_thickness):
    """
    Calculate the pipe inner area

    Parameters
    ----------
    outer_diameter : float
        Outer diameter of the pipe
    wall_thickness : float
        Wall thickness of the pipe

    Returns
    -------
    inner_area : float
        Inner area of the pipe
    """
    inner_diameter = calc_inner_diameter(outer_diameter, wall_thickness)
    inner_area = np.pi / 4.0 * inner_diameter**2
    return inner_area

def calc_outer_area(outer_diameter):
    """
    Calculate the pipe outer area

    Parameters
    ----------
    outer_diameter : float
        Outer diameter of the pipe

    Returns
    -------
    outer_area : float
        Outer area of the pipe
    """
    outer_area = np.pi / 4.0 * outer_diameter**2
    return outer_area

def calc_steel_area(outer_diameter, wall_thickness):
    """
    Calculate the steel cross-sectional area

    Parameters
    ----------
    outer_diameter : float
        Outer diameter of the pipe
    wall_thickness : float
        Wall thickness of the pipe

    Returns
    -------
    steel_area : float
        Steel cross-sectional area
    """
    outer_area = calc_outer_area(outer_diameter)
    inner_area = calc_inner_area(outer_diameter, wall_thickness)
    steel_area = outer_area - inner_area
    return steel_area

def calc_total_outer_diameter(outer_diameter, coating_thickness):
    """
    Calculate the total outer diameter of steel and coating

    Parameters
    ----------
    outer_diameter : float
        Outer diameter of the steel pipe
    coating_thickness : float
        Coating wall thickness

    Returns
    -------
    total_outer_diameter : float
        Total outer diameter of steel and coating
    """
    total_outer_diameter = outer_diameter + 2.0 * coating_thickness
    return total_outer_diameter

def calc_total_outer_area(outer_diameter, coating_thickness):
    """
    Calculate the outer area of steel and coating

    Parameters
    ----------
    outer_diameter : float
        Outer diameter of the steel pipe
    coating_thickness : float
        Coating wall thickness

    Returns
    -------
    total_outer_area : float
        Outer area of steel and coating
    """
    total_outer_diameter = calc_total_outer_diameter(outer_diameter, coating_thickness)
    total_outer_area = np.pi / 4.0 * total_outer_diameter**2
    return total_outer_area

def calc_coating_area(outer_diameter, coating_thickness):
    """
    Calculate the coating cross-sectional area

    Parameters
    ----------
    outer_diameter : float
        Outer diameter of the steel pipe
    coating_thickness : float
        Coating wall thickness

    Returns
    -------
    coating_area : float
        Coating cross-sectional area
    """
    total_outer_area = calc_total_outer_area(outer_diameter, coating_thickness)
    outer_area = calc_outer_area(outer_diameter)
    coating_area = total_outer_area - outer_area
    return coating_area

def calc_axial_stiffness(outer_diameter, wall_thickness, youngs_modulus):
    """
    Calculate the axial stiffness

    Parameters
    ----------
    outer_diameter : float
        Outer diameter of the pipe
    wall_thickness : float
        Wall thickness of the pipe
    youngs_modulus : float
        Young's modulus of the material

    Returns
    -------
    axial_stiffness : float
        Axial stiffness of the pipe
    """
    steel_area = calc_steel_area(outer_diameter, wall_thickness)
    axial_stiffness = youngs_modulus * steel_area
    return axial_stiffness

def calc_area_moment_inertia(outer_diameter, wall_thickness):
    """
    Calculate the area moment inertia

    Parameters
    ----------
    outer_diameter : float
        Outer diameter of the pipe
    wall_thickness : float
        Wall thickness of the pipe

    Returns
    -------
    area_moment_inertia : float
        Area moment inertia
    """
    inner_diameter = calc_inner_diameter(outer_diameter, wall_thickness)
    area_moment_inertia = np.pi / 64.0 * (outer_diameter**4 - inner_diameter**4)
    return area_moment_inertia

def calc_bending_stiffness(outer_diameter, wall_thickness, youngs_modulus):
    """
    Calculate the bending stiffness

    Parameters
    ----------
    outer_diameter : float
        Outer diameter of the pipe
    wall_thickness : float
        Wall thickness of the pipe
    youngs_modulus : float
        Young's modulus

    Returns
    -------
    bending_stiffness : float
        Bending stiffness
    """
    area_moment_inertia = calc_area_moment_inertia(outer_diameter, wall_thickness)
    bending_stiffness = youngs_modulus * area_moment_inertia
    return bending_stiffness

def calc_eaf(outer_diameter, wall_thickness, young_modulus, poisson, expansion_coefficient,
             temperature_installation, pressure, temperature):
    """
    Calculate the effective axial force

    Parameters
    ----------
    outer_diameter : float
        Outer diameter of the pipe
    wall_thickness : float
        Wall thickness of the pipe
    young_modulus : float
        Young's modulus
    poisson : float
        Poisson's ratio
    expansion_coefficient : float
        Thermal expansion coefficient
    temperature_installation : float
        Installation temperature
    pressure : float
        Loadcase pressure
    temperature : float
        Loadcase temperature

    Returns
    -------
    effective_axial_force : float
        The effective axial force
    """
    inner_area = calc_inner_area(outer_diameter, wall_thickness)
    steel_area = calc_steel_area(outer_diameter, wall_thickness)
    effective_axial_force = - (1 - 2 * poisson) * pressure * inner_area - \
        young_modulus * expansion_coefficient * (temperature - temperature_installation) \
            * steel_area
    return effective_axial_force
