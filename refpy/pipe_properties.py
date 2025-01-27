'''
Library of pipeline main properties
'''

import numpy as np

def inner_diameter(outer_diameter: np.ndarray, wall_thickness: np.ndarray) -> np.ndarray:
    """
    Calculate the pipe inner diameter.

    Parameters
    ----------
    outer_diameter : np.ndarray
        Outer diameter of the pipe.
    wall_thickness : np.ndarray
        Wall thickness of the pipe.

    Returns
    -------
    inner_diameter : np.ndarray
        Inner diameter of the pipe.

    Examples
    --------
    >>> inner_diameter(np.array([0.2731, 0.3239]), np.array([0.0127, 0.0159]))
    array([0.2477, 0.2921])
    """
    return outer_diameter - 2.0 * wall_thickness

def inner_area(outer_diameter: np.ndarray, wall_thickness: np.ndarray) -> np.ndarray:
    """
    Calculate the pipe inner area

    Parameters
    ----------
    outer_diameter : np.ndarray
        Outer diameter of the pipe
    wall_thickness : np.ndarray
        Wall thickness of the pipe

    Returns
    -------
    inner_area : np.ndarray
        Inner area of the pipe

    See Also
    --------
    inner_area : Calculate the pipe inner area.

    Examples
    --------
    >>> inner_area(np.array([0.2731, 0.3239]), np.array([0.0127, 0.0159]))
    array([0.04818833, 0.06701206])
    """
    inner_diameter_val = inner_diameter(outer_diameter, wall_thickness)
    return np.pi / 4.0 * inner_diameter_val**2

def outer_area(outer_diameter: np.ndarray) -> np.ndarray:
    """
    Calculate the pipe outer area

    Parameters
    ----------
    outer_diameter : np.ndarray
        Outer diameter of the pipe

    Returns
    -------
    outer_area : np.ndarray
        Outer area of the pipe

    Examples
    --------
    >>> outer_area(np.array([0.2731, 0.3239]))
    array([0.05857783, 0.08239707])
    """
    return np.pi / 4.0 * outer_diameter**2

def steel_area(outer_diameter: np.ndarray, wall_thickness: np.ndarray) -> np.ndarray:
    """
    Calculate the steel cross-sectional area.

    Parameters
    ----------
    outer_diameter : np.ndarray
        Outer diameter of the pipe.
    wall_thickness : np.ndarray
        Wall thickness of the pipe.

    Returns
    -------
    steel_area : np.ndarray
        Steel cross-sectional area.

    See Also
    --------
    inner_area : Calculate the pipe inner area.
    outer_area : Calculate the pipe outer area.

    Examples
    --------
    >>> steel_area(np.array([0.2731, 0.3239]), np.array([0.0127, 0.0159]))
    array([0.0103895 , 0.01538501])
    """
    outer_area_val = outer_area(outer_diameter)
    inner_area_val = inner_area(outer_diameter, wall_thickness)
    return outer_area_val - inner_area_val

def total_outer_diameter(outer_diameter: np.ndarray, coating_thickness: np.ndarray) -> np.ndarray:
    """
    Calculate the total outer diameter of steel and coating.

    Parameters
    ----------
    outer_diameter : np.ndarray
        Outer diameter of the steel pipe.
    coating_thickness : np.ndarray
        Coating wall thickness.

    Returns
    -------
    total_outer_diameter : np.ndarray
        Total outer diameter of steel and coating.

    Examples
    --------
    >>> total_outer_diameter(np.array([0.2731, 0.3239]), np.array([0.003, 0.003]))
    array([0.2791, 0.3299])
    """
    return outer_diameter + 2.0 * coating_thickness

def total_outer_area(outer_diameter: np.ndarray, coating_thickness: np.ndarray) -> np.ndarray:
    """
    Calculate the outer area of steel and coating.

    Parameters
    ----------
    outer_diameter : np.ndarray
        Outer diameter of the steel pipe.
    coating_thickness : np.ndarray
        Coating wall thickness.

    Returns
    -------
    total_outer_area : np.ndarray
        Outer area of steel and coating.

    Examples
    --------
    >>> total_outer_area(np.array([0.2731, 0.3239]), np.array([0.003, 0.003]))
    array([0.06118001, 0.08547803])
    """
    total_outer_dia = total_outer_diameter(outer_diameter, coating_thickness)
    return np.pi / 4.0 * total_outer_dia**2

def coating_area(outer_diameter: np.ndarray, coating_thickness: np.ndarray) -> np.ndarray:
    """
    Calculate the coating cross-sectional area.

    Parameters
    ----------
    outer_diameter : np.ndarray
        Outer diameter of the steel pipe.
    coating_thickness : np.ndarray
        Coating wall thickness.

    Returns
    -------
    coating_area : np.ndarray
        Coating cross-sectional area.

    See Also
    --------
    total_outer_area : Calculate the outer area of steel and coating.
    outer_area_val : Calculate the pipe outer area
    
    Examples
    --------
    >>> coating_area(np.array([0.2731, 0.3239]), np.array([0.003, 0.003]))
    array([0.00260218, 0.00308096])
    """
    total_outer_area_val = total_outer_area(outer_diameter, coating_thickness)
    outer_area_val = outer_area(outer_diameter)
    return total_outer_area_val - outer_area_val

def axial_stiffness(outer_diameter: np.ndarray, wall_thickness: np.ndarray,
                    youngs_modulus: np.ndarray) -> np.ndarray:
    """
    Calculate the axial stiffness.

    Parameters
    ----------
    outer_diameter : np.ndarray
        Outer diameter of the pipe.
    wall_thickness : np.ndarray
        Wall thickness of the pipe.
    youngs_modulus : np.ndarray
        Young's modulus of the material.

    Returns
    -------
    axial_stiffness : np.ndarray
        Axial stiffness of the pipe.

    See Also
    --------
    steel_area : Calculate the steel cross-sectional area.

    Examples
    --------
    >>> axial_stiffness(np.array([0.2731, 0.3239]), np.array([0.0127, 0.0159]), \
        np.array([207.0e+09, 207.0e+09]))
    array([2.15062613e+09, 3.18469656e+09])
    """
    steel_area_val = steel_area(outer_diameter, wall_thickness)
    return youngs_modulus * steel_area_val

def area_moment_inertia(outer_diameter: np.ndarray, wall_thickness: np.ndarray) -> np.ndarray:
    """
    Calculate the area moment inertia.

    Parameters
    ----------
    outer_diameter : np.ndarray
        Outer diameter of the pipe.
    wall_thickness : np.ndarray
        Wall thickness of the pipe.

    Returns
    -------
    area_moment_inertia : np.ndarray
        Area moment inertia.

    See Also
    --------
    inner_diameter : Calculate the pipe inner diameter.

    Examples
    --------
    >>> area_moment_inertia(np.array([0.2731, 0.3239]), np.array([0.0127, 0.0159]))
    array([8.82710601e-05, 1.82921605e-04])
    """
    inner_dia = inner_diameter(outer_diameter, wall_thickness)
    return np.pi / 64.0 * (outer_diameter**4 - inner_dia**4)

def bending_stiffness(outer_diameter: np.ndarray, wall_thickness: np.ndarray,
                      youngs_modulus: np.ndarray) -> np.ndarray:
    """
    Calculate the bending stiffness.

    Parameters
    ----------
    outer_diameter : np.ndarray
        Outer diameter of the pipe.
    wall_thickness : np.ndarray
        Wall thickness of the pipe.
    youngs_modulus : np.ndarray
        Young's modulus.

    Returns
    -------
    bending_stiffness : np.ndarray
        Bending stiffness.

    See Also
    --------
    area_moment_inertia : Calculate the area moment inertia.

    Examples
    --------
    >>> bending_stiffness(np.array([0.2731, 0.3239]), np.array([0.0127, 0.0159]), \
        np.array([207.0e+09, 207.0e+09]))
    array([18272109.437121  , 37864772.21769765])
    """
    area_moment_inertia_val = area_moment_inertia(outer_diameter, wall_thickness)
    return youngs_modulus * area_moment_inertia_val
